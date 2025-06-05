import os
import glob
import threading
import numpy as np
from astropy.io import fits
import time
import gc

import cupy as cp
from .plotting import *
from . import utils as prep_utils
from .cupy_calc import *

from ..utils import get_header, flatten, time_diff_in_seconds
from ..path import PathHandler
from ..config import PreprocConfiguration
from ..services.setup import BaseSetup
from ..const import HEADER_KEY_MAP


class Preprocess(BaseSetup):
    """
    Assumes homogeneous BIAS, DARK, FLAT, SCI frames as input
    taken on the same date with the same
    unit, n_binning, gain, and cameras.
    """

    def __init__(
        self,
        config,
        queue=False,
        logger=None,
        overwrite=False,
        master_frame_only=False,
        **kwargs,
    ):
        # Load Configuration
        super().__init__(config, logger, queue)
        self._flag_name = "preprocess"

    @property
    def sequential_task(self):
        return [
            (1, "load_mbdf", False),
            (2, "data_reduction", True),
            (3, "save_processed_files", False),
            (4, "flagging", False),
        ]

    @classmethod
    def from_list(cls, images):
        image_list = []
        for image in images:
            path = Path(image)
            if not path.is_file():
                print("The file does not exist.")
                return None
            image_list.append(path.parts[-1])
        working_dir = str(path.parent.absolute())
        config = Configuration.base_config(working_dir)
        config.file.processed_files = image_list
        return cls(config=config)

    def run(self, use_eclaire=True):

        self.logger.info("-" * 80)
        self.logger.info(f"Start preprocessing for {self.config.name}")

        self.load_mbdf()

        self.calibrate(use_eclaire=use_eclaire)

        self.config.flag.preprocess = True
        self.logger.info(f"Preprocessing Done for {self.config.name}")
        MemoryMonitor.cleanup_memory()
        self.logger.debug(MemoryMonitor.log_memory_usage)

    def calibrate(self, use_eclaire=True):
        if self.queue:
            if use_eclaire:
                self.queue.add_task(
                    self._calibrate_image_eclaire,
                    priority=Priority.MEDIUM,
                    task_name=f"{self.config.name} - Calibration",
                    gpu=True,
                )
            else:
                self.queue.add_task(
                    self._calibrate_image_cupy,
                    priority=Priority.MEDIUM,
                    task_name=f"{self.config.name} - Calibration",
                    gpu=True,
                )
        else:
            return self.config.preprocess.device

    def load_masterframe(self, device_id=None):
        """
        no raw calib -> fetch from the library of pre-generated master frames
        raw calibs exist
            -> if output master exists, just fetch.
            -> if overwrite, always generate and overwrite

        If there's nothing to fetch, the code will fail.
        """
        device_id = self.get_device_id(device_id)

        self.logger.info(f"Generating masterframes for group {self._current_group+1} in GPU device {device_id}")
        st = time.time()

        for dtype in ["bias", "dark", "flat"]:

            input_data = getattr(self, f"{dtype}_input")
            output_data = getattr(self, f"{dtype}_output")
            self.logger.debug(f"{dtype}_input: {input_data}")
            self.logger.debug(f"{dtype}_output: {output_data}")

            if input_data:  # if the list is not empty
                if not os.path.exists(output_data) or self.overwrite:
                    self._generate_masterframe(dtype, device_id)
                else:
                    self._fetch_masterframe(output_data, dtype, device_id)
            else:
                self._fetch_masterframe(output_data, dtype, device_id)

        self.logger.info(f"Generation/Loading of masterframes completed in {time_diff_in_seconds(st)} seconds")

    def _generate_masterframe(self, dtype, device_id):
        """Generate & Save masterframe and sigma image"""
        self.logger.info(f"Generating master {dtype}")
        input_data = getattr(self, f"{dtype}_input")
        header = self.get_header(dtype)

        if dtype == "bias":
            median, std = combine_images_with_cupy(input_data, device_id=device_id)
            self.bias_data = median

        elif dtype == "dark":
            median, std = combine_images_with_cupy(input_data, subtract=self.bias_data, device_id=device_id)
            self.dark_data = median
            self.dark_exptime = header[HEADER_KEY_MAP["exptime"]]
            n_sigma = self.config.preprocess.n_sigma
            self.generate_bpmask(median, n_sigma=n_sigma, header=header, device_id=device_id)

        elif dtype == "flat":
            dark_scale = self._calc_dark_scale(header[HEADER_KEY_MAP["exptime"]], self.dark_exptime)
            median, std = combine_images_with_cupy(
                input_data, subtract=(self.bias_data + self.dark_data * dark_scale), norm=True, device_id=device_id
            )
            self.config.preprocess.mflat_file = search_with_date_offsets(
                link_to_file(self.config.preprocess.mflat_link), future=True
            )

            # if the found is the same as link: abort processing
            if (
                self.config.preprocess.mbias_file == prep_utils.read_link(self.config.preprocess.mbias_link)
                and self.config.preprocess.mdark_file == prep_utils.read_link(self.config.preprocess.mdark_link)
                and self.config.preprocess.mflat_file == prep_utils.read_link(self.config.preprocess.mflat_link)
            ):
                self.logger.info("All newly found master frames are the same as existing links")  # fmt: skip
                raise ValueError("No new closest master calibration frames. Aborting...")  # fmt: skip
        elif selection == "custom":
            # use self.config.preprocess.m????_file
            if self.config.preprocess.mbias_file is None:
                self.logger.error("No 'mbias_file' given although 'masterframe' is 'custom'")  # fmt:skip
                raise ValueError("mbias_file must be specified when masterframe is 'custom'.")  # fmt:skip

            if self.config.preprocess.mdark_file is None:
                self.logger.error("No 'mdark_file' given although 'masterframe' is 'custom'")  # fmt:skip
                raise ValueError("mdark_file must be specified when masterframe is 'custom'.")  # fmt:skip

            if self.config.preprocess.mflat_file is None:
                self.logger.error("No 'mflat_file' given although 'masterframe' is 'custom'")  # fmt:skip
                raise ValueError("mflat_file must be specified when masterframe is 'custom'.")  # fmt:skip

        # full paths to sigma maps
        self.config.preprocess.biassig_file = self.config.preprocess.mbias_file.replace("bias", "biassig")  # fmt:skip
        self.config.preprocess.darksig_file = self.config.preprocess.mdark_file.replace("dark", "darksig")  # fmt:skip
        self.config.preprocess.flatsig_file = self.config.preprocess.mflat_file.replace("flat", "flatsig")  # fmt:skip
        self.config.preprocess.bpmask_file = self.config.preprocess.darksig_file.replace("darksig", "bpmask")  # fmt:skip

        self.files = {
            "bias": self.config.preprocess.mbias_file,
            "dark": self.config.preprocess.mdark_file,
            "flat": self.config.preprocess.mflat_file,
            "raw": self.config.file.raw_files,
            "processed": self.config.file.processed_files,
        }

    def data_reduction(self, use_eclaire=True):

        if use_eclaire:
            ofc = ec.FitsContainer(self.files["raw"])
            with prep_utils.load_data_gpu(self.files["bias"]) as mbias, \
                prep_utils.load_data_gpu(self.files["dark"]) as mdark, \
                prep_utils.load_data_gpu(self.files["flat"]) as mflat:  # fmt:skip
                ofc.data = ec.reduction(ofc.data, mbias, mdark, mflat)
            self._temp_data = ofc.data.get()
            del ofc

    def save_processed_files(self, make_plots=True):
        self.files = {
            "bias": self.config.preprocess.mbias_file,
            "dark": self.config.preprocess.mdark_file,
            "flat": self.config.preprocess.mflat_file,
            "raw": self.config.file.raw_files,
            "processed": self.config.file.processed_files,
        }
        for idx, (raw_file, out_file) in enumerate(zip(self.files["raw"], self.files["processed"])):
            header = fits.getheader(raw_file)
            header["SATURATE"] = prep_utils.get_saturation_level(
                header, self.bias_output, self.dark_output, self.flat_output
            )
            header = prep_utils.write_IMCMB_to_header(
                header, [self.bias_output, self.dark_output, self.flat_output, raw_file]
            )
            header = prep_utils.add_padding(header, n_head_blocks, copy_header=True)
            
            os.makedirs(os.path.dirname(processed_file), exist_ok=True)
            fits.writeto(
                output_path,
                data=cp.asnumpy(self._temp_data[idx]),
                header=header,
                overwrite=True,
            )
            if make_plots:
                self.make_plots(raw_file, output_path)
        del self._temp_data

    # def _calibrate_image_eclaire(self, make_plots=True):
    #     mbias_file = self.config.preprocess.mbias_file
    #     mdark_file = self.config.preprocess.mdark_file
    #     mflat_file = self.config.preprocess.mflat_file
    #     raw_files = self.config.file.raw_files
    #     processed_files = self.config.file.processed_files

    #     self.logger.debug(f"Calibrating {len(raw_files)} SCI frames: {self.config.obs.filter}, {self.config.obs.exposure}s")  # fmt:skip
    #     # self.logger.debug(f"Current memory usage: {MemoryMonitor.log_memory_usage}")
    #     # batch processing
    #     BATCH_SIZE = 30  # 10
    #     for i in range(0, len(raw_files), BATCH_SIZE):
    #         batch_raw = raw_files[i : min(i + BATCH_SIZE, len(raw_files))]
    #         processed_batch = processed_files[i : min(i + BATCH_SIZE, len(raw_files))]

    #         ofc = ec.FitsContainer(batch_raw)

    #         # 	Reduction
    #         with prep_utils.load_data_gpu(mbias_file) as mbias, \
    #              prep_utils.load_data_gpu(mdark_file) as mdark, \
    #              prep_utils.load_data_gpu(mflat_file) as mflat:  # fmt:skip
    #             ofc.data = ec.reduction(ofc.data, mbias, mdark, mflat)

    #         # Save each slice of the cube as a separate 2D file
    #         for idx in range(len(batch_raw)):
    #             header = fits.getheader(raw_files[i + idx])
    #             header["SATURATE"] = prep_utils.get_saturation_level(
    #                 header, mbias_file, mdark_file, mflat_file
    #             )
    #             header = prep_utils.write_IMCMB_to_header(
    #                 header,
    #                 [mbias_file, mdark_file, mflat_file, raw_files[i + idx]],
    #             )
    #             n_head_blocks = self.config.settings.header_pad
    #             add_padding(header, n_head_blocks, copy_header=False)

    #             path = self.config.path.path_processed
    #             output_path = os.path.join(path, processed_batch[idx])
    #             fits.writeto(
    #                 output_path,
    #                 data=cp.asnumpy(ofc.data[idx]),
    #                 header=header,
    #                 overwrite=True,
    #             )
    #             if make_plots:
    #                 self.make_plots(raw_files[i + idx], output_path)
    #         self.logger.debug(
    #             f"Current memory usage after {i}-th batch: {MemoryMonitor.log_memory_usage}"
    #         )

    #     self.logger.debug(
    #         f"Current memory usage before cleanup: {MemoryMonitor.log_memory_usage}"
    #     )
    #     MemoryMonitor.cleanup_memory()
    #     self.logger.debug(
    #         f"Current memory usage after cleanup: {MemoryMonitor.log_memory_usage}"
    #     )

    # def _calibrate_image_cupy(self):
    #     mbias_file = self.config.preprocess.mbias_file
    #     mdark_file = self.config.preprocess.mdark_file
    #     mflat_file = self.config.preprocess.mflat_file
    #     pass

    def make_plots(self, raw_file, output_file):
        path = Path(output_file)
        os.makedirs(path.parent / "images", exist_ok=True)
        image_name = os.path.basename(output_file).replace(".fits", "")
        raw_image_name = image_name.replace("calib_", "raw_")
        save_fits_as_png(raw_file, path.parent / "images" / f"{raw_image_name}.png")
        save_fits_as_png(output_file, path.parent / "images" / f"{image_name}.png")
