import cupy as cp
import eclaire as ec
from typing import Union, Any
from astropy.io import fits
import os
from pathlib import Path

from . import utils as prep_utils
from ..utils import add_padding

from ..services.memory import MemoryMonitor
from ..services.queue import QueueManager, Priority
from ..config import Configuration

from ..base import BaseSetup
from .plotting import save_fits_as_png


class Preprocess(BaseSetup):
    def __init__(
        self,
        config: Union[str, Any] = None,
        logger: Any = None,
        queue: Union[bool, QueueManager] = False,
    ) -> None:
        """Initialize the astrometry module.

        Args:
            config: Configuration object or path to config file
            logger: Custom logger instance (optional)
            queue: QueueManager instance or boolean to enable parallel processing
        """
        super().__init__(config, logger, queue)

    @property
    def sequential_task(self):
        return [
            (1, "load_mbdf", False),
            (2, "data_reduction", True),
            (3, "save_processed_files", False)
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
        self.logger.info(
            f"Calibrating image with {'Eclaire' if use_eclaire else 'Cupy'}"
        )
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
            try:
                if use_eclaire:
                    self.data_reduction()
                    self.save_processed_files()
                    #self._calibrate_image_eclaire()
                else:
                    self._calibrate_image_cupy()
            except Exception as e:
                self.logger.error(f"Error during preprocessing: {str(e)}")
                raise
        self.logger.info("Calibration Done")

    def load_mbdf(self):
        selection = self.config.preprocess.masterframe

        if selection == "default":  # use links under self.config.preprocess
            mbias_file = prep_utils.read_link(self.config.preprocess.mbias_link)
            self.config.preprocess.mbias_file = mbias_file
            self.logger.debug("Completed reading master BIAS link; 'masterframe' is 'default'")  # fmt:skip

            mdark_file = prep_utils.read_link(self.config.preprocess.mdark_link)
            self.config.preprocess.mdark_file = mdark_file
            self.logger.debug("Completed reading master DARK link; 'masterframe' is 'default'")  # fmt:skip

            mflat_file = prep_utils.read_link(self.config.preprocess.mflat_link)
            self.config.preprocess.mflat_file = mflat_file
            self.logger.debug("Completed reading master FLAT link; 'masterframe' is 'default'")  # fmt:skip

        elif selection == "closest":  # search closest master frames again
            from .utils import link_to_file, search_with_date_offsets

            # looks for real files, not links

            self.config.preprocess.mbias_file = search_with_date_offsets(
                link_to_file(self.config.preprocess.mbias_link), future=True
            )
            self.config.preprocess.mdark_file = search_with_date_offsets(
                link_to_file(self.config.preprocess.mdark_link), future=True
            )
            self.config.preprocess.mflat_file = search_with_date_offsets(
                link_to_file(self.config.preprocess.mflat_link), future=True
            )

            # if the found is the same as link: abort processing
            if (
                self.config.preprocess.mbias_file
                == prep_utils.read_link(self.config.preprocess.mbias_link)
                and self.config.preprocess.mdark_file
                == prep_utils.read_link(self.config.preprocess.mdark_link)
                and self.config.preprocess.mflat_file
                == prep_utils.read_link(self.config.preprocess.mflat_link)
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
            "raw":  self.config.file.raw_files,
            "processed": self.config.file.processed_files,
        }

    def data_reduction(self, with_eclaire=True):
        
        if with_eclaire:
            ofc = ec.FitsContainer(self.files["raw"])
            with prep_utils.load_data_gpu(self.files["bias"]) as mbias, \
                prep_utils.load_data_gpu(self.files["dark"]) as mdark, \
                prep_utils.load_data_gpu(self.files["flat"]) as mflat:  # fmt:skip
                ofc.data = ec.reduction(ofc.data, mbias, mdark, mflat)
            self._temp_data = ofc.data.get()

    def save_processed_files(self, make_plots=True):
        self.files = {
            "bias": self.config.preprocess.mbias_file,
            "dark": self.config.preprocess.mdark_file,
            "flat": self.config.preprocess.mflat_file,
            "raw":  self.config.file.raw_files,
            "processed": self.config.file.processed_files,
        }
        for idx, (raw_file, out_file) in enumerate(zip(self.files["raw"], self.files["processed"])):
            header = fits.getheader(raw_file)
            header["SATURATE"] = prep_utils.get_saturation_level(
                header, self.files["bias"], self.files["dark"], self.files["flat"]
            )
            header = prep_utils.write_IMCMB_to_header(
                header,
                [self.files["bias"], self.files["dark"], self.files["flat"], raw_file]
            )
            n_head_blocks = self.config.settings.header_pad
            add_padding(header, n_head_blocks, copy_header=False)
            output_path = os.path.join(self.config.path.path_processed, out_file)
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
