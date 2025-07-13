import os
import glob
import threading
import numpy as np
from astropy.io import fits
import time

from .plotting import *
from . import utils as prep_utils

from .calc import record_statistics
from ..utils import (
    get_header,
    flatten,
    time_diff_in_seconds,
    update_header_by_overwriting,
    write_header_into_file,
    atleast_1d,
)
from ..config import PreprocConfiguration
from ..services.setup import BaseSetup
from ..const import HEADER_KEY_MAP
from ..services.utils import acquire_available_gpu


class Preprocess(BaseSetup):
    """
    Assumes homogeneous BIAS, DARK, FLAT, SCI frames as input
    taken on the same date with the same
    unit, n_binning, gain, and cameras.
    """

    # IDE autocomplete
    bias_input: list[str]
    dark_input: list[str]
    flat_input: list[str]
    biassig_output: str
    darksig_output: str
    flatsig_output: str
    bias_output: str
    dark_output: str
    flat_output: str
    sci_input: list[str]
    sci_output: list[str]
    bpmask_output: str

    def __init__(
        self,
        config,
        queue=False,
        logger=None,
        overwrite=False,
        master_frame_only=False,
        calib_types=None,
        use_gpu=True,
        **kwargs,
    ):
        # Load Configuration
        super().__init__(config, logger, queue)

        self.overwrite = overwrite
        self.master_frame_only = master_frame_only

        self.calib_types = calib_types or ["bias", "dark", "flat"]

        self._use_gpu = use_gpu
        self.initialize()

        # self.logger.debug(f"Masterframe output folder: {self.path_fdz}")

    @classmethod
    def from_list(cls, images: list, **kwargs):
        config = PreprocConfiguration(atleast_1d(images), **kwargs)
        return cls(config, **kwargs)

    @property
    def sequential_task(self):
        tasks = []
        for i in range(self._n_groups):
            tasks.append((4 * i, f"load_masterframe", True))
            tasks.append((4 * i + 1, f"prepare_headers", False))

            tasks.append((4 * i + 2, f"data_reduction", True))

            if i < self._n_groups - 1:
                tasks.append((4 * i + 3, f"proceed_to_next_group", False))

        return tasks

    def initialize(self):
        from ..path import PathHandler

        self.logger.info("Initializing Preprocess")
        if (hasattr(self.config.input, "masterframe_images") and self.config.input.masterframe_images) or (
            hasattr(self.config.input, "science_images") and self.config.input.science_images
        ):
            bdf_flattened = flatten(self.config.input.masterframe_images)
            input_files = bdf_flattened + list(self.config.input.science_images)
            self.raw_groups = PathHandler.take_raw_inventory(input_files)
            self.logger.debug(f"grouped_raw from manual input: {self.raw_groups}")
        elif self.config.input.raw_dir:
            input_files = glob.glob(os.path.join(self.config.input.raw_dir, "*.fits"))
            self.raw_groups = PathHandler.take_raw_inventory(input_files)
        else:
            raise ValueError("No input files or directory specified")

        self._n_groups = len(self.raw_groups)
        self._current_group = 0

        self.logger.info(f"{self._n_groups} groups are found")
        self.logger.debug(f"raw_groups:\n{self.raw_groups}")
        # raise ValueError("stop")  # for debug

    def run(self, device_id=None, make_plots=True, use_gpu=True, only_with_sci=False):
        self._use_gpu = all([use_gpu, self._use_gpu])

        st = time.time()

        threads_for_making_plots = []
        for i in range(self._n_groups):
            if only_with_sci and len(self.sci_input) == 0:
                self.logger.info(f"No science images for this masterframe. Skipping...")
                continue

            self.load_masterframe(device_id=device_id)

            if not self.master_frame_only:
                self.prepare_headers()
                self.data_reduction(device_id=device_id)

            if make_plots:
                t = threading.Thread(target=self.make_plots, kwargs={"group_index": i})
                t.start()
                threads_for_making_plots.append(t)

            if i < self._n_groups - 1:
                self.proceed_to_next_group()

        if make_plots:
            for t in threads_for_making_plots:
                t.join()

        self.logger.info(f"Preprocessing completed in {time_diff_in_seconds(st)} seconds")

    def make_plot_all(self):
        st = time.time()
        self.logger.info("Generating plots for all groups")
        for i in range(self._n_groups):
            self.make_plots(i)
        self.logger.info(f"Finished generating plots for all groups in {time_diff_in_seconds(st)} seconds")

    def proceed_to_next_group(self):
        self._current_group += 1
        if self._current_group >= self._n_groups:
            raise StopIteration

    def proceed_to_previous_group(self):
        self._current_group -= 1
        if self._current_group < 0:
            raise StopIteration

    def switch_to_group(self, group_index):
        if group_index < 0 or group_index >= self._n_groups:
            raise StopIteration
        self._current_group = group_index

    def __getattr__(self, name):
        """bias_input, dark_input, flat_input, bias_output, dark_output, flat_input are defined here"""
        if name.endswith("_input") or name.endswith("_output"):
            return self._get_raw_group(name, self._current_group)

    @property
    def _key_to_index(self):
        return {"bias": 0, "dark": 1, "flat": 2}

    def _get_raw_group(self, name, group_index):
        if name == "sci_input":
            return self._parse_sci_list(group_index, "input")
        elif name == "sci_output":
            return self._parse_sci_list(group_index, "output")
        elif name == "bpmask_output":
            return getattr(self, f"dark_output").replace("dark", f"bpmask")

        if name.endswith("_input"):
            key = name[:4]  # strip "_input" (e.g., bias_input)
            if key in self._key_to_index:
                return self.raw_groups[group_index][0][self._key_to_index[key]]
        elif name.endswith("_output"):
            key = name[:4]  # strip "_output" (e.g., bias_output)
            if key in self._key_to_index:
                if "sig" in name:
                    return getattr(self, f"{key}_output").replace(key, f"{key}sig")
                else:
                    return self.raw_groups[group_index][1][self._key_to_index[key]]
        raise AttributeError(f"Attribute {name} not found")

    def _parse_sci_list(self, group_index, dtype="input"):
        l = []
        for value in self.raw_groups[group_index][2].values():
            if dtype == "input":
                l += value[0]
            elif dtype == "output":
                l += value[1]
        return l

    def get_header(self, dtype):
        if dtype == "bias":
            header = fits.getheader(self.bias_input[0])
            header = prep_utils.write_IMCMB_to_header(header, self.bias_input)
            header["NFRAMES"] = len(self.bias_input)
        elif dtype == "dark":
            header = fits.getheader(self.dark_input[0])
            header = prep_utils.write_IMCMB_to_header(header, [self.bias_output] + self.dark_input)
            header["NFRAMES"] = len(self.dark_input)
        elif dtype == "flat":
            header = fits.getheader(self.flat_input[0])
            header = prep_utils.write_IMCMB_to_header(header, [self.bias_output, self.dark_output] + self.flat_input)
            header["NFRAMES"] = len(self.flat_input)
        return header

    def _calc_dark_scale(self, flat_exptime, dark_exptime):
        self.logger.debug(f"FLAT DARK SCALING (FLAT / DARK): {flat_exptime} / {dark_exptime}")
        return flat_exptime / dark_exptime

    def load_masterframe(self, device_id=None, use_gpu: bool = True):
        """
        no raw calib -> fetch from the library of pre-generated master frames
        raw calibs exist
            -> if output master exists, just fetch.
            -> if overwrite, always generate and overwrite

        If there's nothing to fetch, the code will fail.
        """
        self._use_gpu = all([use_gpu, self._use_gpu])

        st = time.time()

        for dtype in self.calib_types:

            input_file = getattr(self, f"{dtype}_input")
            output_file = getattr(self, f"{dtype}_output")

            self.logger.debug(f"{dtype}_input: {input_file}")
            self.logger.debug(f"{dtype}_output: {output_file}")

            if input_file:  # if the list is not empty
                if not os.path.exists(output_file) or self.overwrite:
                    self._generate_masterframe(dtype, device_id)
                else:
                    self._fetch_masterframe(output_file, dtype)
            elif isinstance(output_file, str) or len(output_file) > 0:
                self._fetch_masterframe(output_file, dtype)
            else:
                self.logger.warning(f"No input or output data for {dtype}")
                self.logger.debug(f"{dtype}_input: {input_file}")
                self.logger.debug(f"{dtype}_output: {output_file}")

        self.logger.info(f"Generation/Loading of masterframes completed in {time_diff_in_seconds(st)} seconds")

    def _generate_masterframe(self, dtype, device_id):
        """Generate & Save masterframe and sigma image"""

        st = time.time()

        input_data = getattr(self, f"{dtype}_input")
        header = self.get_header(dtype)

        device_id = device_id if self._use_gpu else "CPU"

        with acquire_available_gpu(device_id=device_id) as device_id:
            if device_id is None:
                from .calc import combine_images_with_cpu

                calc_function = combine_images_with_cpu
                self.logger.info(f"Generating masterframe {dtype} for group {self._current_group+1} in CPU")
            else:
                from .calc import combine_images_with_subprocess

                calc_function = combine_images_with_subprocess
                self.logger.info(
                    f"Generating masterframe {dtype} for group {self._current_group+1} in GPU device {device_id}"
                )

            if dtype == "bias":
                calc_function(input_data, device_id=device_id, output=self.bias_output, sig_output=self.biassig_output)

            elif dtype == "dark":
                calc_function(
                    input_data,
                    device_id=device_id,
                    subtract=[self.bias_output],
                    scale=[1],
                    output=self.dark_output,
                    sig_output=self.darksig_output,
                    make_bpmask=self.bpmask_output,
                    bpmask_sigma=self.config.preprocess.n_sigma,
                )
                self.dark_exptime = header[HEADER_KEY_MAP["exptime"]]

            elif dtype == "flat":
                dark_scale = self._calc_dark_scale(header[HEADER_KEY_MAP["exptime"]], self.dark_exptime)
                calc_function(
                    input_data,
                    subtract=[self.bias_output, self.dark_output],
                    scale=[1, dark_scale],
                    norm=True,
                    device_id=device_id,
                    output=self.flat_output,
                    sig_output=self.flatsig_output,
                )

        update_header_by_overwriting(getattr(self, f"{dtype}sig_output"), header)

        header = prep_utils.add_image_id(header)

        header = record_statistics(getattr(self, f"{dtype}_output"), header)

        update_header_by_overwriting(getattr(self, f"{dtype}_output"), header)

        self.logger.info(f"Master {dtype} generated in {time_diff_in_seconds(st)} seconds")
        self.logger.debug(f"FITS Written: {getattr(self, f'{dtype}_output')}")

        if dtype == "dark":
            self.update_bpmask()

    def _fetch_masterframe(self, template, dtype):
        self.logger.info(f"Fetching master {dtype}")
        # existing_data can be either on-date or off-date
        max_offset = self.config.preprocess.max_offset
        self.logger.debug(f"Masterframe Search Template: {template}")
        existing_mframe_file = prep_utils.search_with_date_offsets(template, max_offset=max_offset, future=True)

        if not existing_mframe_file:
            raise FileNotFoundError(
                f"No pre-existing master {dtype} found in place of {template} wihin {max_offset} days"
            )

        setattr(self, f"{dtype}_output", existing_mframe_file)

        output = list(self.raw_groups[self._current_group])
        next_output = list(output[1])
        next_output[self._key_to_index[dtype]] = existing_mframe_file
        output[1] = tuple(next_output)
        self.raw_groups[self._current_group] = tuple(output)

        if dtype == "dark":
            self.dark_exptime = get_header(existing_mframe_file)[HEADER_KEY_MAP["exptime"]]

    def prepare_headers(self):

        st = time.time()
        n_head_blocks = self.config.preprocess.n_head_blocks
        bias, dark, flat = self.bias_output, self.dark_output, self.flat_output
        self._header = []
        # Write results
        for raw_file, processed_file in zip(self.sci_input, self.sci_output):
            header = fits.getheader(raw_file)
            header["SATURATE"] = prep_utils.get_saturation_level(header, bias, dark, flat)
            header = prep_utils.write_IMCMB_to_header(header, [bias, dark, flat, raw_file])
            header = prep_utils.add_padding(header, n_head_blocks, copy_header=True)

            write_header_into_file(processed_file, header)
            self._header.append(header)

        self.logger.info(
            f"Prepare image headers for group {self._current_group+1} in {time_diff_in_seconds(st)} seconds."
        )

    def data_reduction(self, device_id=None, use_gpu: bool = True):
        self._use_gpu = all([use_gpu, self._use_gpu])

        if not self.sci_input:
            self.logger.info(f"No science frames found in group {self._current_group + 1}, skipping data reduction.")
            self.all_results = None
            for attr in ("bias_data", "dark_data", "flat_data"):
                if attr in self.__dict__:
                    del self.__dict__[attr]
            return

        flag = [os.path.exists(file) for file in self.sci_output]
        if all(flag):
            self.logger.info(f"All images in group {self._current_group+1} are already processed")
            return

        st = time.time()
        device_id = device_id if self._use_gpu else "CPU"

        with acquire_available_gpu(device_id=device_id) as device_id:
            if device_id is None:
                from .calc import process_image_with_cpu

                process_kernel = process_image_with_cpu
                self.logger.info(f"Processing {len(self.sci_input)} images in group {self._current_group+1} on CPU")
            else:
                from .calc import process_image_with_subprocess

                process_kernel = process_image_with_subprocess
                self.logger.info(
                    f"Processing {len(self.sci_input)} images in group {self._current_group+1} on GPU device(s): {device_id} "
                )

            process_kernel(
                self.sci_input,
                self.bias_output,
                self.dark_output,
                self.flat_output,
                device_id=device_id,
                output_paths=self.sci_output,
                header=self._header,
                use_gpu=self._use_gpu,
            )

            for i, o in enumerate(self.sci_output):
                with fits.open(o, mode="update") as hdul:
                    hdul[0].header = self._header[i]
                    hdul.flush()

            self.logger.info(
                f"Completed data reduction for {len(self.sci_input)} images in group {self._current_group+1} in {time_diff_in_seconds(st)} seconds ({time_diff_in_seconds(st, return_float=True)/len(self.sci_input):.1f} s/image)"
            )

    def make_plots(self, group_index=None):
        st = time.time()
        if group_index is None:
            group_index = self._current_group

        # generate calib plots
        self.logger.info(f"Generating plots for master calibration frames of group {group_index+1}")
        use_multi_thread = self.config.preprocess.use_multi_thread

        bias_file = self._get_raw_group("bias_output", group_index)
        # if prep_utils.wait_for_masterframe(bias_file, timeout=10):
        if "bias" in self.calib_types:
            plot_bias(bias_file, savefig=True)

        if "dark" in self.calib_types:
            mask = plot_bpmask(self._get_raw_group("bpmask_output", group_index), savefig=True)
            sample_header = fits.getheader(self._get_raw_group("bpmask_output", group_index), ext=1)
            if "BADPIX" in sample_header.keys():
                badpix = sample_header["BADPIX"]
            else:
                self.logger.warning("Header missing BADPIX; using 1")
                badpix = 1

            mask = mask != badpix
            fmask = mask.ravel()
            plot_dark(self._get_raw_group("dark_output", group_index), fmask, savefig=True)

        if "flat" in self.calib_types:
            plot_flat(self._get_raw_group("flat_output", group_index), fmask, savefig=True)

        # generate sci plots
        self.logger.info(
            f"Generating plots for science frames of group {group_index+1} ({len(self._get_raw_group('sci_input', group_index))} images)"
        )
        if use_multi_thread:
            threads = []
            for input_img, output_img in zip(
                self._get_raw_group("sci_input", group_index), self._get_raw_group("sci_output", group_index)
            ):
                thread = threading.Thread(target=plot_sci, args=(input_img, output_img))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
        else:
            for input_img, output_img in zip(
                self._get_raw_group("sci_input", group_index), self._get_raw_group("sci_output", group_index)
            ):
                plot_sci(input_img, output_img)

        self.logger.info(
            f"Completed plot generation for images in group {group_index+1} in {time_diff_in_seconds(st)} seconds"
        )

    def update_bpmask(self):
        header = self.get_header("dark")
        hot_mask = fits.getdata(self.bpmask_output)
        newhdu = fits.CompImageHDU(data=hot_mask)
        if header:
            for key in [
                "INSTRUME",
                "GAIN",
                "EXPTIME",
                "EXPOSURE",
                "JD",
                "MJD",
                "DATE-OBS",
                "DATE-LOC",
                "XBINNING",
                "YBINNING",
            ]:
                newhdu.header[key] = header[key]
            newhdu.header["COMMENT"] = "Header inherited from first dark frame"
        newhdu.header["NHOTPIX"] = (np.sum(hot_mask), "Number of hot pixels.")
        newhdu.header["SIGMAC"] = (self.config.preprocess.n_sigma, "HP threshold in clipped sigma")
        newhdu.header["BADPIX"] = (1, "Pixel Value for Bad pixels")

        primary_hdu = fits.PrimaryHDU()
        newhdul = fits.HDUList([primary_hdu, newhdu])
        newhdul.writeto(self.bpmask_output, overwrite=True)
