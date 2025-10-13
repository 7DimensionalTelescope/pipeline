import os
import glob
import time
import pprint
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import numpy as np
from datetime import datetime
from astropy.io import fits
import copy
from .plotting import *
from . import utils as prep_utils
from .calc import record_statistics, delta_edge_center
from ..utils import (
    get_header,
    flatten,
    time_diff_in_seconds,
    atleast_1d,
)

from ..config import PreprocConfiguration
from ..config.utils import get_key
from ..services.setup import BaseSetup
from ..const import HEADER_KEY_MAP
from ..services.utils import acquire_available_gpu
from .checker import Checker
from ..services.database import QAData
from ..services.database.handler import DatabaseHandler
from ..header import add_padding
from ..const import PipelineError

pp = pprint.PrettyPrinter(indent=2)  # , width=120)


class Preprocess(BaseSetup, Checker, DatabaseHandler):
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
        add_database=True,
        **kwargs,
    ):
        # Load Configuration
        super().__init__(config, logger, queue)

        self.overwrite = overwrite
        self.master_frame_only = master_frame_only

        self.calib_types = calib_types or ["bias", "dark", "flat"]

        self.skip_plotting_flags = {
            "bias": True,
            "dark": True,
            "flat": True,
            "sci": True,
        }  # keys synced with self.calib_types!
        for calib in self.calib_types:
            self.skip_plotting_flags[calib] = False
        self.skip_plotting_flags["sci"] = master_frame_only

        self._use_gpu = use_gpu

        # Initialize DatabaseHandler
        DatabaseHandler.__init__(self, add_database=add_database)
        if self.is_connected:
            self.set_logger(logger)
            self.logger.debug("Initialized DatabaseHandler for pipeline and QA data management")

        self.initialize()
        self._generated_masterframes = []

    @classmethod
    def from_list(cls, images: list, **kwargs):
        config = PreprocConfiguration(atleast_1d(images), **kwargs)
        return cls(config, **kwargs)

    @property
    def sequential_task(self):
        tasks = []
        for i in range(self._n_groups):
            tasks.append((4 * i, f"load_masterframe", True))

            if i < self._n_groups - 1:
                tasks.append((4 * i + 3, f"proceed_to_next_group", False))

        return tasks

    def initialize(self):
        from ..path import PathHandler

        self.logger.info("Initializing Preprocess")

        if get_key(self.config.input, "masterframe_images") or get_key(self.config.input, "science_images"):
            bdf_flattened = flatten(self.config.input.masterframe_images)
            input_files = bdf_flattened + list(self.config.input.science_images)
            self.raw_groups = PathHandler.take_raw_inventory(input_files)
            # self.logger.debug(f"raw_groups initialized: {self.raw_groups}")
        elif self.config.input.raw_dir:
            input_files = glob.glob(os.path.join(self.config.input.raw_dir, "*.fits"))
            self.raw_groups = PathHandler.take_raw_inventory(input_files)
        else:
            raise ValueError("No input files or directory specified")

        self._n_groups = len(self.raw_groups)
        self._original_raw_groups = copy.deepcopy(self.raw_groups)
        self._current_group = 0
        self.load_criteria()

        self.logger.info(f"{self._n_groups} groups are found")
        self.logger.debug(f"raw_groups:\n{pp.pformat(self.raw_groups)}")

        # Create pipeline record in database
        if self.is_connected:
            self.pipeline_id = self.create_pipeline_record(self.config, self.raw_groups, self.overwrite)

    def run(self, device_id=None, make_plots=True, use_gpu=True):
        try:
            self._use_gpu = all([use_gpu, self._use_gpu])

            st = time.time()

            # Update pipeline status to running
            self.update_pipeline_progress(0, "running")

            threads_for_making_plots = []
            for i in range(self._n_groups):
                # Calculate progress percentage
                progress = int((i / self._n_groups) * 100)
                self.update_pipeline_progress(progress)

                self.logger.debug(f"[Group {i+1}] [filter: exptime] {PathHandler.get_group_info(self.raw_groups[i])}")
                # self.logger.info(f"Start processing group {i+1} / {self._n_groups}")
                self.logger.debug("\n" + "#" * 100 + f"\n{' '*30}Start processing group {i+1} / {self._n_groups}\n" + "#" * 100)  # fmt: skip
                self.load_masterframe(device_id=device_id)

                if not self.master_frame_only:
                    self.prepare_header()
                    self.data_reduction(device_id=device_id)

                if make_plots:
                    t = threading.Thread(
                        target=self.make_plots, kwargs={"group_index": i, "skip_flag": self.skip_plotting_flags}
                    )
                    t.start()
                    threads_for_making_plots.append(t)

                if i < self._n_groups - 1:
                    self.proceed_to_next_group()

            if make_plots:
                for t in threads_for_making_plots:
                    t.join()

            # Update pipeline status to completed
            self.update_pipeline_progress(100, "completed")

            self.logger.info(f"Preprocessing completed in {time_diff_in_seconds(st)} seconds")
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            raise

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
        """This parses from the PathHandler.take_raw_inventory output, self.raw_groups"""

        if name == "sci_input":
            return self._parse_sci_list(group_index, "input")
        elif name == "sci_output":
            return self._parse_sci_list(group_index, "output")
        elif name == "bpmask_output":
            dark_out = self._get_raw_group("dark_output", group_index)
            return dark_out.replace("dark", "bpmask")

        if name.endswith("_input"):
            key = name[:4]  # strip "_input" (e.g., bias from bias_input)
            if key in self._key_to_index:
                return self.raw_groups[group_index][0][self._key_to_index[key]]
        elif name.endswith("_output"):
            key = name[:4]  # strip "_output" (e.g., bias from bias_output)
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
        """updates ingredient files"""
        header = fits.getheader(getattr(self, f"{dtype}_input")[0])

        if dtype == "bias":
            header = prep_utils.write_IMCMB_to_header(header, self.bias_input)
        elif dtype == "dark":
            header = prep_utils.write_IMCMB_to_header(header, [self.bias_output] + self.dark_input)
        elif dtype == "flat":
            header = prep_utils.write_IMCMB_to_header(
                header, [self.bias_output, self.flatdark_output] + self.flat_input
            )

        header["NFRAMES"] = len(getattr(self, f"{dtype}_input"))
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

            self.logger.debug(f"[Group {self._current_group+1}] {dtype}_input: {input_file}")
            self.logger.debug(f"[Group {self._current_group+1}] {dtype}_output: {output_file}")

            if dtype == "dark":
                self.logger.debug(f"[Group {self._current_group+1}] flatdark_output: {self.flatdark_output}")

            if os.path.exists(output_file):
                header = fits.getheader(output_file)
                if not QAData.check_header(header, dtype):
                    self.overwrite = True

            if (
                input_file
                and (not os.path.exists(output_file) or self.overwrite)
                and (output_file not in self._generated_masterframes)
            ):
                norminal = self._generate_masterframe(dtype, device_id)
                if not norminal:
                    self._fetch_masterframe(output_file, dtype)
                self._generated_masterframes.append(output_file)
            elif isinstance(output_file, str) and len(output_file) != 0:
                self._fetch_masterframe(output_file, dtype)
                self.skip_plotting_flags[dtype] = True
            else:
                self.logger.warning(f"[Group {self._current_group+1}] {dtype} has no input or output data (to fetch)")
                self.logger.debug(f"[Group {self._current_group+1}] {dtype}_input: {input_file}")
                self.logger.debug(f"[Group {self._current_group+1}] {dtype}_output: {output_file}")
                self.add_warning()

            
            self.write_qa_data(
                dtype=dtype,
                raw_groups=self._original_raw_groups,
                current_group=self._current_group,
                key_to_index=self._key_to_index,
                output_file=getattr(self, f"{dtype}_output"),
                logger=self.logger,
            )

        self.logger.info(f"[Group {self._current_group+1}] Generation/Loading of masterframes completed in {time_diff_in_seconds(st)} seconds")  # fmt: skip

        # Update pipeline progress after masterframe processing
        if self.has_pipeline_id:
            masterframe_progress = int(
                (self._current_group + 1) / self._n_groups * 50
            )  # Masterframes are 50% of total work
            self.update_pipeline_progress(masterframe_progress)

    def _generate_masterframe(self, dtype, device_id):
        """Generate & Save masterframe and sigma image"""

        st = time.time()

        input_files = getattr(self, f"{dtype}_input")
        header = self.get_header(dtype)

        device_id = device_id if self._use_gpu else "CPU"

        with acquire_available_gpu(device_id=device_id) as device_id:
            # cpu
            if device_id is None:
                from .calc import combine_images_with_cpu

                calc_function = combine_images_with_cpu
                self.logger.info(f"[Group {self._current_group+1}] Generating masterframe {dtype} in CPU")
            # gpu
            else:
                from .calc import combine_images_with_subprocess_gpu

                calc_function = combine_images_with_subprocess_gpu
                self.logger.info(f"[Group {self._current_group+1}] Generating masterframe {dtype} in GPU device {device_id}")  # fmt: skip

            if dtype == "bias":
                calc_function(input_files, device_id=device_id, output=self.bias_output, sig_output=self.biassig_output)

            elif dtype == "dark":
                calc_function(
                    input_files,
                    device_id=device_id,
                    subtract=[self.bias_output],
                    scale=[1],
                    output=self.dark_output,
                    sig_output=self.darksig_output,
                    make_bpmask=self.bpmask_output,
                    bpmask_sigma=self.config.preprocess.n_sigma,
                )
                # for flatdark
                self.flatdark_output = self.dark_output  # named _output for consistency, but not written to disk
                self.dark_exptime = header[HEADER_KEY_MAP["exptime"]]

            elif dtype == "flat":
                dark_scale = self._calc_dark_scale(header[HEADER_KEY_MAP["exptime"]], self.dark_exptime)
                calc_function(
                    input_files,
                    subtract=[self.bias_output, self.flatdark_output],
                    scale=[1, dark_scale],
                    norm=True,
                    device_id=device_id,
                    output=self.flat_output,
                    sig_output=self.flatsig_output,
                )

                self.logger.info(f"[Group {self._current_group+1}] Checking the quality and updating header for {dtype}")  # fmt: skip

        prep_utils.update_header_by_overwriting(getattr(self, f"{dtype}sig_output"), header)

        header = prep_utils.add_image_id(header)
        
        flag = self._quality_assessment(header=header, dtype=dtype)

        if flag:
            self.logger.info(f"[Group {self._current_group+1}] Nominal master {dtype} generated successfully in {time_diff_in_seconds(st)} seconds")  # fmt: skip
            self.logger.debug(f"[Group {self._current_group+1}] FITS Written: {getattr(self, f'{dtype}_output')}")
            return True
        else:
            self.logger.warning(f"[Group {self._current_group+1}] Master {dtype} generated but failed quality check")
            self.logger.debug(f"[Group {self._current_group+1}] FITS Written: {getattr(self, f'{dtype}_output')}")
            self.logger.warning(
                f"[Group {self._current_group+1}] Making a plot for the current {dtype} and fetching a new one with better quality"
            )
            self.make_plot(getattr(self, f"{dtype}_output"), dtype, self._current_group)
            self.add_error()
            return False

    def _quality_assessment(self, header, dtype):
        header = record_statistics(getattr(self, f"{dtype}_output"), header, dtype=dtype)

        flag, header = self.apply_criteria(header=header, dtype=dtype)
        
        if dtype == "dark":
            hotpix = self.update_bpmask(sanity=flag)
            header["NHOTPIX"] = (hotpix, "Number of hot pixels")

        prep_utils.update_header_by_overwriting(getattr(self, f"{dtype}_output"), header)

        return flag

    def _fetch_masterframe(self, template, dtype):
        self.logger.info(f"[Group {self._current_group+1}] Fetching a nominal master {dtype}")
        # existing_data can be either on-date or off-date
        max_offset = self.config.preprocess.max_offset
        self.logger.debug(f"[Group {self._current_group+1}] Masterframe Search Template: {template}")
        existing_mframe_file = prep_utils.search_with_date_offsets(template, max_offset=max_offset, future=True)

        if not existing_mframe_file:
            self.add_error()
            self.logger.error(
                f"[Group {self._current_group+1}] No pre-existing master {dtype} found in place of {template} within {max_offset} days"
            )
            raise PipelineError(f"No pre-existing master {dtype} found in place of {template} within {max_offset} days")
        else:
            self.logger.info(
                f"[Group {self._current_group+1}] Found pre-existing nominal master {dtype} at {os.path.basename(existing_mframe_file)}"
            )
        # update the output names in raw_groups
        self.raw_groups[self._current_group][1][self._key_to_index[dtype]] = existing_mframe_file

        # for flatdark
        if dtype == "dark":
            path = PathHandler(template)
            path.name.exptime = "*"
            flatdark_template = path.preprocess.masterframe
            existing_mframe_file = prep_utils.search_with_date_offsets(
                flatdark_template, max_offset=max_offset, future=True
            )  # search closest date first, minimum exptime if multiple found
            if existing_mframe_file:
                setattr(self, "flatdark_output", existing_mframe_file)  # mdark for mflat
                self.dark_exptime = get_header(existing_mframe_file)[HEADER_KEY_MAP["exptime"]]
            else:
                self.logger.error(f"[Group {self._current_group+1}] No pre-existing master flatdark found in place of {flatdark_template} within {max_offset} days")
                self.add_error()
                raise PipelineError(f"No pre-existing master flatdark found in place of {flatdark_template} within {max_offset} days")

    def data_reduction(self, device_id=None, use_gpu: bool = True):
        self._use_gpu = all([use_gpu, self._use_gpu])

        if not self.sci_input:
            self.logger.info(f"No science frames found in group {self._current_group + 1}, skipping data reduction.")
            self.all_results = None
            for attr in ("bias_data", "dark_data", "flat_data"):
                if attr in self.__dict__:
                    del self.__dict__[attr]
            self.skip_plotting_flags["sci"] = True
            return

        flag = [os.path.exists(file) for file in self.sci_output]
        # flag = [list((Path(file).parent.parent / "figures").glob("*.jpg")) for file in self.sci_output]
        # flag = [len(f) > 0 for f in flag]

        if all(flag):  # and not self.overwrite:
            self.logger.info(f"All images in group {self._current_group+1} are already processed")
            self.skip_plotting_flags["sci"] = True
            return

        st = time.time()
        device_id = device_id if self._use_gpu else "CPU"

        with acquire_available_gpu(device_id=device_id) as device_id:
            if device_id is None:
                from .calc import process_image_with_cpu

                process_kernel = process_image_with_cpu
                self.logger.info(f"[Group {self._current_group+1}] Processing {len(self.sci_input)} images on CPU")
            else:
                from .calc import process_image_with_subprocess_gpu

                process_kernel = process_image_with_subprocess_gpu
                self.logger.info(f"[Group {self._current_group+1}] Processing {len(self.sci_input)} images on GPU device(s): {device_id} ")  # fmt: skip

            # Determine number of workers for CPU processing
            n_workers = None
            # if device_id is None:  # CPU processing
            #     # Use up to 32 workers to avoid overwhelming the system
            #     n_workers = min(3, len(self.sci_input), cpu_count())
            #     self.logger.info(
            #         f"[Group {self._current_group+1}] Using {n_workers} parallel workers for CPU processing"
            #     )

            process_kernel(
                self.sci_input,
                self.bias_output,
                self.dark_output,
                self.flat_output,
                output_paths=self.sci_output,
                device_id=device_id,
                use_gpu=self._use_gpu,
                n_workers=n_workers,
            )

        self.logger.info(
            f"[Group {self._current_group+1}] Completed data reduction for {len(self.sci_input)} "
            f"images in {time_diff_in_seconds(st)} seconds "
            f"({time_diff_in_seconds(st, return_float=True)/len(self.sci_input):.1f} s/image)"
        )

        # Update pipeline progress after data reduction
        if self.has_pipeline_id:
            data_reduction_progress = 50 + int(
                (self._current_group + 1) / self._n_groups * 50
            )  # Data reduction is 50-100% of total work
            self.update_pipeline_progress(data_reduction_progress)

        # for raw_file, processed_file in zip(self.sci_input, self.sci_output):
        #     header = fits.getheader(raw_file)
        #     header["SATURATE"] = prep_utils.get_saturation_level(header, bias, dark, flat)
        #     header = prep_utils.write_IMCMB_to_header(header, [bias, dark, flat, raw_file])
        #     header = add_padding(header, n_head_blocks, copy_header=True)

        #     prep_utils.update_header_by_overwriting(processed_file, header)

    def prepare_header(self):
        bias, dark, flat = self.bias_output, self.dark_output, self.flat_output
        n_head_blocks = self.config.preprocess.n_head_blocks
        for raw_file, processed_file in zip(self.sci_input, self.sci_output):
            with fits.open(raw_file) as hdul:
                header = hdul[0].header.copy()
            header["SATURATE"] = prep_utils.get_saturation_level(header, bias, dark, flat)
            header = prep_utils.write_IMCMB_to_header(header, [bias, dark, flat, raw_file])
            header = add_padding(header, n_head_blocks, copy_header=True)
            header = prep_utils.ensure_mjd_in_header(header, logger=self.logger)
            prep_utils.write_header(processed_file, header)

    def make_plot(self, file_path: str, dtype: str, group_index: int):
        if dtype == "bias":
            plot_bias(file_path)
        elif dtype == "dark":
            bpmask_file = file_path.replace("dark", "bpmask")
            plot_bpmask(bpmask_file)
            badpix = fits.getval(bpmask_file, "BADPIX", ext=1) or 1
            mask = fits.getdata(bpmask_file, ext=1) != badpix
            fmask = mask.ravel()
            plot_dark(file_path, fmask)
        elif dtype == "flat":
            bpmask_file = self._get_raw_group("bpmask_output", group_index)
            badpix = fits.getval(bpmask_file, "BADPIX", ext=1) or 1
            mask = fits.getdata(bpmask_file, ext=1) != badpix
            fmask = mask.ravel()
            plot_flat(file_path, fmask)

    def make_plots(self, group_index: int, skip_flag={"bias": False, "dark": False, "flat": False, "sci": False}):
        try:

            all_flag = all(skip_flag.values())
            if all_flag:
                self.logger.info(f"[Group {group_index+1}] Skipping plot generation")
                return

            # generate calib plots
            self.logger.info(f"[Group {group_index+1}] Generating plots for master calibration frames")
            use_multi_thread = self.config.preprocess.use_multi_thread

            # bias
            if "bias" in self.calib_types and not skip_flag["bias"]:
                bias_file = self._get_raw_group("bias_output", group_index)
                if os.path.exists(bias_file):
                    plot_bias(bias_file)
                else:
                    self.logger.warning(f"Bias file {bias_file} does not exist. Skipping bias plot.")
                    self.add_warning()
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping bias plot")

            # bpmask
            if "dark" in self.calib_types:
                bpmask_file = self._get_raw_group("bpmask_output", group_index)
                if os.path.exists(bpmask_file):
                    if not skip_flag["dark"]:
                        plot_bpmask(bpmask_file)
                    badpix = fits.getval(bpmask_file, "BADPIX", ext=1)
                    if badpix is None:
                        self.logger.warning("Header missing BADPIX; using 1")
                        self.add_warning()
                        badpix = 1

                    mask = fits.getdata(bpmask_file, ext=1) != badpix
                    fmask = mask.ravel()
                else:
                    self.logger.warning(f"BPMask file {bpmask_file} does not exist. Skipping bpmask plot.")
                    self.add_warning()
                    fmask = None
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping bpmask plot")

            # dark
            if "dark" in self.calib_types and not skip_flag["dark"]:
                dark_file = self._get_raw_group("dark_output", group_index)
                if os.path.exists(dark_file):
                    plot_dark(dark_file, fmask)
                else:
                    self.logger.warning(f"Dark file {dark_file} does not exist. Skipping dark plot.")
                    self.add_warning()
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping dark plot")

            # flat
            if "flat" in self.calib_types and not skip_flag["flat"]:
                flat_file = self._get_raw_group("flat_output", group_index)
                if os.path.exists(flat_file):
                    plot_flat(flat_file, fmask)
                else:
                    self.logger.warning(f"Flat file {flat_file} does not exist. Skipping flat plot.")
                    self.add_warning()
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping flat plot")

            self.logger.info(f"[Group {group_index+1}] Completed generating plots for master calibration frames")

            # science
            st = time.time()
            num_sci = len(self._get_raw_group("sci_input", group_index))
            if num_sci and not skip_flag["sci"]:
                self.logger.info(f"[Group {group_index+1}] Generating plots for science frames ({num_sci} images)")
                if use_multi_thread:
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        futures = []
                        for input_img, output_img in zip(
                            self._get_raw_group("sci_input", group_index),
                            self._get_raw_group("sci_output", group_index),
                        ):
                            future = executor.submit(plot_sci, input_img, output_img)
                            futures.append(future)
                        # Wait for all plots to complete
                        for future in futures:
                            future.result()
                else:
                    for input_img, output_img in zip(
                        self._get_raw_group("sci_input", group_index), self._get_raw_group("sci_output", group_index)
                    ):
                        plot_sci(input_img, output_img)

                self.logger.info(
                    f"[Group {group_index+1}] Completed plot generation for images in {time_diff_in_seconds(st)} seconds "
                    f"({time_diff_in_seconds(st, return_float=True)/(num_sci or 1):.1f} s/image)"
                )
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping science plot")
        except Exception as e:
            self.logger.error(f"Error making plots: {e}")
            self.add_warning()

    def update_bpmask(self, sanity=True):
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
                if key in header:
                    newhdu.header[key] = header[key]
            newhdu.header["COMMENT"] = "Header inherited from first dark frame"
        newhdu.header["NHOTPIX"] = (np.sum(hot_mask), "Number of hot pixels.")
        newhdu.header["SIGMAC"] = (self.config.preprocess.n_sigma, "HP threshold in clipped sigma")
        newhdu.header["BADPIX"] = (1, "Pixel Value for Bad pixels")
        newhdu.header["SANITY"] = (sanity, "Sanity flag")
        primary_hdu = fits.PrimaryHDU()
        newhdul = fits.HDUList([primary_hdu, newhdu])
        newhdul.writeto(self.bpmask_output, overwrite=True)
        return np.sum(hot_mask)
