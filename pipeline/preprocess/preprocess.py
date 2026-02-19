import os
import glob
import time
import pprint
import threading
import numpy as np
from astropy.io import fits
import copy
import traceback

from .plotting import *
from . import utils as prep_utils
from .calc import record_statistics
from . import ppflag

from ..utils import flatten, time_diff_in_seconds, atleast_1d
from ..config import PreprocConfiguration
from ..config.utils import get_key
from ..services.setup import BaseSetup
from ..const import HEADER_KEY_MAP
from ..services.utils import acquire_available_gpu
from ..services.checker import CheckerMixin
from ..services.database.image_qa import ImageQATable
from ..services.database.handler import DatabaseHandler
from ..utils.header import add_padding, get_header
from ..errors import PreprocessError, MasterFrameNotFoundError
from ..path import PathHandler, NameHandler

pp = pprint.PrettyPrinter(indent=2)  # , width=120)


class Preprocess(BaseSetup, CheckerMixin, DatabaseHandler):
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
        use_gpu=False,
        add_database=True,
        **kwargs,
    ):
        # Load Configuration
        super().__init__(config, logger, queue)
        self.logger.process_error = PreprocessError

        is_too = get_key(self.config_node.settings, "is_too", False)

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
        DatabaseHandler.__init__(self, add_database=add_database if not is_too else False, logger=self.logger)

        if self.is_connected:
            self.logger.debug("Initialized DatabaseHandler for pipeline and QA data management")

        self.is_too = is_too

        self.initialize()
        self._generated_masterframes = []  # this is to avoid re-generating masterframes when overwrite=True is given

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

        self.logger.info("Initializing Preprocess")

        if get_key(self.config_node.input, "masterframe_images") or get_key(self.config_node.input, "science_images"):
            bdf_flattened = flatten(self.config_node.input.masterframe_images)
            input_files = bdf_flattened + list(self.config_node.input.science_images)
            self.raw_groups = PathHandler.take_raw_inventory(input_files, is_too=self.is_too)
            # self.logger.debug(f"raw_groups initialized: {self.raw_groups}")
        elif self.config_node.input.raw_dir:
            input_files = glob.glob(os.path.join(self.config_node.input.raw_dir, "*.fits"))
            self.raw_groups = PathHandler.take_raw_inventory(input_files, is_too=self.is_too)
        else:
            raise ValueError("No input files or directory specified")

        self._n_groups = len(self.raw_groups)
        self._original_raw_groups = copy.deepcopy(self.raw_groups)
        self._current_group = 0  # Do not manipulate it directly; use proceed_to_next_group and so on
        self.log_group_manifest()
        self.load_criteria()

        self.logger.info(f"{self._n_groups} groups are found")
        self.logger.debug(f"raw_groups:\n{pp.pformat(self.raw_groups)}")

        # Create pipeline record in database
        if self.is_connected:
            self.logger.debug(f"is connected: creating pipeline record in database")
            self.process_status_id = self.create_process_data(self.config_node, overwrite=self.overwrite)

            if self.process_status_id is not None:
                from ..services.database.handler import ExceptionHandler

                self.logger.database = ExceptionHandler(self.process_status_id)

    def log_group_manifest(self):
        for i, group in enumerate(self.raw_groups):
            if sci := self._parse_sci_list(i, "input"):
                groupname = NameHandler(sci[0]).groupname
            else:
                for dtype in self.calib_types[::-1]:  # flat is most important
                    if calib := self._get_raw_group(f"{dtype}_input", i):
                        groupname = NameHandler(calib[0]).groupname
                        break
            self.logger.debug(f"[Group {i+1}] {groupname}")

    def run(
        self,
        device_id=None,
        make_plots=True,
        use_gpu=True,
        override_skip_plotting_flags=None,
        dry_run: bool = False,
    ):
        """
        override_skip_plotting_flags is for finer control over which plots to generate.
        e.g., override_skip_plotting_flags={"bias": True, "dark": True, "flat": True, "sci": False} to regenerate sci plots
        dry_run traces execution without modifying data on disk (reads are allowed).
        """

        try:
            self._use_gpu = all([use_gpu, self._use_gpu])

            st = time.time()

            # Reset errors and warnings at the start of processing
            if self.is_connected and not dry_run:
                self.reset_exceptions()

            # Update pipeline status to running
            if not dry_run:
                self.update_progress(0, "running")
            else:
                self.logger.info("Pipeline run started; no files will be created or modified (DRY RUN)")

            threads_for_making_plots = []
            for i in range(self._n_groups):
                # Calculate progress percentage
                if not dry_run:
                    progress = int((i / self._n_groups) * 100)
                    self.update_progress(progress)

                self.logger.debug("\n" + "#" * 100 + f"\n{' '*30}Start processing group {i+1} / {self._n_groups}\n" + "#" * 100)  # fmt: skip
                self.logger.debug(f"[Group {i+1}] [filter: exptime] {PathHandler.get_group_info(self.raw_groups[i])}")

                # ---- group-level work ----
                try:
                    self.load_masterframe(device_id=device_id, dry_run=dry_run)

                    if not self.master_frame_only:
                        self.prepare_header(dry_run=dry_run)
                        self.data_reduction(device_id=device_id, dry_run=dry_run)

                    flags_for_this_group = copy.deepcopy(self.skip_plotting_flags)

                    if make_plots:
                        t = threading.Thread(
                            target=self.make_plots,
                            kwargs={
                                "group_index": i,
                                "skip_flags": flags_for_this_group,
                                "override_skip_flags": override_skip_plotting_flags,
                                "dry_run": dry_run,
                            },
                        )
                        t.start()
                        threads_for_making_plots.append(t)

                except Exception as e:
                    self.logger.error(
                        f"[Group {i+1}] Error during masterframe generation or data reduction: {str(e)}",
                        e,
                        exc_info=False,
                    )
                    self.logger.debug(traceback.format_exc())

                    self.logger.info(f"[Group {i+1}] Skipping to next group")

                finally:
                    if i < self._n_groups - 1:
                        self.proceed_to_next_group()

            if make_plots:
                for t in threads_for_making_plots:
                    t.join()

            # Update pipeline status to completed
            if not dry_run:
                self.update_progress(100, "completed")

            self.logger.info(f"Preprocessing completed in {time_diff_in_seconds(st)} seconds")
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}", PreprocessError.UnknownError, exc_info=True)
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

    def _parse_sci_list(self, group_index, dtype="input") -> list[str]:
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

    def load_masterframe(self, device_id=None, use_gpu: bool = True, dry_run: bool = False):
        """
        no raw calib -> fetch from the library of pre-generated master frames
        raw calibs exist
            -> if output master exists, just fetch.
            -> if overwrite, always generate and overwrite

        If there's nothing to fetch, the code will fail.
        """

        self._use_gpu = all([use_gpu, self._use_gpu])

        st = time.time()
        self._ppflag = {}  # PPFLAG per dtype for current group; also _flatdark_ppflag for flat

        for dtype in self.calib_types:

            input_file = getattr(self, f"{dtype}_input")
            output_file = getattr(self, f"{dtype}_output")

            self.logger.debug(f"[Group {self._current_group+1}] {dtype}_input: {input_file}")
            self.logger.debug(f"[Group {self._current_group+1}] {dtype}_output: {output_file}")

            if dtype == "dark":
                self.logger.debug(f"[Group {self._current_group+1}] flatdark_output: {self.flatdark_output}")

            if (
                input_file
                and (not os.path.exists(output_file) or self.overwrite)
                and (output_file not in self._generated_masterframes)
            ):
                norminal = self._generate_masterframe(dtype, device_id, dry_run=dry_run)
                if not norminal:
                    self._fetch_masterframe(output_file, dtype, dry_run=dry_run)
                self._generated_masterframes.append(output_file)
            elif isinstance(output_file, str) and len(output_file) != 0:
                self._fetch_masterframe(output_file, dtype, dry_run=dry_run)
                self.skip_plotting_flags[dtype] = True
            else:
                # cases like lone bias, where no dark_, flat_output exist
                self.logger.debug(f"[Group {self._current_group+1}] {dtype} has no input or output data to fetch.")
                self.logger.debug(f"[Group {self._current_group+1}] {dtype}_input: {input_file}")
                self.logger.debug(f"[Group {self._current_group+1}] {dtype}_output: {output_file}")
                # self.logger.error(msg, MasterFrameNotFoundError)
                # raise MasterFrameNotFoundError(msg)

            if input_file and not dry_run:

                qa_id = self.create_image_qa_data(
                    getattr(self, f"{dtype}_output"),
                    self.process_status_id,
                )

                self.logger.info(f"[Group {self._current_group+1}] Created QA data for {dtype} with ID: {qa_id}")

        self.logger.info(f"[Group {self._current_group+1}] Generation/Loading of masterframes completed in {time_diff_in_seconds(st)} seconds")  # fmt: skip

        # Update pipeline progress after masterframe processing
        if self.has_process_status_id and not dry_run:
            masterframe_progress = int(
                (self._current_group + 1) / self._n_groups * 50
            )  # Masterframes are 50% of total work
            self.update_progress(masterframe_progress)

    def _generate_masterframe(self, dtype, device_id, dry_run: bool = False):
        """Generate & Save masterframe and sigma image"""

        if dry_run:
            outputs = [getattr(self, f"{dtype}_output"), getattr(self, f"{dtype}sig_output")]
            if dtype == "dark":
                outputs.append(self.bpmask_output)
            self.logger.info(f"[Group {self._current_group+1}] Would create master {dtype} files: {outputs} (DRY RUN)")
            if dtype == "dark":
                self.logger.info(f"[Group {self._current_group+1}] Flatdark will use current dark output (DRY RUN)")
            return True

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
                    bpmask_sigma=self.config_node.preprocess.n_sigma,
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

        # PPFLAG: propagate from dependencies (bias=0, dark=bias, flat=bias|flatdark)
        if dtype == "bias":
            ppflag_val = 0
        elif dtype == "dark":
            ppflag_val = self._ppflag.get("bias", 0)
            self._flatdark_ppflag = ppflag_val  # flatdark = dark when generated
        elif dtype == "flat":
            ppflag_val = ppflag.propagate_ppflag(self._ppflag.get("bias", 0), getattr(self, "_flatdark_ppflag", 0))
        else:
            ppflag_val = 0
        self._ppflag[dtype] = ppflag_val

        flag = self._quality_assessment(header=header, dtype=dtype, ppflag_val=ppflag_val)

        if flag:
            self.logger.info(f"[Group {self._current_group+1}] Nominal master {dtype} generated successfully in {time_diff_in_seconds(st)} seconds")  # fmt: skip
            self.logger.debug(f"[Group {self._current_group+1}] FITS Written: {getattr(self, f'{dtype}_output')}")
            return True
        else:
            self.logger.warning(
                f"[Group {self._current_group+1}] Master {dtype} generated in {time_diff_in_seconds(st)} seconds but failed quality check"
            )
            self.logger.debug(f"[Group {self._current_group+1}] FITS Written: {getattr(self, f'{dtype}_output')}")
            self.logger.warning(
                f"[Group {self._current_group+1}] Making a plot for the current {dtype} and fetching a new one with better quality"
            )
            self.make_masterframe_plots(getattr(self, f"{dtype}_output"), dtype, self._current_group)

        return False

    def _quality_assessment(self, header, dtype, ppflag_val: int = 0):
        header = record_statistics(getattr(self, f"{dtype}_output"), header, dtype=dtype)

        flag, header = self.apply_criteria(header=header, dtype=dtype)

        ppflag.set_ppflag_in_header(header, ppflag_val)

        if dtype == "dark":
            hotpix = self.update_bpmask(sanity=flag)
            header["NHOTPIX"] = (hotpix, "Number of hot pixels")

        prep_utils.update_header_by_overwriting(getattr(self, f"{dtype}_output"), header)
        return flag

    def _fetch_masterframe(self, template, dtype, dry_run: bool = False):
        """
        You get the Fetch log even though you only want to regenerate the plots.
        It's only finding the files in disk, not loading it. The performance
        impact is insignificant.
        """
        self.logger.info(f"[Group {self._current_group+1}] Fetching a nominal master {dtype}")
        # existing_data can be either on-date or off-date
        max_offset = self.config_node.preprocess.max_offset
        ignore_sanity = get_key(self.config_node.preprocess, "ignore_sanity_if_no_match", False)
        ignore_lenient = get_key(self.config_node.preprocess, "ignore_lenient_keys_if_no_match", False)
        self.logger.debug(f"[Group {self._current_group+1}] ignore_sanity: {ignore_sanity}")
        self.logger.debug(f"[Group {self._current_group+1}] ignore_lenient: {ignore_lenient}")
        self.logger.debug(f"[Group {self._current_group+1}] Masterframe Search ({dtype}) Template: {template}")
        existing_mframe_file, ignored_lenient = prep_utils.tolerant_search(
            template,
            dtype,
            max_offset=max_offset,
            future=True,
            ignore_sanity_if_no_match=ignore_sanity,
            ignore_lenient_keys_if_no_match=ignore_lenient,
        )

        if not existing_mframe_file:
            self.logger.error(
                f"[Group {self._current_group+1}] No pre-existing master {dtype} found in place of {template} within {max_offset} days",
                MasterFrameNotFoundError,
            )
            raise MasterFrameNotFoundError(
                f"No pre-existing master {dtype} found in place of {template} within {max_offset} days"
            )
        else:
            sanity_check = fits.getval(existing_mframe_file, "SANITY")
            self.logger.info(
                f"[Group {self._current_group+1}] Found pre-existing nominal (sanity: {sanity_check}) master {dtype} at {os.path.basename(existing_mframe_file)}"
            )

        # PPFLAG: fetched frame gets 1 (different date), 4 (sanity F), 8 (lenient keys) as appropriate
        ppflag_val = ppflag.compute_fetch_ppflag(
            existing_mframe_file,
            template,
            sanity_check,
            ignored_lenient_keys=ignored_lenient,
        )
        self._ppflag[dtype] = ppflag_val
        if not dry_run:
            with fits.open(existing_mframe_file, mode="update") as hdul:
                ppflag.set_ppflag_in_header(hdul[0].header, ppflag_val)
                hdul.flush()

        # update the output names in raw_groups
        self.raw_groups[self._current_group][1][self._key_to_index[dtype]] = existing_mframe_file

        # for flatdark
        if dtype == "dark":
            self.logger.debug(f"[Group {self._current_group+1}] Masterframe Search (flatdark) Template: {template}")
            path = PathHandler(template)
            path.name.exptime = "*"
            flatdark_template = path.preprocess.masterframe
            existing_flatdark_file, flatdark_ignored_lenient = prep_utils.tolerant_search(
                flatdark_template,
                "dark",
                max_offset=max_offset,
                future=True,
                ignore_sanity_if_no_match=ignore_sanity,
                ignore_lenient_keys_if_no_match=ignore_lenient,
            )  # search closest date first, minimum exptime if multiple found
            if existing_flatdark_file:
                flatdark_sanity = fits.getval(existing_flatdark_file, "SANITY")
                setattr(self, "flatdark_output", existing_flatdark_file)  # mdark for mflat
                self.dark_exptime = get_header(existing_flatdark_file)[HEADER_KEY_MAP["exptime"]]
                self.logger.info(
                    f"[Group {self._current_group+1}] Found pre-existing nominal (sanity: {flatdark_sanity}) flatdark at {os.path.basename(existing_flatdark_file)}"
                )
                # Flatdark: PPFLAG 0 if same nightdate as target (user requirement)
                flatdark_same_night = ppflag.is_same_nightdate(existing_flatdark_file, flatdark_template)
                self._flatdark_ppflag = ppflag.compute_fetch_ppflag(
                    existing_flatdark_file,
                    flatdark_template,
                    flatdark_sanity,
                    flatdark_same_nightdate=flatdark_same_night,
                    ignored_lenient_keys=flatdark_ignored_lenient,
                )
            else:
                self.logger.error(
                    f"[Group {self._current_group+1}] No pre-existing master flatdark found in place of {flatdark_template} within {max_offset} days",
                    MasterFrameNotFoundError,
                )

                raise MasterFrameNotFoundError(
                    f"No pre-existing master flatdark found in place of {flatdark_template} within {max_offset} days"
                )

    def data_reduction(self, device_id=None, use_gpu: bool = True, dry_run: bool = False):
        self._use_gpu = all([use_gpu, self._use_gpu])

        if not self.sci_input:
            self.logger.info(f"[Group {self._current_group+1}] No science frames found, skipping data reduction.")
            self.all_results = None
            for attr in ("bias_data", "dark_data", "flat_data"):
                if attr in self.__dict__:
                    del self.__dict__[attr]
            self.skip_plotting_flags["sci"] = True
            return

        flag = [os.path.exists(file) for file in self.sci_output]

        if all(flag) and not (self.overwrite):
            self.logger.info(f"[Group {self._current_group+1}] All images are already processed")
            self.skip_plotting_flags["sci"] = True
            return
        elif self.overwrite:
            input_files = self.sci_input
            output_files = self.sci_output
        else:
            input_files = []
            output_files = []
            for infile, outfile in zip(self.sci_input, self.sci_output):
                if not os.path.exists(outfile):
                    input_files.append(infile)
                    output_files.append(outfile)

        if dry_run:
            self.logger.info(
                f"[Group {self._current_group+1}] Would process {len(output_files)} science images (DRY RUN)"
            )
            self.logger.info(f"[Group {self._current_group+1}] Would create processed files: {output_files} (DRY RUN)")
            return

        st = time.time()
        device_id = device_id if self._use_gpu else "CPU"

        with acquire_available_gpu(device_id=device_id) as device_id:
            if device_id is None:
                from .calc import process_image_with_cpu

                process_kernel = process_image_with_cpu
                self.logger.info(f"[Group {self._current_group+1}] Processing {len(output_files)} images on CPU")
            else:
                from .calc import process_image_with_subprocess_gpu

                process_kernel = process_image_with_subprocess_gpu
                self.logger.info(f"[Group {self._current_group+1}] Processing {len(output_files)} images on GPU device(s): {device_id} ")  # fmt: skip

            # Determine number of workers for CPU processing
            n_workers = None
            # if device_id is None:  # CPU processing
            #     # Use up to 32 workers to avoid overwhelming the system
            #     n_workers = min(3, len(self.sci_input), cpu_count())
            #     self.logger.info(
            #         f"[Group {self._current_group+1}] Using {n_workers} parallel workers for CPU processing"
            #     )

            process_kernel(
                input_files,
                self.bias_output,
                self.dark_output,
                self.flat_output,
                output_paths=output_files,
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
        if self.has_process_status_id:
            data_reduction_progress = 50 + int(
                (self._current_group + 1) / self._n_groups * 50
            )  # Data reduction is 50-100% of total work
            self.update_progress(data_reduction_progress)

        # for raw_file, processed_file in zip(self.sci_input, self.sci_output):
        #     header = fits.getheader(raw_file)
        #     header["SATURATE"] = prep_utils.get_saturation_level(header, bias, dark, flat)
        #     header = prep_utils.write_IMCMB_to_header(header, [bias, dark, flat, raw_file])
        #     header = add_padding(header, n_head_blocks, copy_header=True)

        #     prep_utils.update_header_by_overwriting(processed_file, header)

    def prepare_header(self, dry_run: bool = False):
        if dry_run:
            self.logger.info(
                f"[Group {self._current_group+1}] Would write updated headers to processed files: {self.sci_output} (DRY RUN)"
            )
            return

        bias, dark, flat = self.bias_output, self.dark_output, self.flat_output
        n_head_blocks = self.config_node.preprocess.n_head_blocks
        sci_ppflag = ppflag.propagate_ppflag(
            ppflag.get_ppflag_from_header(bias, raise_if_missing=True),
            ppflag.get_ppflag_from_header(dark, raise_if_missing=True),
            ppflag.get_ppflag_from_header(flat, raise_if_missing=True),
        )
        for raw_file, processed_file in zip(self.sci_input, self.sci_output):
            with fits.open(raw_file) as hdul:
                header = hdul[0].header.copy()
            header["SATURATE"] = prep_utils.get_saturation_level(header, bias, dark, flat)
            header = prep_utils.write_IMCMB_to_header(header, [bias, dark, flat, raw_file])
            ppflag.set_ppflag_in_header(header, sci_ppflag)
            header = prep_utils.ensure_mjd_in_header(header, logger=self.logger)
            header = prep_utils.sanitize_header(header)
            header = add_padding(header, n_head_blocks, copy_header=True)
            self.logger.debug(
                f"Header size: {(x := len(header.tostring()))} bytes, {x//2880} blocks + {(x%2880)/80} lines"
            )
            prep_utils.write_header(processed_file, header)

    def make_masterframe_plots(self, file_path: str, dtype: str, group_index: int, dry_run: bool = False):
        if dtype == "bias":
            plot_bias(file_path, dry_run=dry_run)
        elif dtype == "dark":
            bpmask_file = file_path.replace("dark", "bpmask")
            plot_bpmask(bpmask_file, dry_run=dry_run)
            badpix = fits.getval(bpmask_file, "BADPIX", ext=1) or 1
            mask = fits.getdata(bpmask_file, ext=1) != badpix
            fmask = mask.ravel()
            plot_dark(file_path, fmask, dry_run=dry_run)
        elif dtype == "flat":
            bpmask_file = self._get_raw_group("bpmask_output", group_index)
            badpix = fits.getval(bpmask_file, "BADPIX", ext=1) or 1
            mask = fits.getdata(bpmask_file, ext=1) != badpix
            fmask = mask.ravel()
            plot_flat(file_path, fmask, dry_run=dry_run)

    def make_plots(
        self,
        group_index: int,
        skip_flags={"bias": False, "dark": False, "flat": False, "sci": False},
        override_skip_flags=None,
        dry_run: bool = False,
    ):
        try:
            all_flag = all(skip_flags.values())
            if all_flag and not override_skip_flags:
                self.logger.info(f"[Group {group_index+1}] Skipping plot generation")
                return

            skip_flags = override_skip_flags or skip_flags
            self.logger.debug(f"[Group {group_index+1}] skip_flags after override: {skip_flags}")

            # generate calib plots
            self.logger.info(f"[Group {group_index+1}] Generating plots for master calibration frames")
            # use_multi_thread = self.config.preprocess.use_multi_thread

            # bias
            if "bias" in self.calib_types and not skip_flags["bias"]:
                bias_file = self._get_raw_group("bias_output", group_index)
                if os.path.exists(bias_file):
                    plot_bias(bias_file, dry_run=dry_run)
                else:
                    self.logger.warning(f"[Group {group_index+1}] Bias image does not exist. Skipping bias plot.")

            else:
                self.logger.info(f"[Group {group_index+1}] Skipping bias plot")

            # bpmask
            if "dark" in self.calib_types:
                bpmask_file = self._get_raw_group("bpmask_output", group_index)
                if os.path.exists(bpmask_file):
                    if not skip_flags["dark"]:
                        plot_bpmask(bpmask_file, dry_run=dry_run)
                    badpix = fits.getval(bpmask_file, "BADPIX", ext=1)
                    if badpix is None:
                        self.logger.warning(f"[Group {group_index+1}] Header missing BADPIX; using 1")

                        badpix = 1

                    mask = fits.getdata(bpmask_file, ext=1) != badpix
                    fmask = mask.ravel()
                else:
                    self.logger.warning(f"[Group {group_index+1}] BPMask image does not exist. Skipping bpmask plot.")

                    fmask = None
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping bpmask plot")

            # dark
            if "dark" in self.calib_types and not skip_flags["dark"]:
                dark_file = self._get_raw_group("dark_output", group_index)
                if os.path.exists(dark_file):
                    plot_dark(dark_file, fmask, dry_run=dry_run)
                else:
                    self.logger.warning(f"[Group {group_index+1}] Dark image does not exist. Skipping dark plot.")

            else:
                self.logger.info(f"[Group {group_index+1}] Skipping dark plot")

            # flat
            if "flat" in self.calib_types and not skip_flags["flat"]:
                flat_file = self._get_raw_group("flat_output", group_index)
                if os.path.exists(flat_file):
                    plot_flat(flat_file, fmask, dry_run=dry_run)
                else:
                    self.logger.warning(f"[Group {group_index+1}] Flat image does not exist. Skipping flat plot.")

            else:
                self.logger.info(f"[Group {group_index+1}] Skipping flat plot")

            self.logger.info(f"[Group {group_index+1}] Completed generating plots for master calibration frames")

            # science
            st = time.time()
            num_sci = len(self._get_raw_group("sci_input", group_index))
            if num_sci and not skip_flags["sci"]:
                self.logger.info(f"[Group {group_index+1}] Generating plots for science frames ({num_sci} images)")

                for input_img, output_img in zip(
                    self._get_raw_group("sci_input", group_index), self._get_raw_group("sci_output", group_index)
                ):
                    plot_sci(input_img, output_img, is_too=self.is_too, dry_run=dry_run)

                self.logger.info(
                    f"[Group {group_index+1}] Completed plot generation for images in {time_diff_in_seconds(st)} seconds "
                    f"({time_diff_in_seconds(st, return_float=True)/(num_sci or 1):.1f} s/image)"
                )
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping science plot")
        except Exception as e:
            self.logger.error(f"[Group {group_index+1}] Error making plots: {e}", PreprocessError.UnknownError)
            self.logger.debug(traceback.format_exc())

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
        newhdu.header["SIGMAC"] = (self.config_node.preprocess.n_sigma, "HP threshold in clipped sigma")
        newhdu.header["BADPIX"] = (1, "Pixel Value for Bad pixels")
        newhdu.header["SANITY"] = (sanity, "Sanity flag")
        primary_hdu = fits.PrimaryHDU()
        newhdul = fits.HDUList([primary_hdu, newhdu])
        newhdul.writeto(self.bpmask_output, overwrite=True)
        return np.sum(hot_mask)
