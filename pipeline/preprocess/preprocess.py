import os
import glob
import time
import pprint
import threading
import numpy as np
from astropy.io import fits
import copy
import traceback

from .plotting import plot_bias, plot_bpmask, plot_dark, plot_flat, plot_sci
from . import utils as prep_utils
from .calc import record_masterframe_statistics
from . import ppflag

from ..utils import flatten, time_diff_in_seconds, atleast_1d
from ..config import PreprocConfiguration
from ..config.utils import get_key
from ..services.setup import BaseSetup
from ..const import HEADER_KEY_MAP, CALIB_TYPE_BIAS, CALIB_TYPE_DARK, CALIB_TYPE_FLAT, CALIB_TYPES
from ..services.utils import acquire_available_gpu
from ..services.checker import Checker
from ..services.database.image_qa import ImageQATable
from ..services.database.handler import DatabaseHandler
from ..utils.header import add_padding, get_header
from ..errors import PreprocessError, MasterFrameNotFoundError
from ..path import PathHandler, NameHandler
from .reprocess import ReprocessMixin

pp = pprint.PrettyPrinter(indent=2)  # , width=120)


class Preprocess(BaseSetup, Checker, DatabaseHandler, ReprocessMixin):
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
        use_database=True,
        **kwargs,
    ):
        # Load Configuration
        super().__init__(config, logger, queue)
        self.logger.process_error = PreprocessError

        is_too = get_key(self.config_node.settings, "is_too", False)
        is_pipeline = get_key(self.config_node.settings, "is_pipeline", False)

        self.overwrite = overwrite
        self.master_frame_only = master_frame_only

        self.calib_types = calib_types or list(CALIB_TYPES)
        if list(self.calib_types) != list(CALIB_TYPES)[: len(self.calib_types)]:
            raise PreprocessError.ValueError(
                f"calib_types must be a cumulative prefix of ['bias', 'dark', 'flat'], not {self.calib_types}"
            )

        self._better_match = get_key(self.config_node.preprocess, "reprocess_on_better_match", False)
        self._change_policy = "regenerated+sanity+better-match" if self._better_match else "regenerated+sanity"
        self._is_pipeline = is_pipeline
        self._ingredient_cache = {}  # {master path: (IMAGEID, SANITY)}; reset per group

        self._load_designated_masterframes()

        self._use_gpu = use_gpu

        # Initialize DatabaseHandler
        DatabaseHandler.__init__(
            self, use_database=(use_database and is_pipeline) if not is_too else False, logger=self.logger
        )

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

        # Respect pipeline vs ad-hoc layout from the configuration
        is_pipeline = get_key(self.config_node.settings, "is_pipeline", False)

        if get_key(self.config_node.input, "masterframe_images") or get_key(self.config_node.input, "science_images"):
            bdf_flattened = flatten(self.config_node.input.masterframe_images)
            input_files = bdf_flattened + list(self.config_node.input.science_images)
            self.raw_groups = PathHandler.take_raw_inventory(
                input_files,
                is_too=self.is_too,
                is_pipeline=is_pipeline,
            )
            # self.logger.debug(f"raw_groups initialized: {self.raw_groups}")
        elif self.config_node.input.raw_dir:
            input_files = glob.glob(os.path.join(self.config_node.input.raw_dir, "*.fits"))
            self.raw_groups = PathHandler.take_raw_inventory(
                input_files,
                is_too=self.is_too,
                is_pipeline=is_pipeline,
            )
        else:
            raise PreprocessError.ValueError("No input files or directory specified")

        self._n_groups = len(self.raw_groups)
        self._original_raw_groups = copy.deepcopy(self.raw_groups)
        self._current_group = 0  # Do not manipulate it directly; use proceed_to_next_group and so on
        self.log_group_manifest()
        self.load_qa_criteria()

        self.logger.info(f"{self._n_groups} groups are found")
        self.logger.debug(f"raw_groups:\n{pp.pformat(self.raw_groups)}")

        # designation dispatch is resolved, logged and validated before anything runs
        self._dispatch_designated_masterframes()

        self._expected_outputs = self._collect_expected_outputs()
        self.logger.info(f"This run is expected to produce {len(self._expected_outputs)} output file(s)")

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
        dry_run: bool = False,
    ):
        """
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

            if self.overwrite:
                self.logger.info("Overwrite=True; existing science outputs and plots may be regenerated")
            else:
                self.logger.info("Overwrite=False; existing science outputs and plots will be reused when available")
                self.logger.info(
                    f"Ingredient-change policy: {self._change_policy} — products whose master frames' "
                    "IMAGEID moved are rebuilt; dry_run=True sizes it (a lower bound: a dry run "
                    "regenerates nothing, so work cascading from a rebuilt master is not counted)"
                )

            threads_for_making_plots = []
            failed_groups = []
            for i in range(self._n_groups):
                self.logger.debug("\n" + "#" * 100 + f"\n{' '*30}Start processing group {i+1} / {self._n_groups}\n" + "#" * 100)  # fmt: skip
                self.logger.debug(f"[Group {i+1}] [filter: exptime] {PathHandler.get_group_info(self.raw_groups[i])}")

                # ---- group-level work ----
                try:
                    self.load_masterframe(device_id=device_id, dry_run=dry_run)

                    if not self.master_frame_only:
                        self.prepare_header(dry_run=dry_run)
                        self.data_reduction(device_id=device_id, dry_run=dry_run)

                    if make_plots:
                        t = threading.Thread(
                            target=self.make_plots,
                            kwargs={
                                "group_index": i,
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

                    # existence alone cannot see this on reruns: the previous run's files still pass the check
                    failed_groups.append(i + 1)
                    self.logger.info(f"[Group {i+1}] Skipping to next group")

                finally:
                    # in `finally` so a group that raised still contributes what it owed
                    self._check_group_outputs(i)
                    if not dry_run:
                        self.update_progress(self._output_progress)

                    if i < self._n_groups - 1:
                        self.proceed_to_next_group()

            if make_plots:
                for t in threads_for_making_plots:
                    t.join()

            # Final status: completed only if every expected output exists and no group raised
            missing = [f for f, exists in self._expected_outputs.items() if not exists]
            problems = []
            if missing:
                problems.append(f"{len(missing)} of {len(self._expected_outputs)} expected output file(s) missing")
            if failed_groups:
                problems.append(f"group(s) {', '.join(map(str, failed_groups))} raised an error")
            if not dry_run:
                self.update_progress(self._output_progress, "failed" if problems else "completed")
                self.sync_config_dependencies()

            if problems and not dry_run:
                if missing:
                    self.logger.debug(f"Missing expected outputs:\n{pp.pformat(missing)}")
                self.logger.warning(f"Preprocessing ended incomplete in {time_diff_in_seconds(st)} seconds")
                raise PreprocessError("; ".join(problems))
            self.logger.info(f"Preprocessing completed in {time_diff_in_seconds(st)} seconds")
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}", e, exc_info=True)
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
        return {CALIB_TYPE_BIAS: 0, CALIB_TYPE_DARK: 1, CALIB_TYPE_FLAT: 2}

    def _get_raw_group(self, name, group_index):
        """This parses from the PathHandler.take_raw_inventory output, self.raw_groups"""

        if name == "sci_input":
            return self._parse_sci_list(group_index, "input")
        elif name == "sci_output":
            return self._parse_sci_list(group_index, "output")
        elif name == "bpmask_output":
            dark_out = self._get_raw_group("dark_output", group_index)
            return dark_out.replace("dark", "bpmask")

        # Strip the whole suffix rather than taking name[:4]: "flatdark_output"[:4] is "flat", which silently
        # resolved the flatdark to the master FLAT path.
        if name.endswith("_input"):
            key = name[: -len("_input")]
            if key in self._key_to_index:
                return self.raw_groups[group_index][0][self._key_to_index[key]]
        elif name.endswith("_output"):
            key = name[: -len("_output")]
            if key.endswith("sig") and key[: -len("sig")] in self._key_to_index:
                base = key[: -len("sig")]
                return getattr(self, f"{base}_output").replace(base, f"{base}sig")
            if key in self._key_to_index:
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

        if dtype == CALIB_TYPE_BIAS:
            header = prep_utils.write_IMCMB_to_header(header, self.bias_input)
        elif dtype == CALIB_TYPE_DARK:
            header = prep_utils.write_IMCMB_to_header(header, [self.bias_output] + self.dark_input)
        elif dtype == CALIB_TYPE_FLAT:
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
        # per-group state; leaking any of these across groups corrupts the next flat
        self.flatdark_output = None
        self.dark_exptime = None
        self._flatdark_ppflag = 0
        self._ingredient_cache = {}

        for dtype in self.calib_types:

            input_file = getattr(self, f"{dtype}_input")
            output_file = getattr(self, f"{dtype}_output")

            self.logger.debug(f"[Group {self._current_group+1}] {dtype}_input: {input_file}")
            self.logger.debug(f"[Group {self._current_group+1}] {dtype}_output: {output_file}")

            if dtype == CALIB_TYPE_DARK:
                self.logger.debug(f"[Group {self._current_group+1}] flatdark_output: {self.flatdark_output}")

            generated_now = False
            if designated_file := self._designation_dispatch.get((self._current_group, dtype)):
                self._adopt_designated_masterframe(designated_file, dtype, template=output_file)
            elif (
                input_file
                and (output_file not in self._generated_masterframes)
                and (
                    not os.path.exists(output_file)
                    or self.overwrite
                    or self._master_change(dtype, output_file)
                )
            ):
                norminal = self._generate_masterframe(dtype, device_id, dry_run=dry_run)
                if not norminal:
                    self._fetch_masterframe(output_file, dtype, dry_run=dry_run)
                else:
                    generated_now = True
                self._generated_masterframes.append(output_file)
            elif isinstance(output_file, str) and len(output_file) != 0:
                self._fetch_masterframe(output_file, dtype, dry_run=dry_run)
            else:
                # cases like lone bias, where no dark_, flat_output exist
                self.logger.debug(f"[Group {self._current_group+1}] {dtype} has no input or output data to fetch.")
                self.logger.debug(f"[Group {self._current_group+1}] {dtype}_input: {input_file}")
                self.logger.debug(f"[Group {self._current_group+1}] {dtype}_output: {output_file}")
                # self.logger.error(msg, MasterFrameNotFoundError)
                # raise MasterFrameNotFoundError(msg)

            # settle the flatdark once per group, only when a flat will be built
            if dtype == CALIB_TYPE_DARK and output_file and self.flat_output and CALIB_TYPE_FLAT in self.calib_types:
                try:
                    self._resolve_flatdark(output_file)
                except MasterFrameNotFoundError:
                    if not dry_run:
                        raise
                    self.logger.info(
                        f"[Group {self._current_group+1}] No flatdark on disk yet; a real run "
                        "would resolve it after generating the dark (DRY RUN)"
                    )

            if input_file and not dry_run:

                qa_id = self.create_image_qa_data(
                    getattr(self, f"{dtype}_output"),
                    self.process_status_id,
                )

                self.logger.info(f"[Group {self._current_group+1}] Created QA data for {dtype} with ID: {qa_id}")
                # Register dependencies only for a freshly generated master; a fetched
                # master keeps the dependencies written when it was generated.
                if generated_now:
                    self.create_image_qa_dependencies(getattr(self, f"{dtype}_output"), qa_id)
                    # sci image_qa_dependency is handled in Astrometry, not Preprocess
                    self.logger.info(f"[Group {self._current_group+1}] Created QA dependencies for {dtype}")

        self.logger.info(f"[Group {self._current_group+1}] Generation/Loading of masterframes completed in {time_diff_in_seconds(st)} seconds")  # fmt: skip

        # Update pipeline progress after masterframe processing
        if not dry_run:
            self._check_group_outputs(self._current_group)
            self.update_progress(self._output_progress)

    def _collect_expected_outputs(self) -> dict[str, bool]:
        """Every file this run commits to produce, judged from the input pool alone: {path: seen on disk}."""
        expected = {}
        for i, (raws, masters, sci_dict) in enumerate(self._original_raw_groups):
            for dtype in self.calib_types:
                idx = self._key_to_index[dtype]
                # a master is owed only where the group holds its raw calibs and no designation preempts generation
                if raws[idx] and masters[idx] and (i, dtype) not in self._designation_dispatch:
                    expected[masters[idx]] = False
            if not self.master_frame_only:
                for _, processed in sci_dict.values():
                    expected.update(dict.fromkeys(processed, False))
        return expected

    def _check_group_outputs(self, group_index):
        """Mark this group's share of the expected outputs off against the disk."""
        # read _original_raw_groups: _fetch_masterframe and designation adoption rewrite raw_groups[...][1] in place
        raws, masters, sci_dict = self._original_raw_groups[group_index]
        paths = [masters[self._key_to_index[d]] for d in self.calib_types if raws[self._key_to_index[d]]]
        paths += [f for _, processed in sci_dict.values() for f in processed]
        for path in paths:
            if path in self._expected_outputs:
                self._expected_outputs[path] = os.path.exists(path)

    @property
    def _output_progress(self) -> int:
        """Percent of expected output files existing on disk."""
        if not self._expected_outputs:
            return 100
        return int(100 * sum(self._expected_outputs.values()) / len(self._expected_outputs))

    def _generate_masterframe(self, dtype, device_id, dry_run: bool = False):
        """Generate & Save masterframe and sigma image"""

        if dry_run:
            outputs = [getattr(self, f"{dtype}_output"), getattr(self, f"{dtype}sig_output")]
            if dtype == CALIB_TYPE_DARK:
                outputs.append(self.bpmask_output)
            self.logger.info(f"[Group {self._current_group+1}] Would create master {dtype} files: {outputs} (DRY RUN)")
            if dtype == CALIB_TYPE_DARK:
                self.logger.info(f"[Group {self._current_group+1}] Flatdark will be resolved once the dark is settled (DRY RUN)")  # fmt: skip
            return True

        st = time.time()

        input_files = getattr(self, f"{dtype}_input")
        header = self.get_header(dtype)
        prep_utils.set_pipe_ver_in_header(header)
        outputs = [getattr(self, f"{dtype}_output"), getattr(self, f"{dtype}sig_output")]
        if dtype == CALIB_TYPE_DARK:
            outputs.append(self.bpmask_output)
        for output_path in outputs:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

            if dtype == CALIB_TYPE_BIAS:
                calc_function(
                    input_files,
                    device_id=device_id,
                    output=self.bias_output,
                    sig_output=self.biassig_output,
                    dtype=dtype,
                )

            elif dtype == CALIB_TYPE_DARK:
                calc_function(
                    input_files,
                    device_id=device_id,
                    subtract=[self.bias_output],
                    scale=[1],
                    output=self.dark_output,
                    sig_output=self.darksig_output,
                    make_bpmask=self.bpmask_output,
                    bpmask_sigma=self.config_node.preprocess.n_sigma,
                    dtype=dtype,
                )
                # flatdark comes from _resolve_flatdark, never from this group's dark

            elif dtype == CALIB_TYPE_FLAT:
                dark_scale = self._calc_dark_scale(header[HEADER_KEY_MAP["exptime"]], self.dark_exptime)
                calc_function(
                    input_files,
                    subtract=[self.bias_output, self.flatdark_output],
                    scale=[1, dark_scale],
                    norm=True,
                    device_id=device_id,
                    output=self.flat_output,
                    sig_output=self.flatsig_output,
                    dtype=dtype,
                )

                self.logger.info(f"[Group {self._current_group+1}] Checking the quality and updating header for {dtype}")  # fmt: skip

            else:
                raise PreprocessError.ValueError(
                    f"[Group {self._current_group+1}] _generate_masterframe: unknown dtype {dtype!r}"
                )

        prep_utils.update_header_by_overwriting(getattr(self, f"{dtype}sig_output"), header)

        # PPFLAG: propagate from dependencies (bias=0, dark=bias, flat=bias|flatdark)
        if dtype == CALIB_TYPE_BIAS:
            ppflag_val = 0
            ingredient_ppflags = {}
        elif dtype == CALIB_TYPE_DARK:
            ppflag_val = self._ppflag.get(CALIB_TYPE_BIAS, 0)
            ingredient_ppflags = {"bias": self._ppflag.get(CALIB_TYPE_BIAS, 0)}
        elif dtype == CALIB_TYPE_FLAT:
            ppflag_val = ppflag.propagate_ppflag(
                self._ppflag.get(CALIB_TYPE_BIAS, 0), getattr(self, "_flatdark_ppflag", 0)
            )
            ingredient_ppflags = {
                "bias": self._ppflag.get(CALIB_TYPE_BIAS, 0),
                "flatdark": getattr(self, "_flatdark_ppflag", 0),
            }
        else:
            raise PreprocessError.ValueError(
                f"[Group {self._current_group+1}] Undefined behavior: _generate_masterframe is called but dtype is not bias, dark, or flat"
            )
        self._ppflag[dtype] = ppflag_val
        sanity_f_ingredients = [
            name for name, val in ingredient_ppflags.items() if val & ppflag.PPFLAG_SANITY_F_USED
        ]

        sanity_flag = self._assess_masterframe_quality_and_update_header(
            header=header,
            dtype=dtype,
            ppflag_val=ppflag_val,
            sanity_f_ingredients=sanity_f_ingredients,
        )

        if sanity_flag:
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

    def _assess_masterframe_quality_and_update_header(
        self,
        header,
        dtype,
        ppflag_val: int = 0,
        sanity_f_ingredients: list = None,
    ):
        """Merge pre-computed raw QA + pixel statistics from the master ``.header`` text file."""
        header = record_masterframe_statistics(
            getattr(self, f"{dtype}_output"),
            header,
            # dtype=dtype,
        )

        # mint AFTER the sidecar merge, or the replaced version's IMAGEID would be restored
        header = prep_utils.add_image_id(header)

        sanity_flag = self.apply_qa_criteria(header=header, dtype=dtype)  # evaluates sanity of the image itself
        if ppflag_val & ppflag.PPFLAG_SANITY_F_USED:  # consider propagated sanity flag of the ingredient frames
            if sanity_flag:
                culprits = ", ".join(sanity_f_ingredients) if sanity_f_ingredients else "bias/dark/flatdark"
                self.logger.warning(
                    f"[Group {self._current_group+1}] Master {dtype} rejected: ingredient frame(s) with SANITY=False were used ({culprits})"
                )
            sanity_flag = False

        self.update_sanity_header(header, sanity_flag)

        ppflag.set_ppflag_in_header(header, ppflag_val)

        if dtype == CALIB_TYPE_DARK:
            hotpix = self.update_bpmask(sanity=sanity_flag)
            header["NHOTPIX"] = (hotpix, "Number of hot pixels")

        prep_utils.update_header_by_overwriting(getattr(self, f"{dtype}_output"), header)
        # consumers read the id sidecar-first; the sidecar must carry the new identity too
        prep_utils.update_header_file(
            getattr(self, f"{dtype}_output"),
            fits.Header([("IMAGEID", header["IMAGEID"], header.comments["IMAGEID"])]),
        )
        return sanity_flag

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
        existing_mframe_file, relaxation_flags = prep_utils.tolerant_search(
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
            raise PreprocessError.MasterFrameNotFoundError(
                f"No pre-existing master {dtype} found in place of {template} within {max_offset} days"
            )

        if relaxation_flags.ignored_binning:
            if dry_run:
                self.logger.info(
                    f"[Group {self._current_group+1}] Would generate a binned master {dtype} "
                    f"from {os.path.basename(existing_mframe_file)} (DRY RUN)"
                )
            else:
                existing_mframe_file = self._generated_binned_master_frame(existing_mframe_file, template, dtype=dtype)
        elif not dry_run:
            self._reregister_adopted_binned_master(existing_mframe_file)

        existing_header_sanity = fits.getval(existing_mframe_file, "SANITY")
        self.logger.info(
            f"[Group {self._current_group+1}] Found pre-existing nominal (sanity: {existing_header_sanity}) master {dtype} at {os.path.basename(existing_mframe_file)}"
        )

        # PPFLAG: fetched frame gets 1 (different date), 4 (sanity F), 8 (lenient keys) as appropriate
        ppflag_val = ppflag.compute_fetch_ppflag(
            existing_mframe_file,
            template,
            existing_header_sanity,
            ignored_lenient_keys=relaxation_flags.ignored_lenient_keys,
        )
        self._ppflag[dtype] = ppflag_val

        # update the output names in raw_groups
        self.raw_groups[self._current_group][1][self._key_to_index[dtype]] = existing_mframe_file

    def _resolve_flatdark(self, template):
        """Flatdark = designated dark if dispatched, else the minimum-exptime dark on the closest date."""
        if designated_file := self._designation_dispatch.get((self._current_group, "flatdark")):
            self.flatdark_output = designated_file
            self.dark_exptime = fits.getval(designated_file, HEADER_KEY_MAP["exptime"])
            try:
                flatdark_sanity = fits.getval(designated_file, "SANITY")
            except Exception:
                flatdark_sanity = None
            self.logger.info(
                f"[Group {self._current_group+1}] Using designated (sanity: {flatdark_sanity}) flatdark "
                f"{os.path.basename(designated_file)} ({self.dark_exptime}s)"
            )
            self._flatdark_ppflag = ppflag.compute_fetch_ppflag(designated_file, template, flatdark_sanity)
            return

        max_offset = self.config_node.preprocess.max_offset
        path = PathHandler(template)
        path.name.exptime = "*"
        flatdark_template = path.preprocess._masterframe
        self.logger.debug(f"[Group {self._current_group+1}] Masterframe Search (flatdark) Template: {flatdark_template}")  # fmt: skip

        existing_flatdark_file, flatdark_relaxation_flags = prep_utils.tolerant_search(
            flatdark_template,
            CALIB_TYPE_DARK,
            max_offset=max_offset,
            future=True,
            ignore_sanity_if_no_match=get_key(self.config_node.preprocess, "ignore_sanity_if_no_match", False),
            ignore_lenient_keys_if_no_match=get_key(self.config_node.preprocess, "ignore_lenient_keys_if_no_match", False),  # fmt: skip
        )  # search closest date first, minimum exptime if multiple found
        if not existing_flatdark_file:
            self.logger.error(
                f"[Group {self._current_group+1}] No pre-existing master flatdark found in place of {flatdark_template} within {max_offset} days",
                MasterFrameNotFoundError,
            )

            raise PreprocessError.MasterFrameNotFoundError(
                f"No pre-existing master flatdark found in place of {flatdark_template} within {max_offset} days"
            )

        flatdark_sanity = fits.getval(existing_flatdark_file, "SANITY")
        self.flatdark_output = existing_flatdark_file  # mdark for mflat
        self.dark_exptime = fits.getval(existing_flatdark_file, HEADER_KEY_MAP["exptime"])
        self.logger.info(
            f"[Group {self._current_group+1}] Using nominal (sanity: {flatdark_sanity}) flatdark "
            f"{os.path.basename(existing_flatdark_file)} ({self.dark_exptime}s)"
        )
        self._flatdark_ppflag = ppflag.compute_fetch_ppflag(
            existing_flatdark_file,
            flatdark_template,
            flatdark_sanity,
            ignored_lenient_keys=flatdark_relaxation_flags.ignored_lenient_keys,
        )

    def _generated_binned_master_frame(self, existing_mframe_file, template, dtype):
        if not (dtype == CALIB_TYPE_FLAT):
            self.logger.error(
                f"[Group {self._current_group+1}] Undefined behavior: _generated_binned_master_frame is called but dtype is not flat",
                ValueError,
            )
            raise PreprocessError.ValueError(
                f"[Group {self._current_group+1}] Undefined behavior: _generated_binned_master_frame is called but dtype is not flat"
            )

        self.logger.info(f"[Group {self._current_group+1}] Generating binned master frame for {dtype}")

        from .calc import bin_image
        from .utils import set_inspcomm_in_header
        from .ppflag import propagate_ppflag, set_ppflag_in_header

        n_binning = NameHandler(template).n_binning
        data, header = fits.getdata(existing_mframe_file, header=True)
        binned_mflat = bin_image(data, bin_x=n_binning, bin_y=n_binning, method="mean")

        name = NameHandler(existing_mframe_file)
        name.n_binning = n_binning
        binned_mflat_path = os.path.join(os.path.dirname(existing_mframe_file), name.mflat_basename)

        header["XBINNING"] = n_binning
        header["YBINNING"] = n_binning
        header["NAXIS1"] = binned_mflat.shape[1]
        header["NAXIS2"] = binned_mflat.shape[0]
        # true lineage is the source flat alone; its own ingredients are reached through its image_qa row
        for card in [k for k in header if k.startswith("IMCMB") or k.startswith("IMCID")]:
            del header[card]
        header = prep_utils.write_IMCMB_to_header(header, [existing_mframe_file])
        # header["SANITY"] = False
        header = prep_utils.add_image_id(header)  # fresh id: this is a new file, not the source master
        header = prep_utils.set_pipe_ver_in_header(header)  # this run made the copy, not the source's version
        header = set_inspcomm_in_header(header, "Auto-generated binned master frame")
        header = set_ppflag_in_header(header, propagate_ppflag(header["PPFLAG"], 2))
        fits.writeto(binned_mflat_path, binned_mflat, header=header, overwrite=True)

        self.logger.debug(
            f"[Group {self._current_group+1}] Generated binned master frame for binning {n_binning}x{n_binning} {dtype} at {os.path.basename(existing_mframe_file)}"
        )

        # a new product of this run: register it even though the group holds no raw flats
        qa_id = self.create_image_qa_data(binned_mflat_path, self.process_status_id)
        self.create_image_qa_dependencies(binned_mflat_path, qa_id)

        return binned_mflat_path

    def _reregister_adopted_binned_master(self, path):
        """A re-adopted auto-binned copy self-heals its image_qa row and edges (upsert by name)."""
        if not self.is_connected:
            return
        try:
            marker = fits.getval(path, "INSPCOMM")
        except KeyError:
            return
        if marker == "Auto-generated binned master frame":
            qa_id = self.create_image_qa_data(path, self.process_status_id)
            self.create_image_qa_dependencies(path, qa_id)

    def data_reduction(self, device_id=None, use_gpu: bool = True, dry_run: bool = False):
        self._use_gpu = all([use_gpu, self._use_gpu])

        if not self.sci_input:
            self.logger.info(f"[Group {self._current_group+1}] No science frames found, skipping data reduction.")
            self.all_results = None
            for attr in ("bias_data", "dark_data", "flat_data"):
                if attr in self.__dict__:
                    del self.__dict__[attr]
            return

        reasons = [self._sci_change(file) for file in self.sci_output]
        todo = [r is not None for r in reasons]
        self.logger.info(
            f"[Group {self._current_group+1}] {len(reasons)} science image(s), policy={self._change_policy}: "
            + ", ".join(
                f"{sum(1 for r in reasons if r == kind)} {kind}"
                for kind in ("missing", "designated", "regenerated", "sanity", "better-match")
            )
            + f", {len(reasons) - sum(todo)} up to date"
        )

        if not any(todo) and not (self.overwrite):
            self.logger.info(f"[Group {self._current_group+1}] All images are already processed")
            return
        elif self.overwrite:
            input_files = self.sci_input
            output_files = self.sci_output
        else:
            input_files = [infile for infile, t in zip(self.sci_input, todo) if t]
            output_files = [outfile for outfile, t in zip(self.sci_output, todo) if t]

        if dry_run:
            self.logger.info(
                f"[Group {self._current_group+1}] Would process {len(output_files)} science images (DRY RUN)"
            )
            self.logger.info(f"[Group {self._current_group+1}] Would create processed files: {output_files} (DRY RUN)")
            return

        st = time.time()
        device_id = device_id if self._use_gpu else "CPU"
        n_head_blocks = get_key(self.config_node.preprocess, "n_head_blocks", 8)

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
                n_head_blocks=n_head_blocks,
            )

        self.logger.info(
            f"[Group {self._current_group+1}] Completed data reduction for {len(self.sci_input)} "
            f"images in {time_diff_in_seconds(st)} seconds "
            f"({time_diff_in_seconds(st, return_float=True)/len(self.sci_input):.1f} s/image)"
        )

        # Update pipeline progress after data reduction
        self._check_group_outputs(self._current_group)
        self.update_progress(self._output_progress)

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

        if self.overwrite:
            pairs = list(zip(self.sci_input, self.sci_output))
        else:
            # must stay the same predicate as data_reduction: this stages the sidecar it bakes in
            pairs = [(i, o) for i, o in zip(self.sci_input, self.sci_output) if self._sci_change(o)]

        if self.sci_input and not pairs:
            self.logger.info(
                f"[Group {self._current_group+1}] All science outputs already exist; skipping header preparation"
            )
            return

        bias, dark, flat = self.bias_output, self.dark_output, self.flat_output
        n_head_blocks = get_key(self.config_node.preprocess, "n_head_blocks", 8)

        sci_ppflag = ppflag.propagate_ppflag(
            self._ppflag.get(CALIB_TYPE_BIAS, ppflag.get_ppflag_from_header(bias)),
            self._ppflag.get(CALIB_TYPE_DARK, ppflag.get_ppflag_from_header(dark)),
            self._ppflag.get(CALIB_TYPE_FLAT, ppflag.get_ppflag_from_header(flat)),
        )

        for raw_file, processed_file in pairs:
            header = get_header(raw_file)
            header["SATURATE"] = prep_utils.get_saturation_level(header, bias, dark, flat)
            header = prep_utils.write_IMCMB_to_header(header, [bias, dark, flat, raw_file])
            ppflag.set_ppflag_in_header(header, sci_ppflag)
            header = prep_utils.ensure_mjd_in_header(header, logger=self.logger)
            header = prep_utils.sanitize_header(header)
            header = prep_utils.add_image_id(header)  # mint fresh; do not inherit the raw frame's IMAGEID
            header = add_padding(header, n_head_blocks, copy_header=True)
            self.logger.debug(
                f"Header size: {(x := len(header.tostring()))} bytes, {x//2880} blocks + {(x%2880)/80} lines"
            )
            prep_utils.write_header(processed_file, header)

    def make_masterframe_plots(self, file_path: str, dtype: str, group_index: int, dry_run: bool = False):
        if dtype == CALIB_TYPE_BIAS:
            plot_bias(file_path, overwrite=self.overwrite, dry_run=dry_run)
        elif dtype == CALIB_TYPE_DARK:
            bpmask_file = file_path.replace("dark", "bpmask")
            plot_bpmask(bpmask_file, overwrite=self.overwrite, dry_run=dry_run)
            badpix = fits.getval(bpmask_file, "BADPIX", ext=1) or 1
            mask = fits.getdata(bpmask_file, ext=1) != badpix
            fmask = mask.ravel()
            plot_dark(file_path, fmask, overwrite=self.overwrite, dry_run=dry_run)
        elif dtype == CALIB_TYPE_FLAT:
            bpmask_file = self._get_raw_group("bpmask_output", group_index)
            badpix = fits.getval(bpmask_file, "BADPIX", ext=1) or 1
            mask = fits.getdata(bpmask_file, ext=1) != badpix
            fmask = mask.ravel()
            plot_flat(file_path, fmask, overwrite=self.overwrite, dry_run=dry_run)

    def make_plots(self, group_index: int, dry_run: bool = False):
        try:
            self.logger.info(f"[Group {group_index+1}] Generating plots for master calibration frames")

            # bias
            if CALIB_TYPE_BIAS in self.calib_types:
                bias_file = self._get_raw_group("bias_output", group_index)
                if os.path.exists(bias_file):
                    plot_bias(bias_file, overwrite=self.overwrite, dry_run=dry_run)
                else:
                    self.logger.warning(f"[Group {group_index+1}] Bias image does not exist. Skipping bias plot.")
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping bias plot")

            dark_file = self._get_raw_group("dark_output", group_index)
            flat_file = self._get_raw_group("flat_output", group_index)
            bpmask_file = self._get_raw_group("bpmask_output", group_index)

            # bpmask
            if CALIB_TYPE_DARK in self.calib_types:
                if os.path.exists(bpmask_file):
                    plot_bpmask(bpmask_file, overwrite=self.overwrite, dry_run=dry_run)
                else:
                    self.logger.warning(f"[Group {group_index+1}] BPMask image does not exist. Skipping bpmask plot.")
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping bpmask plot")

            # dark
            if CALIB_TYPE_DARK in self.calib_types:
                if os.path.exists(dark_file):
                    plot_dark(
                        dark_file,
                        bpmask_file=bpmask_file if os.path.exists(bpmask_file) else None,
                        overwrite=self.overwrite,
                        dry_run=dry_run,
                    )
                else:
                    self.logger.warning(f"[Group {group_index+1}] Dark image does not exist. Skipping dark plot.")
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping dark plot")

            # flat
            if CALIB_TYPE_FLAT in self.calib_types:
                if os.path.exists(flat_file):
                    plot_flat(
                        flat_file,
                        bpmask_file=bpmask_file if os.path.exists(bpmask_file) else None,
                        overwrite=self.overwrite,
                        dry_run=dry_run,
                    )
                else:
                    self.logger.warning(f"[Group {group_index+1}] Flat image does not exist. Skipping flat plot.")
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping flat plot")

            self.logger.info(f"[Group {group_index+1}] Completed generating plots for master calibration frames")

            # science
            st = time.time()
            sci_pairs = list(
                zip(self._get_raw_group("sci_input", group_index), self._get_raw_group("sci_output", group_index))
            )
            num_sci = len(sci_pairs)
            if num_sci and not self.master_frame_only:
                self.logger.info(f"[Group {group_index+1}] Generating plots for science frames ({num_sci} images)")

                for input_img, output_img in sci_pairs:
                    plot_sci(input_img, output_img, is_too=self.is_too, overwrite=self.overwrite, dry_run=dry_run)

                self.logger.info(
                    f"[Group {group_index+1}] Completed plot generation for images in {time_diff_in_seconds(st)} seconds "
                    f"({time_diff_in_seconds(st, return_float=True)/(num_sci or 1):.1f} s/image)"
                )
            else:
                self.logger.info(f"[Group {group_index+1}] Skipping science plot")
        except Exception as e:
            self.logger.error(f"[Group {group_index+1}] Error making plots: {e}", e)
            self.logger.debug(traceback.format_exc())

    def update_bpmask(self, sanity=True):
        header = self.get_header(CALIB_TYPE_DARK)
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
        prep_utils.set_pipe_ver_in_header(newhdu.header)
        primary_hdu = fits.PrimaryHDU()
        newhdul = fits.HDUList([primary_hdu, newhdu])
        newhdul.writeto(self.bpmask_output, overwrite=True)
        return np.sum(hot_mask)
