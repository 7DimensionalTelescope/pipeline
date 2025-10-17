import os
import re
import time
import shutil
import numpy as np
from typing import Any, List, Tuple, Union, Any, Optional
from dataclasses import dataclass, field
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.wcs import WCS
from astropy.table import Table
import astropy.units as u
from functools import cached_property
from concurrent.futures import ThreadPoolExecutor, as_completed


from .. import external
from ..const import PIXSCALE, PipelineError
from ..utils import swap_ext, add_suffix, force_symlink, time_diff_in_seconds, unique_filename, atleast_1d
from ..header import update_padded_header, reset_header, fitsrec_to_header
from ..services.memory import MemoryMonitor
from ..config import SciProcConfiguration
from ..config.utils import get_key
from ..services.setup import BaseSetup
from ..io.cfitsldac import write_ldac
from .utils import (
    polygon_info_header,
    read_scamp_header,
    build_wcs,
    polygon_info_header,
    get_fov_quad,
    strip_wcs,
    read_text_header,
)
from .evaluation import evaluate_single_wcs, evaluate_joint_wcs

from ..services.database.handler import DatabaseHandler
from ..services.database.table import QAData

# from .generate_refcat_gaia import get_refcat_gaia


class Astrometry(BaseSetup, DatabaseHandler):
    """A class to handle astrometric solutions for astronomical images.

    This class manages the complete astrometric pipeline including plate solving,
    source extraction, and WCS header updates for astronomical images. It supports
    both sequential and parallel processing modes.

    Attributes:
        config (Configuration): Configuration object containing pipeline settings
        logger: Logger instance for recording operations and debugging
        queue (Optional[QueueManager]): Queue manager for parallel processing tasks

    Example:
        >>> astro = Astrometry(config="/data/...7DT05/m550/calib_7DT05_T00176_20250102_012738_m550_100.0.yml")
        >>> astro.run(solve_field=True, joint_scamp=True)
    """

    start_time = None

    def __init__(
        self,
        config: Union[str, SciProcConfiguration] = None,
        logger: Any = None,
        queue: Union[bool, Any] = False,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the astrometry module.

        Args:
            config: Configuration object or path to config file
            logger: Custom logger instance (optional)
            queue: QueueManager instance or boolean to enable parallel processing
        """
        super().__init__(config, logger, queue)
        self._flag_name = "astrometry"
        self.logger.debug(f"Astrometry Queue is '{queue}'")

        self.start_time = time.time()
        self.define_paths()

        self.qa_ids = []
        DatabaseHandler.__init__(self, add_database=self.config.settings.is_pipeline)

        if self.is_connected:
            self.set_logger(logger)
            self.logger.debug("Initialized DatabaseHandler for pipeline and QA data management")
            self.pipeline_id = self.create_pipeline_data(self.config)
            if self.pipeline_id is not None:
                self.update_pipeline_progress(0, "configured")
                for image in self.input_images:
                    qa_id = self.create_qa_data("science", image=image, output_file=image)
                    self.qa_ids.append(qa_id)
            else:
                self.logger.warning("Pipeline record creation failed, skipping QA data creation")

    @classmethod
    def from_list(cls, images, working_dir=None):
        images = [os.path.abspath(image) for image in sorted(images)]  # this is critical for soft links
        for image in images:
            if not os.path.exists(image):
                raise FileNotFoundError(f"File does not exist: {image}")

        config = SciProcConfiguration.base_config(input_images=images, working_dir=working_dir, logger=True)
        return cls(config=config)

    @property
    def sequential_task(self):
        """[(number, name, use_gpu), ...]"""
        return [(1, "run", False)]

    def run(
        self,
        se_preset: str = "prep",
        joint_scamp: bool = False,
        force_solve_field: bool = False,
        evaluate_prep_sol: bool = True,
        max_scamp: int = 3,
        use_threading: bool = False,
        # processes=["sextractor", "scamp", "header_update"],
        # use_gpu: bool = False,
        overwrite=False,
        solvefield_args=[],
        avoid_solvefield=True,
    ) -> None:
        """Execute the complete astrometry pipeline.

        Performs a sequence of operations including plate solving, source extraction,
        and WCS header updates. Supports both parallel and sequential processing.

        Args:
            solve_field: Whether to run plate-solving
            joint_scamp: Whether to run SCAMP jointly on all images
            use_missfits: Whether to use missfits for header updates
            processes: List of operations to perform
            prefix: Prefix for sextractor
        """
        try:
            self.start_time = time.time()

            self.logger.info(f"Start 'Astrometry'")

            self.inject_wcs_guess(self.input_images)
            # Source Extractor
            self.run_sextractor(self.soft_links_to_input_images, se_preset=se_preset, overwrite=overwrite)

            # flexibly iterate to refine
            for i, (image_info, prep_cat) in enumerate(zip(self.images_info, self.prep_cats)):
                try:
                    if force_solve_field:
                        raise Exception("force_solve_field")
                    # run initial solve: scamp or solve-field
                    self.run_scamp(prep_cat, scamp_preset="prep", joint=False, overwrite=overwrite)
                    self.update_pipeline_progress(5, "astrometry-scamp-prep")

                    if evaluate_prep_sol:
                        self.evaluate_solution(
                            input_images=image_info.image_path,
                            images_info=image_info,
                            prep_cats=prep_cat,
                            suffix="prepwcs",
                            use_threading=use_threading,
                            export_eval_cards=True,
                        )

                    self.logger.info(f"Running main scamp iteration [{i+1}/{len(self.prep_cats)}] for {prep_cat}")
                    # main scamp iteration
                    self._iterate_scamp(
                        prep_cat, image_info, evaluate_prep_sol, use_threading, max_scamp, overwrite=overwrite
                    )

                    self.update_pipeline_progress(10, "astrometry-scamp-main")
                    self.logger.info(f"Scamp iteration completed [{i+1}/{len(self.prep_cats)}] for {prep_cat}")

                    if image_info.bad:
                        self.logger.warning(
                            f"Bad solution. UNMATCH: {image_info.rsep_stats.unmatched_fraction}, "
                            f"RSEP_P95: {image_info.reference_sep_p95}"
                            f"after {max_scamp} iterations for {image_info.image_path}"
                        )
                        if not avoid_solvefield:
                            raise Exception(f"Bad solution")

                except Exception as e:
                    # if evaluate_prep_sol:
                    #     self.evaluate_solution(suffix="prepwcs", use_threading=use_threading, export_eval_cards=True)
                    self.logger.warning(e)
                    self.logger.warning(f"Solve-field triggered, better solution not guaranteed for {prep_cat}")

                    # self.run_solve_field(input_catalogs=self.prep_cats, output_images=self.solved_images)
                    self.run_solve_field(input_catalogs=prep_cat, output_images=None, solvefield_args=solvefield_args)
                    if evaluate_prep_sol:
                        self.evaluate_solution(
                            suffix="solvefieldwcs", use_threading=use_threading, export_eval_cards=True
                        )
                    self._iterate_scamp(
                        prep_cat, image_info, evaluate_prep_sol, use_threading, max_scamp, overwrite=overwrite
                    )

            # evaluate main scamp
            self.evaluate_solution(suffix="wcs", isep=True, use_threading=use_threading, scamp_preset="main")
            self.update_pipeline_progress(15, "astrometry-scamp-main-eval")
            # update the input image
            self.update_header()
            self.update_pipeline_progress(20, "astrometry")

            for image, qa_id in zip(self.input_images, self.qa_ids):
                qa_data = QAData.from_header(
                    fits.getheader(image), "science", "science", self.pipeline_id, os.path.basename(image)
                )
                qa_dict = qa_data.to_dict()
                qa_dict["qa_id"] = qa_id
                qa_id = self.qa_db.update_qa_data(**qa_dict)

            self.config.flag.astrometry = True

            self.logger.info(
                f"'Astrometry' is completed in {time_diff_in_seconds(self.start_time)} seconds "
                f"({time_diff_in_seconds(self.start_time, return_float=True) / len(self.input_images):.2f} seconds per image)"
            )
            self.logger.debug(MemoryMonitor.log_memory_usage)
        except Exception as e:
            self.logger.error(f"Error during astrometry processing: {str(e)}", exc_info=True)
            raise

    def _iterate_scamp(self, prep_cat, image_info, evaluate_prep_sol, use_threading, max_scamp, overwrite=True):
        """main scamp iteration"""
        for _ in range(max_scamp):
            self.run_scamp([prep_cat], scamp_preset="main", joint=False, overwrite=(_ == max_scamp - 1 or overwrite))
            if evaluate_prep_sol:
                self.evaluate_solution(
                    input_images=image_info.image_path,
                    images_info=image_info,
                    prep_cats=prep_cat,
                    suffix="iterwcs",
                    plot=True,
                    isep=False,
                    use_threading=use_threading,
                    export_eval_cards=True,
                )
                self.logger.debug(f"SEP_P95: {image_info.reference_sep_p95} for {image_info.image_path}")
                if image_info.good:
                    self.logger.debug(f"Stop iterating at {_}")
                    break

    # deprecated
    def _run_joint(
        self,
        se_preset: str = "prep",
        joint_scamp: bool = True,
        force_solve_field: bool = False,
        evaluate_prep_sol: bool = True,
        refine_init_wcs: bool = True,
        use_threading: bool = False,
        # processes=["sextractor", "scamp", "header_update"],
        # use_gpu: bool = False,
    ) -> None:
        """Execute the complete astrometry pipeline.

        Performs a sequence of operations including plate solving, source extraction,
        and WCS header updates. Supports both parallel and sequential processing.

        Args:
            solve_field: Whether to run plate-solving
            joint_scamp: Whether to run SCAMP jointly on all images
            use_missfits: Whether to use missfits for header updates
            processes: List of operations to perform
            prefix: Prefix for sextractor
        """
        try:
            start_time = self.start_time or time.time()
            self.logger.info(f"Start 'Astrometry'")
            self.define_paths()

            self.inject_wcs_guess(self.input_images)
            # Source Extractor
            self.run_sextractor(self.soft_links_to_input_images, se_preset=se_preset)

            # run initial solve: scamp or solve-field
            try:
                if force_solve_field:
                    raise Exception("Force solve field")
                self.run_scamp(self.prep_cats, scamp_preset="prep", joint=False)
                if evaluate_prep_sol:
                    self.evaluate_solution(suffix="prepwcs", use_threading=use_threading, export_eval_cards=True)
            except Exception as e:
                if evaluate_prep_sol:
                    self.evaluate_solution(suffix="prepwcs", use_threading=use_threading, export_eval_cards=True)
                self.logger.warning(e)
                self.run_solve_field(self.soft_links_to_input_images, self.solved_images)
                if evaluate_prep_sol:
                    self.evaluate_solution(suffix="solvefieldwcs", use_threading=use_threading, export_eval_cards=True)

            # even more refinement
            if refine_init_wcs:
                self.run_scamp(self.prep_cats, scamp_preset="main", joint=False)
                if evaluate_prep_sol:
                    self.evaluate_solution(suffix="prep2wcs", use_threading=use_threading, export_eval_cards=True)

            # run main scamp
            self.run_scamp(self.prep_cats, scamp_preset="main", joint=joint_scamp)
            self.evaluate_solution(use_threading=use_threading)

            # update the input image
            self.update_header()

            self.config.flag.astrometry = True

            self.logger.info(
                f"'Astrometry' is completed in {time_diff_in_seconds(start_time)} seconds "
                f"({time_diff_in_seconds(start_time, return_float=True) / len(self.input_images):.2f} seconds per image)"
            )
            self.logger.debug(MemoryMonitor.log_memory_usage)
        except Exception as e:
            self.logger.error(f"Error during astrometry processing: {str(e)}", exc_info=True)
            raise

    def define_paths(self) -> Tuple[List[str], List[str], List[str]]:
        self.logger.info("Defining paths for astrometry")
        # self.path_astrometry = self.path.astrometry.tmp_dir

        # override if astrometry.input_images is set
        local_inim = get_key(self.config, "astrometry.input_images")
        if local_inim is not None:
            inims = local_inim
        # otherwise use the common input
        else:
            inims = self.config.input.calibrated_images
            self.config.astrometry.input_images = inims

        self.input_images = inims

        # soft_links = [os.path.join(self.path_astrometry, os.path.basename(s)) for s in inims]
        soft_links = self.path.astrometry.soft_link
        for inim, soft_link in zip(inims, soft_links):
            force_symlink(inim, soft_link)
            self.logger.debug(f"Soft link created: {inim} -> {soft_link}")
        self.soft_links_to_input_images = soft_links

        self.prep_cats = atleast_1d(self.path.astrometry.catalog)  # fits_ldac sextractor output
        self.solved_images = [add_suffix(s, "solved") for s in soft_links]  # solve-field output
        self.solved_heads = [swap_ext(s, "head") for s in self.prep_cats]  # scamp output
        self.images_info = [ImageInfo.parse_image_header_info(image) for image in self.input_images]
        for i, image_info in enumerate(self.images_info):
            image_info.logger = self.logger  # pass the logger
            image_info.head = self.solved_heads[i]

        local_astref = (
            self.config.astrometry.local_astref or self.path.astrometry.astrefcat
        )  # None if no local astrefcat
        if not os.path.exists(local_astref):
            try:
                raise PipelineError(f"Local astrefcat {local_astref} does not exist")
                get_refcat_gaia(self.input_images[0])
            except:
                local_astref = None
        self.config.astrometry.local_astref = local_astref

    def inject_wcs_guess(
        self, input_images: List[str], wcs_list: List[WCS | fits.Header] = None, reset_image_header: bool = True
    ) -> None:
        """Inject WCS into image header."""

        self.logger.info(f"Injecting initial WCS into {len(input_images)} image(s)")
        input_images = input_images or self.input_images

        if wcs_list:
            self.logger.debug("WCS to inject provided.")
            assert len(wcs_list) == len(input_images)
        else:
            self.logger.debug("No WCS provided. Using coarse WCS from image header.")
            wcs_list = [image_info.coarse_wcs for image_info in self.images_info]
        assert len(input_images) == len(wcs_list)

        self.injected_wcs = []
        for image, wcs in zip(input_images, wcs_list):
            self.logger.debug(f"Injecting WCS into {image}")
            self.injected_wcs.append(wcs)

            if reset_image_header:
                reset_header(image)

            with fits.open(image, mode="update") as hdul:
                if isinstance(wcs, fits.Header):
                    hdul[0].header.update(wcs)
                elif isinstance(wcs, WCS):
                    hdul[0].header.update(wcs.to_header(relax=True))
                else:
                    raise ValueError(f"Invalid WCS type: {type(wcs)}")

    def reset_header(self, input_images: List[str] = None) -> None:
        """Reset header of input images."""
        input_images = input_images or self.input_images
        for image in input_images:
            self.logger.debug(f"Resetting header of {image}")
            reset_header(image)

    def run_solve_field(
        self,
        input_catalogs: str | List[str] = None,
        input_images: str | List[str] = None,
        output_images: str | List[str] = None,  # only for input_images
        overwrite: bool = False,
        solvefield_args: list = [],
    ) -> None:
        """Runs astrometry.net's solve-field to determine WCS solution for each image."""
        self.logger.info(f"Start solve-field")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        input_catalogs = atleast_1d(input_catalogs) if input_catalogs is not None else input_catalogs  # priority
        input_images = atleast_1d(input_images) if input_images is not None else input_images
        output_images = atleast_1d(output_images) if output_images is not None else output_images
        self.logger.debug(f"input_catalogs: {input_catalogs}")
        self.logger.debug(f"input_images: {input_images}")
        self.logger.debug(f"output_images: {output_images}")

        if not input_catalogs and not input_images:
            raise ValueError("Either input_catalogs or input_images must be provided for run_solve_field")

        # solved_flag_files = [swap_ext(input, ".solved") for input in input_catalogs or input_images]

        # filtered_data = [
        #     (inp, out)
        #     for inp, out, solved_flag_file in zip(input_images, output_images, solved_flag_files)
        #     if not os.path.exists(solved_flag_file)
        # ]

        # if not filtered_data:
        #     self.logger.info("All input images have already been solved. Exiting solve-field.")
        #     return

        # input_images, output_images = zip(*filtered_data)

        # if self.queue:
        #     self._submit_task(
        #         external.solve_field,
        #         input_image=input_images,
        #         output_image=output_images,
        #         # dump_dir=self.path_astrometry,
        #         pixscale=self.path.pixscale,  # PathHandler brings pixscale from NameHandler
        #     )
        # else:
        solvefield_wcs_list = []
        if input_catalogs:
            for i, (slink, pixscale) in enumerate(zip(input_catalogs, self.path.pixscale)):
                if os.path.exists(swap_ext(slink, ".solved")):
                    self.logger.info(f"Solve-field already run: {swap_ext(slink, '.solved')}. Skipping...")
                    continue
                wcs_file = external.solve_field(
                    input_catalog=slink,
                    # dump_dir=self.path_astrometry,
                    pixscale=pixscale,
                    overwrite=overwrite,
                    solvefield_args=solvefield_args,
                )
                self.logger.info(f"Completed solve-field [{i+1}/{len(input_catalogs)}]")
                self.logger.debug(f"Solve-field input: {slink}, output: {wcs_file}")

                self.images_info[i].sip_wcs = wcs_file
                solvefield_wcs_list.append(wcs_file)

        elif input_images:  # this is deprecated. just for code reuse.
            assert len(input_images) == len(output_images)
            for i, (slink, sfile, pixscale) in enumerate(zip(input_images, output_images, self.path.pixscale)):
                if os.path.exists(swap_ext(slink, ".solved")):
                    self.logger.info(f"Solve-field already run: {swap_ext(slink, '.solved')}. Skipping...")
                    continue
                external.solve_field(
                    input_image=slink,
                    output_image=sfile,
                    # dump_dir=self.path_astrometry,
                    pixscale=pixscale,
                    overwrite=overwrite,
                    solvefield_args=solvefield_args,
                )
                self.logger.info(f"Completed solve-field [{i+1}/{len(input_images)}]")
                self.logger.debug(f"Solve-field input: {slink}, output: {sfile}")
        else:
            pass

        self.logger.debug(MemoryMonitor.log_memory_usage)

        self.update_catalog(input_catalogs, solvefield_wcs_list)

    def run_sextractor(
        self,
        input_images: List[str],
        output_catalogs: List[str] = None,
        se_preset: str = "prep",
        overwrite=False,
    ) -> List[str]:
        """Run Source Extractor on solved images.

        Extracts sources from solved images for use in SCAMP calibration.
        Creates FITS_LDAC format catalogs required by SCAMP.

        Args:
            files: Paths to astrometrically solved FITS files

        Returns:
            List of paths to generated source catalogs
        """
        # parallelize if queue=True
        self.logger.info("Start pre-sextractor")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        output_catalogs = output_catalogs or self.prep_cats

        if self.queue:
            self._submit_task(
                external.sextractor,
                input_images,
                outcat=output_catalogs,
                prefix=se_preset,
                logger=self.logger,
                fits_ldac=True,
                overwrite=overwrite,
            )
        else:
            for i, (solved_image, prep_cat) in enumerate(zip(input_images, output_catalogs)):
                external.sextractor(
                    solved_image,
                    outcat=prep_cat,
                    se_preset=se_preset,
                    logger=self.logger,
                    fits_ldac=True,
                    overwrite=overwrite,
                )
                self.logger.info(f"Completed sextractor (prep) [{i+1}/{len(input_images)}]")
                self.logger.debug(f"{solved_image}")

        self.logger.debug(MemoryMonitor.log_memory_usage)

        return output_catalogs

    def run_scamp(
        self,
        input_catalogs: List[str] = None,
        joint: bool = True,
        astrefcat: str = None,
        path_ref_scamp: str = None,
        scampconfig: str = None,
        scamp_preset: str = "prep",
        scamp_args: str = None,
        apply_wcs_to_catalog: bool = True,
        overwrite=False,
    ) -> None:
        """Run SCAMP for astrometric calibration.

        Performs astrometric calibration using SCAMP, either jointly on all images
        or individually. Supports parallel processing for individual mode.

        Args:
            files: Paths to astrometrically solved FITS files
            joint: Whether to process all images together
            apply_wcs_to_catalog: Whether to apply the solved WCS to the catalog.
                If it's the last scamp, it's not necessary unless you want to evaluate the solution.
        """
        st = time.time()
        self.logger.info(f"Start {'joint' if joint else 'individual'} scamp")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        # presex_cats = [os.path.splitext(s)[0] + f".{prefix}.cat" for s in files]
        input_catalogs = atleast_1d(input_catalogs or self.prep_cats)
        self.logger.debug(f"Scamp input catalogs: {input_catalogs}")

        # path for scamp refcat download
        if path_ref_scamp is False:
            self.logger.debug("SCAMP REFOUT_CATPATH is CWD")
            path_ref_scamp = False
        else:
            path_ref_scamp = path_ref_scamp or self.path.astrometry.ref_query_dir

        # use local astrefcat if tile obs
        astrefcat = astrefcat or self.config.astrometry.local_astref
        self.logger.debug(f"Using astrefcat: {astrefcat}")

        # joint scamp
        if joint:
            # write target files into a text file
            cat_to_scamp = os.path.join(self.path.astrometry.tmp_dir, "scamp_input.cat")
            with open(cat_to_scamp, "w") as f:
                for precat in input_catalogs:
                    f.write(f"{precat}\n")

            solved_heads = external.scamp(
                cat_to_scamp,
                path_ref_scamp=path_ref_scamp,
                local_astref=astrefcat,
                scampconfig=scampconfig,
                scamp_args=scamp_args,
                scamp_preset=scamp_preset,
            )
            self.logger.debug(f"Joint run solved_heads: {solved_heads}")

        # individual, parallel: needs update to get the return (solved_heads)
        elif self.queue:
            self._submit_task(
                external.scamp,
                input_catalogs,
                path_ref_scamp=path_ref_scamp,
                local_astref=astrefcat,
                scampconfig=scampconfig,
                scamp_args=scamp_args,
                scamp_preset=scamp_preset,
            )
            solved_heads = self.queue.results[0]
            self.logger.debug(f"Parallel run solved_heads: {solved_heads}")

        # individual, sequential
        else:
            solved_heads = []
            for precat in input_catalogs:
                solved_head = external.scamp(
                    precat,
                    path_ref_scamp=path_ref_scamp,
                    local_astref=astrefcat,
                    scampconfig=scampconfig,
                    scamp_args=scamp_args,
                    scamp_preset=scamp_preset,
                )
                solved_heads.extend(solved_head)
                self.logger.debug(f"Completed scamp for {precat}")
            self.logger.debug(f"Individual run solved_heads: {solved_heads}")

        # back up prep solutions in a different name. The original name must be accessible by ImageInfo.head
        if scamp_preset != "main":
            for i, head in enumerate(solved_heads):
                backup_head = unique_filename(add_suffix(head, "prep"))
                shutil.copy(head, backup_head)
                # solved_heads[i] = backup_head
                self.logger.debug(f"Backed up prep head {head} to {backup_head}")

        # update WCS in ImageInfo
        for solved_head, input_catalog in zip(solved_heads, input_catalogs):
            # finding the right image_info is difficult
            # idx = self.input_images.index(self.images_info[2].image_path)
            # image_info.head = solved_head
            # self.logger.debug(f"Updated WCS in ImageInfo {image_info.head}")
            # self.logger.debug(f"ImageInfo of {image_info.image_path} is updated")
            with open(solved_head, "r") as f:
                if scamp_preset == "main" and not any("PV1_0" in line for line in f):
                    self.logger.error(f"No PV1_0 in {solved_head} - solution invalid")
                    # raise PipelineError(f"No PV1_0 in {solved_head} - solution invalid")

        self.logger.info(f"Completed {scamp_preset} scamp in {time_diff_in_seconds(st)} seconds")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        if apply_wcs_to_catalog:
            self.update_catalog(input_catalogs, solved_heads)
            self.logger.info(f"Updated catalogs with solved heads")

    def update_catalog(self, input_catalogs: List[str], solved_heads: List[str], update_fwhm: bool = False) -> None:
        """
        Update catalog with solved heads.
        Currently only updates RA/Dec.
        FWHM_WORLD not updated.
        """
        input_catalogs = input_catalogs or self.prep_cats
        solved_heads = solved_heads or self.solved_heads
        assert len(input_catalogs) == len(solved_heads)

        for input_catalog, solved_head in zip(input_catalogs, solved_heads):
            # update wcs in LDAC_IMHEAD
            image_header = fitsrec_to_header(
                fits.getdata(input_catalog, ext=1)
            )  # assuming LDAC_IMHED is the second extension
            image_header = strip_wcs(image_header)  # strip the previous WCS
            wcs_header = read_scamp_header(solved_head)
            image_header.update(wcs_header)

            # update coordinates in LDAC_OBJECT
            # wcs = WCS(image_header)  # not the same as the scamp header
            wcs = read_scamp_header(solved_head, return_wcs=True)
            tbl = Table.read(input_catalog, hdu=2)  # assuming LDAC_OBJECT is the third extension
            ra, dec = wcs.all_pix2world(tbl["X_IMAGE"], tbl["Y_IMAGE"], 1)
            tbl["ALPHA_J2000"] = ra
            tbl["DELTA_J2000"] = dec

            if update_fwhm:
                # tbl["FWHM_WORLD"] = something
                pass

            # update (overwrite) the catalog
            write_ldac(image_header, tbl, input_catalog)  # internally c script with cfitsio
            self.logger.debug(f"Updated catalog {input_catalog} with {solved_head}")

    def evaluate_solution(
        self,
        input_images: List[str] = None,
        images_info=None,
        prep_cats: List[str] = None,
        plot=True,
        isep=True,
        suffix=None,
        num_sci=200,
        num_ref=200,
        use_threading=True,
        export_eval_cards=False,
        scamp_preset="prep",  # for determining error
    ) -> None:
        """Evaluate the solution. This was developed in lack of latest scamp version."""
        self.logger.debug("Start evaluate_solution")

        input_images = atleast_1d(input_images or self.input_images)
        images_info = atleast_1d(images_info or self.images_info)
        prep_cats = atleast_1d(prep_cats or self.prep_cats)
        suffix = suffix or "wcs"

        refcat = Table.read(self.config.astrometry.local_astref, hdu=2)

        def _eval_one(idx: int):
            "threading helper"
            input_image = input_images[idx]
            image_info = images_info[idx]
            prep_cat = prep_cats[idx]

            bname = os.path.basename(input_image)
            plot_path = unique_filename(
                os.path.join(self.path.astrometry.figure_dir, swap_ext(add_suffix(bname, suffix), "jpg"))
            )

            # update FOV polygon to ImageInfo. It will be cached and put into final image header later
            try:
                image_info.fov_ra, image_info.fov_dec = get_fov_quad(
                    image_info.wcs, image_info.naxis1, image_info.naxis2
                )
                self.logger.debug(f"Updated FOV polygon. RA: {image_info.fov_ra}, Dec: {image_info.fov_dec}")
            except Exception as e:
                self.logger.error(f"Failed to update FOV polygon for {image_info.image_path}: {e}")

            eval_result = evaluate_single_wcs(
                image=input_image,
                ref_cat=refcat,
                source_cat=prep_cat,
                date_obs=image_info.dateobs,
                wcs=image_info.wcs,
                H=image_info.naxis2,
                W=image_info.naxis1,
                fov_ra=image_info.fov_ra,
                fov_dec=image_info.fov_dec,
                plot_save_path=plot_path if plot else None,
                num_sci=num_sci,
                num_ref=num_ref,
                cutout_size=30,
                logger=self.logger,
            )
            return idx, eval_result

        def _apply(idx: int, eval_result):
            """apply helper to avoid code duplication"""
            matched = eval_result.matched
            rsep_stats = eval_result.rsep_stats
            psf_stats = eval_result.psf_stats

            if scamp_preset == "main" and rsep_stats.unmatched_fraction == 1.0:
                self.logger.error(
                    f"Unmatched fraction is 1.0 for {input_images[idx]}. "
                    f"Check the initial WCS guess or the reference catalog."
                )

            info = images_info[idx]
            info.matched_catalog = matched
            info.rsep_stats = rsep_stats
            # info.ref_max_mag = rsep_stats.ref_max_mag
            # info.sci_max_mag = rsep_stats.sci_max_mag
            # info.num_ref_sources = rsep_stats.num_ref
            # info.unmatched_fraction = rsep_stats.unmatched_fraction
            # info.subpixel_fraction = rsep_stats.subpixel_fraction
            # info.subsecond_fraction = rsep_stats.subsecond_fraction
            info.set_reference_sep_stats(rsep_stats.separation_stats)
            info.psf_stats = psf_stats
            self.logger.debug(f"Evaluated solution for {input_images[idx]}")
            return

        def _produce_sequential():
            for idx in range(len(input_images)):
                yield _eval_one(idx)  # yield makes a generator

        def _produce_parallel():
            max_workers = min(len(input_images), int(os.cpu_count() / 2))
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="wcs_eval") as ex:
                futures = {ex.submit(_eval_one, idx): idx for idx in range(len(input_images))}
                for fut in as_completed(futures):
                    yield fut.result()

        try:
            # Single evaluation (external rms)
            producer = _produce_parallel if use_threading else _produce_sequential
            for idx, bundle in producer():
                _apply(idx, bundle)

            # Optional joint evaluation (internal rms)
            if len(input_images) > 1 and isep:
                isep_stats_list, match_stat_list = evaluate_joint_wcs(images_info)
                for image_info, internal_sep_stats, match_stats in zip(images_info, isep_stats_list, match_stat_list):
                    image_info.set_internal_sep_stats(internal_sep_stats)
                    image_info.set_internal_match_stats(match_stats)

            if export_eval_cards:
                for image_info, prep_cat in zip(images_info, prep_cats):
                    cards = image_info.wcs_eval_cards
                    text_path = unique_filename(swap_ext(add_suffix(prep_cat, f"{suffix}_eval_cards"), "txt"))
                    with open(text_path, "w") as f:
                        f.write(fits.Header(cards).tostring(sep="\n"))

        except Exception as e:
            self.logger.error(f"Failed to evaluate solution: {e}")
            # don't raise error

        self.logger.debug(f"Completed evaluate_solution")
        self.logger.debug(MemoryMonitor.log_memory_usage)

    def update_qa_config(self) -> None:
        """Update the QA configuration."""
        self.config.qa.ellipticity = [image_info.ellipticity for image_info in self.images_info]
        self.config.qa.seeing = [image_info.seeing for image_info in self.images_info]
        self.config.qa.pa = [fits.getheader(img)["ROTANG"] for img in self.input_images]
        self.logger.debug(f"SEEING     : {self.config.qa.seeing:.3f} arcsec")
        self.logger.debug(f"ELLIPTICITY: {self.config.qa.ellipticity:.3f}")
        self.logger.debug(f"PA         : {self.config.qa.pa:.3f} deg")

    def update_header(
        self,
        inims: List[str] = None,
        heads: List[str] = None,
        reset_image_header: bool = True,
        add_polygon_info: bool = True,
    ) -> None:
        """
        Updates WCS solutions found by SCAMP to original FITS image files

        Args:
            files: Paths to solved FITS files
            inims: Paths to original input images
            links: Paths to symbolic links (need for use_missfits)
            use_missfits: Whether to use missfits for updates
        """

        heads = heads or self.solved_heads
        inims = inims or self.input_images
        self.logger.info(f"Updating WCS to header(s) of {len(heads)} image(s)")

        for image_info, solved_head, target_fits in zip(self.images_info, heads, inims):
            solved_header = read_scamp_header(solved_head)

            if reset_image_header:
                reset_header(target_fits)

            if image_info.wcs_evaluated:
                # solved_header.update(image_info.wcs_eval_cards)  # this removes duplicate COMMENT cards
                solved_header.extend(image_info.wcs_eval_cards)
                self.logger.debug(f"WCS evaluation is added to {target_fits}")
            else:
                self.logger.debug(f"WCS evaluation is missing for {target_fits}")

            if add_polygon_info:
                polygon_header = polygon_info_header(image_info)
                # solved_header.update(polygon_header)
                solved_header.extend(polygon_header)

            # update the header, consuming the COMMENT padding
            update_padded_header(target_fits, solved_header)

    def _submit_task(self, func: callable, items: List[Any], **kwargs: Any) -> None:
        """Submit tasks to the queue manager for parallel processing.

        Handles task submission and monitoring for parallel processing operations.
        Automatically assigns appropriate priorities and resource requirements.

        Args:
            func: Function to execute
            items: Items to process
            **kwargs: Additional arguments for the function
        """
        from ..services.queue import Priority

        task_ids = []

        for i, item in enumerate(items):
            if type(item) is not tuple:
                item = (item,)

            # build per‐item kwargs: if the original kwarg is a list/tuple,
            # grab its i-th element, otherwise pass it through unchanged
            per_item_kwargs = {
                key: (value[i] if isinstance(value, (list, tuple)) else value) for key, value in kwargs.items()
            }

            task_id = self.queue.add_task(
                func,
                args=item,
                kwargs=per_item_kwargs,
                priority=Priority.MEDIUM,
                gpu=False,
                task_name=f"{func.__name__}_{i}",  # Dynamic task name
            )
            task_ids.append(task_id)

        self.queue.wait_until_task_complete(task_ids)


@dataclass
class ImageInfo:
    """Stores information needed for astrometry. Mostly extracted from the FITS image header."""

    image_path: str

    # Header information
    dateobs: str  # Observation date/time
    naxis1: int  # Image width
    naxis2: int  # Image height
    racent: float  # RA of image center
    decent: float  # DEC of image center
    xcent: float  # X coordinate of image center
    ycent: float  # Y coordinate of image center
    n_binning: int  # Binning factor
    pixscale: float  # Pixel scale [arcsec/pix]

    # WCS solution
    head: Optional[str] = field(default=None)
    sip_wcs: Optional[str] = field(default=None)

    # FOV polygon
    fov_ra: list = field(default=None)
    fov_dec: list = field(default=None)

    # WCS evaluation parameters
    # ref_max_mag: Optional[float] = field(default=None)
    # sci_max_mag: Optional[float] = field(default=None)
    # num_ref_sources: Optional[int] = field(default=None)
    # unmatched_fraction: Optional[float] = field(default=None)
    # subpixel_fraction: Optional[float] = field(default=None)
    # subsecond_fraction: Optional[float] = field(default=None)

    rsep_stats: Optional["RSEPSStats"] = field(default=None)
    # matched catalog itself
    matched_catalog: Optional[Table] = field(default=None)

    # separation statistics from reference catalog
    reference_sep_min: Optional[float] = field(default=None)
    reference_sep_max: Optional[float] = field(default=None)
    reference_sep_rms: Optional[float] = field(default=None)
    reference_sep_q1: Optional[float] = field(default=None)
    reference_sep_q2: Optional[float] = field(default=None)
    reference_sep_q3: Optional[float] = field(default=None)
    # sep_mean: Optional[float] = field(default=None)
    # sep_median: Optional[float] = field(default=None)
    # sep_std: Optional[float] = field(default=None)

    # psf qa statistics
    psf_stats: Optional["PsfStats"] = field(default=None)

    # Internal separation statistics (between multiple images)
    internal_sep_min: Optional[float] = field(default=None)
    internal_sep_max: Optional[float] = field(default=None)
    internal_sep_rms: Optional[float] = field(default=None)
    internal_sep_q1: Optional[float] = field(default=None)
    internal_sep_q2: Optional[float] = field(default=None)
    internal_sep_q3: Optional[float] = field(default=None)
    internal_match_counts: Optional[list] = field(default=None)
    internal_match_recall: Optional[list] = field(default=None)

    # dependency logger
    logger: Any = None

    _joint_cards_up_to_date = False  # prevent single and joint eval cards coming from different solutions

    def _log(self, msg: str, level: str = "error", *args, **kwargs):
        """Log or fallback to print/raise if logger missing."""
        if self.logger is not None:
            log_fn = getattr(self.logger, level, None)
            if callable(log_fn):
                log_fn(msg, *args, **kwargs)
                return
        # no logger available
        if level.lower() in ("error", "critical"):
            raise PipelineError(msg % args if args else msg)
        else:
            print(f"[ImageInfo:{level.upper()}] {msg % args if args else msg}")

    def __repr__(self) -> str:
        """Returns a string representation of the ImageInfo."""
        return ",\n".join(f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def parse_image_header_info(cls, image_path: str) -> "ImageInfo":
        """Parses image information from a FITS header."""

        hdr = fits.getheader(image_path)

        x, y = hdr["NAXIS1"], hdr["NAXIS2"]

        # Center coord for reference catalog query
        xcent = (x + 1) / 2.0
        ycent = (y + 1) / 2.0
        n_binning = hdr["XBINNING"]
        assert n_binning == hdr["YBINNING"]
        pixscale = n_binning * PIXSCALE

        # racent, decent = hdr["OBJCTRA"], hdr["OBJCTDEC"]  # desired pointing
        racent, decent = hdr["RA"], hdr["DEC"]  # actual pointing
        # racent = Angle(racent, unit="hourangle").deg
        racent = Angle(racent, unit="deg").deg
        decent = Angle(decent, unit="deg").deg

        # wcs = WCS(image_path)
        # racent, decent = wcs.all_pix2world(xcent, ycent, 1)
        # racent = float(racent)
        # decent = float(decent)

        # time_obj = Time(hdr["DATE-OBS"], format="isot")
        time_obj = hdr["DATE-OBS"]

        # if solved
        # if "CTYPE1" not in hdr.keys():
        #     raise PipelineError("Check Astrometry solution: no WCS information for Photometry")

        return cls(
            image_path=image_path,
            dateobs=time_obj,  # hdr["DATE-OBS"],
            naxis1=x,
            naxis2=y,
            racent=racent,
            decent=decent,
            xcent=xcent,
            ycent=ycent,
            n_binning=n_binning,
            pixscale=pixscale,
        )

    @property
    def coarse_wcs(self):
        pa = 0
        return build_wcs(self.racent, self.decent, self.xcent, self.ycent, self.pixscale, pa, flip=True)

    def set_reference_sep_stats(self, sep_stats: dict) -> None:
        self._joint_cards_up_to_date = False
        self.reference_sep_min = sep_stats["min"]
        self.reference_sep_max = sep_stats["max"]
        self.reference_sep_rms = sep_stats["rms"]
        self.reference_sep_q1 = sep_stats["q1"]
        self.reference_sep_q2 = sep_stats["q2"]
        self.reference_sep_q3 = sep_stats["q3"]
        self.reference_sep_p95 = sep_stats["p95"]
        self.reference_sep_p99 = sep_stats["p99"]

    def set_internal_sep_stats(self, sep_stats: dict) -> None:
        self._joint_cards_up_to_date = True
        self.internal_sep_rms_x = sep_stats["rms_x"]
        self.internal_sep_rms_y = sep_stats["rms_y"]
        self.internal_sep_rms = sep_stats["rms"]
        self.internal_sep_min = sep_stats["min"]
        self.internal_sep_q1 = sep_stats["q1"]
        self.internal_sep_q2 = sep_stats["q2"]
        self.internal_sep_q3 = sep_stats["q3"]
        self.internal_sep_p95 = sep_stats["p95"]
        self.internal_sep_p99 = sep_stats["p99"]
        self.internal_sep_max = sep_stats["max"]

    def set_internal_match_stats(self, match_stats: dict) -> None:
        self.internal_match_counts = match_stats["counts_by_group_size"]  # ex) {2: 70, 3: 180, 1: 16}
        self.internal_match_recall = match_stats["recall"]

    @property
    def wcs_evaluated(self) -> bool:
        """only checks the existence of reference separation statistics.
        Ensures that all the keys are updated by set_sep_stats."""
        sep_keys = [k for k in self.__dict__.keys() if k.startswith("sep_")]
        return all(getattr(self, k) is not None for k in sep_keys)

    @property
    def wcs_eval_cards(self) -> List[Tuple[str, float]]:

        cards = (
            self._single_wcs_eval_cards + self._joint_wcs_eval_cards
            if self._joint_cards_up_to_date
            else self._single_wcs_eval_cards
        )
        # Clean invalid values
        for i, (k, v, c) in enumerate(cards):
            if np.isnan(v):
                self._log(f"WCS statistics contains nan for {k}", level="error")
                cards[i] = (k, None, c)
            elif not isinstance(v, (float, int, str, np.float32, np.int32)):  # sometimes the value can be masked
                self._log(f"WCS statistics contains type {type(v)} {v} for {k}")
        return cards

    @property
    def _single_wcs_eval_cards(self) -> List[Tuple[str, float]]:
        cards = [
            (f"REFMXMAG", self.rsep_stats.ref_max_mag, "Highest g mag of selected reference sources"),
            (f"SCIMXMAG", self.rsep_stats.sci_max_mag, "Highest inst mag of selected science sources"),
            (f"NUM_REF", self.rsep_stats.num_ref_sources, "Number of reference sources selected"),
            (f"UNMATCH", self.rsep_stats.unmatched_fraction, "Fraction of unmatched reference sources"),
            (f"SUBPIXEL", self.rsep_stats.subpixel_fraction, "Fraction of matched with sep < PIXSCALE"),
            (f"SUBSEC", self.rsep_stats.subsecond_fraction, "Fraction of matched with sep < 1"),
            (f"RSEP_MIN", self.reference_sep_min, "Min separation from reference catalog [arcsec]"),
            (f"RSEP_MAX", self.reference_sep_max, "Max separation from reference catalog [arcsec]"),
            (f"RSEP_RMS", self.reference_sep_rms, "RMS separation from reference catalog [arcsec]"),
            (f"RSEP_Q1", self.reference_sep_q1, "Q1 separation from ref catalog [arcsec]"),
            (f"RSEP_Q2", self.reference_sep_q2, "Q2 separation from ref catalog [arcsec]"),
            (f"RSEP_Q3", self.reference_sep_q3, "Q3 separation from ref catalog [arcsec]"),
            (f"RSEP_P95", self.reference_sep_p95, "95 percentile sep from ref catalog [arcsec]"),
            (f"RSEP_P99", self.reference_sep_p99, "99 percentile sep from ref catalog [arcsec]"),
            # (f"RSEP_MED", self.sep_median, "Median separation from ref catalog [arcsec]"),
            # (f"RSEP_STD", self.sep_std, "STD of separation from ref catalog [arcsec]"),
            (f"FWHMCRMN", self.psf_stats.FWHMCRMN, "Mean corner/center FWHM ratio (9 PSFs)"),
            (f"FWHMCRSD", self.psf_stats.FWHMCRSD, "STD of corner/center FWHM ratio (9 PSFs)"),
            (f"AWINCRMN", self.psf_stats.AWINCRMN, "Mean corner/center AWIN ratio (9 PSFs)"),
            (f"AWINCRSD", self.psf_stats.AWINCRSD, "STD of corner/center AWIN ratio (9 PSFs)"),
            (f"PA_ALIGN", self.psf_stats.PA_ALIGN, "PA alignment score of 3x3 selected PSFs"),
            (f"ELLIPMN", self.psf_stats.ELLIPMN, "Mean ellipticity of the 9 PSFs"),
            (f"ELLIPSTD", self.psf_stats.ELLIPSTD, "STD of ellipticities of the 9 PSFs"),
        ]
        return cards

    @property
    def _joint_wcs_eval_cards(self) -> List[Tuple[str, float]]:
        # Include internal separation statistics if they exist on the instance.
        cards: List[Tuple[str, float]] = []
        internal_vals = {
            "ISEPRMSX": getattr(self, "internal_sep_rms_x", None),
            "ISEPRMSY": getattr(self, "internal_sep_rms_y", None),
            "ISEP_RMS": getattr(self, "internal_sep_rms", None),
            "ISEP_MIN": getattr(self, "internal_sep_min", None),
            "ISEP_Q1": getattr(self, "internal_sep_q1", None),
            "ISEP_Q2": getattr(self, "internal_sep_q2", None),
            "ISEP_Q3": getattr(self, "internal_sep_q3", None),
            "ISEP_MAX": getattr(self, "internal_sep_max", None),
            "ISEP_P95": getattr(self, "internal_sep_p95", None),
            "ISEP_P99": getattr(self, "internal_sep_p99", None),
            # "IMATCH_COUNTS": getattr(self, "internal_match_counts", None),
            "I_RECALL": getattr(self, "internal_match_recall", None),
        }
        descriptions = {
            "ISEPRMSX": "RMS x internal sep of outer-matched [arcsec]",
            "ISEPRMSY": "RMS y internal sep of outer-matched [arcsec]",
            "ISEP_RMS": "RMS internal sep of outer-matched [arcsec]",
            "ISEP_MIN": "Min internal sep of outer-matched [arcsec]",
            "ISEP_Q1": "Q1 internal sep of outer-matched [arcsec]",
            "ISEP_Q2": "Q2 internal sep of outer-matched [arcsec]",
            "ISEP_Q3": "Q3 internal sep of outer-matched [arcsec]",
            "ISEP_MAX": "Max internal sep of outer-matched [arcsec]",
            "ISEP_P95": "95 percentile of outer-matched [arcsec]",
            "ISEP_P99": "99 percentile of outer-matched [arcsec]",
            # "IMATCH_COUNTS": "Counts of matched sources by group size",
            "I_RECALL": "Recovery fraction in outer-matched cat",
        }
        for key, val in internal_vals.items():
            if val is not None:
                if isinstance(val, u.Quantity):
                    v = val.to(u.arcsec).value
                else:
                    v = val
                cards.append((key, v, descriptions[key]))
        return cards

    # @cached_property  # cache prevents update
    @property
    def wcs(self) -> WCS:
        """Updated every time run_scamp is completed"""
        if os.path.exists(self.head):

            wcs_header = read_scamp_header(self.head)
            return WCS(wcs_header)

        if self.sip_wcs:
            return WCS(read_text_header(self.sip_wcs))

        # raise PipelineError("No self.head in ImageInfo")
        self._log("No self.head in ImageInfo. Using coarse_wcs", level="warning")
        return self.coarse_wcs

    @property
    def seeing(self) -> float:
        """seeing from gaia-matched point sources"""
        return np.ma.median(self.matched_catalog["FWHM_WORLD"]) * 3600  # in arcsec

    @property
    def seeing_error(self) -> float:
        return np.ma.std(self.matched_catalog["FWHM_WORLD"]) * 3600

    @property
    def ellipticity(self) -> float:
        return np.ma.median(self.matched_catalog["ELLIPTICITY"])

    @property
    def ellipticity_error(self) -> float:
        return np.ma.std(self.matched_catalog["ELLIPTICITY"])

    @property
    def good(self) -> bool:
        """iteration condition"""
        # not checking unmatched fraction as it's better to exit _iterate_scamp quickly and go to solve-field
        return self.reference_sep_p95 < PIXSCALE

    @property
    def bad(self) -> bool:
        """iteration condition"""
        return self.rsep_stats.unmatched_fraction > 0.9 or self.reference_sep_p95 > 2 * PIXSCALE
