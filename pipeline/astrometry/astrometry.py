from __future__ import annotations  # for ImageInfo
import os
import re
import json
import time
import shutil
import numpy as np
from typing import Any, List, Tuple, Union, Any, Optional
from dataclasses import dataclass, field
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.wcs import WCS
from astropy.table import Table
import astropy.units as u
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import external
from ..const import PIXSCALE, REF_DIR
from ..errors import AstrometryError, ExceptionArg
from ..utils import swap_ext, add_suffix, force_symlink, time_diff_in_seconds, unique_filename, atleast_1d
from ..utils.header import update_padded_header, reset_header, fitsrec_to_header
from ..services.memory import MemoryMonitor
from ..config import SciProcConfiguration
from ..config.utils import get_key
from ..services.setup import BaseSetup
from ..io.cfitsldac import write_ldac
from ..services.database.handler import DatabaseHandler
from ..services.database.image_qa import ImageQATable
from ..services.checker import Checker
from ..services.logger import Logger

from .utils import (
    polygon_info_header,
    read_scamp_header,
    build_wcs,
    polygon_info_header,
    get_fov_quad,
    strip_wcs,
    read_text_header,
    get_source_num_frac,
)
from .evaluation import (
    evaluate_single_wcs,
    evaluate_joint_wcs,
    EvaluationResult,
    CornerStats,
    ImageStats,
    RSEPStats,
    RadialStats,
)
from .generate_refcat_gaia import get_refcat_gaia


class Astrometry(BaseSetup, DatabaseHandler, Checker):
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
        logger: Logger = None,
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
        self.logger.process_error = AstrometryError
        self.logger.debug(f"Astrometry Queue is '{queue}'")

        self.start_time = time.time()
        self.define_paths()

        self.load_criteria(dtype="science")
        self.qa_ids = []
        DatabaseHandler.__init__(
            self,
            add_database=self.config_node.settings.is_pipeline,
            is_too=get_key(self.config_node.settings, "is_too", False),
        )

        if self.is_connected:
            self.logger.add_exception_code = self.add_exception_code
            self.process_status_id = self.create_process_data(self.config_node)
            if self.too_id is not None:
                self.logger.debug(f"Initialized DatabaseHandler for ToO data management, ToO ID: {self.too_id}")
            else:
                self.logger.debug(
                    f"Initialized DatabaseHandler for pipeline and QA data management, Pipeline ID: {self.process_status_id}"
                )
            self.update_progress(0, "astrometry-configured")
            if self.process_status_id is not None:
                for image in self.input_images:
                    qa_id = self.create_image_qa_data(image, self.process_status_id)
                    self.qa_ids.append(qa_id)

    @classmethod
    def from_list(cls, images: List[str], working_dir: str = None):
        images = [os.path.abspath(image) for image in sorted(atleast_1d(images))]  # this is critical for soft links
        for image in images:
            if not os.path.exists(image):
                raise FileNotFoundError(f"File does not exist: {image}")

        config = SciProcConfiguration.user_config(input_images=images, working_dir=working_dir, logger=True)
        return cls(config=config)

    @property
    def sequential_task(self):
        """[(number, name, use_gpu), ...]"""
        return [(1, "run", False)]

    def clear_cache(self) -> None:
        """Clear the cache directory"""
        import shutil

        self.logger.info(f"Clearing Astrometry factory")
        factory = self.path.astrometry.tmp_dir  # this generates factory if not exists
        self.logger.debug(f"Deleting {factory}")
        shutil.rmtree(factory)
        # PathHandler's mkdir cache prevents regeneration; manually regenerate
        os.makedirs(factory, exist_ok=True)
        self.logger.debug(f"Re-generated {self.path.astrometry.tmp_dir}")

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
        sex_args=[],
        debug_plot: bool = False,
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
            if overwrite:
                self.clear_cache()
                self.define_paths()

            self.inject_wcs_guess(self.input_images)
            # Source Extractor
            self.run_sextractor(
                self.soft_links_to_input_images, se_preset=se_preset, sex_args=sex_args, overwrite=overwrite
            )

            # flexibly iterate to refine
            for i, (image_info, prep_cat) in enumerate(zip(self.images_info, self.prep_cats)):
                try:

                    if force_solve_field:
                        raise AstrometryError.NotImplementedError("force_solve_field")

                    # early QA
                    image_info.set_early_qa_stats(sci_cat=prep_cat, ref_cat=self.config_node.astrometry.local_astref)
                    flag, _ = self.apply_criteria(header=fits.Header(image_info.early_qa_cards), dtype="science")
                    image_info.SANITY = flag  # true if nothing to check
                    if not image_info.SANITY:
                        self.logger.info(
                            f"Early QA rejected {os.path.basename(image_info.image_path)}! "
                            f"Skipping all subsequent processing, including Astrometry and Photometry."
                        )
                        self.logger.error(
                            f"Early QA rejected {os.path.basename(image_info.image_path)}! Skipping all subsequent processing, including Astrometry and Photometry.",
                            AstrometryError.BlankImageError,
                        )
                        update_padded_header(image_info.image_path, fits.Header(image_info.early_qa_cards))
                        continue

                    # run initial solve: scamp or solve-field
                    self.run_scamp(prep_cat, scamp_preset="prep", joint=joint_scamp, overwrite=overwrite)
                    self.update_progress(5, "astrometry-scamp-prep")

                    if evaluate_prep_sol:
                        self.evaluate_solution(
                            input_images=image_info.image_path,
                            images_info=image_info,
                            prep_cats=prep_cat,
                            suffix="prepwcs",
                            use_threading=use_threading,
                            export_eval_cards=True,
                            overwrite=overwrite,
                            plot=debug_plot,
                        )

                    self.logger.info(f"Running main scamp iteration [{i+1}/{len(self.prep_cats)}] for {prep_cat}")
                    # main scamp iteration
                    self._iterate_scamp(
                        prep_cat, image_info, evaluate_prep_sol, use_threading, max_scamp, overwrite=overwrite
                    )

                    self.update_progress(10, "astrometry-scamp-main")
                    self.logger.info(f"Scamp iteration completed [{i+1}/{len(self.prep_cats)}] for {prep_cat}")

                    if image_info.bad:

                        self.logger.warning(
                            f"Bad solution. UNMATCH: {image_info.rsep_stats.unmatched_fraction if image_info.rsep_stats.unmatched_fraction is not None else 'None'}, "
                            f"RSEP_P95: {image_info.rsep_stats.separation_stats.P95 if image_info.rsep_stats.separation_stats.P95 is not None else 'None'} "
                            f"after {max_scamp} iterations for {image_info.image_path}",
                            AstrometryError.BadWcsSolutionError,
                        )

                        if not avoid_solvefield:
                            raise AstrometryError.ScampError(f"Bad SCAMP solution")

                except AstrometryError.ScampError as e:
                    # re-raise if not bad SCAMP error
                    if "Bad SCAMP solution" not in str(e):
                        raise

                    # if evaluate_prep_sol:
                    #     self.evaluate_solution(suffix="prepwcs", use_threading=use_threading, export_eval_cards=True)

                    self.logger.warning(
                        f"Solve-field triggered, better solution not guaranteed for {prep_cat}",
                        AstrometryError.AlternativeSolverError,
                    )

                    # self.run_solve_field(input_catalogs=self.prep_cats, output_images=self.solved_images)
                    self.run_solve_field(input_catalogs=prep_cat, output_images=[None], solvefield_args=solvefield_args)

                    if evaluate_prep_sol:
                        self.evaluate_solution(
                            suffix="solvefieldwcs",
                            plot=debug_plot,
                            use_threading=use_threading,
                            export_eval_cards=True,
                            overwrite=overwrite,
                        )
                    self._iterate_scamp(
                        prep_cat,
                        image_info,
                        evaluate_prep_sol,
                        use_threading,
                        max_scamp,
                        overwrite=overwrite,
                        plot=debug_plot,
                    )

                except Exception as e:
                    self.logger.error(
                        f"Unexpected error during astrometry processing: {str(e)}",
                        AstrometryError.UnknownError,
                        exc_info=True,
                    )
                    raise AstrometryError.UnknownError(f"Unexpected error during astrometry processing: {str(e)}")

            # evaluate main scamp
            self.evaluate_solution(
                suffix="wcs", isep=True, use_threading=use_threading, scamp_preset="main", overwrite=overwrite
            )
            self.update_progress(15, "astrometry-scamp-main-eval")
            # update the input image
            self.update_header()
            self.update_progress(20, "astrometry")

            if self.is_connected:
                for image, qa_id in zip(self.input_images, self.qa_ids):
                    qa_data = ImageQATable.from_file(image, process_status_id=self.process_status_id)
                    # update_data expects (target_id, **kwargs)
                    self.image_qa.update_data(qa_id, **qa_data.to_dict())

            self.config_node.flag.astrometry = True

            self.logger.info(
                f"'Astrometry' is completed in {time_diff_in_seconds(self.start_time)} seconds "
                f"({time_diff_in_seconds(self.start_time, return_float=True) / len(self.input_images):.2f} seconds per image)"
            )
            self.logger.debug(MemoryMonitor.log_memory_usage)
        except Exception as e:
            self.logger.error(
                f"Error during astrometry processing: {str(e)}",
                AstrometryError.UnknownError,
                exc_info=True,
            )
            raise

    def _iterate_scamp(
        self,
        prep_cat,
        image_info: ImageInfo,
        evaluate_prep_sol,
        use_threading,
        max_scamp,
        overwrite=True,
        plot=False,
        joint=False,
    ):
        """main scamp iteration"""
        for _ in range(max_scamp):
            self.run_scamp([prep_cat], scamp_preset="main", joint=joint, overwrite=(_ == max_scamp - 1 or overwrite))
            if evaluate_prep_sol:
                self.evaluate_solution(
                    input_images=image_info.image_path,
                    images_info=image_info,
                    prep_cats=prep_cat,
                    suffix="iterwcs",
                    plot=plot,
                    isep=False,
                    use_threading=use_threading,
                    export_eval_cards=True,
                    overwrite=overwrite,
                )
                self.logger.debug(f"SEP_P95: {image_info.rsep_stats.separation_stats.P95} for {image_info.image_path}")
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
                self.logger.warning(e, AstrometryError.UnknownError)

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

            self.config_node.flag.astrometry = True

            self.logger.info(
                f"'Astrometry' is completed in {time_diff_in_seconds(start_time)} seconds "
                f"({time_diff_in_seconds(start_time, return_float=True) / len(self.input_images):.2f} seconds per image)"
            )
            self.logger.debug(MemoryMonitor.log_memory_usage)
        except Exception as e:
            self.logger.error(
                f"Error during astrometry processing: {str(e)}", AstrometryError.UnknownError, exc_info=True
            )
            raise

    def define_paths(self) -> Tuple[List[str], List[str], List[str]]:
        self.logger.info("Defining paths for astrometry")
        # self.path_astrometry = self.path.astrometry.tmp_dir

        # override if astrometry.input_images is set
        local_input_images = get_key(self.config_node, "astrometry.input_images")
        if local_input_images is not None:
            input_images = local_input_images
        # otherwise use the common input
        else:
            input_images = self.config_node.input.calibrated_images
            self.config_node.astrometry.input_images = input_images

        self.input_images = input_images  # must be in sync with self.images_info

        # soft_links = [os.path.join(self.path_astrometry, os.path.basename(s)) for s in inims]
        soft_links = atleast_1d(self.path.astrometry.soft_link)
        for inim, soft_link in zip(input_images, soft_links):
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

        local_astref = self.config_node.astrometry.local_astref or self.path.astrometry.astrefcat
        if local_astref and not os.path.exists(local_astref):
            # Try to generate the reference catalog automatically
            try:
                self.logger.info(f"Local astrefcat {local_astref} does not exist. Generating from image header...")
                # Extract necessary info from first image
                image_info = self.images_info[0]
                get_refcat_gaia(
                    output_path=local_astref,
                    ra=image_info.racent,
                    dec=image_info.decent,
                    naxis1=image_info.naxis1,
                    naxis2=image_info.naxis2,
                    pixscale=image_info.pixscale,
                )
                self.logger.info(f"Generated reference catalog: {local_astref}")
            except Exception as e:
                self.logger.error(
                    f"Failed to generate reference catalog: {e}. Proceeding without local astrefcat.",
                    AstrometryError.AstrometryReferenceGenerationError,
                    exc_info=True,
                )

                local_astref = None
        self.config_node.astrometry.local_astref = local_astref

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

        def _update_header(image, wcs, reset_image_header):
            self.logger.debug(f"Injecting WCS into {image}")

            if reset_image_header:
                self.reset_headers(image)

            if isinstance(wcs, fits.Header):
                wcs_header = wcs
            elif isinstance(wcs, WCS):
                wcs_header = wcs.to_header(relax=True)
            else:
                raise ValueError(f"Invalid WCS type: {type(wcs)}")

            with fits.open(image, mode="update") as hdul:
                hdul[0].header.update(wcs_header)

        with ThreadPoolExecutor(max_workers=min(len(input_images), 10)) as executor:
            futures = [
                executor.submit(_update_header, image, wcs, reset_image_header)
                for image, wcs in zip(input_images, wcs_list)
            ]
            for future in as_completed(futures):
                future.result()

    def reset_headers(self, input_images: str | List[str] = None) -> None:
        """Reset header of input images."""
        input_images = atleast_1d(input_images or self.input_images)
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
        timeout=None,
    ) -> None:
        """Runs astrometry.net's solve-field to determine WCS solution for each image."""
        self.logger.info(f"Start solve-field")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        input_catalogs = (
            atleast_1d(input_catalogs)
            if input_catalogs is not None
            else [s for s, ii in zip(self.prep_cats, self.images_info) if ii.sane]
        )  # priority
        input_images = (
            atleast_1d(input_images)
            if input_images is not None
            else [ii.image_path for ii in self.images_info if ii.sane]
        )
        output_images = (
            atleast_1d(output_images)
            if output_images is not None
            else [s for s, ii in zip(self.solved_images, self.images_info) if ii.sane]
        )
        self.logger.debug(f"input_catalogs: {input_catalogs}")
        self.logger.debug(f"input_images: {input_images}")
        self.logger.debug(f"output_images: {output_images}")

        timeout = timeout or self.config_node.astrometry.solvefield_timeout

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
                    timeout=timeout,
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
                    timeout=timeout,
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
        sex_args: list = [],
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
        output_catalogs = output_catalogs or [s for s, ii in zip(self.prep_cats, self.images_info) if ii.sane]

        # if self.queue:
        #     self._submit_task(
        #         external.sextractor,
        #         input_images,
        #         outcat=output_catalogs,
        #         prefix=se_preset,
        #         sex_args=sex_args,
        #         logger=self.logger,
        #         fits_ldac=True,
        #         overwrite=overwrite,
        #     )
        # else:
        # Run sextractor sequentially
        for i, (solved_image, prep_cat) in enumerate(zip(input_images, output_catalogs)):
            try:
                external.sextractor(
                    solved_image,
                    outcat=prep_cat,
                    se_preset=se_preset,
                    logger=self.logger,
                    sex_args=sex_args,
                    fits_ldac=True,
                    overwrite=overwrite,
                )
                self.logger.info(f"Completed sextractor (prep) [{i+1}/{len(input_images)}]")
                self.logger.debug(f"{solved_image}")
            except Exception as e:
                self.logger.error(f"Sextractor failed for {solved_image}: {e}", AstrometryError.SextractorError)
                raise AstrometryError.SextractorError(f"Sextractor failed for {solved_image}: {e}") from e

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
        timeout=None,
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

        timeout = timeout or self.config_node.astrometry.scamp_timeout

        # presex_cats = [os.path.splitext(s)[0] + f".{prefix}.cat" for s in files]
        input_catalogs = atleast_1d(input_catalogs or [s for s, ii in zip(self.prep_cats, self.images_info) if ii.sane])
        self.logger.debug(f"Scamp input catalogs: {input_catalogs}")

        # path for scamp refcat download
        if path_ref_scamp is False:
            self.logger.debug("SCAMP REFOUT_CATPATH is CWD")
            path_ref_scamp = False
        else:
            path_ref_scamp = path_ref_scamp or self.path.astrometry.ref_query_dir

        # use local astrefcat if tile obs
        astrefcat = astrefcat or self.config_node.astrometry.local_astref
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
                timeout=timeout,
                logger=self.logger,
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
                timeout=timeout,
                logger=self.logger,
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
                    timeout=timeout,
                    logger=self.logger,
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
                    self.logger.error(f"No PV1_0 in {solved_head} - solution invalid", AstrometryError.ScampError)
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
        input_catalogs = input_catalogs or [s for s, ii in zip(self.prep_cats, self.images_info) if ii.sane]
        solved_heads = solved_heads or [s for s, ii in zip(self.solved_heads, self.images_info) if ii.sane]
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

            if update_fwhm:  # TODO
                self.logger.debug("Updating FWHM_WORLD is under development. Skipping...")
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
        suffix: str = "wcs",
        num_sci=200,
        num_ref=200,
        use_threading=True,
        export_eval_cards=False,
        scamp_preset="prep",  # for determining error
        overwrite=True,
    ) -> None:
        """Evaluate the solution. This was developed in lack of latest scamp version."""
        self.logger.debug("Start evaluate_solution")

        input_images = atleast_1d(input_images or [ii.image_path for ii in self.images_info if ii.sane])
        images_info = atleast_1d(images_info or [ii for ii in self.images_info if ii.sane])
        prep_cats = atleast_1d(prep_cats or [s for s, ii in zip(self.prep_cats, self.images_info) if ii.sane])

        refcat = Table.read(self.config_node.astrometry.local_astref, hdu=2)

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
                self.logger.error(
                    f"Failed to update FOV polygon for {image_info.image_path}: {e}",
                    AstrometryError.UnknownError,
                    exc_info=True,
                )

            try:
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
                    match_radius=self.config_node.astrometry.eval_match_radius,
                    cutout_size=30,
                    logger=self.logger,
                    overwrite=overwrite,
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate solution for {input_image}: {e}", AstrometryError.UnknownError, exc_info=True
                )

                raise
            return idx, eval_result

        def _apply(idx: int, eval_result: EvaluationResult):
            """apply helper to avoid code duplication"""
            matched = eval_result.matched
            rsep_stats = eval_result.rsep_stats
            corner_stats = eval_result.corner_stats
            image_stats = eval_result.image_stats
            radial_stats = eval_result.radial_stats

            if scamp_preset == "main" and rsep_stats.unmatched_fraction == 1.0:
                self.logger.error(
                    f"Unmatched fraction is 1.0 for {input_images[idx]}. "
                    f"Check the initial WCS guess or the reference catalog.",
                    AstrometryError.InvalidWcsSolution,
                )
                raise AstrometryError.InvalidWcsSolution(f"Unmatched fraction is 1.0 for {input_images[idx]}")

            info = images_info[idx]
            info.matched_catalog = matched
            info.rsep_stats = rsep_stats
            info.corner_stats = corner_stats
            info.image_stats = image_stats
            info.radial_stats = radial_stats
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
                isep_stats_list, match_stat_list = evaluate_joint_wcs([ii.matched_catalog for ii in self.images_info])
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
            self.logger.error(f"Failed to evaluate solution: {e}", AstrometryError.UnknownError, exc_info=True)

            # don't raise error

        self.logger.debug(f"Completed evaluate_solution")
        self.logger.debug(MemoryMonitor.log_memory_usage)

    # def update_qa_config(self) -> None:
    #     """Update the QA configuration."""
    #     self.config_node.qa.ellipticity = [image_info.ellipticity for image_info in self.images_info]
    #     self.config_node.qa.seeing = [image_info.seeing for image_info in self.images_info]
    #     self.config_node.qa.pa = [fits.getheader(img)["ROTANG"] for img in self.input_images]
    #     self.logger.debug(f"SEEING     : {self.config_node.qa.seeing:.3f} arcsec")
    #     self.logger.debug(f"ELLIPTICITY: {self.config_node.qa.ellipticity:.3f}")
    #     self.logger.debug(f"PA         : {self.config_node.qa.pa:.3f} deg")

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

        # exclude images rejected by early QA
        heads = heads or [s for s, ii in zip(self.solved_heads, self.images_info) if ii.sane]
        inims = inims or [ii.image_path for ii in self.images_info if ii.sane]
        self.logger.info(f"Updating WCS to header(s) of {len(heads)} image(s)")
        assert len(heads) == len(inims)

        for image_info, solved_head, target_fits in zip(self.images_info, heads, inims):
            solved_header = read_scamp_header(solved_head)

            if reset_image_header:
                self.reset_headers(target_fits)

            if image_info.wcs_evaluated:
                # solved_header.update(image_info.wcs_eval_cards)  # this removes duplicate COMMENT cards
                self.logger.debug(
                    f"WCS evaluation cards:\n WCS evaluation cards:\n\t"
                    + fits.Header(image_info.cards).tostring(sep="\n\t")
                )
                solved_header.extend(image_info.cards)
                self.logger.debug(f"WCS evaluation has been added to {target_fits}")
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
    """
    Stores information needed for astrometry.
    Pristine image header + derived information in Astrometry.

    The goal is to load all required header information once and keep it frozen.
    It helps generate new header cards to be added to the images' headers.
    """

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
    filter: str  # Filter

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

    # matched catalog itself
    matched_catalog: Optional[Table] = field(default=None)

    # evaluation results
    rsep_stats: Optional[RSEPStats] = field(default=None)
    corner_stats: Optional[CornerStats] = field(default=None)
    image_stats: Optional[ImageStats] = field(default=None)
    radial_stats: Optional[RadialStats] = field(default=None)

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

    _joint_cards_are_up_to_date = False  # prevent single and joint eval cards coming from different solutions

    # early qa
    num_frac: Optional[float] = field(default=None)
    SANITY: Optional[bool] = field(default=True)
    # late qa
    # UNMATCH, PA_ALIGN, ELLIPMN, ELLIPSTD

    def _log(self, msg: str, level: str = "error", exception: ExceptionArg = None, *args, **kwargs):
        """Log or fallback to print/raise if logger missing."""
        if self.logger is not None:
            log_fn = getattr(self.logger, level, None)
            if callable(log_fn):
                log_fn(msg, exception, *args, **kwargs)
                return
        # if no logger available
        if level.lower() in ("error", "critical"):
            raise AstrometryError(msg % args if args else msg)
        else:
            print(f"[ImageInfo:{level.upper()}] {msg % args if args else msg}")

    def __repr__(self) -> str:
        """Returns a string representation of the ImageInfo."""
        return ",\n".join(f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def parse_image_header_info(cls, image_path: str) -> ImageInfo:
        """Parses image information from a FITS header."""

        hdr = fits.getheader(image_path)

        x, y = hdr["NAXIS1"], hdr["NAXIS2"]
        # Center coord for reference catalog query
        xcent = (x + 1) / 2.0
        ycent = (y + 1) / 2.0
        n_binning = hdr["XBINNING"]
        assert n_binning == hdr["YBINNING"]
        pixscale = n_binning * PIXSCALE

        # wcs = WCS(image_path)
        # racent, decent = wcs.all_pix2world(xcent, ycent, 1)
        # racent = float(racent)
        # decent = float(decent)

        # if solved
        # if "CTYPE1" not in hdr.keys():
        #     raise PipelineError("Check Astrometry solution: no WCS information for Photometry")

        return cls(
            image_path=image_path,
            dateobs=hdr["DATE-OBS"],  # Time(hdr["DATE-OBS"], format="isot")
            naxis1=x,
            naxis2=y,
            racent=Angle(hdr["RA"], unit="deg").deg,  # mount-reported RA. Desired RA is OBJCTRA.
            decent=Angle(hdr["DEC"], unit="deg").deg,
            xcent=xcent,
            ycent=ycent,
            n_binning=n_binning,
            pixscale=pixscale,
            filter=hdr["FILTER"],
        )

    @property
    def coarse_wcs(self):
        pa = 0
        return build_wcs(self.racent, self.decent, self.xcent, self.ycent, self.pixscale, pa, flip=True)

    def set_internal_sep_stats(self, sep_stats: dict) -> None:
        self._joint_cards_are_up_to_date = True
        self.internal_sep_rms_x = sep_stats["rms_x"] if sep_stats["rms_x"] is not None else None
        self.internal_sep_rms_y = sep_stats["rms_y"] if sep_stats["rms_y"] is not None else None
        self.internal_sep_rms = sep_stats["rms"] if sep_stats["rms"] is not None else None
        self.internal_sep_min = sep_stats["min"] if sep_stats["min"] is not None else None
        self.internal_sep_q1 = sep_stats["q1"] if sep_stats["q1"] is not None else None
        self.internal_sep_q2 = sep_stats["q2"] if sep_stats["q2"] is not None else None
        self.internal_sep_q3 = sep_stats["q3"] if sep_stats["q3"] is not None else None
        self.internal_sep_p95 = sep_stats["p95"] if sep_stats["p95"] is not None else None
        self.internal_sep_p99 = sep_stats["p99"] if sep_stats["p99"] is not None else None
        self.internal_sep_max = sep_stats["max"] if sep_stats["max"] is not None else None

    def set_internal_match_stats(self, match_stats: dict) -> None:
        self.internal_match_counts = match_stats["counts_by_group_size"]  # ex) {2: 70, 3: 180, 1: 16}
        self.internal_match_recall = match_stats["recall"]

    def set_early_qa_stats(self, sci_cat: str, ref_cat: str):
        """sets self.early_qa_cards"""
        if not ref_cat:
            self._log("No refcat to perform early QA. Skipping...", level="error")
            return
        if not os.path.exists(ref_cat):
            self._log(f"Refcat {ref_cat} not found. Skipping early QA...", level="error")
            return

        with open(os.path.join(REF_DIR, "zeropoints.json"), "r") as f:
            zp_per_filter = json.load(f)
        with open(os.path.join(REF_DIR, "depths.json"), "r") as f:
            depths_per_filter = json.load(f)
        if self.filter in zp_per_filter:
            zp = zp_per_filter[self.filter]
        else:
            zp = zp_per_filter["unknown"]
            self._log(
                f"Filter {self.filter} not in zeropoints.json for early QA. Using default: {zp}.",
                exception=AstrometryError.PrerequisiteNotMet,
            )
        if self.filter in depths_per_filter:
            depth = depths_per_filter[self.filter]
        else:
            depth = depths_per_filter["unknown"]
            self._log(
                f"Filter {self.filter} not in depths.json for early QA. Using default: {depth}.",
                exception=AstrometryError.PrerequisiteNotMet,
            )

        self.num_frac = get_source_num_frac(sci_cat, ref_cat, sci_zp=zp, depth=depth - 0.5)
        self._log(f"Early QA: NUMFRAC = {self.num_frac}", level="info")
        return

    @property
    def cards(self) -> List[Tuple[str, float]]:
        return self.early_qa_cards + self.wcs_eval_cards

    @property
    def early_qa_cards(self) -> List[Tuple[str, float]]:
        cards = [
            (f"SANITY", self.SANITY, "Sanity flag for SCIPROCESS"),
            (f"NUMFRAC", self.num_frac, "Source number fraction vs Gaia expected"),
        ]
        return cards

    @property
    def wcs_evaluated(self) -> bool:
        return self.rsep_stats is not None

    @property
    def wcs_eval_cards(self) -> List[Tuple[str, float]]:

        cards = (
            self._single_wcs_eval_cards + self._joint_wcs_eval_cards
            if self._joint_cards_are_up_to_date
            else self._single_wcs_eval_cards
        )
        # Clean invalid values
        import numpy.ma as ma

        for i, (k, v, c) in enumerate(cards):
            # Handle MaskedConstant (from numpy masked arrays)
            if isinstance(v, ma.core.MaskedConstant):
                self._log(f"WCS statistics contains masked value for {k}, converting to None", level="error")
                cards[i] = (k, None, c)
            elif isinstance(v, (float, np.floating)) and np.isnan(v):
                self._log(f"WCS statistics contains nan for {k}", level="error")
                cards[i] = (k, None, c)
            elif not isinstance(v, (float, int, str, np.float32, np.int32, type(None))):
                # Convert other invalid types to None
                self._log(f"WCS statistics contains type {type(v)} {v} for {k}, converting to None", level="error")
                cards[i] = (k, None, c)

        return cards

    @property
    def _single_wcs_eval_cards(self) -> List[Tuple[str, float]]:

        rsep_stats_cards = self.rsep_stats.fits_header_cards_for_metadata
        if not rsep_stats_cards:
            self._log("No RSEPStats metadata ", level="error")

        ref_sep_cards = self.rsep_stats.separation_stats.fits_header_cards
        if not ref_sep_cards:
            self._log("No reference_sep ", level="error")

        image_stats_cards = self.image_stats.fits_header_cards
        if not image_stats_cards:
            self._log("No image_stats ", level="error")

        if self.corner_stats is not None:
            corner_stats_cards = self.corner_stats.fits_header_cards
            if not corner_stats_cards:
                self._log("No corner_stats ", level="error")
        else:
            corner_stats_cards = []

        if self.radial_stats is not None:
            radial_stats_cards = self.radial_stats.fits_header_cards
            if not radial_stats_cards:
                self._log("No radial_stats ", level="error")
        else:
            radial_stats_cards = []

        cards = rsep_stats_cards + ref_sep_cards + image_stats_cards + corner_stats_cards + radial_stats_cards

        if cards is None:
            self._log("No cards ", level="error")
            return []
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
        if self.rsep_stats.separation_stats.P95 is None:
            return False
        else:
            return self.rsep_stats.separation_stats.P95 < PIXSCALE

    @property
    def bad(self) -> bool:
        """iteration condition"""

        return self.rsep_stats.unmatched_fraction > 0.9 or self.rsep_stats.separation_stats.P95 > 2 * PIXSCALE

    @property
    def sane(self) -> bool:
        """whether or not to process the image judging by early QA"""
        return self.SANITY is None or self.SANITY
