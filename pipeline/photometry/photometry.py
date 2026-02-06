from __future__ import annotations
import os
import getpass
from typing import Any, List, Dict, Tuple, Optional, Union
import datetime
import numpy as np
import itertools
import time
from dataclasses import dataclass, fields

# astropy
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.table import Table, hstack, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

from ..services.logger import Logger
from ..services.memory import MemoryMonitor
from ..services.queue import QueueManager
from ..services.database.handler import DatabaseHandler
from ..services.database.image_qa import ImageQATable
from ..services.checker import CheckerMixin

from ..config.utils import get_key
from ..utils import time_diff_in_seconds, force_symlink, collapse
from ..config import SciProcConfiguration
from ..config.base import ConfigNode
from .. import external
from ..const import PIXSCALE, MEDIUM_FILTERS, BROAD_FILTERS, ALL_FILTERS
from ..services.setup import BaseSetup
from ..tools.table import match_two_catalogs, build_condition_mask
from ..path.path import PathHandler
from ..utils.header import get_header_key, update_padded_header
from ..utils.tile import is_ris_tile
from ..errors import (
    # process errors
    SinglePhotometryError,
    CoaddPhotometryError,
    DifferencePhotometryError,
    SystemError,
    PathHandlerError,
    # kind errors
    NotEnoughSourcesError,
    NoReferenceSourceError,
    FilterCheckError,
    FilterInventoryError,
    InferredFilterMismatchError,
    ConnectionError,
    PreviousStageError,
    PrerequisiteNotMetError,
    UnknownError,
)

from . import utils as phot_utils
from .plotting import plot_zp, plot_filter_check


class Photometry(BaseSetup, DatabaseHandler, CheckerMixin):
    """
    A wrapper class of PhotometrySingle.
    Dispatches to PhotometrySingle based on the photometry_mode:
        - single_photometry
        - coadd_photometry
        - difference_photometry
    """

    def __init__(
        self,
        config: Any = None,
        logger: Any = None,
        queue: Union[bool, QueueManager] = False,
        images: Optional[List[str]] = None,
        photometry_mode: Optional[str] = None,
        ref_cat_type: Optional[str] = None,
    ) -> None:
        """
        Initialize the Photometry class.

        Photometry will know its mode based on self.config.flag., but it is
        safer to explicitly set the mode when re-running scidata reduction,
        where the flags can be out of sync.
        """
        # Load Configuration
        super().__init__(config, logger, queue)
        # self._flag_name = "photometry"

        self.ref_cat_type = ref_cat_type or self.config_node.photometry.refcatname

        if photometry_mode == "single_photometry" or (
            not self.config_node.flag.single_photometry
            # and (not self.config.input.coadd_image and not self.config.input.difference_image)  # hassle when rerunning
        ):
            self.logger.process_error = SinglePhotometryError
            self.input_images = images or self.config_node.input.calibrated_images
            self.apply_sanity_filter_and_report()  # overrides self.input_images
            self.config_node.photometry.input_images = self.input_images

            self.logger.debug("Running single photometry")
            self._photometry_mode = "single_photometry"
        elif photometry_mode == "coadd_photometry" or (
            not self.config_node.flag.coadd_photometry
            # and (self.config.input.coadd_image and not self.config.input.difference_image)
        ):
            self.logger.process_error = CoaddPhotometryError
            self.config_node.photometry.input_images = (
                images or [x] if (x := self.config_node.input.coadd_image) else None
            )
            self.input_images = self.config_node.photometry.input_images
            self.logger.debug("Running coadd photometry")
            self._photometry_mode = "coadd_photometry"
        elif photometry_mode == "difference_photometry" or not self.config_node.flag.difference_photometry:
            self.logger.process_error = DifferencePhotometryError
            self.config_node.photometry.input_images = (
                images or [x] if (x := self.config_node.input.difference_image) else None
            )
            self.input_images = self.config_node.photometry.input_images
            self.logger.debug("Running difference photometry")
            self._photometry_mode = "difference_photometry"

        else:
            self.logger.debug(f"photometry mode undefined: {photometry_mode}")
            raise self.logger.process_error.exception(ValueError)(
                "Unexpected photometry mode: check if flags are sequentially turned on for single_photometry, combined_photometry, and difference_photometry"
            )

        self.logger.info(f"Photometry mode: {self._photometry_mode}")

        self.qa_ids = []
        DatabaseHandler.__init__(
            self, add_database=self.config_node.settings.is_pipeline, is_too=self.config_node.settings.is_too
        )

        if self.is_connected:

            if self._photometry_mode == "single_photometry":
                self.reset_exceptions("single_photometry")
            elif self._photometry_mode == "coadd_photometry":
                self.reset_exceptions("coadd_photometry")
            elif self._photometry_mode == "difference_photometry":
                self.reset_exceptions("difference_photometry")

            if self.process_status_id is not None:
                from ..services.database.handler import ExceptionHandler

                self.logger.database = ExceptionHandler(self.process_status_id)

            self.process_status_id = self.create_process_data(self.config_node)
            if self.too_id is not None:
                self.logger.debug(f"Initialized DatabaseHandler for ToO data management, ToO ID: {self.too_id}")
            else:
                self.logger.debug(
                    f"Initialized DatabaseHandler for pipeline and QA data management, Pipeline ID: {self.process_status_id}"
                )
            self.update_progress(self._progress_by_mode[0], f"{self._photometry_mode}-configured")

    @property
    def _progress_by_mode(self):
        if self._photometry_mode == "single_photometry":
            return 40, 20
        elif self._photometry_mode == "coadd_photometry":
            return 70, 10
        elif self._photometry_mode == "difference_photometry":
            return 90, 10
        else:
            raise SinglePhotometryError.ValueError(f"Undefined photometry mode: {self._photometry_mode}")

    @classmethod
    def from_list(cls, images: List[str], working_dir=None) -> Optional["Photometry"]:
        """
        Create Photometry instance from a list of image paths.

        Args:
            images: List of paths to image files

        Returns:
            Photometry instance
        """
        for image in images:
            if not os.path.exists(image):
                raise FileNotFoundError(f"File does not exist: {image}")

        config = SciProcConfiguration.user_config(input_images=images, working_dir=working_dir, logger=True)
        return cls(config=config)

    @property
    def sequential_task(self):
        """[(number, name, use_gpu), ...]"""
        return [(1, "run", False)]

    def run(self, overwrite=True) -> None:
        """
        Run photometry on all configured images.

        Processes images either sequentially or in parallel depending on queue configuration.
        Updates configuration flags and performs memory cleanup after completion.

        Overwrite does nothing for now. Overwrite all by default
        """
        st = time.time()
        self.logger.info(f"Start 'Photometry'")

        if not self.input_images:  # exception for when input.difference_image is not set.
            self.logger.debug(f"input_images: {self.input_images}")
            if self._photometry_mode == "difference_photometry":
                self.logger.info(f"No input images found. Skipping {self._photometry_mode}.")

                self.update_progress(100, f"{self._photometry_mode}-completed")
                if self.is_too and self.too_id is not None:
                    interim_notice = self.too_db.read_data_by_id(self.too_id).get("interim_notice")

                    if interim_notice == 0:
                        self.too_db.send_interim_notice_email(self.too_id, sed_data=sed_data, dtype="coadd")

                    self.too_db.mark_completed(self.too_id)
                return
            else:
                self.logger.error(
                    f"No input images found in {self._photometry_mode}.",
                    SinglePhotometryError.EmptyInput,
                )
                raise SinglePhotometryError.EmptyInputError(f"No input images found in {self._photometry_mode}.")
        try:
            if self.queue:
                self._run_parallel(overwrite=overwrite)
            else:
                self._run_sequential(overwrite=overwrite)

            if self.is_connected:
                for image, qa_id in zip(self.input_images, self.qa_ids):
                    qa_data = ImageQATable.from_file(
                        image,
                        process_status_id=self.process_status_id,
                    )
                    qa_id = self.image_qa.update_data(qa_id, **qa_data.to_dict())

            self.update_progress(sum(self._progress_by_mode), f"{self._photometry_mode}-completed")

            if self.is_too and self.too_id is not None and self._photometry_mode == "difference_photometry":
                interim_notice = self.too_db.read_data_by_id(self.too_id).get("interim_notice")

                if interim_notice == 0:
                    self.too_db.send_interim_notice_email(self.too_id, sed_data=sed_data, dtype="difference")

                self.too_db.mark_completed(self.too_id)

            if self._photometry_mode == "single_photometry":
                self.config_node.flag.single_photometry = True
            elif self._photometry_mode == "coadd_photometry":
                self.config_node.flag.coadd_photometry = True
            elif self._photometry_mode == "difference_photometry":
                self.config_node.flag.difference_photometry = True
            else:
                raise SinglePhotometryError.ValueError(f"Undefined photometry mode: {self._photometry_mode}")

            self.logger.info(f"'Photometry' is Completed in {time_diff_in_seconds(st)} seconds")
            self.logger.debug(MemoryMonitor.log_memory_usage)
        except Exception as e:
            self.logger.error(f"Photometry failed: {str(e)}", UnknownError, exc_info=True)
            raise

    def _run_parallel(self, overwrite=True) -> None:
        """Process images in parallel using queue system."""
        task_ids = []
        for i, image in enumerate(self.input_images):
            process_name = f"{self.config_node.name} - single photometry - {i+1} of {len(self.input_images)}"
            single_config = self.config_node.extract_single_image_config(i)
            diff_phot = self._photometry_mode == "difference_photometry"

            phot_single = PhotometrySingle(
                image,
                single_config,  # self.config,
                self.logger,
                ref_cat_type=self.ref_cat_type,
                reset_count=i == 0,
            )
            task_id = self.queue.add_task(
                phot_single.run,
                kwargs={"name": process_name, "difference_photometry": diff_phot, "overwrite": overwrite},
                gpu=False,
                task_name=process_name,
            )
            task_ids.append(task_id)
        self.queue.wait_until_task_complete(task_ids)

    def _run_sequential(self, overwrite=True) -> None:
        """Process images sequentially."""
        for i, image in enumerate(self.input_images):
            single_config = self.config_node.extract_single_image_config(i)
            diff_phot = self._photometry_mode == "difference_photometry"
            PhotometrySingle(
                # image,
                single_config,  # self.config,
                logger=self.logger,
                ref_cat_type=self.ref_cat_type,
                total_image=len(self.input_images),
                difference_photometry=diff_phot,
                reset_count=i == 0,
            ).run(overwrite=overwrite)

            self.update_progress(
                self._progress_by_mode[0] + (self._progress_by_mode[1] / len(self.input_images)) * (i + 1),
                f"{self._photometry_mode}-{i}/{len(self.input_images)}",
            )


class PhotometrySingle:
    """
    Class for performing photometry on a single astronomical image.

    Handles the complete photometry pipeline for one image, including:
    - Source extraction
    - Reference catalog matching
    - Zero point calculation
    - Header updates
    """

    _id_counter = itertools.count(1)

    def __init__(
        self,
        # image: str,
        config_node: ConfigNode,
        logger: Logger = None,
        name: Optional[str] = None,
        ref_cat_type: str = "GaiaXP",
        total_image: int = 1,
        check_filter=True,
        difference_photometry=False,
        reset_count=False,
    ) -> None:
        """Initialize PhotometrySingle instance."""

        if reset_count:
            self._id_counter = itertools.count(1)

        # if config_node is a SciProcConfiguration
        if hasattr(config_node, "node"):
            config_node = config_node.node

        self.config_node = config_node
        self.logger = logger or self._setup_logger(config_node)
        self.ref_cat_type = ref_cat_type
        # self.image = os.path.join(self.config.path.path_processed, image)
        # self.input_image = image
        self.phot_conf = self.config_node.photometry
        self.input_image = collapse(self.phot_conf.input_images, raise_error=True)
        self.logger.debug(f"input_image: {self.input_image}")
        self.logger.debug(f"=" * 100)
        self.image_info = ImageInfo.parse_image_header_info(self.input_image)
        self.name = name or os.path.basename(self.input_image)
        self.logger.debug(f"{self.name}: ImageInfo: {self.image_info}")
        self.phot_header = PhotometryHeader(self.image_info)  # mind its update structure
        self.phot_header.AUTHOR = getpass.getuser()
        self.phot_header.REFCAT = self.ref_cat_type
        self.logger.debug(f"{self.name}: self.phot_header at PhotometrySingle.__init__(): {self.phot_header}")

        self._id = str(next(self._id_counter)) + "/" + str(total_image)

        self.path = PathHandler(self.input_image, is_too=self.config_node.settings.is_too)
        self.path_tmp = self.path.photometry.tmp_dir

        self._trust_header_seeing = difference_photometry
        self._trust_header_zp = difference_photometry
        self._check_filter = check_filter and not difference_photometry

    def _setup_logger(self, config: Any) -> Any:
        """Initialize logger instance."""
        if hasattr(config, "logger") and config.logger is not None:
            return config.logger
        else:
            from ..services.logger import Logger

            return Logger(name="7DT pipeline logger", slack_channel="pipeline_report")

    # @classmethod
    # def from_file(cls, image: str) -> Optional["PhotometrySingle"]:
    #     """Create instance from single image file."""
    #     path = Path(image)
    #     if not path.is_file():
    #         print("The file does not exist.")
    #         return None
    #     working_dir = str(path.parent.absolute())
    #     config = Configuration.base_config(working_dir)
    #     image = path.parts[-1]
    #     return cls(image, config, name="user-input")

    def run(self, overwrite=True) -> None:
        """
        Execute complete photometry pipeline for single image.

        Performs the following steps:
        1. Loads reference catalog
        2. Calculates the seeing
        3. Runs source extraction
        4. Matches detected sources with reference catalog
        5. Calculates zero point corrections
        6. Updates image header
        7. Writes photometry catalog

        Times the complete process and performs memory cleanup after completion.
        """
        try:
            start_time = time.time()
            self.logger.info(f"Start 'PhotometrySingle' for the image {self.name} [{self._id}]")
            self.logger.debug(f"{'=' * 13} {os.path.basename(self.input_image)} {'=' * 13}")

            self.calculate_seeing(use_header_seeing=(self._trust_header_seeing and not overwrite), overwrite=overwrite)
            obs_src_table = self.photometry_with_sextractor(overwrite=overwrite)

            if self._check_filter:
                self.logger.info("Performing filter check")
                filters_to_check = get_key(self.config_node.photometry, "filters_to_check") or ALL_FILTERS
                obs_src_table = self.add_matched_reference_catalog(obs_src_table, filters=filters_to_check)
                zp_src_table = self.get_zp_src_table(obs_src_table)

                temp_headers = {}
                self.logger.debug(f"Starting filter check for {filters_to_check}")
                for i, filt in enumerate(filters_to_check):
                    temp_phot_header = PhotometryHeader()  # PEEING None -> ZP_AUTO only
                    temp_phot_header = self.calculate_zp(
                        zp_src_table, filt=filt, phot_header=temp_phot_header, save_plots=False
                    )
                    if temp_phot_header is not None:
                        temp_headers[filt] = temp_phot_header
                        self.logger.debug(f"Filter check photometry for {filt} is completed")

                # save the photometry result for the filename filter as default
                self.calculate_zp(zp_src_table)  # uses self.phot_header below
                self.add_calibrated_columns(obs_src_table)

                # if inferred filter is different, save it to header and catalog, too
                inferred_filter = self.determine_filter(temp_headers)
                if inferred_filter != self.image_info.filter:
                    self.logger.info(f"Saving {inferred_filter} photometry to catalog too")
                    inferred_phot_header = temp_headers[inferred_filter]
                    self.add_calibrated_columns(obs_src_table, filt=inferred_filter, phot_header=inferred_phot_header)
                    # self.update_image_header(inferred_phot_header)  # namespace collision
                # update header after filter inference so INF_FILT is persisted
                self.update_image_header()

            else:
                self.logger.info("Skipping filter check")
                # zp_src_table = self.get_zp_src_table(obs_src_table)
                # obs_src_table = self.add_matched_reference_catalog(zp_src_table)
                if not self._trust_header_zp:
                    obs_src_table = self.add_matched_reference_catalog(obs_src_table)
                    zp_src_table = self.get_zp_src_table(obs_src_table)
                    self.calculate_zp(zp_src_table)

                self.add_calibrated_columns(obs_src_table)

                if not self._trust_header_zp:
                    self.update_image_header()

            self.write_catalog(obs_src_table)

            self.logger.debug(MemoryMonitor.log_memory_usage)
            self.logger.info(
                f"'PhotometrySingle' is completed for the image [{self._id}]"
                f" in {time_diff_in_seconds(start_time)} seconds"
            )
        except Exception as e:
            self.logger.error(
                f"PhotometrySingle failed for the image [{self._id}]: {str(e)}", UnknownError, exc_info=True
            )
            raise

    def calculate_seeing(
        self, phot_header: PhotometryHeader = None, use_header_seeing: bool = False, overwrite=False
    ) -> None:
        """
        Calculate seeing conditions from stellar sources in the image.

        Uses source extraction to identify stars and calculate median FWHM,
        ellipticity, and elongation values stored by self.phot_header.

        Args:
            low_mag_cut: Lower magnitude limit for star selection
        """

        phot_header = phot_header or self.phot_header

        if use_header_seeing:
            if self.image_info.has_psf_stats_from_astrometry:
                phot_header.SEEING = self.image_info.SEEINGMN
                phot_header.PEEING = self.image_info.PEEINGMN
                phot_header.ELLIP = self.image_info.ELLIPMN
                phot_header.ELONG = self.image_info.ELONGMN
                self.logger.debug(f"Trusting seeing/peeing from the header")
                self.logger.debug(f"SEEING     : {phot_header.SEEING:.3f} arcsec")
                self.logger.debug(f"PEEING     : {phot_header.PEEING:.3f} pixel")
                return
            else:
                self.logger.warning(
                    "No PSF stats from astrometry. Using sextractor to calculate seeing.", PreviousStageError
                )

        # config_seeing = get_key(self.config.qa, "seeing")
        # config_ellipticity = get_key(self.config.qa, "ellipticity")
        # if config_seeing and config_ellipticity:
        #     self.logger.debug(f"Using seeing, ellipticity, and pa from the config")
        #     phot_header.seeing = config_seeing
        #     phot_header.ellipticity = config_ellipticity  # 1 - b/a
        #     phot_header.elongation = 1 / (1 - config_ellipticity)  # a/b

        else:
            prep_cat = self.path.photometry.prep_catalog
            self.logger.debug(f"(photometry) prep_cat: {prep_cat}")

            # load astrometry prep cat if it exists
            astrometry_prep_cat = self.path.astrometry.catalog
            self.logger.debug(f"astrometry_prep_cat: {astrometry_prep_cat}")
            if os.path.exists(astrometry_prep_cat):
                self.logger.debug(f"Creating symlink: {astrometry_prep_cat} -> {prep_cat}")
                force_symlink(astrometry_prep_cat, prep_cat)

            if os.path.exists(prep_cat):
                self.logger.info("Calculating seeing from a pre-existing 'prep' catalog")
                self.logger.debug(f"Loading prep catalog: {prep_cat}")
                if prep_cat.endswith(".fits"):
                    obs_src_table = Table.read(prep_cat, hdu=2, format="fits")
                elif prep_cat.endswith(".cat"):
                    obs_src_table = Table.read(prep_cat, format="ascii.sextractor")
                else:
                    raise ValueError(f"Invalid catalog format: {prep_cat}")
            else:
                obs_src_table = self._run_sextractor(sex_preset="prep", fits_ldac=True, overwrite=overwrite)

            post_match_table = self.add_matched_reference_catalog(obs_src_table)
            post_match_table = self.filter_catalog(post_match_table, snr_cut=False, low_mag_cut=11.75)

            if len(post_match_table) == 0:
                self.logger.error(
                    f"No star-like sources found. Skipping photometry. Check the catalog: {prep_cat}",
                    NotEnoughSourcesError,
                )
                raise self.logger.process_error.exception(NotEnoughSourcesError)
                return

            phot_header.SEEING = np.median(post_match_table["FWHM_WORLD"] * 3600)
            phot_header.PEEING = phot_header.SEEING / self.image_info.pixscale
            phot_header.ELLIP = round(np.median(post_match_table["ELLIPTICITY"]), 3)
            phot_header.ELONG = round(np.median(post_match_table["ELONGATION"]), 3)

        self.logger.debug(f"{len(post_match_table)} Star-like Sources Found")
        self.logger.debug(f"SEEING     : {phot_header.SEEING:.3f} arcsec")
        self.logger.debug(f"ELONGATION : {phot_header.ELONG:.3f}")
        self.logger.debug(f"ELLIPTICITY: {phot_header.ELLIP:.3f}")

        return

    def _load_ref_catalog(self, filters=None) -> None:
        """
        Load reference catalog for photometric calibration.

        Handles both standard and corrected GaiaXP catalogs.
        Creates new catalog if it doesn't exist by parsing Gaia data.
        """
        # ref_ris_dir = get_key(self.config_node.photometry, "path.ref_ris_dir") or self.path.photometry.get_ref_cat(self.ref_catalog)

        predefined_ref_cat = get_key(self.config_node.photometry, "path.ref_cat")
        if predefined_ref_cat:
            ref_cat = predefined_ref_cat
        else:
            ref_cat = self.path.photometry.get_ref_cat(self.image_info.obj, ref_cat_type=self.ref_cat_type)

        self.logger.debug(f"Using photometric reference catalog: {ref_cat}")

        if is_ris_tile(self.image_info.obj) and os.path.exists(ref_cat):
            ref_src_table = Table.read(ref_cat)
        else:  # generate the ref_cat on the fly

            if hasattr(self.phot_conf, "query_radius"):
                query_radius = self.phot_conf.query_radius
            else:
                query_radius = 1.0

            self.logger.info(f"Generating Gaia reference catalog on the fly for {self.image_info.obj}")
            ref_src_table = phot_utils.aggregate_gaia_catalogs(
                target_coord=SkyCoord(self.image_info.racent, self.image_info.decent, unit="deg"),
                query_radius=query_radius,
                path_calibration_field=self.path.photometry.ref_gaia_dir,
            )
            self.logger.debug(f"ref_src_table[:5]:\n{ref_src_table[:5].pprint(max_width=150)}")

            # ref_src_table.write(ref_cat, overwrite=True)  # let's not do this.
            # 1) creates confusion for moving targets
            # 2) needs lock files under parallelization
            # 3) better not modify the requisite dir in runtime
            # TODO: maybe save it in factory (tmp_dir) for debug?

        filters = filters or [self.image_info.filter]
        synthetic_ref_mag_keys = [f"mag_{filt}" for filt in filters]

        self.gaia_columns = ["source_id", "ra", "dec", "bp_rp", "phot_g_mean_mag"]

        for col in synthetic_ref_mag_keys:
            if col not in ref_src_table.colnames:
                if col == self.image_info.filter:
                    self.logger.error(f"Column {col} not found in reference catalog", KeyError)
                    raise self.logger.process_error.exception(KeyError)
                else:
                    self.logger.warning(f"Column {col} not found in reference catalog", KeyError)
                self.logger.debug(f"Reference catalog: {ref_src_table.colnames}")
            else:
                self.gaia_columns.append(col)

        return ref_src_table[self.gaia_columns]

    def add_matched_reference_catalog(
        self,
        obs_src_table,
        filters=None,
    ) -> Table:
        """
        Adds reference source columns to obs_src_table
        by matches detected sources with reference catalog.

        Applies proper motion corrections and performs spatial matching.

        Returns:
            Table of matched sources
        """

        self.logger.debug("Loading reference catalog.")
        ref_src_table = self._load_ref_catalog(filters=filters)

        self.logger.debug("Matching sources with reference catalog.")
        r = self.phot_conf.match_radius * u.arcsec
        post_match_table = match_two_catalogs(obs_src_table, ref_src_table, x1="ra", y1="dec", radius=r, join="left")

        self.logger.info(f"Matched sources: {len(post_match_table)} (r = {r.to_value(u.arcsec):.3f} arcsec)")
        if len(post_match_table) == 0:
            self.logger.error(
                "There is no matched source for photometry. It will cause a problem in the next step.",
                NoReferenceSourceError,
            )
            raise self.logger.process_error.exception(NoReferenceSourceError)

        return post_match_table

    def filter_catalog(
        self,
        table: Table,
        snr_cut: Union[float, bool] = 20,
        low_mag_cut: Union[float, bool] = None,
        high_mag_cut: Union[float, bool] = None,
        phot_header: PhotometryHeader = None,
    ):
        """
        Filters matches based on separation, signal-to-noise, and magnitude limits.

        Args:
            snr_cut: Signal-to-noise ratio cut for filtering
            low_mag_cut: Lower magnitude limit
            high_mag_cut: Upper magnitude limit

        Returns:
            Table of sources meeting all criteria
        """
        self.logger.debug(f"Filtering catalog with columns: {table.colnames}")
        self.logger.debug(
            f"Filtering catalog with snr_cut: {snr_cut}, low_mag_cut: {low_mag_cut}, high_mag_cut: {high_mag_cut}"
        )
        low_mag_cut = low_mag_cut or self.phot_conf.ref_mag_lower
        high_mag_cut = high_mag_cut or self.phot_conf.ref_mag_upper

        phot_header = phot_header or self.phot_header

        phot_header.MAGLOW = low_mag_cut
        phot_header.MAGUP = high_mag_cut

        # Copy the original table to avoid mutation
        post_match_table = table.copy()

        post_match_table["within_ellipse"] = phot_utils.is_within_ellipse(
            np.array(post_match_table["X_IMAGE"], dtype=np.float32),
            np.array(post_match_table["Y_IMAGE"], dtype=np.float32),
            self.image_info.xcent,
            self.image_info.ycent,
            self.phot_conf.photfraction * self.image_info.naxis1 / 2,
            self.phot_conf.photfraction * self.image_info.naxis1 / 2,  # originally naxis2, changed to circle
        )

        post_match_table = phot_utils.filter_table(post_match_table, "FLAGS", 0)
        post_match_table = phot_utils.filter_table(post_match_table, "within_ellipse", True)

        if low_mag_cut:
            post_match_table = phot_utils.filter_table(
                post_match_table, self.image_info.ref_mag_key, low_mag_cut, method="lower"
            )

        if high_mag_cut:
            post_match_table = phot_utils.filter_table(
                post_match_table, self.image_info.ref_mag_key, high_mag_cut, method="upper"
            )

        if snr_cut:
            post_match_table = phot_utils.filter_table(post_match_table, "SNR_AUTO", snr_cut, method="lower")

        return post_match_table

    def photometry_with_sextractor(self, overwrite=False) -> None:
        """
        Run source extraction on the image.

        Configures and executes SExtractor with appropriate parameters
        based on seeing conditions and image characteristics.
        """
        self.logger.info(f"Start source extractor (sextractor)")

        satur_level = self.image_info.satur_level * self.phot_conf.satur_margin
        self.logger.debug(f"Saturation level with margin {1 - self.phot_conf.satur_margin}: {satur_level}")

        self.logger.debug("Setting apertures for photometry.")
        sex_options = phot_utils.get_sex_options(
            self.input_image,
            self.phot_conf,
            egain=self.image_info.egain,
            peeing=self.phot_header.PEEING,
            pixscale=self.image_info.pixscale,
            satur_level=satur_level,
        )

        # If hot pixels are present, do not convolve the image
        if not self.image_info.bpx_interp:
            self.logger.debug("Hot pixels present. Skip SEx conv.")
            sex_options["-FILTER"] = "N"

        # run sextractor with 'main' preset
        obs_src_table = self._run_sextractor(
            sex_preset="main",
            sex_options=sex_options,
            overwrite=overwrite,
        )

        # add snr columns
        self.logger.debug("Adding columns to the sextracted table")
        suffixes = [key.replace("FLUXERR_", "") for key in obs_src_table.keys() if "FLUXERR_" in key]
        for suffix in suffixes:
            obs_src_table[f"SNR_{suffix}"] = obs_src_table[f"FLUX_{suffix}"] / obs_src_table[f"FLUXERR_{suffix}"]

        return obs_src_table

    def _run_sextractor(
        self,
        sex_preset: str = "prep",
        sex_options: Optional[Dict] = None,
        output: str = None,
        fits_ldac: bool = False,
        phot_header: PhotometryHeader = None,
        overwrite=False,
        **kwargs,
    ) -> Table:
        """
        Execute SExtractor on the image.

        Args:
            output: Path for output catalog
            prefix: Prefix for temporary files
            sex_options: Additional options for SExtractor
            **kwargs: Additional keyword arguments for SExtractor

        Returns:
            Sextracted Table
        """
        self.logger.info(f"Run source extractor (sextractor) ({sex_preset})")

        if output is None:
            # new PathHandler for single
            output = getattr(
                PathHandler(self.input_image, is_too=self.config_node.settings.is_too).photometry,
                f"{sex_preset}_catalog",
            )

        self.logger.debug(f"PhotometrySingle _run_sextractor input image: {self.input_image}")
        self.logger.debug(f"PhotometrySingle _run_sextractor output catalog: {output}")
        self.logger.debug(f"PhotometrySingle _run_sextractor sex_options: {sex_options}")
        self.logger.debug(f"PhotometrySingle _run_sextractor **kwargs: {kwargs}")

        _, outcome = external.sextractor(
            self.input_image,
            outcat=output,
            sex_preset=sex_preset,
            logger=self.logger,
            sex_options=sex_options,
            return_sex_output=True,
            fits_ldac=fits_ldac,
            overwrite=overwrite,
            **kwargs,
        )

        # self.logger.debug(f"sextractor outcome: {outcome}")  # too long

        if sex_preset == "main":
            phot_header = phot_header or self.phot_header
            outcome = [s for s in outcome.split("\n") if "RMS" in s][0]
            phot_header.SKYVAL = float(outcome.split("Background:")[1].split("RMS:")[0])
            phot_header.SKYSIG = float(outcome.split("RMS:")[1].split("/")[0])

        if fits_ldac:
            return Table.read(output, hdu=2)
        return Table.read(output, format="ascii.sextractor")

    def get_zp_src_table(self, obs_src_table: Table, phot_header: PhotometryHeader = None) -> Table:
        """Returns the filtered table of image sources to be compared with the reference catalog for zp calculation"""
        self.logger.debug(f"Filtering source catalog for zp calculation")
        zp_src_table = self.filter_catalog(obs_src_table)
        self.logger.debug(f"After filtering: {len(zp_src_table)}/{len(obs_src_table)} sources")
        self.logger.info(f"Calculating zero points with {len(zp_src_table)} sources")
        phot_header = phot_header or self.phot_header
        phot_header.STDNUMB = len(zp_src_table)
        return zp_src_table

    def calculate_zp(
        self, zp_src_table: Table, filt: str = None, phot_header: PhotometryHeader = None, save_plots: bool = True
    ) -> PhotometryHeader | None:
        """
        Updates header.aperture_info in-place with zero point and aperture information.
        If header is not provided, uses self.phot_header.
        """

        phot_header = phot_header or self.phot_header

        if filt:
            ref_mag_key = f"mag_{filt}"  # column name in the reference catalog
        else:
            ref_mag_key = self.image_info.ref_mag_key

        if ref_mag_key not in zp_src_table.keys():
            self.logger.warning(
                f"Reference magnitude key {ref_mag_key} not found in the source table; consider adding it. Skipping for now...",
                KeyError,
            )
            return None

        aperture_dict = phot_utils.get_aperture_dict(phot_header.PEEING, self.image_info.pixscale)

        for aperture_key in aperture_dict.keys():
            mag_key, magerr_key = phot_utils.get_mag_key(aperture_key)

            # zp for the given aperture
            zps = zp_src_table[ref_mag_key] - zp_src_table[mag_key]
            zperrs = zp_src_table[magerr_key]  # gaia synphot err unavailable; ignored
            # zperrs = phot_utils.rss(zp_src_table[magerr_key], zp_src_table(self.image_info.ref_magerr_key))

            mask = sigma_clip(zps, sigma=2.0).mask

            input_arr = np.array(zps[~mask].value)
            if len(input_arr) == 0:
                self.logger.warning(
                    f"No valid zero point sources found for {filt} {mag_key}. Skipping.", NotEnoughSourcesError
                )
                self.logger.debug(f"input_arr: {input_arr}")
                continue
            zp, zperr = phot_utils.compute_median_nmad(input_arr, normalize=True)

            if mag_key == "MAG_AUTO":
                ul_3sig, ul_5sig = 0.0, 0.0
            else:
                aperture_size, _ = aperture_dict[aperture_key]
                ul_3sig, ul_5sig = phot_utils.limitmag(np.array([3, 5]), zp, aperture_size, phot_header.SKYSIG)
                self.logger.debug(f"filter: {ref_mag_key}, aper: {aperture_size}, SKYSIG: {phot_header.SKYSIG}")
                self.logger.debug(f"filter: {ref_mag_key}, ul_3sig: {ul_3sig}, ul_5sig: {ul_5sig}")

            phot_header.aperture_info[aperture_key] = {
                "value": aperture_dict[aperture_key][0],
                "COMMENT": aperture_dict[aperture_key][1],
                "suffix": phot_utils.get_aperture_suffix(aperture_key),
                "ZP": zp,
                "ZPERR": zperr,
                "UL3": ul_3sig,
                "UL5": ul_5sig,
            }

            if save_plots:
                plot_zp(self, mag_key, zp_src_table, zps, zperrs, zp, zperr, mask, filt=filt)
        return phot_header

    def add_calibrated_columns(
        self, obs_src_table: Table, filt: str = None, phot_header: PhotometryHeader = None
    ) -> None:
        """
        *implicit mutation & no return

        Adds columns of calibrated magnitudes, errors, fluxes, flux errors, and SNRs to the observation source table.
        """
        keyset_filter = filt or self.image_info.filter
        phot_header = phot_header or self.phot_header
        aperture_dict = phot_header.aperture_dict
        aperture_info = phot_header.aperture_info

        for aperture_key in aperture_dict.keys():
            mag_key, magerr_key = phot_utils.get_mag_key(aperture_key)

            zp = aperture_info[aperture_key]["ZP"]
            zperr = aperture_info[aperture_key]["ZPERR"]

            keys = phot_utils.keyset(mag_key, keyset_filter)
            columns = phot_utils.apply_zp(obs_src_table[mag_key], obs_src_table[magerr_key], zp, zperr)

            for key, arr in zip(keys, columns):
                obs_src_table[key] = arr
                obs_src_table[key].format = ".3f"

    # def add_reference_columns(self, obs_src_table: Table, cols: Dict[str, np.ndarray]) -> None:
    #     """
    #     *implicit mutation & no return

    #     Adds reference columns to the observation source table.
    #     """
    #     for key, arr in cols.items():
    #         obs_src_table[key] = arr
    #         obs_src_table[key].format = ".3f"
    #     return

    def determine_filter(self, phot_headers: Dict[str, PhotometryHeader], save_plot=True) -> str:
        """Updates PhotometryHeader.INF_FILT with the best-matching filter inferred by the pipeline"""
        zp_cut = 27.2  # 26.8
        alleged_filter = self.image_info.filter
        filters_checked = [k for k in phot_headers.keys()]

        self.logger.debug(f"phot_headers: {phot_headers}")
        self.logger.debug(f"alleged_filter: {alleged_filter}")
        self.logger.debug(f"filters_checked: {filters_checked}")

        if len(phot_headers) == 0:
            self.logger.warning(f"No phot_headers found. Using alleged filter.", FilterCheckError)
            inferred_filter = alleged_filter
            # Persist inferred filter via PhotometryHeader (written by update_image_header())
            self.phot_header.INF_FILT = inferred_filter
            return alleged_filter

        # filter out when phot_headers[filt] is None
        dicts = {
            filt: (header.zp_dict, header.aperture_dict) for filt, header in phot_headers.items() if header is not None
        }
        dicts_for_plotting = dicts.copy()
        test_dicts = dicts.copy()

        self.logger.debug(f"filters_checked: {filters_checked}")
        # self.logger.debug(f"dicts: {dicts}")  # too long

        # (1) rule out filters that were not present at the time
        active_filters = self.get_active_filters()
        self.logger.debug(f"Active filters: {active_filters}")

        for filt in list(set(filters_checked) - set(active_filters)):
            if filt in test_dicts:
                test_dicts.pop(filt)
                self.logger.debug(f"Filter {filt} is not active. Removing from checklist.")

        # self.logger.debug(f"Filtered dicts: {test_dicts}")  # too long

        # (2) apply prior knowledge of zp for broad and medium band filters
        while True:
            narrowed_filters, zps, zperrs = phot_utils.dicts_to_lists(test_dicts)
            self.logger.debug(f"Narrowed filters: {narrowed_filters}")
            self.logger.debug(f"ZPs: {zps}")
            self.logger.debug(f"ZP errors: {zperrs}")
            if not zperrs:
                self.logger.warning(
                    f"No valid zero point sources found. Falling back to header filter '{alleged_filter}'",
                    FilterCheckError,
                )
                return alleged_filter

            idx = zperrs.index(min(zperrs))
            inferred_filter = narrowed_filters[idx]
            zp = zps[idx]
            zperr = zperrs[idx]

            # if (inferred_filter in BROAD_FILTERS and zp <= zp_cut) or (
            #     inferred_filter in MEDIUM_FILTERS and zp > zp_cut
            # ):
            if inferred_filter in MEDIUM_FILTERS and zp > zp_cut:  # only this is safe. TODO: adhoc
                self.logger.debug(
                    f"Filter {inferred_filter} is a {'broadband' if inferred_filter in BROAD_FILTERS else 'mediumband'} filter and has a zero point (zp) of {zp} which is less than the zp cut, {zp_cut}. Removing from potential filters."
                )
                test_dicts.pop(inferred_filter)
            else:
                self.logger.debug(
                    f"Found the best-matching filter, '{inferred_filter}', with zp = {zp}+/-{zperr}. Breaking the loop."
                )
                break

            if len(test_dicts) == 0:
                zp, zperr = phot_utils.get_zp_from_dict(dicts, alleged_filter)

                self.logger.error(
                    f"Filter determination process eliminated all candidates. Falling back to header filter '{alleged_filter}' (zp = {zp:.2f}±{zperr:.2f})",
                    FilterCheckError,
                )
                inferred_filter = alleged_filter
                break

        if save_plot:
            filters, zps, zperrs = phot_utils.dicts_to_lists(dicts_for_plotting)
            plot_filter_check(self, alleged_filter, inferred_filter, narrowed_filters, filters_checked, zps, zperrs)

        if alleged_filter != inferred_filter:
            orig_zp, orig_zperr = phot_utils.get_zp_from_dict(dicts, alleged_filter)
            self.logger.warning(
                f"The filter in header ({alleged_filter}; zp = {orig_zp:.2f}±{orig_zperr:.2f}) is not the best matching ({inferred_filter}; zp = {zp:.2f}±{zperr:.2f})",
                InferredFilterMismatchError,
            )
        else:
            self.logger.info(f"The inferred filter matches the original filter, '{alleged_filter}'")

        self.phot_header.INF_FILT = inferred_filter

        return inferred_filter

    def get_active_filters(self) -> set:
        """TODO: ad-hoc before DB integration"""
        from glob import glob
        from ..path import NameHandler
        from ..services.database.query import RawImageQuery

        query = RawImageQuery()
        # Django-postresql main image db
        if query.is_connected():
            self.logger.debug(f"get_active_filters: Connected to Django-PostgreSQL raw image DB")
            name = NameHandler(self.input_image)
            all_filter_files = (
                query.on_date(name.nightdate).by_units([name.unit]).of_types(["flat", "sci"]).image_files()
            )
            if all_filter_files:
                active_filters = set(NameHandler(all_filter_files).filter)
                self.logger.debug(f"Fetched active filters from Django-PostgreSQL raw image DB: {active_filters}")
                return active_filters
            self.logger.warning(
                f"get_active_filters: empty DB result; falling back to local filesystem", FilterInventoryError
            )
        else:
            self.logger.warning(
                f"get_active_filters: cannot connect to Django-PostgreSQL main raw image DB", ConnectionError
            )

        # sqlite pipeline db
        # TODO
        if False:
            self.logger.debug(f"Fetched active filters from sqlite filter inventory: {active_filters}")

        self.logger.debug(f"get_active_filters: DB connection failed, fallback to local filesystem")
        # fallback to local filesystem
        try:
            template = PathHandler(self.input_image, is_too=self.config_node.settings.is_too).conjugate
            self.logger.debug(f"Filter check glob template (PathHandler.conjugate): {template}")
            flist = glob(os.path.join(os.path.dirname(template), "*.fits"))
            flats = NameHandler(flist).pick_type("raw_flat")
            scis = NameHandler(flist).pick_type("raw_science")
            active_filters = set(NameHandler(flats + scis).filter)
            self.logger.debug(f"Fetched active filters from local filesystem: {active_filters}")
            return active_filters
        except (SystemError, PathHandlerError) as e:
            self.logger.warning(
                f"Unable to fetch active filters from on-date inventory. Using all default filters: {e}",
                FilterInventoryError,
            )
            return ALL_FILTERS
        except Exception as e:
            self.logger.warning(
                f"Fetching active filters failed. Using all default filters: {e}", FilterInventoryError, exc_info=True
            )
            return ALL_FILTERS

    def update_image_header(
        self,
        phot_header: PhotometryHeader = None,
    ) -> None:
        """
        Update the input fits image's header with photometry information.
        """
        if phot_header is not None:
            self.logger.debug(f"Using provided phot_header at update_image_header(): {phot_header}")
        else:
            phot_header = self.phot_header
            self.logger.debug(f"Using self.phot_header at update_image_header(): {phot_header}")
        update_padded_header(self.input_image, phot_header.dict)
        self.logger.info(f"Image header has been updated with photometry information (aperture, zero point, ...).")

    def write_catalog(self, obs_src_table) -> None:
        """
        Write photometry catalog to disk.

        Saves the source detection and photometry results to a catalog file.
        The catalog includes all detected sources with their measured properties
        and calculated photometric values.
        """

        metadata = self.image_info.metadata
        obs_src_table.meta = metadata
        # self.obs_src_table.meta["comments"] = [
        #     "Zero point based on Gaia DR3",
        #     "Zero point uncertainty is not included in FLUX and MAG errors",
        # ]  # this interferes with table description

        # reorder gaia columns to be at the end
        if hasattr(self, "gaia_columns") and not (self._trust_header_zp):
            gaia_cols = self.gaia_columns + [
                "separation"
            ]  # e.g. ["source_id", "bp_rp", f"mag_{self.image_info.filter}"]
            present_gaia_cols = [c for c in gaia_cols if c in obs_src_table.colnames]
            missing_gaia_cols = [c for c in gaia_cols if c not in obs_src_table.colnames]
            if missing_gaia_cols:
                self.logger.warning(
                    f"Some Gaia columns are missing and being omitted: {missing_gaia_cols}.",
                    PrerequisiteNotMetError,
                )

            other_cols = [c for c in obs_src_table.colnames if c not in gaia_cols]
            obs_src_table = obs_src_table[other_cols + present_gaia_cols]

        # save
        output_catalog_file = self.path.photometry.final_catalog
        obs_src_table.write(output_catalog_file, format="fits", overwrite=True)  # "ascii.tab" "ascii.ecsv"
        self.logger.info(f"Photometry catalog is written in {os.path.basename(output_catalog_file)}")

        return


@dataclass(slots=True, frozen=True)
class ImageInfo:
    """
    Stores information extracted from a FITS image header for photometry.
    The goal is to load all required header information once and keep it frozen.
    """

    obj: str  # Object name
    filter: str  # Filter used
    unit: str  # 7DT Unit used
    dateobs: str  # Observation date/time
    egain: float  # gain in e-/ADU, not camera gain
    naxis1: int  # Image width
    naxis2: int  # Image height
    ref_mag_key: str  # Reference magnitude key
    ref_magerr_key: str  # Reference magnitude error key
    jd: float  # Julian Date
    mjd: float  # Modified Julian Date
    racent: float  # RA of image center
    decent: float  # DEC of image center
    xcent: float  # X coordinate of image center
    ycent: float  # Y coordinate of image center
    n_binning: int  # Binning factor
    pixscale: float  # Pixel scale [arcsec/pix]
    satur_level: float = 2**16 - 1  # Saturation level
    bpx_interp: bool = False  # whether bad pixels have been interpolated
    SEEINGMN: float = None
    PEEINGMN: float = None
    ELLIPMN: float = None
    ELONGMN: float = None

    # keys needed in PhotometryHeader
    phot_header_keys: dict = None  # don't make this dict here; can be shared by multiple instances

    def __repr__(self) -> str:
        """Returns a string representation of the ImageInfo."""
        # no __dict__ with slots=True
        # return ",\n".join(f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return ",\n".join(f"  {f.name}: {getattr(self, f.name)}" for f in fields(self) if not f.name.startswith("_"))

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns a dictionary of metadata."""
        return {
            "OBJECT": self.obj,
            "FILTER": self.filter,
            "TELESCOP": self.unit,
            "DATE-OBS": self.dateobs,
            "JD": self.jd,
            "MJD": self.mjd,
        }

    @property
    def has_psf_stats_from_astrometry(self) -> bool:
        return (
            self.SEEINGMN is not None
            and self.PEEINGMN is not None
            and self.ELLIPMN is not None
            and self.ELONGMN is not None
        )

    @classmethod
    def parse_image_header_info(cls, image_path: str) -> ImageInfo:
        """Parses image information from a FITS header."""

        hdr = fits.getheader(image_path)
        wcs = WCS(image_path)

        # Center coord for reference catalog query
        xcent = (hdr["NAXIS1"] + 1) / 2.0
        ycent = (hdr["NAXIS2"] + 1) / 2.0
        racent, decent = wcs.all_pix2world(xcent, ycent, 1)
        racent = float(racent)
        decent = float(decent)

        time_obj = Time(hdr["DATE-OBS"], format="isot")
        jd = float(time_obj.jd)
        mjd = float(time_obj.mjd)

        if "INTERP" not in hdr.keys():  # if badpix uninterpolated
            interped = False
        else:
            interped = True

        if "CTYPE1" not in hdr.keys():
            # just single phot error. propagating the 3 kinds of photometry errors is not worth the effort.
            raise SinglePhotometryError.PreviousStageError(
                "Check Astrometry solution: no WCS information for Photometry"
            )

        ELONGMN = hdr.get("ELONGMN", None)
        if ELONGMN is None and hdr.get("ELLIPMN", None) is not None:
            ELONGMN = 1 / (1 - hdr["ELLIPMN"])

        kwargs = dict(
            obj=hdr["OBJECT"],
            filter=hdr["FILTER"],
            unit=hdr["TELESCOP"],
            dateobs=hdr["DATE-OBS"],
            egain=float(hdr["EGAIN"]),  # effective gain for combined images
            naxis1=int(hdr["NAXIS1"]),
            naxis2=int(hdr["NAXIS2"]),
            ref_mag_key=f"mag_{hdr['FILTER']}",
            ref_magerr_key=f"magerr_{hdr['FILTER']}",
            jd=jd,
            mjd=mjd,
            racent=racent,
            decent=decent,
            xcent=xcent,
            ycent=ycent,
            n_binning=hdr["XBINNING"],
            pixscale=hdr["XBINNING"] * PIXSCALE,
            satur_level=get_header_key(image_path, "SATURATE", default=60000),
            bpx_interp=interped,
            SEEINGMN=hdr.get("SEEINGMN", None),
            PEEINGMN=hdr.get("PEEINGMN", None),
            ELLIPMN=hdr.get("ELLIPMN", None),
            ELONGMN=ELONGMN,
        )

        # pick up header keys needed by PhotometryHeader
        phot_header_keys: Dict[str, Any] = {}
        for f in fields(PhotometryHeader):
            key = f.name
            if key in hdr:
                phot_header_keys[key] = hdr[key]

        for key in hdr.keys():
            if key.startswith(("AUTO", "APER", "ZP", "EZP", "UL3", "UL5")):
                phot_header_keys[key] = hdr[key]

        kwargs["phot_header_keys"] = phot_header_keys

        return cls(**kwargs)


@dataclass
class PhotometryHeader:
    """
    Builds the fits header about photometry to be written to the image.
    If the image already has the photometry result, PhotometryHeader reads it.

    This is dynamic, unlike ImageInfo. Keys like AUTHOR can be overwritten.
    """

    # must match the actual header keys
    AUTHOR: str = "pipeline"
    PHOTIME: str = None
    INF_FILT: str = None
    JD: float = None
    MJD: float = None
    SEEING: float = None
    PEEING: float = None
    ELLIP: float = None
    ELONG: float = None
    SKYSIG: float = None
    SKYVAL: float = None
    REFCAT: str = None  # "GaiaXP"
    MAGLOW: float = None
    MAGUP: float = None
    STDNUMB: int = None

    # a dict of all information accompanying each aperture
    aperture_info: Dict = None
    # e.g.,
    # {
    #     "APER_1": {
    #         "value": ,
    #         "COMMENT": "...",
    #         "suffix": "1",
    #         "ZP": ,
    #         "ZPERR": ,
    #         "UL3": ,
    #         "UL5":
    #     },
    #    ....
    # }

    def __init__(self, image_info: ImageInfo = None):
        """If photometry result already exists, load it from the image header"""
        self.aperture_info = {}  # set here to avoid sharing between instances

        if image_info is not None:
            self.PHOTIME = datetime.date.today().isoformat()
            self.image_info = image_info

            for f in fields(self):
                key = f.name

                # don't skip. overridden anyway.
                # # skip these fields
                # if key in ["AUTHOR", "PHOTIME"]:
                #     continue

                if key in image_info.phot_header_keys:
                    # try:
                    setattr(self, f.name, image_info.phot_header_keys[key])
                    # except Exception:
                    #     pass  # silently ignore any conversion errors

                if "ZP_AUTO" in image_info.phot_header_keys:
                    self._set_aperture_info_from_header(image_info.phot_header_keys)

    def _set_aperture_info_from_header(self, phot_header_keys: dict) -> None:
        """
        Sets aperture information from the image header.
        Relies on ImageInfo.phot_header_keys.
        """

        aperture_dict = phot_utils.get_aperture_dict(self.PEEING, self.image_info.pixscale)

        for aperture_key in aperture_dict.keys():
            mag_key, magerr_key = phot_utils.get_mag_key(aperture_key)
            suffix = phot_utils.get_aperture_suffix(aperture_key)

            zp = phot_header_keys[f"ZP_{suffix}"]
            zperr = phot_header_keys[f"EZP_{suffix}"]
            if mag_key == "MAG_AUTO":
                ul_3sig, ul_5sig = 0.0, 0.0
            else:
                ul_3sig = phot_header_keys[f"UL3_{suffix}"]
                ul_5sig = phot_header_keys[f"UL5_{suffix}"]

            self.aperture_info[aperture_key] = {
                "value": aperture_dict[aperture_key][0],
                "COMMENT": aperture_dict[aperture_key][1],
                "suffix": suffix,
                "ZP": zp,
                "ZPERR": zperr,
                "UL3": ul_3sig,
                "UL5": ul_5sig,
            }

    @property
    def aperture_dict(self) -> Dict:
        """same as what phot_utils.get_aperture_dict returns"""
        return {k: (v["value"], v["COMMENT"]) for k, v in self.aperture_info.items()}

    @property
    def zp_dict(self) -> Dict:
        temp = {}
        for aper, bundle in self.aperture_info.items():
            mag_key, magerr_key = phot_utils.get_mag_key(aper)
            suffix = bundle["suffix"]
            zp = bundle["ZP"]
            zperr = bundle["ZPERR"]
            ul_3sig = bundle["UL3"]
            ul_5sig = bundle["UL5"]
            temp.update(
                {
                    f"ZP_{suffix}": (zp, f"ZERO POINT for {mag_key}"),
                    f"EZP_{suffix}": (zperr, f"ZERO POINT ERROR for {mag_key}"),
                    f"UL3_{suffix}": (ul_3sig, f"3 SIGMA LIMITING MAG FOR {mag_key}"),
                    f"UL5_{suffix}": (ul_5sig, f"5 SIGMA LIMITING MAG FOR {mag_key}"),
                }
            )
        return temp

    @property
    def dict(self) -> Dict[str, Tuple[Any, str]]:
        """This is updated to the image header"""

        phot_header_dict = {}

        misc_dict = {
            "AUTHOR": (self.AUTHOR, "user who last updated photometry header"),
            "PHOTIME": (self.PHOTIME, "PHOTOMETRY TIME [KST]"),
            "INF_FILT": (self.INF_FILT, "BEST-MATCHING FILTER INFERRED BY PIPELINE"),
            "JD": (self.JD, "Julian Date of the observation"),
            "MJD": (self.MJD, "Modified Julian Date of the observation"),
            "SEEING": (round(self.SEEING, 3) if self.SEEING is not None else 0, "SEEING [arcsec]"),
            "PEEING": (round(self.PEEING, 3) if self.PEEING is not None else 0, "SEEING [pixel]"),
            "ELLIP": (round(self.ELLIP, 3) if self.ELLIP is not None else 0, "ELLIPTICITY 1-B/A [0-1]"),
            "ELONG": (round(self.ELONG, 3) if self.ELONG is not None else 0, "ELONGATION A/B [1-]"),
            "SKYSIG": (round(self.SKYSIG, 3) if self.SKYSIG is not None else 0, "SKY SIGMA VALUE"),
            "SKYVAL": (round(self.SKYVAL, 3) if self.SKYVAL is not None else 0, "SKY MEDIAN VALUE"),
            "REFCAT": (self.REFCAT, "REFERENCE CATALOG TYPE"),
            "MAGLOW": (self.MAGLOW, "REF MAG RANGE, LOWER LIMIT"),
            "MAGUP": (self.MAGUP, "REF MAG RANGE, UPPER LIMIT"),
            "STDNUMB": (self.STDNUMB, "# OF STD STARS TO CALIBRATE ZP"),
        }

        phot_header_dict.update(misc_dict)
        # round float values to .3f
        phot_header_dict.update({k: (round(v[0], 3), v[1]) for k, v in self.aperture_dict.items()})
        phot_header_dict.update({k: (round(v[0], 3), v[1]) for k, v in self.zp_dict.items()})

        # Filter out entries where the value is None
        return {k: v for k, v in phot_header_dict.items() if v[0] is not None}

    def __repr__(self) -> str:
        # return ",\n".join(f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return ",\n".join(f"  {k}: {v}" for k, v in self.dict.items())


# @dataclass
# class PhotometryCatalog:
#     """Stores the catalog-side results of photometry for a single image."""

#     image_info: ImageInfo  # link back to the image
#     obs_src_table: Table  # final catalog (with ref cols)
#     zp_src_table: Table  # subset used for ZP
#     column_data: Dict[str, np.ndarray]  # optional, if you want
