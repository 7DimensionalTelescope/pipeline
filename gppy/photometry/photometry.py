from __future__ import annotations
import os
import getpass
from reprlib import recursive_repr
from typing import Any, List, Dict, Tuple, Optional, Union
import datetime
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from pathlib import Path
from dataclasses import dataclass, fields

# astropy
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.table import Table, hstack, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

# gppy modules
from ..config.utils import get_key
from ..utils import time_diff_in_seconds, get_header_key, force_symlink, collapse
from ..config import SciProcConfiguration
from ..config.base import ConfigurationInstance
from ..services.memory import MemoryMonitor
from ..services.queue import QueueManager, Priority
from .. import external
from ..const import PIXSCALE, MEDIUM_FILTERS, BROAD_FILTERS, ALL_FILTERS, PipelineError
from ..services.setup import BaseSetup
from ..tools.table import match_two_catalogs, build_condition_mask
from ..path.path import PathHandler
from ..header import update_padded_header

from ..services.database.handler import DatabaseHandler
from ..services.database.table import QAData

from . import utils as phot_utils
from .plotting import plot_zp, plot_filter_check


class Photometry(BaseSetup, DatabaseHandler):
    """
    A class to perform photometric analysis on astronomical images.

    This class handles both single-image and batch photometry processing,
    with support for parallel processing through a queue system.

    Attributes:
        config (Configuration): Configuration settings for photometry
        logger: Logger instance for output messaging
        queue (Optional[QueueManager]): Queue manager for parallel processing
        ref_catalog (str): Name of reference catalog to use
        images (List[str]): List of image files to process
    """

    def __init__(
        self,
        config: Any = None,
        logger: Any = None,
        queue: Union[bool, QueueManager] = False,
        images: Optional[List[str]] = None,
        photometry_mode: Optional[str] = None,
        ref_catalog: Optional[str] = None,
    ) -> None:
        """
        Initialize the Photometry class.

        Photometry will know its mode based on self.config.flag., but it is
        safer to explicitly set the mode when re-running scidata reduction,
        where the flags can be out of sync.

        Args:
            config: Configuration object or path to config yaml
            logger: Logger instance for output messaging
            queue: Queue manager for parallel processing or boolean to create one
            images: List of image files to process
            photometry_mode: None, single_photometry, combined_photometry, difference_photometry
            ref_catalog: Name of reference catalog to use
        """
        # Load Configuration
        super().__init__(config, logger, queue)
        # self._flag_name = "photometry"

        self.ref_catalog = ref_catalog or self.config.photometry.refcatname

        if photometry_mode == "single_photometry" or (
            not self.config.flag.single_photometry
            # and (not self.config.input.stacked_image and not self.config.input.difference_image)  # hassle when rerunning
        ):
            self.config.photometry.input_images = images or self.config.input.calibrated_images
            self.input_images = self.config.photometry.input_images
            self.logger.debug("Running single photometry")
            self._photometry_mode = "single_photometry"
        elif photometry_mode == "combined_photometry" or (
            not self.config.flag.combined_photometry
            # and (self.config.input.stacked_image and not self.config.input.difference_image)
        ):
            self.config.photometry.input_images = images or [x] if (x := self.config.input.stacked_image) else None
            self.input_images = self.config.photometry.input_images
            self.logger.debug("Running combined photometry")
            self._photometry_mode = "combined_photometry"
        elif photometry_mode == "difference_photometry" or not self.config.flag.difference_photometry:
            self.config.photometry.input_images = images or [x] if (x := self.config.input.difference_image) else None
            self.input_images = self.config.photometry.input_images
            self.logger.debug("Running difference photometry")
            self._photometry_mode = "difference_photometry"

        else:
            self.logger.debug(f"photometry mode undefined: {photometry_mode}")
            raise PipelineError(
                "Unexpected photometry mode: check if flags are sequentially turned on for single_photometry, combined_photometry, and difference_photometry"
            )

        self.logger.info(f"Photometry mode: {self._photometry_mode}")

        self.qa_ids = []
        DatabaseHandler.__init__(self, add_database=self.config.settings.is_pipeline)

        if self.is_connected:
            self.set_logger(logger)
            self.logger.debug("Initialized DatabaseHandler for pipeline and QA data management")
            self.pipeline_id = self.create_pipeline_data(self.config)
            if self.pipeline_id is not None:
                self.update_pipeline_progress(self._progress_by_mode[0], f"{self._photometry_mode}-configured")

    @property
    def _progress_by_mode(self):
        if self._photometry_mode == "single_photometry":
            return 40, 20
        elif self._photometry_mode == "combined_photometry":
            return 70, 10
        elif self._photometry_mode == "difference_photometry":
            return 90, 10
        else:
            raise PipelineError(f"Undefined photometry mode: {self._photometry_mode}")

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

        config = SciProcConfiguration.base_config(input_images=images, working_dir=working_dir, logger=True)
        return cls(config=config)

    @property
    def sequential_task(self):
        """[(number, name, use_gpu), ...]"""
        return [(1, "run", False)]

    def run(self) -> None:
        """
        Run photometry on all configured images.

        Processes images either sequentially or in parallel depending on queue configuration.
        Updates configuration flags and performs memory cleanup after completion.
        """
        st = time.time()
        self.logger.info(f"Start 'Photometry'")
        if not self.input_images:  # exception for when input.difference_image is not set.
            self.logger.debug(f"input_images: {self.input_images}")
            self.logger.info(f"No input images found. Skipping {self._photometry_mode}.")
            return
        try:
            if self.queue:
                self._run_parallel()
            else:
                self._run_sequential()

            if self._photometry_mode == "single_photometry":
                self.config.flag.single_photometry = True
            elif self._photometry_mode == "combined_photometry":
                self.config.flag.combined_photometry = True
            elif self._photometry_mode == "difference_photometry":
                self.config.flag.difference_photometry = True
            else:
                raise PipelineError(f"Undefined photometry mode: {self._photometry_mode}")

            if self.is_connected:
                for image, qa_id in zip(self.input_images, self.qa_ids):
                    qa_data = QAData.from_header(
                        fits.getheader(image), "science", "science", self.pipeline_id, os.path.basename(image)
                    )
                    qa_dict = qa_data.to_dict()
                    qa_dict["qa_id"] = qa_id
                    qa_id = self.qa_db.update_qa_data(**qa_dict)

            self.update_pipeline_progress(sum(self._progress_by_mode), f"{self._photometry_mode}-completed")

            self.logger.info(f"'Photometry' is Completed in {time_diff_in_seconds(st)} seconds")
            self.logger.debug(MemoryMonitor.log_memory_usage)
        except Exception as e:
            self.logger.critical(f"Photometry failed: {str(e)}", exc_info=True)
            raise

    def _run_parallel(self) -> None:
        """Process images in parallel using queue system."""
        task_ids = []
        for i, image in enumerate(self.input_images):
            process_name = f"{self.config.name} - single photometry - {i+1} of {len(self.input_images)}"
            single_config = self.config.extract_single_image_config(i)
            diff_phot = self._photometry_mode == "difference_photometry"

            phot_single = PhotometrySingle(
                image,
                single_config,  # self.config,
                self.logger,
                ref_catalog=self.ref_catalog,
            )
            task_id = self.queue.add_task(
                phot_single.run,
                kwargs={"name": process_name, "difference_photometry": diff_phot},
                priority=Priority.MEDIUM,
                gpu=False,
                task_name=process_name,
            )
            task_ids.append(task_id)
        self.queue.wait_until_task_complete(task_ids)

    def _run_sequential(self) -> None:
        """Process images sequentially."""
        for i, image in enumerate(self.input_images):
            single_config = self.config.extract_single_image_config(i)
            diff_phot = self._photometry_mode == "difference_photometry"
            PhotometrySingle(
                # image,
                single_config,  # self.config,
                logger=self.logger,
                ref_catalog=self.ref_catalog,
                total_image=len(self.input_images),
                difference_photometry=diff_phot,
            ).run()

            self.update_pipeline_progress(
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
    - Results output

    Attributes:
        config: Configuration settings
        logger: Logger instance
        ref_catalog (str): Reference catalog name
        image (str): Path to image file
        image_info (ImageInfo): Parsed image metadata
        phot_conf: Photometry configuration settings
        name (str): Process name for logging
        header (ImageHeader): Header information container
        _id (int): Unique identifier for this instance
    """

    _id_counter = itertools.count(1)

    def __init__(
        self,
        # image: str,
        config: ConfigurationInstance,
        logger: Any = None,
        name: Optional[str] = None,
        ref_catalog: str = "GaiaXP",
        total_image: int = 1,
        check_filter=True,
        difference_photometry=False,
    ) -> None:
        """Initialize PhotometrySingle instance."""

        if hasattr(config, "config"):
            config = config.config

        self.config = config
        self.logger = logger or self._setup_logger(config)
        self.ref_catalog = ref_catalog
        # self.image = os.path.join(self.config.path.path_processed, image)
        # self.input_image = image
        self.phot_conf = self.config.photometry
        self.input_image = collapse(self.phot_conf.input_images, raise_error=True)
        self.logger.debug(f"input_image: {self.input_image}")
        self.logger.debug(f"=" * 100)
        self.image_info = ImageInfo.parse_image_header_info(self.input_image)
        self.name = name or os.path.basename(self.input_image)
        self.logger.debug(f"{self.name}: ImageInfo: {self.image_info}")
        self.phot_header = PhotometryHeader(self.image_info)
        self.phot_header.AUTHOR = getpass.getuser()

        self._id = str(next(self._id_counter)) + "/" + str(total_image)

        self.path = PathHandler(self.input_image)
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

    def run(self) -> None:
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

            self.calculate_seeing(use_header_seeing=self._trust_header_seeing)
            obs_src_table = self.photometry_with_sextractor()

            if self._check_filter:
                self.logger.info("Performing filter check")
                filters_to_check = get_key(self.config.photometry, "filters_to_check") or ALL_FILTERS
                obs_src_table = self.add_matched_reference_catalog(obs_src_table, filters=filters_to_check)
                zp_src_table = self.get_zp_src_table(obs_src_table)

                temp_headers = {}
                self.logger.debug(f"Starting filter check for {filters_to_check}")
                for i, filt in enumerate(filters_to_check):
                    phot_header = PhotometryHeader()  # just a container of aperture_info;
                    # can't save alternative photometry result to image header anyway due to namespace collision
                    # PhotometryHeader(self.image_info) if you want to save to image header

                    phot_header = self.calculate_zp(zp_src_table, filt=filt, phot_header=phot_header, save_plots=False)
                    temp_headers[filt] = phot_header
                    self.logger.debug(f"Filter check photometry for {filt} is completed")

                # save the photometry result for the filename filter as default
                # placed after filter inference because in the future this will be the main photometry
                phot_header = self.calculate_zp(zp_src_table)
                self.add_calibrated_columns(obs_src_table, phot_header=phot_header)
                self.update_image_header(phot_header)

                # if inferred filter is different, save it to header and catalog, too
                inferred_filter = self.determine_filter(temp_headers)
                if inferred_filter != self.image_info.filter:
                    self.logger.info(f"Saving {inferred_filter} photometry to catalog too")
                    inferred_phot_header = temp_headers[inferred_filter]
                    self.add_calibrated_columns(obs_src_table, filt=inferred_filter, phot_header=inferred_phot_header)
                    # self.update_image_header(inferred_phot_header)  # you can't save it too due to namespace collision

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
            self.logger.critical(f"PhotometrySingle failed for the image [{self._id}]: {str(e)}", exc_info=True)
            raise

    def calculate_seeing(self, use_header_seeing: bool = False) -> None:
        """
        Calculate seeing conditions from stellar sources in the image.

        Uses source extraction to identify stars and calculate median FWHM,
        ellipticity, and elongation values stored by self.phot_header.

        Args:
            low_mag_cut: Lower magnitude limit for star selection
        """

        if use_header_seeing:
            hdr = fits.getheader(self.input_image)
            self.phot_header.SEEING = hdr["SEEING"]
            self.phot_header.PEEING = hdr["PEEING"]
            self.phot_header.ELLIP = hdr["ELLIP"]
            self.phot_header.ELONG = hdr["ELONG"]
            self.logger.debug(f"Trusting seeing/peeing from the header")
            self.logger.debug(f"SEEING     : {self.phot_header.SEEING:.3f} arcsec")
            self.logger.debug(f"PEEING     : {self.phot_header.PEEING:.3f} pixel")
            return

        # config_seeing = get_key(self.config.qa, "seeing")
        # config_ellipticity = get_key(self.config.qa, "ellipticity")
        # if config_seeing and config_ellipticity:
        #     self.logger.debug(f"Using seeing, ellipticity, and pa from the config")
        #     self.phot_header.seeing = config_seeing
        #     self.phot_header.ellipticity = config_ellipticity  # 1 - b/a
        #     self.phot_header.elongation = 1 / (1 - config_ellipticity)  # a/b

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
                    obs_src_table = Table.read(prep_cat, hdu=2)
                elif prep_cat.endswith(".cat"):
                    obs_src_table = Table.read(prep_cat, format="ascii.sextractor")
                else:
                    raise ValueError(f"Invalid catalog format: {prep_cat}")
            else:
                obs_src_table = self._run_sextractor(se_preset="prep", fits_ldac=True)

            post_match_table = self.add_matched_reference_catalog(obs_src_table)
            post_match_table = self.filter_catalog(post_match_table, snr_cut=False, low_mag_cut=11.75)

            self.phot_header.SEEING = np.median(post_match_table["FWHM_WORLD"] * 3600)
            self.phot_header.PEEING = self.phot_header.SEEING / self.image_info.pixscale
            self.phot_header.ELLIP = round(np.median(post_match_table["ELLIPTICITY"]), 3)
            self.phot_header.ELONG = round(np.median(post_match_table["ELONGATION"]), 3)

        self.logger.debug(f"{len(post_match_table)} Star-like Sources Found")
        self.logger.debug(f"SEEING     : {self.phot_header.SEEING:.3f} arcsec")
        self.logger.debug(f"ELONGATION : {self.phot_header.ELONG:.3f}")
        self.logger.debug(f"ELLIPTICITY: {self.phot_header.ELLIP:.3f}")

        return

    def _load_ref_catalog(self, filters=None) -> None:
        """
        Load reference catalog for photometric calibration.

        Handles both standard and corrected GaiaXP catalogs.
        Creates new catalog if it doesn't exist by parsing Gaia data.
        """
        ref_ris_dir = get_key(self.config.photometry, "path.ref_ris_dir") or self.path.photometry.ref_ris_dir
        if self.ref_catalog == "GaiaXP_cor":
            ref_cat = f"{ref_ris_dir}/cor_gaiaxp_dr3_synphot_{self.image_info.obj}.csv"
        elif self.ref_catalog == "GaiaXP":
            ref_cat = f"{ref_ris_dir}/gaiaxp_dr3_synphot_{self.image_info.obj}.csv"
        else:
            raise ValueError(f"Invalid reference catalog: {self.ref_catalog}. It should be 'GaiaXP' or 'GaiaXP_cor'")

        # generate the missing ref_cat and save on disk
        ref_gaia_dir = get_key(self.config.photometry, "path.ref_gaia_dir") or self.path.photometry.ref_gaia_dir
        if not os.path.exists(ref_cat):  # and "gaia" in self.ref_catalog:
            ref_src_table = phot_utils.aggregate_gaia_catalogs(
                target_coord=SkyCoord(self.image_info.racent, self.image_info.decent, unit="deg"),
                path_calibration_field=ref_gaia_dir,
                matching_radius=self.phot_conf.match_radius * 1.5,
                path_save=ref_cat,
            )
            ref_src_table.write(ref_cat, overwrite=True)

        else:
            ref_src_table = Table.read(ref_cat)

        filters = filters or [self.image_info.filter]
        synthetic_mag_keys = [f"mag_{filt}" for filt in filters]

        self.gaia_columns = ["source_id", "ra", "dec", "bp_rp", "phot_g_mean_mag"] + synthetic_mag_keys
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
        post_match_table = match_two_catalogs(obs_src_table, ref_src_table, x1="ra", y1="dec", join="left")

        self.logger.info(f"Matched sources: {len(post_match_table)} (r = {self.phot_conf.match_radius:.3f} arcsec)")
        if len(post_match_table) == 0:
            self.logger.critical("There is no matched source for photometry. It will cause a problem in the next step.")

        return post_match_table

    def filter_catalog(
        self,
        table: Table,
        snr_cut: Union[float, bool] = 20,
        low_mag_cut: Union[float, bool] = None,
        high_mag_cut: Union[float, bool] = None,
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

        # Copy the original table to avoid mutation
        post_match_table = table.copy()

        post_match_table["within_ellipse"] = phot_utils.is_within_ellipse(
            np.array(post_match_table["X_IMAGE"], dtype=np.float32),
            np.array(post_match_table["Y_IMAGE"], dtype=np.float32),
            self.image_info.xcent,
            self.image_info.ycent,
            self.phot_conf.photfraction * self.image_info.naxis1 / 2,
            self.phot_conf.photfraction * self.image_info.naxis2 / 2,
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

    def photometry_with_sextractor(self) -> None:
        """
        Run source extraction on the image.

        Configures and executes SExtractor with appropriate parameters
        based on seeing conditions and image characteristics.
        """
        self.logger.info(f"Start source extractor (sextractor)")

        satur_level = self.image_info.satur_level * self.phot_conf.satur_margin
        self.logger.debug(f"Saturation level with margin {1 - self.phot_conf.satur_margin}: {satur_level}")

        self.logger.debug("Setting apertures for photometry.")
        sex_args = phot_utils.get_sex_args(
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
            sex_args.extend(["-FILTER", "N"])

        # run sextractor with 'main' preset
        obs_src_table = self._run_sextractor(
            se_preset="main",
            sex_args=sex_args,
        )

        # add snr columns
        self.logger.debug("Adding columns to the sextracted table")
        suffixes = [key.replace("FLUXERR_", "") for key in obs_src_table.keys() if "FLUXERR_" in key]
        for suffix in suffixes:
            obs_src_table[f"SNR_{suffix}"] = obs_src_table[f"FLUX_{suffix}"] / obs_src_table[f"FLUXERR_{suffix}"]

        return obs_src_table

    def _run_sextractor(
        self,
        se_preset: str = "prep",
        sex_args: Optional[Dict] = None,
        output: str = None,
        fits_ldac: bool = False,
        **kwargs,
    ) -> Table:
        """
        Execute SExtractor on the image.

        Args:
            output: Path for output catalog
            prefix: Prefix for temporary files
            sex_args: Additional arguments for SExtractor
            **kwargs: Additional keyword arguments for SExtractor

        Returns:
            Sextracted Table
        """
        self.logger.info(f"Run source extractor (sextractor) ({se_preset})")

        if output is None:
            # new PathHandler for single
            output = getattr(PathHandler(self.input_image).photometry, f"{se_preset}_catalog")

        self.logger.debug(f"PhotometrySingle _run_sextractor input image: {self.input_image}")
        self.logger.debug(f"PhotometrySingle _run_sextractor output catalog: {output}")
        self.logger.debug(f"PhotometrySingle _run_sextractor sex_args: {sex_args}")
        self.logger.debug(f"PhotometrySingle _run_sextractor **kwargs: {kwargs}")

        _, outcome = external.sextractor(
            self.input_image,
            outcat=output,
            se_preset=se_preset,
            logger=self.logger,
            sex_args=sex_args,
            return_sex_output=True,
            fits_ldac=fits_ldac,
            **kwargs,
        )

        if se_preset == "main":
            outcome = [s for s in outcome.split("\n") if "RMS" in s][0]
            self.phot_header.SKYMED = float(outcome.split("Background:")[1].split("RMS:")[0])
            self.phot_header.SKYSIG = float(outcome.split("RMS:")[1].split("/")[0])

        if fits_ldac:
            return Table.read(output, hdu=2)
        return Table.read(output, format="ascii.sextractor")

    def get_zp_src_table(self, obs_src_table: Table) -> Table:
        """Returns the filtered table of image sources to be compared with the reference catalog for zp calculation"""
        self.logger.debug(f"Filtering source catalog for zp calculation")
        zp_src_table = self.filter_catalog(obs_src_table)
        self.logger.debug(f"After filtering: {len(zp_src_table)}/{len(obs_src_table)} sources")
        self.logger.info(f"Calculating zero points with {len(zp_src_table)} sources")
        return zp_src_table

    def calculate_zp(
        self, zp_src_table: Table, filt: str = None, phot_header: PhotometryHeader = None, save_plots: bool = True
    ) -> Tuple[Dict]:
        """
        Updates header.aperture_info in-place with zero point and aperture information.
        If header is not provided, uses self.phot_header.
        """

        phot_header = phot_header or self.phot_header

        if filt:
            ref_mag_key = f"mag_{filt}"  # column name in the reference catalog
        else:
            ref_mag_key = self.image_info.ref_mag_key

        aperture_dict = phot_utils.get_aperture_dict(phot_header.PEEING, self.image_info.pixscale)

        for aperture_key in aperture_dict.keys():
            mag_key, magerr_key = phot_utils.get_mag_key(aperture_key)

            # zp for the given aperture
            zps = zp_src_table[ref_mag_key] - zp_src_table[mag_key]
            zperrs = zp_src_table[magerr_key]  # gaia synphot err unavailable; ignored
            # zperrs = phot_utils.rss(zp_src_table[magerr_key], zp_src_table(self.image_info.ref_magerr_key))

            mask = sigma_clip(zps, sigma=2.0).mask

            zp, zperr = phot_utils.compute_median_nmad(np.array(zps[~mask].value), normalize=True)

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

    def determine_filter(self, phot_headers: Dict[str, PhotometryHeader], save_plot=True):
        zp_cut = 27.2  # 26.8
        alleged_filter = self.image_info.filter
        filters_checked = [k for k in phot_headers.keys()]
        dicts = {filt: (header.zp_dict, header.aperture_dict) for filt, header in phot_headers.items()}
        dicts_for_plotting = dicts.copy()
        test_dicts = dicts.copy()

        self.logger.debug(f"filters_checked: {filters_checked}")
        # self.logger.debug(f"dicts: {dicts}")  # too long

        # (1) rule out filters that were not present at the time
        active_filters = self.get_active_filters()
        self.logger.debug(f"Active filters: {active_filters}")

        for filt in list(set(filters_checked) - set(active_filters)):
            test_dicts.pop(filt)
            self.logger.debug(f"Filter {filt} is not active. Removing from viable filters.")

        # self.logger.debug(f"Filtered dicts: {test_dicts}")  # too long

        # (2) apply prior knowledge of zp for broad and medium band filters
        while True:
            narrowed_filters, zps, zperrs = phot_utils.dicts_to_lists(test_dicts)

            idx = zperrs.index(min(zperrs))
            inferred_filter = narrowed_filters[idx]
            zp = zps[idx]
            zperr = zperrs[idx]

            if (inferred_filter in BROAD_FILTERS and zp <= zp_cut) or (
                inferred_filter in MEDIUM_FILTERS and zp > zp_cut
            ):
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
                self.logger.warning(
                    f"Filter determination process eliminated all candidates. Falling back to header filter '{alleged_filter}' (zp = {zp:.2f}±{zperr:.2f})"
                )
                inferred_filter = alleged_filter
                break

        if save_plot:
            filters, zps, zperrs = phot_utils.dicts_to_lists(dicts_for_plotting)
            plot_filter_check(self, alleged_filter, inferred_filter, narrowed_filters, filters_checked, zps, zperrs)

        if alleged_filter != inferred_filter:
            self.logger.warning(f"The filter in header ({alleged_filter}) is not the best matching ({inferred_filter})")
            self.logger.warning(f"The best-matching filter is {inferred_filter} with zp = {zp}+/-{zperr}")
            orig_zp, orig_zperr = phot_utils.get_zp_from_dict(dicts, alleged_filter)
            self.logger.warning(f"The original filter is {alleged_filter} with zp = {orig_zp:.2f}±{orig_zperr:.2f}")
        else:
            self.logger.info(f"The inferred filter is matched to the original filter, '{alleged_filter}'")

        return inferred_filter

    def get_active_filters(self) -> set:
        """ad-hoc before DB integration"""
        from glob import glob
        from ..path import NameHandler

        try:
            f = PathHandler(self.input_image).conjugate
            if not os.path.exists(f):
                raise PipelineError(f"During get_active_filters, no conjugate image found for {f}")
            flist = glob(os.path.join(os.path.dirname(f), "*.fits"))
            flats = NameHandler(flist).pick_type("raw_flat")
            scis = NameHandler(flist).pick_type("raw_science")
            active_filters = set(NameHandler(flats + scis).filter)
            self.logger.debug(f"Fetched active filters {active_filters}")
            return active_filters
        except:
            self.logger.warning("Fetching active filters failed. Using all default filters")
            return ALL_FILTERS

    def update_image_header(
        self,
        phot_header: PhotometryHeader,
    ) -> None:
        """
        Update the input fits image's header with photometry information.
        """
        phot_header = phot_header or self.phot_header
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
        if hasattr(self, "gaia_columns"):
            gaia_cols = self.gaia_columns + [
                "separation"
            ]  # e.g. ["source_id", "bp_rp", f"mag_{self.image_info.filter}"]
            other_cols = [c for c in obs_src_table.colnames if c not in gaia_cols]
            obs_src_table = obs_src_table[other_cols + gaia_cols]

        # save
        output_catalog_file = self.path.photometry.final_catalog
        obs_src_table.write(output_catalog_file, format="fits", overwrite=True)  # "ascii.tab" "ascii.ecsv"
        self.logger.info(f"Photometry catalog is written in {os.path.basename(output_catalog_file)}")

        return


@dataclass(frozen=True)
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

    # keys needed in PhotometryHeader
    phot_header_keys: dict = None  # don't make this dict here; can be shared by multiple instances

    def __repr__(self) -> str:
        """Returns a string representation of the ImageInfo."""
        return ",\n".join(f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

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

    @classmethod
    def parse_image_header_info(cls, image_path: str) -> "ImageInfo":
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
            raise PipelineError("Check Astrometry solution: no WCS information for Photometry")

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
    JD: float = 0.0
    MJD: float = 0.0
    SEEING: float = 0.0
    PEEING: float = 0.0
    ELLIP: float = 0.0
    ELONG: float = 0.0
    SKYSIG: float = 0.0
    SKYMED: float = 0.0
    REFCAT: str = "GaiaXP"
    MAGLOW: float = 0.0
    MAGUP: float = 0.0
    STDNUMB: int = 0

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

                # skip these fields
                if key in ["AUTHOR", "PHOTIME"]:
                    continue

                if key in image_info.phot_header_keys:
                    # try:
                    setattr(self, f.name, image_info.phot_header_keys[key])
                    # except Exception:
                    #     pass  # silently ignore any conversion errors

                if "ZP_AUTO" in image_info.phot_header_keys:
                    self._set_aperture_info_from_header(image_info.phot_header_keys)

    def __repr__(self) -> str:
        """Returns a string representation of the ImageHeader."""
        return ",\n".join(f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    def _set_aperture_info_from_header(self, phot_header_keys: dict) -> None:
        """Sets aperture information from the image header."""

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
            "JD": (self.JD, "Julian Date of the observation"),
            "MJD": (self.MJD, "Modified Julian Date of the observation"),
            "SEEING": (round(self.SEEING, 3), "SEEING [arcsec]"),
            "PEEING": (round(self.PEEING, 3), "SEEING [pixel]"),
            "ELLIP": (round(self.ELLIP, 3), "ELLIPTICITY 1-B/A [0-1]"),
            "ELONG": (round(self.ELONG, 3), "ELONGATION A/B [1-]"),
            "SKYSIG": (round(self.SKYSIG, 3), "SKY SIGMA VALUE"),
            "SKYVAL": (round(self.SKYMED, 3), "SKY MEDIAN VALUE"),
            "REFCAT": (self.REFCAT, "REFERENCE CATALOG NAME"),
            "MAGLOW": (self.MAGLOW, "REF MAG RANGE, LOWER LIMIT"),
            "MAGUP": (self.MAGUP, "REF MAG RANGE, UPPER LIMIT"),
            "STDNUMB": (self.STDNUMB, "# OF STD STARS TO CALIBRATE ZP"),
        }

        phot_header_dict.update(misc_dict)
        # round float values to .3f
        phot_header_dict.update({k: (round(v[0], 3), v[1]) for k, v in self.aperture_dict.items()})
        phot_header_dict.update({k: (round(v[0], 3), v[1]) for k, v in self.zp_dict.items()})

        return phot_header_dict


# @dataclass
# class PhotometryCatalog:
#     """Stores the catalog-side results of photometry for a single image."""

#     image_info: ImageInfo  # link back to the image
#     obs_src_table: Table  # final catalog (with ref cols)
#     zp_src_table: Table  # subset used for ZP
#     column_data: Dict[str, np.ndarray]  # optional, if you want
