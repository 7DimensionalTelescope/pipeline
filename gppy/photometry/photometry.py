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
from dataclasses import dataclass

# astropy
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.table import Table, hstack, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

# gppy modules
from . import utils as phot_utils
from ..config.utils import get_key
from ..utils import update_padded_header, time_diff_in_seconds
from ..config import SciProcConfiguration
from ..config.base import ConfigurationInstance
from ..services.memory import MemoryMonitor
from ..services.queue import QueueManager, Priority
from .. import external
from ..const import PIXSCALE, MEDIUM_FILTERS, BROAD_FILTERS, ALL_FILTERS, PipelineError
from ..services.setup import BaseSetup
from ..tools.table import match_two_catalogs, build_condition_mask
from ..path.path import PathHandler


def log_once(method):
    """not working"""
    attr = f"_{method.__name__}_logged"

    def wrapper(self, *args, **kwargs):
        if not getattr(self, attr, False):
            self.logger.info(f"Starting {method.__name__}")
            setattr(self, attr, True)
        return method(self, *args, **kwargs)

    return wrapper


class Photometry(BaseSetup):
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
        ref_catalog: Optional[str] = None,
    ) -> None:
        """
        Initialize the Photometry class.

        Args:
            config: Configuration object or path to config yaml
            logger: Logger instance for output messaging
            queue: Queue manager for parallel processing or boolean to create one
            images: List of image files to process
            ref_catalog: Name of reference catalog to use
        """
        # Load Configuration
        super().__init__(config, logger, queue)
        # self._flag_name = "photometry"

        self.ref_catalog = ref_catalog or self.config.photometry.refcatname

        self.run_single_photometry = (
            True if not self.config.flag.single_photometry and not self.config.flag.combined_photometry else False
        )

        if self.run_single_photometry:
            self.config.photometry.input_images = images or self.config.input.calibrated_images
            self.input_images = self.config.photometry.input_images
            self.logger.debug("Running single photometry")
            self._flag_name = "single_photometry"
        else:
            self.config.photometry.input_images = images or [self.config.input.stacked_image]
            self.input_images = self.config.photometry.input_images
            self.logger.debug("Running combined photometry")
            self._flag_name = "combined_photometry"

    @classmethod
    def from_list(cls, images: List[str], working_dir=None) -> Optional["Photometry"]:
        """
        Create Photometry instance from a list of image paths.

        Args:
            images: List of paths to image files

        Returns:
            Photometry instance or None if files don't exist
        """
        image_list = []
        for image in images:
            path = Path(image)
            if not path.is_file():
                print("The file does not exist.")
                return None
            image_list.append(path.parts[-1])
        working_dir = working_dir or str(path.parent.absolute())
        config = SciProcConfiguration.base_config(working_dir)
        config.config.input.calibrated_images = image_list
        config.path = PathHandler(image_list)
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
        try:
            if self.queue:
                self._run_parallel()
            else:
                self._run_sequential()

            if self.run_single_photometry:
                self.config.flag.single_photometry = True
            else:
                self.config.flag.combined_photometry = True

            self.logger.info(f"'Photometry' is Completed in {time_diff_in_seconds(st)} seconds")
            self.logger.debug(MemoryMonitor.log_memory_usage)
        except Exception as e:
            self.logger.error(f"Photometry failed: {str(e)}")
            raise

    def _run_parallel(self) -> None:
        """Process images in parallel using queue system."""
        task_ids = []
        for i, image in enumerate(self.input_images):
            process_name = f"{self.config.name} - single photometry - {i+1} of {len(self.input_images)}"
            single_config = self.config.extract_single_image_config(i)

            phot_single = PhotometrySingle(
                image,
                single_config,  # self.config,
                self.logger,
                ref_catalog=self.ref_catalog,
            )
            task_id = self.queue.add_task(
                phot_single.run,
                kwargs={"name": process_name},
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
            PhotometrySingle(
                # image,
                single_config,  # self.config,
                logger=self.logger,
                ref_catalog=self.ref_catalog,
                total_image=len(self.input_images),
            ).run()


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
        trust_header_seeing=False,
        check_filter=True,
        calculate_zp=True,
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
        self.input_image = self.phot_conf.input_images[0]
        self.image_info = ImageInfo.parse_image_header_info(self.input_image)
        self.name = name or os.path.basename(self.input_image)
        self.phot_header = PhotometryHeader()
        self.phot_header.author = getpass.getuser()

        self._id = str(next(self._id_counter)) + "/" + str(total_image)

        self.path = PathHandler(self.input_image)
        self.path_tmp = self.path.photometry.tmp_dir

        self._trust_header_seeing = trust_header_seeing
        self._calculate_zp = calculate_zp
        self.check_filter = check_filter

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
        2. Calculates seeing conditions
        3. Runs source extraction
        4. Matches detected sources with reference catalog
        5. Calculates zero point corrections
        6. Updates image header
        7. Writes photometry catalog

        Times the complete process and performs memory cleanup after completion.
        """
        start_time = time.time()
        self.logger.info(f"Start 'PhotometrySingle' for the image {self.name} [{self._id}]")
        self.logger.debug(f"{'=' * 13} {os.path.basename(self.input_image)} {'=' * 13}")

        self.calculate_seeing()
        obs_src_table = self.photometry_with_sextractor()

        if self.check_filter:
            filters_to_check = get_key(self.config.photometry, "filters_to_check") or ALL_FILTERS
            obs_src_table = self.add_matched_reference_catalog(obs_src_table, filters=filters_to_check)
            temp_results = {}
            self.logger.debug(f"Starting filter check for {filters_to_check}")
            for i, filt in enumerate(filters_to_check):
                zp_dict, aper_dict, cols = self.calculate_zp(obs_src_table, filt=filt, save_plots=False, recursive=i)
                temp_results[filt] = (zp_dict, aper_dict, cols)

            dicts = {f: (zp, ap) for f, (zp, ap, _) in temp_results.items()}
            inferred_filter = self.determine_filter(dicts)
            self.logger.info(f"Saving photometry info to header for {inferred_filter}")

            _, _, cols = temp_results[inferred_filter]
            self.add_reference_columns(obs_src_table, cols)
            self.update_image_header(*dicts[inferred_filter])
            self.write_catalog(obs_src_table)

        else:
            self.logger.info("Skipping filter check")
            obs_src_table = self.add_matched_reference_catalog(obs_src_table)
            zp_dict, aper_dict, cols = self.calculate_zp(obs_src_table)
            self.add_reference_columns(obs_src_table, cols)
            self.update_image_header(zp_dict, aper_dict)
            self.write_catalog(obs_src_table)

        self.logger.debug(MemoryMonitor.log_memory_usage)
        self.logger.info(
            f"'PhotometrySingle' is completed for the image [{self._id}]"
            f" in {time_diff_in_seconds(start_time)} seconds"
        )

    def calculate_seeing(self, use_header_seeing=False) -> None:
        """
        Calculate seeing conditions from stellar sources in the image.

        Uses source extraction to identify stars and calculate median FWHM,
        ellipticity, and elongation values stored by self.phot_header.

        Args:
            low_mag_cut: Lower magnitude limit for star selection
        """

        if use_header_seeing:
            header = fits.getheader(self.input_image)
            self.phot_header.seeing = header["SEEING"]
            self.phot_header.peeing = header["PEEING"]
            self.phot_header.ellipticity = header["ELLIP"]
            self.phot_header.elongation = header["ELONG"]
            self.logger.debug(f"{len(post_match_table)} Star-like Sources Found")
            self.logger.debug(f"SEEING     : {self.phot_header.seeing:.3f} arcsec")
            self.logger.debug(f"ELONGATION : {self.phot_header.elongation:.3f}")
            self.logger.debug(f"ELLIPTICITY: {self.phot_header.ellipticity:.3f}")
            return

        prep_cat = self.path.photometry.prep_catalog
        if os.path.exists(prep_cat):
            self.logger.info("Calculating seeing from a pre-existing 'prep' catalog")
            obs_src_table = Table.read(prep_cat, format="ascii.sextractor")
        else:
            obs_src_table = self._run_sextractor(se_preset="prep")

        post_match_table = self.add_matched_reference_catalog(obs_src_table)
        post_match_table = self.filter_catalog(post_match_table, snr_cut=False, low_mag_cut=11.75)

        self.phot_header.seeing = np.median(post_match_table["FWHM_WORLD"] * 3600)
        self.phot_header.peeing = self.phot_header.seeing / self.image_info.pixscale
        self.phot_header.ellipticity = round(np.median(post_match_table["ELLIPTICITY"]), 3)
        self.phot_header.elongation = round(np.median(post_match_table["ELONGATION"]), 3)
        self.logger.debug(f"{len(post_match_table)} Star-like Sources Found")
        self.logger.debug(f"SEEING     : {self.phot_header.seeing:.3f} arcsec")
        self.logger.debug(f"ELONGATION : {self.phot_header.elongation:.3f}")
        self.logger.debug(f"ELLIPTICITY: {self.phot_header.ellipticity:.3f}")

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
        Match detected sources with reference catalog.

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
        table,
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
        low_mag_cut = low_mag_cut or self.phot_conf.ref_mag_lower
        high_mag_cut = high_mag_cut or self.phot_conf.ref_mag_upper

        # Copy the original table to avoid mutation
        post_match_table = table.copy()

        post_match_table["within_ellipse"] = phot_utils.is_within_ellipse(
            post_match_table["X_IMAGE"],
            post_match_table["Y_IMAGE"],
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
            peeing=self.phot_header.peeing,
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
            return_sex_output=True,
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
            output = getattr(PathHandler(self.input_image).photometry, f"{se_preset}_catalog")

        self.logger.debug(f"PhotometrySingle _run_sextractor output catalog: {output}")

        outcome = external.sextractor(
            self.input_image,
            outcat=output,
            se_preset=se_preset,
            logger=self.logger,
            sex_args=sex_args,
            **kwargs,
        )

        if se_preset == "main":
            _, outcome = outcome
            outcome = [s for s in outcome.split("\n") if "RMS" in s][0]
            self.phot_header.skymed = float(outcome.split("Background:")[1].split("RMS:")[0])
            self.phot_header.skysig = float(outcome.split("RMS:")[1].split("/")[0])

        return Table.read(output, format="ascii.sextractor")

    @log_once
    def calculate_zp(self, obs_src_table, save_plots=True, filt=None, recursive=False) -> Tuple[Dict, Dict]:
        """
        Calculate photometric zero point.

        Computes zero points and their errors for different apertures,
        creates diagnostic plots, and calculates limiting magnitudes.

        Args:
            zp_src_table: Table of matched sources for ZP calculation

        Returns:
            Tuple of dictionaries containing zero point and aperture information
        """

        if bool(recursive):
            self.logger.debug(f"Filtering source catalog for zp calculation")
        zp_src_table = self.filter_catalog(obs_src_table)
        
        if bool(recursive):
            self.logger.debug(f"After filtering: {len(zp_src_table)}/{len(obs_src_table)} sources")

        if bool(recursive):
            self.logger.info(f"Calculating zero points with {len(zp_src_table)} sources")
        apertures = phot_utils.get_aperture_dict(self.phot_header.peeing, self.image_info.pixscale)

        if filt:
            ref_mag_key = f"mag_{filt}"
            keyset_filt = filt
        else:
            ref_mag_key = self.image_info.ref_mag_key
            keyset_filt = self.image_info.filter

        zp_dict = {}
        aper_dict = {}
        column_data: Dict[str, np.ndarray] = {}  # filter: (zp, aper, col)
        for mag_key in apertures.keys():
            magerr_key = mag_key.replace("MAG", "MAGERR")

            zps = zp_src_table[ref_mag_key] - zp_src_table[mag_key]
            zperrs = zp_src_table[magerr_key]  # gaia synphot err unavailable; ignored
            # zperrs = phot_utils.rss(zp_src_table[magerr_key], zp_src_table(self.image_info.ref_magerr_key))

            mask = sigma_clip(zps, sigma=2.0).mask

            zp, zperr = phot_utils.compute_median_nmad(np.array(zps[~mask].value), normalize=True)
            keys = phot_utils.keyset(mag_key, keyset_filt)
            values = phot_utils.apply_zp(obs_src_table[mag_key], obs_src_table[magerr_key], zp, zperr)

            # stash the results for later
            for key, val in zip(keys, values):
                column_data[key] = val

            if mag_key == "MAG_AUTO":
                ul_3sig, ul_5sig = 0.0, 0.0
            else:
                ul_3sig, ul_5sig = phot_utils.limitmag(np.array([3, 5]), zp, apertures[mag_key][0], self.phot_header.skysig)  # fmt:skip

            suffix = mag_key.replace("MAG_", "")
            aper_dict[suffix] = round(apertures[mag_key][0], 3), apertures[mag_key][1]

            suffix = suffix.replace("APER", "0").replace("0_", "")
            zp_dict.update(
                {
                    f"ZP_{suffix}": (round(zp, 3), f"ZERO POINT for {mag_key}"),
                    f"EZP_{suffix}": (round(zperr, 3), f"ZERO POINT ERROR for {mag_key}"),
                    f"UL3_{suffix}": (round(ul_3sig, 3), f"3 SIGMA LIMITING MAG FOR {mag_key}"),
                    f"UL5_{suffix}": (round(ul_5sig, 3), f"5 SIGMA LIMITING MAG FOR {mag_key}"),
                }
            )  # fmt: skip

            if save_plots:
                self.plot_zp(mag_key, zp_src_table, zps, zperrs, zp, zperr, mask)

        return zp_dict, aper_dict, column_data

    def add_reference_columns(self, obs_src_table, cols):
        """Mind the implicit mutation"""
        for key, arr in cols.items():
            obs_src_table[key] = arr
            obs_src_table[key].format = ".3f"
        return

    def determine_filter(self, dicts: dict, save_plot=True):
        zp_cut = 26.8
        alleged_filter = self.image_info.filter
        filters_checked = [k for k in dicts.keys()]
        dicts_for_plotting = dicts.copy()

        # (1) rule out filters that were not present at the time
        active_filters = self.get_active_filters()
        self.logger.info(f"Active filters: {active_filters}")
        for filt in list(set(filters_checked) - set(active_filters)):
            dicts.pop(filt)

        # (2) apply prior knowledge of zp for broad and medium band filters
        while True:
            narrowed_filters, zps, zperrs = phot_utils.dicts_to_lists(dicts)
            idx = zperrs.index(min(zperrs))
            inferred_filter = narrowed_filters[idx]
            zp = zps[idx]

            if (inferred_filter in BROAD_FILTERS and zp <= zp_cut) or (
                inferred_filter in MEDIUM_FILTERS and zp > zp_cut
            ):
                dicts.pop(inferred_filter)
            else:
                break

        if save_plot:
            filters, zps, zperrs = phot_utils.dicts_to_lists(dicts_for_plotting)
            self.plot_filter_check(alleged_filter, inferred_filter, narrowed_filters, filters_checked, zps, zperrs)

        if alleged_filter != inferred_filter:
            self.logger.warning(f"The filter in header ({alleged_filter}) is not the best matching ({inferred_filter})")

        return inferred_filter

    def get_active_filters(self) -> set:
        """ad-hoc before DB integration"""
        from glob import glob
        from ..path import NameHandler

        try:
            f = PathHandler(self.input_image).conjugate
            flist = glob(os.path.join(os.path.dirname(f), "*.fits"))
            flats = NameHandler(flist).pick_type("raw_flat")
            scis = NameHandler(flist).pick_type("raw_science")
            return set(NameHandler(flats + scis).filter)
        except:
            self.logger.info("Fetching active filters failed. Using all default filters")
            return ALL_FILTERS

    def plot_filter_check(self, alleged_filter, inferred_filter, narrowed_filters, filters_checked, zps, zperrs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # 1) Plot EZP on ax1
        c_ezp = "tab:blue"
        ax1.plot(
            filters_checked,
            zperrs,
            marker="x",
            linestyle="-",
            label="EZP",
            color=c_ezp,
        )
        ax1.set_ylabel("Zero Point Error (EZP)", color=c_ezp)
        ax1.tick_params(axis="y", labelcolor=c_ezp)

        # 2) Shade backgrounds for ruled-out filters
        for i, flt in enumerate(filters_checked):
            if flt not in narrowed_filters:
                ax1.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.3)
        gray_patch = mpatches.Patch(color="gray", alpha=0.3)  # patch for legend

        inf_idx = filters_checked.index(inferred_filter)
        ax1.axvspan(inf_idx - 0.5, inf_idx + 0.5, color="dodgerblue", alpha=0.3)
        yellow_patch = mpatches.Patch(color="dodgerblue", alpha=0.3, label=f"Inferred Filter {inferred_filter}")

        # x-ticks and tilted labels
        ax1.set_xticks(range(len(filters_checked)))
        ax1.set_xticklabels(filters_checked, rotation=45, ha="right")

        # 3) Create twin axis for ZP
        c_zp = "orange"
        ax2 = ax1.twinx()
        ax2.plot(
            filters_checked,
            zps,
            marker="o",
            ls="--",
            label="ZP",
            color=c_zp,
        )
        ax2.set_ylabel("Zero Point (ZP)", color=c_zp)
        ax2.tick_params(axis="y", labelcolor=c_zp)
        ax2.invert_yaxis()
        ax1.margins(x=0)

        # Combine legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2 + [gray_patch, yellow_patch], l1 + l2 + ["Ruled Out", "Inferred Filter"], loc="upper left")

        plt.title(f"Filter Sanity Check for {alleged_filter}")
        plt.tight_layout()

        img_stem = os.path.splitext(os.path.basename(self.input_image))[0]
        f = os.path.join(self.path.figure_dir, f"{img_stem}_filtercheck.png")
        plt.savefig(f, dpi=100)
        plt.close()
        return

    def plot_zp(
        self,
        mag_key: str,
        src_table: Table,
        zp_arr: np.ndarray,
        zperr_arr: np.ndarray,
        zp: float,
        zperr: float,
        mask: np.ndarray,
    ) -> None:
        """Generates and saves a zero-point calibration plot.
        The plot shows the zero-point values for each source and the final calibrated zero-point.
        Sources inside and outside the magnitude limits are plotted with different markers.
        Parameters
        ----------
        mag_key : str
            Key for the magnitude column in the source table
        src_table : astropy.table.Table
            Table containing source measurements and reference magnitudes
        zp_arr : array-like
            Array of individual zero-point values for each source
        zperr_arr : array-like
            Array of zero-point uncertainties for each source
        zp : float
            Final calibrated zero-point value
        zperr : float
            Uncertainty in the final zero-point
        mask : numpy.ndarray
            Boolean mask indicating which sources are within magnitude limits
        Returns
        -------
        None
            Saves plot as PNG file in the processed/images directory
        """
        ref_mag = src_table[self.image_info.ref_mag_key]
        obs_mag = src_table[mag_key]

        plt.errorbar(ref_mag, zp_arr, xerr=0, yerr=zperr_arr, ls="none", c="grey", alpha=0.5)

        plt.plot(
            ref_mag[~mask],
            ref_mag[~mask] - obs_mag[~mask],
            ".",
            c="dodgerblue",
            alpha=0.75,
            zorder=999,
            label=f"{len(ref_mag[~mask])}",
        )

        plt.plot(
            ref_mag[mask], ref_mag[mask] - obs_mag[mask], "x", c="tomato", alpha=0.75, label=f"{len(ref_mag[mask])}"
        )

        plt.axhline(y=zp, ls="-", lw=1, c="grey", zorder=1, label=f"ZP: {zp:.3f}+/-{zperr:.3f}")
        plt.axhspan(ymin=zp - zperr, ymax=zp + zperr, color="silver", alpha=0.5, zorder=0)
        plt.axvspan(xmin=0, xmax=self.phot_conf.ref_mag_lower, color="silver", alpha=0.25, zorder=0)
        plt.axvspan(xmin=self.phot_conf.ref_mag_upper, xmax=25, color="silver", alpha=0.25, zorder=0)

        plt.xlim([10, 20])
        plt.ylim([zp - 0.25, zp + 0.25])

        plt.xlabel(self.image_info.ref_mag_key)
        plt.ylabel(f"ZP_{mag_key}")

        plt.legend(loc="upper center", ncol=3)
        plt.tight_layout()

        # im_path = os.path.join(self.config.path.path_processed, "images")
        im_path = self.path.figure_dir

        if not os.path.exists(im_path):
            os.makedirs(im_path)

        img_stem = os.path.splitext(os.path.basename(self.input_image))[0]
        plt.savefig(f"{im_path}/{img_stem}_{mag_key}.png", dpi=100)
        plt.close()

    def update_image_header(
        self,
        zp_dict: Dict[str, Tuple[float, str]],
        aper_dict: Dict[str, Tuple[float, str]],
    ) -> None:
        """
        Update the FITS image header with photometry information.

        Combines header information from multiple sources and updates the FITS file header.
        This includes photometry statistics, aperture information, and zero point data.

        Args:
            zp_dict: Dictionary containing zero point values and descriptions
                    Format: {'key': (value, description)}
            aper_dict: Dictionary containing aperture values and descriptions
                    Format: {'key': (value, description)}
        """
        header_to_add = {}
        header_to_add.update(self.phot_header.dict)
        header_to_add.update(aper_dict)
        header_to_add.update(zp_dict)
        update_padded_header(self.input_image, header_to_add)
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
        gaia_cols = self.gaia_columns + ["separation"]  # e.g. ["source_id", "bp_rp", f"mag_{self.image_info.filter}"]
        other_cols = [c for c in obs_src_table.colnames if c not in gaia_cols]
        obs_src_table = obs_src_table[other_cols + gaia_cols]

        # save
        output_catalog_file = self.path.photometry.final_catalog
        obs_src_table.write(output_catalog_file, format="fits", overwrite=True)  # "ascii.tab" "ascii.ecsv"
        self.logger.info(f"Photometry catalog is written in {os.path.basename(output_catalog_file)}")

        return


@dataclass
class PhotometryHeader:
    """Stores image header information related to photometry to be written to the image."""

    author: str = "pipeline"
    photime: str = datetime.date.today().isoformat()
    jd: float = 0.0
    mjd: float = 0.0
    seeing: float = 0.0
    peeing: float = 0.0
    ellipticity: float = 0.0
    elongation: float = 0.0
    skysig: float = 0.0
    skymed: float = 0.0
    refcat: str = "GaiaXP"
    maglow: float = 0.0
    magup: float = 0.0
    stdnumb: int = 0

    def __repr__(self) -> str:
        """Returns a string representation of the ImageHeader."""
        return ",\n".join(f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    @property
    def dict(self) -> Dict[str, Tuple[Any, str]]:
        """Generates a dictionary of header information for FITS."""
        return {
            "AUTHOR": (self.author, "LAST UPDATED PHOTOMETRY HEADER"),
            "PHOTIME": (self.photime, "PHOTOMETRY TIME [KST]"),
            "JD": (self.jd, "Julian Date of the observation"),
            "MJD": (self.mjd, "Modified Julian Date of the observation"),
            "SEEING": (round(self.seeing, 3), "SEEING [arcsec]"),
            "PEEING": (round(self.peeing, 3), "SEEING [pixel]"),
            "ELLIP": (round(self.ellipticity, 3), "ELLIPTICITY 1-B/A [0-1]"),
            "ELONG": (round(self.elongation, 3), "ELONGATION A/B [1-]"),
            "SKYSIG": (round(self.skysig, 3), "SKY SIGMA VALUE"),
            "SKYVAL": (round(self.skymed, 3), "SKY MEDIAN VALUE"),
            "REFCAT": (self.refcat, "REFERENCE CATALOG NAME"),
            "MAGLOW": (self.maglow, "REF MAG RANGE, LOWER LIMIT"),
            "MAGUP": (self.magup, "REF MAG RANGE, UPPER LIMIT"),
            "STDNUMB": (self.stdnumb, "# OF STD STARS TO CALIBRATE ZP"),
        }


@dataclass
class ImageInfo:
    """Stores information extracted from a FITS image header."""

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

        return cls(
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
            satur_level=hdr["SATURATE"],
            bpx_interp=interped,
        )
