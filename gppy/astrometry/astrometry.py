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
from functools import cached_property

from .. import external
from ..const import PIXSCALE, PipelineError
from ..utils import swap_ext, add_suffix, force_symlink, time_diff_in_seconds
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
    evaluate_single_wcs,
    evaluate_joint_wcs,
)


class Astrometry(BaseSetup):
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
        joint_scamp: bool = True,
        force_solve_field: bool = False,
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
            except Exception as e:
                self.logger.warning(e)
                self.run_solve_field(self.soft_links_to_input_images, self.solved_images)

            # update the sextractor catalogs with the wcs solutions for next scamp
            self.update_catalog(self.prep_cats, self.solved_heads)

            # self.evaluate_solution()
            # print(self.images_info[0]._single_wcs_eval_cards)

            # run main scamp
            self.run_scamp(self.prep_cats, scamp_preset="main", joint=joint_scamp)

            self.evaluate_solution()

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
        self.path_astrometry = self.path.astrometry.tmp_dir

        # override if astrometry.input_images is set
        local_inim = get_key(self.config, "astrometry.input_images")
        if local_inim is not None:
            inims = local_inim
        # otherwise use the common input
        else:
            inims = self.config.input.calibrated_images
            self.config.astrometry.input_images = inims

        self.input_images = inims

        soft_links = [os.path.join(self.path_astrometry, os.path.basename(s)) for s in inims]
        for inim, soft_link in zip(inims, soft_links):
            force_symlink(inim, soft_link)
            self.logger.debug(f"Soft link created: {inim} -> {soft_link}")
        self.soft_links_to_input_images = soft_links

        self.prep_cats = [add_suffix(inim, "cat") for inim in soft_links]  # fits_ldac sextractor output
        self.solved_images = [add_suffix(s, "solved") for s in soft_links]  # solve-field output
        self.solved_heads = [swap_ext(s, "head") for s in self.prep_cats]  # scamp output
        self.images_info = [ImageInfo.parse_image_header_info(image) for image in self.input_images]

        self.config.astrometry.local_astref = self.path.astrometry.astrefcat  # None if no local astrefcat

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

    def run_solve_field(self, input_images: List[str], output_images: List[str]) -> None:
        """Run astrometric plate-solving on input images.

        Uses astrometry.net's solve-field to determine WCS solution for each image.
        Supports parallel processing through queue system if enabled.

        Args:
            inputs: Paths to input FITS files
            outputs: Paths where solved FITS files will be written
        """
        # parallelize if queue=True
        self.logger.info(f"Start solve-field")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        solved_flag_files = [swap_ext(input, ".solved") for input in input_images]
        filtered_data = [
            (inp, out)
            for inp, out, solved_flag_file in zip(input_images, output_images, solved_flag_files)
            if not os.path.exists(solved_flag_file)
        ]

        if not filtered_data:
            self.logger.info("All input images have already been solved. Exiting solve-field.")
            return

        input_images, output_images = zip(*filtered_data)

        if self.queue:
            self._submit_task(
                external.solve_field,
                zip(input_images, output_images),
                # dump_dir=self.path_astrometry,
                pixscale=self.path.pixscale,  # PathHandler brings pixscale from NameHandler
            )
        else:
            for i, (slink, sfile, pixscale) in enumerate(zip(input_images, output_images, self.path.pixscale)):
                external.solve_field(
                    slink,
                    outim=sfile,
                    # dump_dir=self.path_astrometry,
                    pixscale=pixscale,
                )
                self.logger.info(f"Completed solve-field [{i+1}/{len(input_images)}]")
                self.logger.debug(f"Solve-field input: {slink}, output: {sfile}")

        self.logger.debug(MemoryMonitor.log_memory_usage)

    def run_sextractor(
        self, input_images: List[str], output_catalogs: List[str] = None, se_preset: str = "prep"
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
            )
        else:
            for i, (solved_image, prep_cat) in enumerate(zip(input_images, output_catalogs)):
                external.sextractor(
                    solved_image,
                    outcat=prep_cat,
                    se_preset=se_preset,
                    logger=self.logger,
                    fits_ldac=True,
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
    ) -> None:
        """Run SCAMP for astrometric calibration.

        Performs astrometric calibration using SCAMP, either jointly on all images
        or individually. Supports parallel processing for individual mode.

        Args:
            files: Paths to astrometrically solved FITS files
            joint: Whether to process all images together
        """
        st = time.time()
        self.logger.info(f"Start {'joint' if joint else 'individual'} scamp")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        # presex_cats = [os.path.splitext(s)[0] + f".{prefix}.cat" for s in files]
        input_catalogs = input_catalogs or self.prep_cats
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
            cat_to_scamp = os.path.join(self.path_astrometry, "scamp_input.cat")
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

        # individual, sequential
        else:
            for precat in input_catalogs:
                solved_heads = external.scamp(
                    precat,
                    path_ref_scamp=path_ref_scamp,
                    local_astref=astrefcat,
                    scampconfig=scampconfig,
                    scamp_args=scamp_args,
                    scamp_preset=scamp_preset,
                )
                self.logger.info(f"Completed scamp for {precat}]")  # fmt:skip
                self.logger.debug(f"{precat}")  # fmt:skip

        if scamp_preset == "prep":
            for head in solved_heads:
                backup_head = add_suffix(head, "prep")
                shutil.copy(head, backup_head)
                self.logger.debug(f"Backed up prep head {head} to {backup_head}")

        # update WCS in ImageInfo
        for image_info, solved_head in zip(self.images_info, solved_heads):
            image_info.head = solved_head

        self.logger.info(f"Completed scamp in {time_diff_in_seconds(st)} seconds")
        self.logger.debug(MemoryMonitor.log_memory_usage)

    def update_catalog(self, input_catalogs: List[str], solved_heads: List[str]) -> None:
        """Update catalog with solved heads."""
        input_catalogs = input_catalogs or self.prep_cats
        solved_heads = solved_heads or self.solved_heads
        assert len(input_catalogs) == len(solved_heads)

        for input_catalog, solved_head in zip(input_catalogs, solved_heads):

            # update wcs in LDAC_IMHEAD
            image_header = fitsrec_to_header(
                fits.getdata(input_catalog, ext=1)
            )  # assuming LDAC_IMHED is the second extension
            wcs_header = read_scamp_header(solved_head)
            image_header.update(wcs_header)

            # update coordinates in LDAC_OBJECT
            # wcs = WCS(image_header)  # not the same as the scamp header
            wcs = read_scamp_header(solved_head, return_wcs=True)
            tbl = Table.read(input_catalog, hdu=2)  # assuming LDAC_OBJECT is the third extension
            ra, dec = wcs.all_pix2world(tbl["X_IMAGE"], tbl["Y_IMAGE"], 1)
            tbl["ALPHA_J2000"] = ra
            tbl["DELTA_J2000"] = dec

            # update (overwrite) the catalog
            write_ldac(image_header, tbl, input_catalog)  # internally c script with cfitsio
            self.logger.debug(f"Updated catalog {input_catalog} with {solved_head}")

    def evaluate_solution(self, isep=False) -> None:
        """Evaluate the solution. This was developed in lack of latest scamp version."""
        self.logger.info("Evaluating the solution")

        self.update_catalog(self.prep_cats, self.solved_heads)  # this wcs is used for evaluation

        # update FOV polygon to ImageInfo. It will be cached and put into final image header later
        for image_info in self.images_info:
            image_info.fov_ra, image_info.fov_dec = get_fov_quad(image_info.wcs, image_info.naxis1, image_info.naxis2)
            self.logger.debug(f"Updated FOV polygon. RA: {image_info.fov_ra}, Dec: {image_info.fov_dec}")

        refcat = Table.read(self.config.astrometry.local_astref, hdu=2)

        # evaluate for each image
        for input_image, image_info, prep_cat in zip(self.input_images, self.images_info, self.prep_cats):
            bname = os.path.basename(input_image)
            plot_path = os.path.join(self.path.figure_dir, swap_ext(add_suffix(bname, "wcs"), "jpg"))
            return_bundle = evaluate_single_wcs(
                image=input_image,
                ref_cat=refcat,
                source_cat=prep_cat,
                date_obs=image_info.dateobs,
                wcs=image_info.wcs,
                fov_ra=image_info.fov_ra,
                fov_dec=image_info.fov_dec,
                plot_save_path=plot_path,
            )
            (ref_max_mag, sci_max_mag, num_ref, unmatched_fraction, separation_stats) = return_bundle

            image_info.ref_max_mag = ref_max_mag
            image_info.sci_max_mag = sci_max_mag
            image_info.num_ref_sources = num_ref
            image_info.unmatched_fraction = unmatched_fraction
            image_info.set_sep_stats(separation_stats)
            self.logger.debug(f"Evaluated solution for {input_image}")

        # evaluate the internal consistency of all images
        if len(self.input_images) > 1 and isep:
            internal_sep_stats_list = evaluate_joint_wcs(self.images_info)
            for image_info, internal_sep_stats in zip(self.images_info, internal_sep_stats_list):
                image_info.set_internal_sep_stats(internal_sep_stats)

        self.logger.debug(f"Evaluated solution for all images")
        self.logger.debug(MemoryMonitor.log_memory_usage)

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
                solved_header.update(image_info.wcs_eval_cards)
                self.logger.debug(f"WCS evaluation is added to {target_fits}")
            else:
                self.logger.debug(f"WCS evaluation is missing for {target_fits}")

            if add_polygon_info:
                polygon_header = polygon_info_header(image_info)
                solved_header.update(polygon_header)

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

    dateobs: str  # Observation date/time
    naxis1: int  # Image width
    naxis2: int  # Image height
    racent: float  # RA of image center
    decent: float  # DEC of image center
    xcent: float  # X coordinate of image center
    ycent: float  # Y coordinate of image center
    n_binning: int  # Binning factor
    pixscale: float  # Pixel scale [arcsec/pix]

    # solution
    head: Optional[fits.Header] = field(default=None)

    # FOV polygon
    fov_ra: list = field(default=None)
    fov_dec: list = field(default=None)

    # WCS evaluation parameters
    ref_max_mag: Optional[float] = field(default=None)
    sci_max_mag: Optional[float] = field(default=None)
    num_ref_sources: Optional[int] = field(default=None)
    unmatched_fraction: Optional[float] = field(default=None)

    # separation statistics from reference catalog
    sep_min: Optional[float] = field(default=None)
    sep_max: Optional[float] = field(default=None)
    sep_rms: Optional[float] = field(default=None)
    sep_mean: Optional[float] = field(default=None)
    sep_median: Optional[float] = field(default=None)
    sep_std: Optional[float] = field(default=None)

    # Internal separation statistics (between multiple images)
    internal_sep_min: Optional[float] = field(default=None)
    internal_sep_max: Optional[float] = field(default=None)
    internal_sep_rms: Optional[float] = field(default=None)
    internal_sep_mean: Optional[float] = field(default=None)
    internal_sep_median: Optional[float] = field(default=None)
    internal_sep_std: Optional[float] = field(default=None)

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

    def set_sep_stats(self, sep_stats: dict) -> None:
        self.sep_min = sep_stats["min"]
        self.sep_max = sep_stats["max"]
        self.sep_rms = sep_stats["rms"]
        self.sep_median = sep_stats["median"]
        self.sep_std = sep_stats["std"]

    def set_internal_sep_stats(self, sep_stats: dict) -> None:
        self.internal_sep_min = sep_stats["min"]
        self.internal_sep_max = sep_stats["max"]
        self.internal_sep_rms = sep_stats["rms"]
        self.internal_sep_median = sep_stats["median"]
        self.internal_sep_std = sep_stats["std"]

    @property
    def wcs_evaluated(self) -> bool:
        """only checks the existence of reference separation statistics"""
        sep_keys = [k for k in self.__dict__.keys() if k.startswith("sep_")]
        return all(getattr(self, k) is not None for k in sep_keys)

    @property
    def wcs_eval_cards(self) -> List[Tuple[str, float]]:
        return self._single_wcs_eval_cards + self._joint_wcs_eval_cards

    @property
    def _single_wcs_eval_cards(self) -> List[Tuple[str, float]]:
        cards = [
            (f"REFMXMAG", self.ref_max_mag, "Highest g mag of selected reference sources"),
            (f"SCIMXMAG", self.sci_max_mag, "Highest inst mag of selected science sources"),
            (f"NUM_REF", self.num_ref_sources, "Number of reference sources selected"),
            (f"UNMATCH", self.unmatched_fraction, "Fraction of unmatched reference sources"),
            (f"RSEP_MIN", self.sep_min, "Min separation from reference catalog [arcsec]"),
            (f"RSEP_MAX", self.sep_max, "Max separation from reference catalog [arcsec]"),
            (f"RSEP_RMS", self.sep_rms, "RMS separation from reference catalog [arcsec]"),
            (f"RSEP_MED", self.sep_median, "Median separation from ref catalog [arcsec]"),
            (f"RSEP_STD", self.sep_std, "STD of separation from ref catalog [arcsec]"),
        ]
        for k, v, c in cards:
            if not isinstance(v, (float, int, str, np.float32, np.int32)):
                raise PipelineError(f"WCS statistics contains type {type(v)} {v} for {k}")
        return cards

    @property
    def _joint_wcs_eval_cards(self) -> List[Tuple[str, float]]:
        # Include internal separation statistics if they exist on the instance.
        cards: List[Tuple[str, float]] = []
        internal_vals = {
            "ISEP_MIN": getattr(self, "internal_sep_min", None),
            "ISEP_MAX": getattr(self, "internal_sep_max", None),
            "ISEP_RMS": getattr(self, "internal_sep_rms", None),
            "ISEP_MED": getattr(self, "internal_sep_median", None),
            "ISEP_STD": getattr(self, "internal_sep_std", None),
        }
        descriptions = {
            "ISEP_MIN": "Min internal separation between matched sources [arcsec]",
            "ISEP_MAX": "Max internal separation between matched sources [arcsec]",
            "ISEP_RMS": "RMS internal separation between matched sources [arcsec]",
            "ISEP_MED": "Median internal separation between matched sources [arcsec]",
            "ISEP_STD": "STD of internal separation between matched sources [arcsec]",
        }
        for key, val in internal_vals.items():
            if val is not None:
                cards.append((key, val, descriptions[key]))
        return cards

    @cached_property
    def wcs(self) -> WCS:
        """Updated every time run_scamp is completed"""
        if not self.head:
            raise PipelineError("No self.head in ImageInfo")
        wcs_header = read_scamp_header(self.head)
        return WCS(wcs_header)
