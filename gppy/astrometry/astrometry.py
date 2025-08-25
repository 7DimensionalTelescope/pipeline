import os
import re
import time
from typing import Any, List, Tuple, Union, Any
from dataclasses import dataclass
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import Angle

from .. import external
from ..const import PIXSCALE, PipelineError
from ..utils import swap_ext, add_suffix, force_symlink, time_diff_in_seconds
from ..header import update_padded_header
from ..services.memory import MemoryMonitor
from ..config import SciProcConfiguration
from ..config.utils import get_key
from ..services.setup import BaseSetup
from .utils import read_scamp_header, build_wcs


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

        self.define_paths()
        self.images_info = [ImageInfo.parse_image_header_info(image) for image in self.input_images]

    @classmethod
    def from_list(cls, images, working_dir=None):
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
        solve_field: bool = True,
        joint_scamp: bool = True,
        processes=["sextractor", "scamp", "header_update"],
        se_preset: str = "prep",
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

            # solve-field
            if solve_field:
                self.run_solve_field(self.soft_links_to_input_images, self.solved_images)
            else:
                # add manual WCS update feature
                pass

            # Source Extractor
            if "sextractor" in processes:
                self.run_sextractor(self.solved_images, se_preset=se_preset)

            if "scamp" in processes:
                self.run_scamp(self.solved_images, joint=joint_scamp)

            # add polygon info to header - field rotation

            if "header_update" in processes:
                self.update_header(
                    # self.solved_images,
                    # self.input_images,
                    # self.soft_links_to_input_images,
                )

            self.config.flag.astrometry = True

            self.logger.info(f"'Astrometry' is completed in {time_diff_in_seconds(start_time)} seconds")
            self.logger.debug(MemoryMonitor.log_memory_usage)
        except Exception as e:
            self.logger.error(f"Error during astrometry processing: {str(e)}")
            raise

    def define_paths(self) -> Tuple[List[str], List[str], List[str]]:
        self.path_astrometry = self.path.astrometry.tmp_dir

        # override if astrometry.input_images is set
        local_inim = get_key(self.config, "astrometry.input_images")
        if local_inim is not None:
            inims = local_inim
        # otherwise use the common input
        else:
            inims = self.config.input.calibrated_images
            self.config.astrometry.input_images = inims

        soft_links = [os.path.join(self.path_astrometry, os.path.basename(s)) for s in inims]

        for inim, soft_link in zip(inims, soft_links):
            force_symlink(inim, soft_link)

        solved_files = [add_suffix(s, "solved") for s in soft_links]
        # return solved_files, soft_links, inims
        self.soft_links_to_input_images = soft_links
        self.solved_images = solved_files
        self.input_images = inims
        # self.prep_cats = PathHandler(soft_links).astrometry.catalog
        self.prep_cats = [add_suffix(inim, "cat") for inim in soft_links]  # fits_ldac
        self.solved_heads = [swap_ext(s, "head") for s in self.prep_cats]

    def run_solve_field(self, inputs: List[str], outputs: List[str]) -> None:
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

        solved_flag_files = [swap_ext(input, ".solved") for input in inputs]
        filtered_data = [
            (inp, out)
            for inp, out, solved_flag_file in zip(inputs, outputs, solved_flag_files)
            if not os.path.exists(solved_flag_file)
        ]

        if not filtered_data:
            self.logger.info("All input images have already been solved. Exiting solve-field.")
            return

        inputs, outputs = zip(*filtered_data)

        if self.queue:
            self._submit_task(
                external.solve_field,
                zip(inputs, outputs),
                # dump_dir=self.path_astrometry,
                pixscale=self.path.pixscale,  # PathHandler brings pixscale from NameHandler
            )
        else:
            for i, (slink, sfile, pixscale) in enumerate(zip(inputs, outputs, self.path.pixscale)):
                external.solve_field(
                    slink,
                    outim=sfile,
                    # dump_dir=self.path_astrometry,
                    pixscale=pixscale,
                )
                self.logger.info(f"Completed solve-field [{i+1}/{len(inputs)}]")
                self.logger.debug(f"Solve-field input: {slink}, output: {sfile}")

        self.logger.debug(MemoryMonitor.log_memory_usage)

    def run_sextractor(self, files: List[str], se_preset: str = "prep") -> List[str]:
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

        if self.queue:
            self._submit_task(
                external.sextractor,
                files,
                outcat=self.prep_cats,
                prefix=se_preset,
                logger=self.logger,
                fits_ldac=True,
            )
        else:
            for i, (solved_image, prep_cat) in enumerate(zip(files, self.prep_cats)):
                external.sextractor(
                    solved_image,
                    outcat=prep_cat,
                    se_preset=se_preset,
                    logger=self.logger,
                    fits_ldac=True,
                )
                self.logger.info(f"Completed sextractor (prep) [{i+1}/{len(files)}]")
                self.logger.debug(f"{solved_image}")

        self.logger.debug(MemoryMonitor.log_memory_usage)

    def run_scamp(
        self,
        files: List[str],
        joint: bool = True,
        astrefcat: str = None,
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
        presex_cats = self.prep_cats
        self.logger.debug(f"Scamp input catalogs: {presex_cats}")

        # path for scamp refcat download
        path_ref_scamp = self.path.astrometry.ref_query_dir

        # use local astrefcat if tile obs
        self.config.astrometry.local_astref = self.path.astrometry.astrefcat
        astrefcat = astrefcat or self.config.astrometry.local_astref
        self.logger.debug(f"Using astrefcat: {astrefcat}")

        # joint scamp
        if joint:
            # write target files into a text file
            cat_to_scamp = os.path.join(self.path_astrometry, "scamp_input.cat")
            with open(cat_to_scamp, "w") as f:
                for precat in presex_cats:
                    f.write(f"{precat}\n")

            external.scamp(cat_to_scamp, path_ref_scamp=path_ref_scamp, local_astref=astrefcat)

        # individual, parallel
        elif self.queue:
            self._submit_task(
                external.scamp,
                presex_cats,
                path_ref_scamp=path_ref_scamp,
                local_astref=astrefcat,
            )

        # individual, sequential
        else:
            for precat in presex_cats:
                external.scamp(precat, path_ref_scamp=path_ref_scamp, local_astref=astrefcat)
                self.logger.info(f"Completed scamp for {precat}]")  # fmt:skip
                self.logger.debug(f"{precat}")  # fmt:skip

        self.logger.info(f"Completed scamp in {time_diff_in_seconds(st)} seconds")
        self.logger.debug(MemoryMonitor.log_memory_usage)

    def update_header(
        self,
        inims: List[str] = None,
        heads: List[str] = None,
        use_missfits: bool = False,
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

        for solved_head, target_fits in zip(heads, inims):
            if os.path.exists(solved_head):
                solved_head = read_scamp_header(solved_head)
                update_padded_header(target_fits, solved_head)
            else:
                self.logger.error(f"Check SCAMP output. Check access to the online VizieR catalog, disk space or the field characteristics.")  # fmt: skip
                raise FileNotFoundError(f"SCAMP output (.head) does not exist: {solved_head}")  # fmt: skip
        # self.logger.info("Correcting WCS in image headers is completed.")

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
    # naxis1: int  # Image width
    # naxis2: int  # Image height
    racent: float  # RA of image center
    decent: float  # DEC of image center
    xcent: float  # X coordinate of image center
    ycent: float  # Y coordinate of image center
    n_binning: int  # Binning factor
    pixscale: float  # Pixel scale [arcsec/pix]

    def __repr__(self) -> str:
        """Returns a string representation of the ImageInfo."""
        return ",\n".join(f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    @classmethod
    def parse_image_header_info(cls, image_path: str) -> "ImageInfo":
        """Parses image information from a FITS header."""

        hdr = fits.getheader(image_path)

        # Center coord for reference catalog query
        xcent = (hdr["NAXIS1"] + 1) / 2.0
        ycent = (hdr["NAXIS2"] + 1) / 2.0
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
