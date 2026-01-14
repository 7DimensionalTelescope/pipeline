import os
import time
from pathlib import Path
from glob import glob
from astropy.io import fits
from astropy.table import Table
from datetime import datetime, timedelta

from .. import external
from ..utils import add_suffix, swap_ext, collapse, time_diff_in_seconds
from ..tools.table import match_two_catalogs
from ..config.utils import get_key
from ..errors import SubtractionError
from ..path import PathHandler
from ..preprocess.plotting import save_fits_as_figures

from ..services.setup import BaseSetup
from ..services.database.handler import DatabaseHandler
from ..services.database.image_qa import ImageQATable
from ..services.checker import Checker, SanityFilterMixin
from ..services.database.query import RawImageQuery

from .utils import create_ds9_region_file, select_sources


class ImSubtract(BaseSetup, DatabaseHandler, Checker, SanityFilterMixin):
    def __init__(
        self,
        config=None,
        logger=None,
        queue=None,
        overwrite=False,
    ) -> None:

        super().__init__(config, logger, queue)
        # self._flag_name = "subtract"
        self.logger.process_error = SubtractionError
        self.overwrite = overwrite
        self.name = self.config_node.name
        self.reference_images = None

        self.qa_id = None
        self.is_too = self.config_node.settings.is_too
        DatabaseHandler.__init__(self, add_database=self.config_node.settings.is_pipeline, is_too=self.is_too)

        if self.is_connected:

            self.reset_exceptions("subtraction")

            if self.process_status_id is not None:
                from ..services.database.handler import ExceptionHandler

                self.logger.add_exception_code = ExceptionHandler(self.process_status_id)

            self.process_status_id = self.create_process_data(self.config_node)
            if self.too_id is not None:
                self.logger.debug(f"Initialized DatabaseHandler for ToO data management, ToO ID: {self.too_id}")
            else:
                self.logger.debug(
                    f"Initialized DatabaseHandler for pipeline and QA data management, Pipeline ID: {self.process_status_id}"
                )
            self.update_progress(80, "imsubtract-configured")

    @classmethod
    def from_list(cls, input_images):
        from ..config import SciProcConfiguration

        image_list = []
        for image in input_images:
            path = Path(image)
            if not path.is_file():
                print(f"{image} does not exist or is not a file.")
                return None
            image_list.append(path.parts[-1])  # str

        working_dir = str(path.parent.absolute())

        config = SciProcConfiguration.user_config(working_dir=working_dir)
        config.input.calibrated_images = image_list
        return cls(config=config)

    @property
    def sequential_task(self):
        """[(number, name, use_gpu), ...]"""
        return [(1, "run", False)]

    def run(self, overwrite=False):
        st = time.time()
        self.logger.info(f"Start 'ImSubtract'")
        try:
            self.find_reference_image()
            if self.reference_images is None:  # if not found, do not run
                self.logger.info(f"No reference image found for {self.name}; Skipping transient search.")
                self.config_node.flag.subtraction = True
                self.logger.info(f"'ImSubtract' is Completed in {time_diff_in_seconds(st)} seconds")
                self.update_progress(100, "imsubtract-completed")
                return

            self.define_paths()
            self.update_progress(82, "imsubtract-define-paths-completed")

            if not overwrite and os.path.exists(self.subt_image_file):
                self.logger.info(f"Subtracted image already exists: {self.subt_image_file}; Skipping subtraction.")
                self.config_node.flag.subtraction = True
                self.logger.info(f"'ImSubtract' is Completed in {time_diff_in_seconds(st)} seconds")
                self.update_progress(100, "imsubtract-completed")
                return

            self.create_substamps()
            self.update_progress(84, "imsubtract-create-substamps-completed")

            self.create_masks()
            self.update_progress(86, "imsubtract-create-masks-completed")

            self.run_hotpants()
            self.update_progress(88, "imsubtract-run-hotpants-completed")

            self.mask_unsubtracted()

            # Create QA data for subtracted image if database is connected
            if self.is_connected and self.process_status_id is not None:
                subt_image = self.subt_image_file
                if subt_image and os.path.exists(subt_image):
                    self.qa_id = self.create_qa_data("science", image=subt_image, output_file=subt_image)

            # Update QA data from header if database is connected
            if self.is_connected and self.qa_id is not None:
                subt_image = self.subt_image_file
                if subt_image and os.path.exists(subt_image):
                    qa_data = ImageQATable.from_file(
                        subt_image,
                        process_status_id=self.process_status_id,
                    )
                    self.image_qa.update_data(qa_data.id, **qa_data.to_dict())

            self.plot_subtracted_image()

            self.update_progress(90, "imsubtract-completed")

            self.config_node.flag.subtraction = True
            self.logger.info(f"'ImSubtract' is Completed in {time_diff_in_seconds(st)} seconds")

        except Exception as e:
            self.logger.error(
                f"Error during imsubtract processing: {str(e)}",
                SubtractionError.UnknownError,
                exc_info=True,
            )
            raise

    def find_reference_image(self):

        obs, filt, date = self.path.name.obj[0], self.path.name.filter[0], self.path.name.date[0]
        image_list = RawImageQuery().for_target(obs).with_filter(filt).fetch()["sci"]

        self.logger.debug(f"Image list: {len(image_list)} images")
        available_dates = list(set([img["obstime"].date().strftime("%Y%m%d") for img in image_list]))
        self.logger.debug(f"Full dates: {available_dates}")

        if date in available_dates:
            available_dates.remove(date)

        if available_dates:
            self.logger.debug(f"Available dates: {available_dates}")
            # Convert input date to datetime
            input_date = datetime.strptime(date, "%Y%m%d")

            # Find the date that is farthest from the input date
            farthest_date = max(available_dates, key=lambda d: abs((datetime.strptime(d, "%Y%m%d") - input_date).days))
            farthest_date_dt = datetime.strptime(farthest_date, "%Y%m%d")

            # Check farthest date and +/- 5 day
            for i in range(5):
                search_dates = [
                    farthest_date_dt - timedelta(days=i),
                    farthest_date_dt + timedelta(days=i),
                ]

            # Search in both processed and too directories
            base_paths = ["/lyman/data2/processed", "/lyman/data2/too"]
            ref_image = None

            for search_date in search_dates:
                date_formatted = search_date.strftime("%Y-%m-%d")
                for base_path in base_paths:
                    ref_pattern = f"{base_path}/{date_formatted}/{obs}/{filt}/coadd/*coadd.fits"
                    ref_images = glob(ref_pattern)
                    if ref_images:
                        ref_image = ref_images[0]
                        self.logger.info(f"Reference image found: {ref_image}")
                        break
                if ref_image:
                    break

            if not ref_image:

                self.logger.warning(
                    f"The reference images are likely to exist but they have not been processed yet. Check observation dates: {available_dates} for {obs}/{filt}",
                    SubtractionError.ReferenceImageNotFoundError,
                )

    def define_paths(self):
        # always consider a single coadd image as input, not a list of images
        local_input_images = get_key(self.config_node, "imsubtract.input_image")
        # set from the common input if not set locally
        self.input_images = [local_input_images] or [self.config_node.input.coadd_image]
        self.apply_sanity_filter_and_report()
        input_image = collapse(self.input_images, raise_error=True)
        self.config_node.imsubtract.input_image = input_image

        self.logger.debug(f"ImSubtract inim: {input_image}")
        self.sci_image_file = input_image  # self.path.imcoadd.coadd_image
        # self.sci_source_table_file = get_derived_product_path(self.sci_image_file)
        # self.sci_source_table_file = add_suffix(self.sci_image_file, "cat")
        self.sci_source_table_file = PathHandler(self.sci_image_file, is_too=self.is_too).catalog

        self.ref_image_file = self.config_node.imsubtract.reference_image or self.reference_images[0]
        self.config_node.imsubtract.reference_image = self.ref_image_file  # sync
        self.ref_source_table_file = self.ref_image_file.replace(".fits", "_cat.fits")
        # self.ref_source_table_file = swap_ext(self.ref_image_file, "phot.cat")

        # self.subt_image_file = get_derived_product_path(self.sci_image_file, "transient", "subt.fits")
        self.subt_image_file = PathHandler(self.sci_image_file, is_too=self.is_too).imsubtract.diffim
        self.config_node.input.difference_image = self.subt_image_file
        self.logger.debug(f"subt_image_file: {self.subt_image_file}")
        self.logger.debug(f"config.input.difference_image: {self.config_node.input.difference_image}")
        # transient_dir = os.path.dirname(self.subt_image_file)
        # if not os.path.exists(transient_dir):
        #     os.makedirs(transient_dir)

        self.path_tmp = self.path.imsubtract.tmp_dir
        self.substamp_file = swap_ext(self.subt_image_file, "ssf.txt")
        self.ds9_file = swap_ext(self.subt_image_file, "ssf.reg")

    def create_substamps(self, ds9_region=True):
        sci_source_table = Table.read(self.sci_source_table_file)
        ref_source_table = Table.read(self.ref_source_table_file)
        # ref_source_table = Table.read(self.ref_source_table_file, format="ascii")

        # Select substamp sources
        selected_sci_table = select_sources(sci_source_table)
        self.logger.info(
            f"{len(selected_sci_table)} selected for substamp from {len(sci_source_table)} science catalog sources"
        )

        # Match two catalogs
        matched_source_table = match_two_catalogs(selected_sci_table, ref_source_table)
        self.logger.info(
            f"{len(matched_source_table)} sources matched" f"out of {len(selected_sci_table)}, {len(ref_source_table)}"
        )

        # Write substamp file
        f = open(self.substamp_file, "w")
        for x, y in zip(matched_source_table["X_IMAGE"], matched_source_table["Y_IMAGE"]):
            f.write(f"{x} {y}\n")
        f.close()

        # Create DS9 region file for substamp sources
        if ds9_region:
            create_ds9_region_file(
                matched_source_table["ALPHA_J2000"],
                matched_source_table["DELTA_J2000"],
                filename=self.ds9_file,
            )

    def create_masks(self):
        """FOV masks"""
        # Create mask for science image

        # weight_file = swap_ext(self.sci_image_file, "weight.fits")
        weight_file = PathHandler(self.sci_image_file).weight

        sci_mask = self._create_mask(weight_file if os.path.exists(weight_file) else self.sci_image_file)
        ref_mask = self._create_mask(self.ref_image_file)

        self.sci_mask_file = self._save_mask(sci_mask, self.sci_image_file)
        self.ref_mask_file = self._save_mask(ref_mask, self.ref_image_file)

        # Create common mask
        common_mask = sci_mask | ref_mask  # leave 0, mask 1

        self.common_mask_file = self._save_mask(common_mask, self.subt_image_file)
        self.common_mask = common_mask

    def _create_mask(self, file):
        data = fits.getdata(file)
        return (data == 0).astype("uint8")

    def _save_mask(self, mask, image_file):
        image_name = os.path.basename(image_file)
        filename = os.path.join(self.path_tmp, add_suffix(image_name, "mask"))
        fits.writeto(filename, data=mask, overwrite=True)
        return filename

    def run_hotpants(self):
        out_conv_im = os.path.join(
            self.path_tmp,
            swap_ext(os.path.basename(self.ref_image_file), "conv.fits"),
        )
        external.hotpants(
            self.sci_image_file,
            self.ref_image_file,
            self.sci_mask_file,
            self.ref_mask_file,
            ssf=self.substamp_file,
            outim=self.subt_image_file,
            out_conv_im=out_conv_im,
        )

    def mask_unsubtracted(self):
        # Apply final mask on both SUBT & CONV images
        mask = fits.getdata(self.common_mask_file)

        # Subt
        subt_data, subt_header = fits.getdata(self.subt_image_file, header=True)
        new_hddata = subt_data * (~mask + 2)
        fits.writeto(self.subt_image_file, new_hddata, header=subt_header, overwrite=True)

        # # Conv
        # hcdata, hchdr = fits.getdata(self.ref_image_file, header=True)
        # new_hcdata = hcdata * (~mask + 2)
        # fits.writeto(out_conv_im, new_hcdata, header=hchdr, overwrite=True)

    def plot_subtracted_image(self):
        subt_img = self.subt_image_file
        basename = os.path.basename(subt_img)
        save_fits_as_figures(fits.getdata(subt_img), self.path.figure_dir_to_path / swap_ext(basename, "png"))
        return
