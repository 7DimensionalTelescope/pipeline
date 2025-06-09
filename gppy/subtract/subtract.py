import os
import sys
from pathlib import Path
from glob import glob
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import numpy as np

from ..const import REF_DIR
from ..config import SciProcConfiguration
from .. import external
from ..services.setup import BaseSetup
from ..utils import add_suffix, swap_ext, collapse
from ..tools.table import match_two_catalogs
from ..path import PathHandler

from .utils import create_ds9_region_file, select_sources


class ImSubtract(BaseSetup):
    def __init__(
        self,
        config=None,
        logger=None,
        queue=None,
        overwrite=False,
    ) -> None:

        super().__init__(config, logger, queue)
        self._flag_name = "subtract"
        self.overwrite = overwrite

    @classmethod
    def from_list(cls, input_images):

        image_list = []
        for image in input_images:
            path = Path(image)
            if not path.is_file():
                print(f"{image} does not exist or is not a file.")
                return None
            image_list.append(path.parts[-1])  # str

        working_dir = str(path.parent.absolute())

        config = SciProcConfiguration.base_config(working_dir=working_dir)
        config.input.calibrated_images = image_list
        return cls(config=config)

    @property
    def sequential_task(self):
        """[(number, name, use_gpu), ...]"""
        return [(1, "run", False), (2, "flagging", False)]

    def run(self):

        self.find_reference_image()
        if not self.reference_images:  # if not found, do not run
            self.logger.info(f"No reference image found for {self.name}; Skipping transient search.")
            return

        self.define_paths()

        self.create_substamps()

        self.create_masks()

        self.run_hotpants()

        self.mask_unsubtracted()

        self.config.flag.subtraction = True

        self.logger.info(f"ImSubtract completed.")

    def find_reference_image(self):
        # obj = self.config.obs.object
        # filte = self.config.obs.filter
        obj = collapse(self.path.name.obj, raise_error=True)  # assume same sci group: same obj, filter
        filte = collapse(self.path.name.filter, raise_error=True)
        refim_dir = os.path.join(self.path.imsubtract.ref_image_dir, filte)

        ref_imgs_ps1 = glob(f"{refim_dir}/ref_PS1_{obj}_*_*_{filte}_0.fits")
        ref_imgs_7dt = glob(f"{refim_dir}/ref_7DT_{obj}_*_*_{filte}_*.fits")
        ref_imgs = ref_imgs_7dt + ref_imgs_ps1
        ref_imgs = [ref for ref in ref_imgs if "mask" not in ref]
        self.reference_images = ref_imgs
        # return True if len(ref_imgs) > 0 else False

    def define_paths(self):
        self.sci_image_file = self.config.input.stacked_image  # self.path.imstack.stacked_image
        # self.sci_source_table_file = get_derived_product_path(self.sci_image_file)
        # self.sci_source_table_file = add_suffix(self.sci_image_file, "cat")
        self.sci_source_table_file = PathHandler(self.sci_image_file).catalog

        self.ref_image_file = self.config.imsubtract.reference_image or self.reference_images[0]
        self.config.imstack.reference_image = self.ref_image_file
        self.ref_source_table_file = swap_ext(self.ref_image_file, "phot.cat")

        # self.subt_image_file = get_derived_product_path(self.sci_image_file, "transient", "subt.fits")
        self.subt_image_file = PathHandler(self.sci_image_file).imsubtract.diffim
        # transient_dir = os.path.dirname(self.subt_image_file)
        # if not os.path.exists(transient_dir):
        #     os.makedirs(transient_dir)

        self.path_tmp = self.path.imsubtract.tmp_dir
        self.substamp_file = swap_ext(self.subt_image_file, "ssf.txt")
        self.ds9_file = swap_ext(self.subt_image_file, "ssf.reg")

    def create_substamps(self, ds9_region=True):
        sci_source_table = Table.read(self.sci_source_table_file)
        ref_source_table = Table.read(self.ref_source_table_file, format="ascii")

        # Select substamp sources
        selected_sci_table = select_sources(sci_source_table)
        self.logger.info(
            f"{len(selected_sci_table)} selected for substamp from" f"{len(sci_source_table)} science catalog sources"
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
