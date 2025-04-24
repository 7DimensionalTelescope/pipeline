import os
import sys
from glob import glob
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import numpy as np

from ..const import REF_DIR
from ..config import Configuration
from .. import external
from ..services.setup import BaseSetup
from ..utils import get_derived_product_path, swap_ext, create_common_table

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
        self.overwrite = overwrite

    def run(self):

        self.find_reference_image()
        if not self.reference_images:  # if not found, do not run
            self.logger.info(
                f"No reference image found for {self.name}; Skipping transient search."
            )
            return

        self.define_paths()

        self.create_substamps()

        self.create_masks()

        self.run_hotpants()

        self.mask_unsubtracted()

    def find_reference_image(self):
        obj = self.config.obs.object
        filte = self.config.obs.filter
        refim_dir = os.path.join(self.config.path.path_refim, filte)

        ref_imgs_ps1 = glob(f"{refim_dir}/ref_PS1_{obj}_*_*_{filte}_0.fits")
        ref_imgs_7dt = glob(f"{refim_dir}/ref_7DT_{obj}_*_*_{filte}_*.fits")
        ref_imgs = ref_imgs_7dt + ref_imgs_ps1
        ref_imgs = [ref for ref in ref_imgs if "mask" not in ref]
        self.reference_images = ref_imgs
        # return True if len(ref_imgs) > 0 else False

    def define_paths(self):
        self.sci_image_file = self.config.file.stacked_file
        self.sci_source_table_file = get_derived_product_path(self.sci_image_file)

        self.ref_image_file = self.reference_images[0]
        self.ref_source_table_file = swap_ext(self.ref_image_file, "phot.cat")

        self.subt_image_file = get_derived_product_path(
            self.sci_image_file, "transient", "subt.fits"
        )
        transient_dir = os.path.dirname(self.subt_image_file)
        if not os.path.exists(transient_dir):
            os.makedirs(transient_dir)

        self.path_tmp = os.path.join(
            self.config.path.path_factory,
            "subt",
        )

    def create_substamps(self, ds9_region=True):
        sci_source_table = Table.read(self.sci_source_table_file)
        ref_source_table = Table.read(self.ref_source_table_file)

        # Select substamp sources
        selected_sci_table = select_sources(sci_source_table)
        self.logger.info(
            f"{len(selected_sci_table)} selected for substamp from"
            f"{len(sci_source_table)} science catalog sources"
        )

        # Match two catalogs
        matched_source_table = create_common_table(selected_sci_table, ref_source_table)
        self.logger.info(
            f"{len(matched_source_table)} sources matched"
            f"out of {len(selected_sci_table)}, {len(ref_source_table)}"
        )

        self.substamp_file = swap_ext(self.sci_image_file, "ssf.txt")

        # Write substamp file
        f = open(self.substamp_file, "w")
        for x, y in zip(
            matched_source_table["X_IMAGE"], matched_source_table["Y_IMAGE"]
        ):
            f.write(f"{x} {y}\n")
        f.close()

        # Create DS9 region file for substamp sources
        if ds9_region:
            create_ds9_region_file(
                matched_source_table["ALPHA_J2000"],
                matched_source_table["DELTA_J2000"],
                filename=swap_ext(self.sci_image_file, "ssf.reg"),
            )

    def create_masks(self):
        # Create mask for science image

        weight_file = swap_ext(self.sci_image_file, "weight.fits")

        sci_mask = self._create_mask(
            weight_file if os.path.exists(weight_file) else self.sci_image_file
        )
        ref_mask = self._create_mask(self.ref_image_file)

        self.sci_mask_file = self._save_mask(sci_mask, self.sci_image_file)
        self.ref_mask_file = self._save_mask(ref_mask, self.ref_image_file)

        # Create common mask
        common_mask = sci_mask | ref_mask  # leave 0, mask 1

        self.common_mask_file = self._save_mask(common_mask, self.subt_image_file)
        self.common_mask = common_mask

    def _create_mask(self, data):
        return (data == 0).astype("uint8")

    def _save_mask(self, mask, image_file):
        filename = os.path.join(self.path_tmp, swap_ext(image_file, "mask.fits"))
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
            out_conv_im=out_conv_im,
        )

    def mask_unsubtracted(self):
        # Apply final mask on both SUBT & CONV images
        mask = fits.getdata(self.common_mask_file)

        # Subt
        subt_data, subt_header = fits.getdata(self.subt_image_file, header=True)
        new_hddata = subt_data * (~mask + 2)
        fits.writeto(
            self.subt_image_file, new_hddata, header=subt_header, overwrite=True
        )

        # # Conv
        # hcdata, hchdr = fits.getdata(self.ref_image_file, header=True)
        # new_hcdata = hcdata * (~mask + 2)
        # fits.writeto(out_conv_im, new_hcdata, header=hchdr, overwrite=True)
