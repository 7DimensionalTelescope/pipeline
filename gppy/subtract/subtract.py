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
from ..base import BaseSetup
from ..utils import get_derived_product_path, swap_ext

from .utils import create_ds9_region_file, select_sources, create_common_table


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

    def create_substamps(self):
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

        substamp_file = swap_ext(self.sci_image_file, "ssf.txt")

        # Write substamp file
        f = open(substamp_file, "w")
        for x, y in zip(
            matched_source_table["X_IMAGE"], matched_source_table["Y_IMAGE"]
        ):
            f.write(f"{x} {y}\n")
        f.close()

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
        common_mask = sci_mask | ref_mask

        self.common_mask_file = self._save_mask(common_mask, self.subt_image_file)

    def _create_mask(self, data):
        return (data == 0).astype("uint8")

    def _save_mask(self, mask, image_file):
        filename = os.path.join(
            self.config.path.path_factory, swap_ext(image_file, "mask.fits")
        )
        fits.writeto(filename, data=mask, overwrite=True)
        return filename

    def run_hotpants(self):
        external.hotpants()
        return

    def legacy_gppy(self):
        # ============================================================
        # 	Function
        # ------------------------------------------------------------

        # ============================================================
        # %%
        # 	Input
        # ------------------------------------------------------------
        # 	Input
        # ------------------------------------------------------------
        # inim = input(f"Science Image:")
        # refim = input(f"Reference Image:")
        # inmask_image = input(f"Science Mask Image:")
        # refmask_image = input(f"Reference Mask Image:")
        # ------------------------------------------------------------
        # 	argument
        # ------------------------------------------------------------
        inim = sys.argv[1]
        refim = sys.argv[2]
        inmask_image = sys.argv[3]
        refmask_image = sys.argv[4]
        allmask_image = sys.argv[5]
        # ------------------------------------------------------------
        # 	Test Images
        # ------------------------------------------------------------
        # inim = "/large_data/processed_1x1_gain2750/T09614/7DT03/r/calib_7DT03_T09614_20240423_020757_r_360.com.fits"
        # refim = "/large_data/factory/ref_frame/r/ref_PS1_T09614_00000000_000000_r_0.fits"
        # inmask_image = "/large_data/processed_1x1_gain2750/T09614/7DT03/r/calib_7DT03_T09614_20240423_020757_r_360.com.mask.fits"
        # refmask_image = "/large_data/factory/ref_frame/r/ref_PS1_T09614_00000000_000000_r_0.mask.fits"
        #
        path_sci = os.path.dirname(inim)
        # ============================================================
        # 	Setting
        # ------------------------------------------------------------
        # 	For SSF selection
        aperture_suffix = "AUTO"
        snr_lower = 10
        classstar_lower = 0.2
        flags_upper = 0
        sep_upper = 1.0
        region_radius = 10
        # 	For Hotpants
        n_sigma = 5
        ##	Template
        # tl, tu = refskyval - n_sigma * refskysig, 60000
        # tl, tu = refskyval - n_sigma * refskysig, 60000000
        tl, tu = -60000000, 60000000
        # tl, tu = -20000, 5100000
        ##	Region Split (y, x = 6800, 10200)
        nrx, nry = 3, 2
        # nrx, nry = 1, 1
        # nrx, nry = 6, 4
        # ============================================================
        # %%
        # 	Data
        # ------------------------------------------------------------
        # 	Science
        # ------------------------------------------------------------
        inhdr = fits.getheader(inim)
        inobj = inhdr["OBJECT"]
        infilter = inhdr["FILTER"]
        ingain = inhdr["EGAIN"]
        inellip = inhdr["ELLIP"]
        inskyval = inhdr["SKYVAL"]
        inskysig = inhdr["SKYSIG"]
        incat = inim.replace("fits", "phot.cat")
        intbl = Table.read(incat, format="ascii")
        # ------------------------------------------------------------
        # 	Select substamp sources
        # ------------------------------------------------------------
        #
        indx_input_select = np.where(
            (intbl[f"SNR_{aperture_suffix}_{infilter}"] > snr_lower)
            & (intbl[f"CLASS_STAR"] > classstar_lower)
            & (intbl["FLAGS"] <= flags_upper)
        )
        selected_intbl = intbl[indx_input_select]
        c_in = SkyCoord(
            selected_intbl["ALPHA_J2000"], selected_intbl["DELTA_J2000"], unit="deg"
        )
        print(
            f"{len(selected_intbl)} selected from {len(intbl)} ({len(selected_intbl)/len(intbl):.1%})"
        )
        # ------------------------------------------------------------
        # %%
        # 	Reference
        # ------------------------------------------------------------
        # refim = f'{path_ref}/ref_PS1_T09614_00000000_000000_r_0.fits'
        refhdr = fits.getheader(refim)
        # reffilter = refhdr['FILTER']
        # refgain = refhdr['EGAIN']
        refcat = refim.replace("fits", "phot.cat")
        reftbl = Table.read(refcat, format="ascii")
        refskyval = np.median(reftbl["BACKGROUND"])
        refskysig = np.std(reftbl["BACKGROUND"])
        c_ref = SkyCoord(reftbl["ALPHA_J2000"], reftbl["DELTA_J2000"], unit="deg")
        # ------------------------------------------------------------

        # %%
        # 	Select substamp sources
        # ------------------------------------------------------------
        reftbl[f"SNR_{aperture_suffix}"] = (
            reftbl[f"FLUX_{aperture_suffix}"] / reftbl[f"FLUXERR_{aperture_suffix}"]
        )

        indx_ref_select = np.where(
            (reftbl[f"SNR_{aperture_suffix}"] > snr_lower)
            & (reftbl[f"CLASS_STAR"] > classstar_lower)
            & (reftbl["FLAGS"] <= flags_upper)
        )
        selected_reftbl = reftbl[indx_ref_select]
        c_ref = SkyCoord(
            selected_reftbl["ALPHA_J2000"], selected_reftbl["DELTA_J2000"], unit="deg"
        )
        print(
            f"{len(selected_reftbl)} selected from {len(reftbl)} ({len(selected_reftbl)/len(reftbl):.1%})"
        )

        # %%
        # 	Matching
        # indx_match, sep_match, _ = c_ref.match_to_catalog_sky(c_in)
        indx_match, sep_match, _ = c_in.match_to_catalog_sky(c_ref)
        matched_table = selected_intbl[sep_match.arcsec < sep_upper]
        print(f"{len(matched_table)} sources matched")
        ssf = f"{path_sci}/{os.path.basename(inim).replace('fits', 'ssf.txt')}"
        f = open(ssf, "w")
        for x, y in zip(matched_table["X_IMAGE"], matched_table["Y_IMAGE"]):
            f.write(f"{x} {y}\n")
        f.close()

        # %%
        # 	Output
        # hdim = f"{os.path.dirname(inim)}/hd{os.path.basename(inim)}"
        # hcim = f"{os.path.dirname(inim)}/hc{os.path.basename(inim)}"
        path_data = os.path.dirname(inim)
        hdim = inim.replace("fits", "subt.fits")
        _hcim = f"{os.path.basename(refim).replace('fits', 'conv.fits')}"
        dateobs = os.path.basename(inim).split("_")[3]
        timeobs = os.path.basename(inim).split("_")[4]
        part_hcim = _hcim.split("_")
        part_hcim[3] = dateobs
        part_hcim[4] = timeobs
        hcim = f"{path_data}/{'_'.join(part_hcim)}"

        print(f"Output Image   : {hdim}")
        print(f"Convolved Image: {hcim}")

        # %%
        il, iu = inskyval - n_sigma * inskysig, 60000
        # il, iu = 0, 6000000
        # 	Run
        com = (
            f"hotpants -c t -n i "
            f"-iu {iu} -il {il} -tu {tu} -tl {tl} "
            f"-inim {inim} -tmplim {refim} -outim {hdim} -oci {hcim} "
            f"-imi {inmask_image} -tmi {refmask_image} "
            f"-v 0 "
            f"-nrx {nrx} -nry {nry} "
            f"-ssf {ssf}"
        )
        print(com)
        os.system(com)

        # %%
        ds9region_file = (
            f"{path_sci}/{os.path.basename(inim).replace('fits', 'ssf.reg')}"
        )
        create_ds9_region_file(
            ra_array=matched_table["ALPHA_J2000"],
            dec_array=matched_table["DELTA_J2000"],
            radius=region_radius,
            filename=f"{ds9region_file}",
        )

        # ds9com = f"ds9 -tile column -frame lock wcs {inim} -region load {ds9region_file} {hcim} -region load {ds9region_file} {hdim} -region load {ds9region_file} &"
        # print(ds9com)
        # %%
        print(f"Apply Final Mask on both SUBT & CONV images")
        mask = fits.getdata(allmask_image)
        # 	Subt
        hddata, hdhdr = fits.getdata(hdim, header=True)
        new_hddata = hddata * (~mask + 2)
        fits.writeto(hdim, new_hddata, header=hdhdr, overwrite=True)
        # 	Conv
        hcdata, hchdr = fits.getdata(hcim, header=True)
        new_hcdata = hcdata * (~mask + 2)
        fits.writeto(hcim, new_hcdata, header=hchdr, overwrite=True)
