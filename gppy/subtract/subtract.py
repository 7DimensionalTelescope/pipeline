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
        if not self.ref_imgs:  # if not found, do not run
            self.logger.info(
                f"Reference image not found for {self.name}; Skipping transient search."
            )
            return

        self.define_parameters()

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
        self.ref_imgs = ref_imgs
        # return True if len(ref_imgs) > 0 else False

    def run_hotpants(self):
        external.hotpants()
        return

    def legacy_gppy(self):
        # ============================================================
        # 	Function
        # ------------------------------------------------------------
        def create_ds9_region_file(
            ra_array, dec_array, radius, filename="ds9_regions.reg"
        ):
            """
            RA, Dec 배열과 반경을 입력으로 받아 DS9 region 파일을 생성하는 함수.

            Parameters:
            - ra_array: RA 좌표 배열
            - dec_array: Dec 좌표 배열
            - radius: 원의 반경 (단위: arcsec)
            - filename: 생성될 DS9 region 파일의 이름
            """
            # Region 파일 시작 부분에 필요한 헤더
            header = 'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5'

            # 파일 쓰기 시작
            with open(filename, "w") as file:
                file.write(header + "\n")

                # 각 좌표에 대한 원 형태의 region 추가
                for ra, dec in zip(ra_array, dec_array):
                    region_line = f'circle({ra},{dec},{radius}")\n'
                    file.write(region_line)

            print(f"DS9 region file '{filename}' has been created.")

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
