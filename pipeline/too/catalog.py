import astropy.units as u
import numpy as np
import os
from shapely.geometry import Polygon
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.time import Time
from astropy.table import Table, vstack
from astroquery.vizier import Vizier


class SkyCatalogHistory:
    HISTORY_FIELDS = ["objname", "ra", "dec", "fov_ra", "fov_dec", "filename", "cat_type", "save_date"]

    def __init__(self, **kwargs):
        self.history = {step: None for step in self.HISTORY_FIELDS}
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key] = value

    def to_dict(self):
        return self.history

    def update(self, key, value):
        if key in self.history:
            self.history[key] = value
        else:
            print(f"WARNING: Invalid key: {key}")

    def __repr__(self):
        history_list = [f"{key}: {value}" for key, value in self.history.items()]
        return (
            "History =====================================\n  "
            + "\n  ".join(history_list)
            + "\n==================================================="
        )


class SkyCatalog:
    """SkyCatalog class is used to query the sky catalog and get the catalog data."""

    CATALOG_CONFIGS = {
        "APASS": {"catalog": "II/336/apass9", "columns": ["*"], "maxsources": 100000},
        "SDSS": {"catalog": "V/147/sdss12", "columns": ["*"], "maxsources": 100000},
        "PS1": {"catalog": "II/349/ps1", "columns": ["*"], "maxsources": 500000},
        "SMSS": {
            "catalog": "II/379/smssdr4",
            "columns": [
                "ObjectId",
                "RAICRS",
                "DEICRS",
                "Niflags",
                "flags",
                "Ngood",
                "Ngoodu",
                "Ngoodv",
                "Ngoodg",
                "Ngoodr",
                "Ngoodi",
                "Ngoodz",
                "ClassStar",
                "uPSF",
                "e_uPSF",
                "vPSF",
                "e_vPSF",
                "gPSF",
                "e_gPSF",
                "rPSF",
                "e_rPSF",
                "iPSF",
                "e_iPSF",
                "zPSF",
                "e_zPSF",
            ],
            "maxsources": 1000000,
        },
        "GAIA": {
            "catalog": "I/360/syntphot",
            "columns": [
                "RA_ICRS",
                "DE_ICRS",
                "E_BP_RP_corr",
                "Bmag",
                "BFlag",
                "Vmag",
                "VFlag",
                "Rmag",
                "RFlag",
                "gmag",
                "gFlag",
                "rmag",
                "rFlag",
                "imag",
                "iFlag",
            ],
            "maxsources": 1000000,
        },
    }

    def __init__(
        self,
        objname: str = None,
        ra=None,
        dec=None,
        fov_ra: float = 1.3,
        fov_dec: float = 0.9,
        catalog_type: str = "GAIAXP",
        overlapped_fraction: float = 0.8,
        verbose: bool = True,
    ):
        if catalog_type not in ["GAIAXP", "GAIA", "APASS", "PS1", "SDSS", "SMSS"]:
            raise ValueError(f"Invalid catalog type: {catalog_type}")
        catalog_dir = os.environ.get("CATALOG_DIR", os.path.join(os.getcwd(), "catalog_archive"))
        self.config = {"CATALOG_DIR": catalog_dir}
        self.verbose = verbose
        self.objname = objname
        self.ra = ra
        self.dec = dec
        self.fov_ra = fov_ra
        self.fov_dec = fov_dec
        self.catalog_type = catalog_type
        self.overlapped_fraction = overlapped_fraction
        self.filename = None
        self.save_date = None
        self.data = None

        self._register_objinfo(
            objname=objname, ra=ra, dec=dec, fov_ra=fov_ra, fov_dec=fov_dec, catalog_type=catalog_type
        )
        self._get_catalog(catalog_type=catalog_type, verbose=verbose)

    def __repr__(self):
        return f"SkyCatalog[objname = {self.objname}, type = {self.catalog_type}]"

    def _query(self, catalog_name: str = "APASS", verbose: bool = False):
        def _vizier_query(ra_deg, dec_deg, rad_deg, catalog, columns, maxsources=100000):
            vquery = Vizier(columns=columns, row_limit=maxsources)
            field = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame="icrs")
            query_data = vquery.query_region(field, width=("%fd" % rad_deg), catalog=catalog)

            if len(query_data) > 0:
                result = query_data[0]
                if catalog_name == "GAIA":
                    for col in ["e_Bmag", "e_Vmag", "e_Rmag", "e_gmag", "e_rmag", "e_imag"]:
                        result[col] = 0.02
                return result
            return None

        if catalog_name not in self.CATALOG_CONFIGS:
            raise ValueError(f"{self.objname} does not exist in {catalog_name}")

        self.catalog_type = catalog_name
        if verbose:
            print(f"Start {catalog_name} query...")

        config = self.CATALOG_CONFIGS[catalog_name]
        data = _vizier_query(
            ra_deg=float(self.ra),
            dec_deg=float(self.dec),
            rad_deg=np.max([self.fov_ra, self.fov_dec]),
            catalog=config["catalog"],
            columns=config["columns"],
            maxsources=config["maxsources"],
        )

        if not data:
            raise ValueError(f"{catalog_name} is not registered")

        return self._filter_sources_in_fov(data)

    def get_reference_sources(self, mag_lower: float = 10, mag_upper: float = 20, **kwargs):
        if not self.data:
            raise RuntimeError(f"No catalog data found for {self.objname}")

        cutlines = {
            "APASS": {"e_ra": [0, 0.5], "e_dec": [0, 0.5], "e_V_mag": [0.01, 0.05], "V_mag": [mag_lower, mag_upper]},
            "GAIA": {"V_flag": [0, 1], "V_mag": [mag_lower, mag_upper]},
            "GAIAXP": {"pmra": [-20, 20], "pmdec": [-20, 20], "bp-rp": [0.0, 1.5], "g_mean": [mag_lower, mag_upper]},
            "PS1": {"gFlags": [0, 10], "g_mag": [mag_lower, mag_upper]},
            "SMSS": {"ngood": [20, 999], "class_star": [0.8, 1.0], "g_mag": [mag_lower, mag_upper]},
        }

        if self.catalog_type not in cutlines:
            raise ValueError(f"Invalid catalog type: {self.catalog_type}")

        cutline = {**cutlines[self.catalog_type], **kwargs}
        ref_sources = self.data
        applied_kwargs = []
        for key, value in cutline.items():
            if key in ref_sources.colnames:
                applied_kwargs.append({key: value})
                ref_sources = ref_sources[(ref_sources[key] > value[0]) & (ref_sources[key] < value[1])]
        return ref_sources, applied_kwargs

    @property
    def catalog_summary(self):
        return os.path.join(self.config["CATALOG_DIR"], "summary.ascii_fixed_width")

    def _get_catalog(self, catalog_type: str, verbose: bool = True):
        method_map = {
            "GAIAXP": self._get_GAIAXP,
            "GAIA": self._get_GAIA,
            "APASS": self._get_APASS,
            "PS1": self._get_PS1,
            "SDSS": self._get_SDSS,
            "SMSS": self._get_SMSS,
        }
        if catalog_type not in method_map:
            raise ValueError(f"Invalid catalog type: {catalog_type}")
        method_map[catalog_type](
            objname=self.objname, ra=self.ra, dec=self.dec, fov_ra=self.fov_ra, fov_dec=self.fov_dec, verbose=verbose
        )

    def _load_and_format_catalog(self, catalog_name, format_func, verbose):
        # Check for catalog-specific filename in archive
        expected_filename = f"{self.objname}_{catalog_name}.csv"
        catalog_file = os.path.join(self.config["CATALOG_DIR"], catalog_name, expected_filename)

        if os.path.exists(catalog_file):
            if verbose:
                print(f"Catalog file found in archive: {expected_filename}")
            data = self._get_catalog_from_archive(catalog_name=catalog_name, filename=expected_filename)
        else:
            # Don't overwrite existing objinfo - use current values
            self._register_objinfo(
                objname=self.objname,
                ra=self.ra,
                dec=self.dec,
                fov_ra=self.fov_ra,
                fov_dec=self.fov_dec,
                catalog_type=catalog_name,
            )

            # Check if _register_objinfo found a file for this catalog type
            if self.filename and catalog_name in self.filename:
                if verbose:
                    print(f"Catalog file found in archive: {self.filename}")
                data = self._get_catalog_from_archive(catalog_name=catalog_name, filename=self.filename)
            else:
                try:
                    data = self._query(catalog_name=catalog_name, verbose=verbose)
                    self._save_catalog(data, catalog_name)
                except Exception:
                    raise ValueError(f"{self.objname} does not exist in {catalog_name} catalog")

        self.data = None
        if data:
            formatted_data = format_func(data)
            self.data = self._filter_sources_in_fov(formatted_data)

    def _save_catalog(self, data, catalog_name):
        # Catalog saving disabled - do not save any catalog files
        return

    def _get_GAIAXP(self, objname=None, ra=None, dec=None, fov_ra=1.3, fov_dec=0.9, verbose=False):
        def format_func(catalog):
            original = (
                "source_id",
                "ra",
                "dec",
                "parallax",
                "pmra",
                "pmdec",
                "phot_g_mean_mag",
                "bp_rp",
                "mag_u",
                "mag_g",
                "mag_r",
                "mag_i",
                "mag_z",
                "mag_m375w",
                "mag_m400",
                "mag_m412",
                "mag_m425",
                "mag_m425w",
                "mag_m437",
                "mag_m450",
                "mag_m462",
                "mag_m475",
                "mag_m487",
                "mag_m500",
                "mag_m512",
                "mag_m525",
                "mag_m537",
                "mag_m550",
                "mag_m562",
                "mag_m575",
                "mag_m587",
                "mag_m600",
                "mag_m612",
                "mag_m625",
                "mag_m637",
                "mag_m650",
                "mag_m662",
                "mag_m675",
                "mag_m687",
                "mag_m700",
                "mag_m712",
                "mag_m725",
                "mag_m737",
                "mag_m750",
                "mag_m762",
                "mag_m775",
                "mag_m787",
                "mag_m800",
                "mag_m812",
                "mag_m825",
                "mag_m837",
                "mag_m850",
                "mag_m862",
                "mag_m875",
                "mag_m887",
            )
            format_ = (
                "id",
                "ra",
                "dec",
                "parallax",
                "pmra",
                "pmdec",
                "g_mean",
                "bp-rp",
                "u_mag",
                "g_mag",
                "r_mag",
                "i_mag",
                "z_mag",
                "m375w_mag",
                "m400_mag",
                "m412_mag",
                "m425_mag",
                "m425w_mag",
                "m437_mag",
                "m450_mag",
                "m462_mag",
                "m475_mag",
                "m487_mag",
                "m500_mag",
                "m512_mag",
                "m525_mag",
                "m537_mag",
                "m550_mag",
                "m562_mag",
                "m575_mag",
                "m587_mag",
                "m600_mag",
                "m612_mag",
                "m625_mag",
                "m637_mag",
                "m650_mag",
                "m662_mag",
                "m675_mag",
                "m687_mag",
                "m700_mag",
                "m712_mag",
                "m725_mag",
                "m737_mag",
                "m750_mag",
                "m762_mag",
                "m775_mag",
                "m787_mag",
                "m800_mag",
                "m812_mag",
                "m825_mag",
                "m837_mag",
                "m850_mag",
                "m862_mag",
                "m875_mag",
                "m887_mag",
            )
            catalog.rename_columns(original, format_)
            return self._match_digit_tbl(catalog)

        if self.filename:
            if verbose:
                print(f"Catalog file found in archive: {self.filename}")
            data = self._get_catalog_from_archive(catalog_name="GAIAXP", filename=self.filename)
        else:
            raise ValueError(f"{self.objname} does not exist in GAIAXP catalog")

        self.data = None
        if data:
            self.data = self._filter_sources_in_fov(format_func(data))

    def _get_GAIA(self, objname=None, ra=None, dec=None, fov_ra=1.3, fov_dec=0.9, verbose=True):
        def format_func(catalog):
            original = (
                "RA_ICRS",
                "DE_ICRS",
                "Bmag",
                "e_Bmag",
                "BFlag",
                "Vmag",
                "e_Vmag",
                "VFlag",
                "Rmag",
                "e_Rmag",
                "RFlag",
                "gmag",
                "e_gmag",
                "gFlag",
                "rmag",
                "e_rmag",
                "rFlag",
                "imag",
                "e_imag",
                "iFlag",
            )
            format_ = (
                "ra",
                "dec",
                "B_mag",
                "e_B_mag",
                "B_flag",
                "V_mag",
                "e_Vmag",
                "V_flag",
                "R_mag",
                "e_Rmag",
                "R_flag",
                "g_mag",
                "e_gmag",
                "g_flag",
                "r_mag",
                "e_rmag",
                "r_flag",
                "i_mag",
                "e_imag",
                "i_flag",
            )
            catalog.rename_columns(original, format_)
            if "E_BP_RP_corr" in catalog.colnames:
                catalog.rename_columns(["E_BP_RP_corr"], ["c_star"])
            else:
                catalog["c_star"] = 0
            return self._match_digit_tbl(catalog)

        self._load_and_format_catalog("GAIA", format_func, verbose)

    def _get_APASS(self, objname=None, ra=None, dec=None, fov_ra=1.3, fov_dec=0.9, verbose=True):
        def format_func(catalog):
            original = (
                "RAJ2000",
                "DEJ2000",
                "e_RAJ2000",
                "e_DEJ2000",
                "Bmag",
                "e_Bmag",
                "Vmag",
                "e_Vmag",
                "g'mag",
                "e_g'mag",
                "r'mag",
                "e_r'mag",
                "i'mag",
                "e_i'mag",
            )
            format_ = (
                "ra",
                "dec",
                "e_ra",
                "e_dec",
                "B_mag",
                "e_B_mag",
                "V_mag",
                "e_V_mag",
                "g_mag",
                "e_g_mag",
                "r_mag",
                "e_r_mag",
                "i_mag",
                "e_i_mag",
            )
            catalog.rename_columns(original, format_)
            return self._match_digit_tbl(catalog)

        self._load_and_format_catalog("APASS", format_func, verbose)

    def _get_PS1(self, objname=None, ra=None, dec=None, fov_ra=1.3, fov_dec=0.9, verbose=True):
        def format_func(catalog):
            original = (
                "objID",
                "RAJ2000",
                "DEJ2000",
                "e_RAJ2000",
                "e_DEJ2000",
                "gmag",
                "e_gmag",
                "rmag",
                "e_rmag",
                "imag",
                "e_imag",
                "zmag",
                "e_zmag",
                "ymag",
                "e_ymag",
                "gKmag",
                "e_gKmag",
                "rKmag",
                "e_rKmag",
                "iKmag",
                "e_iKmag",
                "zKmag",
                "e_zKmag",
                "yKmag",
                "e_yKmag",
            )
            format_ = (
                "ID",
                "ra",
                "dec",
                "e_ra",
                "e_dec",
                "g_mag",
                "e_g_mag",
                "r_mag",
                "e_r_mag",
                "i_mag",
                "e_i_mag",
                "z_mag",
                "e_z_mag",
                "y_mag",
                "e_y_mag",
                "g_Kmag",
                "e_g_Kmag",
                "r_Kmag",
                "e_r_Kmag",
                "i_Kmag",
                "e_i_Kmag",
                "z_Kmag",
                "e_z_Kmag",
                "y_Kmag",
                "e_y_Kmag",
            )
            catalog.rename_columns(original, format_)
            return self._match_digit_tbl(catalog)

        self._load_and_format_catalog("PS1", format_func, verbose)

    def _get_SMSS(self, objname=None, ra=None, dec=None, fov_ra=1.3, fov_dec=0.9, verbose=True):
        def format_func(catalog):
            original = (
                "ObjectId",
                "RAICRS",
                "DEICRS",
                "Niflags",
                "flags",
                "Ngood",
                "Ngoodu",
                "Ngoodv",
                "Ngoodg",
                "Ngoodr",
                "Ngoodi",
                "Ngoodz",
                "ClassStar",
                "uPSF",
                "e_uPSF",
                "vPSF",
                "e_vPSF",
                "gPSF",
                "e_gPSF",
                "rPSF",
                "e_rPSF",
                "iPSF",
                "e_iPSF",
                "zPSF",
                "e_zPSF",
            )
            format_ = (
                "ID",
                "ra",
                "dec",
                "nimflag",
                "flag",
                "ngood",
                "ngoodu",
                "ngoodv",
                "ngoodg",
                "ngoodr",
                "ngoodi",
                "ngoodz",
                "class_star",
                "u_mag",
                "e_u_mag",
                "v_mag",
                "e_v_mag",
                "g_mag",
                "e_g_mag",
                "r_mag",
                "e_r_mag",
                "i_mag",
                "e_i_mag",
                "z_mag",
                "e_z_mag",
            )
            catalog.rename_columns(original, format_)
            return self._match_digit_tbl(catalog)

        self._load_and_format_catalog("SMSS", format_func, verbose)

    def _get_SDSS(self, objname=None, ra=None, dec=None, fov_ra=1.3, fov_dec=0.9, verbose=True):
        def format_func(catalog):
            original = (
                "RA_ICRS",
                "DE_ICRS",
                "umag",
                "e_umag",
                "gmag",
                "e_gmag",
                "rmag",
                "e_rmag",
                "imag",
                "e_imag",
                "zmag",
                "e_zmag",
            )
            format_ = (
                "ra",
                "dec",
                "umag",
                "e_umag",
                "gmag",
                "e_gmag",
                "rmag",
                "e_rmag",
                "imag",
                "e_imag",
                "zmag",
                "e_zmag",
            )
            catalog.rename_columns(original, format_)
            return self._match_digit_tbl(catalog)

        self._load_and_format_catalog("SDSS", format_func, verbose)

    def _filter_sources_in_fov(self, data):
        if data is None or len(data) == 0:
            return data

        ra_col = None
        dec_col = None
        for col in data.colnames:
            col_lower = col.lower()
            if ra_col is None and ("ra" in col_lower and "err" not in col_lower and "e_" not in col_lower):
                ra_col = col
            if (
                dec_col is None
                and ("dec" in col_lower or "de" in col_lower)
                and "err" not in col_lower
                and "e_" not in col_lower
            ):
                dec_col = col

        if ra_col is None or dec_col is None:
            return data

        ra_min = self.ra - self.fov_ra / 2.0
        ra_max = self.ra + self.fov_ra / 2.0
        dec_min = self.dec - self.fov_dec / 2.0
        dec_max = self.dec + self.fov_dec / 2.0

        if ra_min < 0:
            mask = (
                ((data[ra_col] >= (ra_min + 360)) | (data[ra_col] <= ra_max))
                & (data[dec_col] >= dec_min)
                & (data[dec_col] <= dec_max)
            )
        elif ra_max > 360:
            mask = (
                ((data[ra_col] >= ra_min) | (data[ra_col] <= (ra_max - 360)))
                & (data[dec_col] >= dec_min)
                & (data[dec_col] <= dec_max)
            )
        else:
            mask = (
                (data[ra_col] >= ra_min)
                & (data[ra_col] <= ra_max)
                & (data[dec_col] >= dec_min)
                & (data[dec_col] <= dec_max)
            )

        return data[mask]

    def _match_digit_tbl(self, tbl):
        for column in tbl.columns:
            if tbl[column].dtype == "float64":
                tbl[column].format = "{:.5f}"
        return tbl

    def _get_catalog_from_archive(self, catalog_name: str, filename: str):
        catalog_file = os.path.join(self.config["CATALOG_DIR"], catalog_name, filename)
        if os.path.exists(catalog_file):
            return ascii.read(catalog_file, format="csv")
        return None

    def _get_cataloginfo_by_coord(
        self,
        coord: SkyCoord,
        fov_ra: float = 1.5,
        fov_dec: float = 1.5,
        overlapped_fraction: float = 0.9,
        verbose: bool = False,
    ) -> Table:
        try:
            catalog_summary_tbl = ascii.read(self.catalog_summary, format="fixed_width")
            ra_min, ra_max = coord.ra.deg - 5, coord.ra.deg + 5
            dec_min, dec_max = coord.dec.deg - 5, coord.dec.deg + 5
            cut_tiles_mask = (
                (catalog_summary_tbl["ra"] >= ra_min)
                & (catalog_summary_tbl["ra"] <= ra_max)
                & (catalog_summary_tbl["dec"] >= dec_min)
                & (catalog_summary_tbl["dec"] <= dec_max)
            )
            catalog_summary_tbl = catalog_summary_tbl[cut_tiles_mask]

            overlap_catalogs = []
            for idx, (cat_ra, cat_dec, cat_fov_ra, cat_fov_dec) in enumerate(
                zip(
                    catalog_summary_tbl["ra"],
                    catalog_summary_tbl["dec"],
                    catalog_summary_tbl["fov_ra"],
                    catalog_summary_tbl["fov_dec"],
                )
            ):
                target_polygon = Polygon(
                    [
                        (coord.ra.deg - fov_ra / 2, coord.dec.deg - fov_dec / 2),
                        (coord.ra.deg + fov_ra / 2, coord.dec.deg - fov_dec / 2),
                        (coord.ra.deg + fov_ra / 2, coord.dec.deg + fov_dec / 2),
                        (coord.ra.deg - fov_ra / 2, coord.dec.deg + fov_dec / 2),
                    ]
                )
                tile_polygon = Polygon(
                    [
                        (cat_ra - cat_fov_ra / 2, cat_dec - cat_fov_dec / 2),
                        (cat_ra + cat_fov_ra / 2, cat_dec - cat_fov_dec / 2),
                        (cat_ra + cat_fov_ra / 2, cat_dec + cat_fov_dec / 2),
                        (cat_ra - cat_fov_ra / 2, cat_dec + cat_fov_dec / 2),
                    ]
                )
                if target_polygon.intersects(tile_polygon):
                    intersection = target_polygon.intersection(tile_polygon)
                    target_area = fov_ra * fov_dec
                    fraction_overlap = intersection.area / target_area
                    if fraction_overlap >= overlapped_fraction:
                        overlap_catalogs.append(catalog_summary_tbl[idx])

            if overlap_catalogs:
                return vstack(overlap_catalogs)
            raise ValueError("No catalog found with sufficient overlap.")
        except Exception as e:
            raise RuntimeError(f"Failed to access catalog summary: {e}")

    def _get_cataloginfo_by_objname(self, objname, catalog_type, fov_ra, fov_dec):
        catalog_summary_file = os.path.join(self.config["CATALOG_DIR"], "catalog_summary.ascii_fixed_width")
        catalog_summary_tbl = ascii.read(catalog_summary_file, format="fixed_width")
        idx = (
            (catalog_summary_tbl["objname"] == objname)
            & (catalog_summary_tbl["cat_type"] == catalog_type)
            & (catalog_summary_tbl["fov_ra"] * 1.1 > fov_ra)
            & (catalog_summary_tbl["fov_dec"] * 1.1 > fov_dec)
        )
        if np.sum(idx) > 0:
            return catalog_summary_tbl[idx]
        raise ValueError(f"{objname} not found in catalog_summary")

    def _update_history(self):
        self.history = SkyCatalogHistory(
            objname=self.objname,
            ra=self.ra,
            dec=self.dec,
            fov_ra=self.fov_ra,
            fov_dec=self.fov_dec,
            filename=self.filename,
            cat_type=self.catalog_type,
            save_date=self.save_date,
        )

    def _register_objinfo(self, objname, ra, dec, fov_ra, fov_dec, catalog_type):
        self.objname = objname
        self.ra = ra
        self.dec = dec
        self.fov_ra = fov_ra
        self.fov_dec = fov_dec
        self.catalog_type = catalog_type

        if (objname is None) and (ra is None) and (dec is None):
            raise ValueError("objname or (ra, dec) must be provided")

        if objname is not None and (ra is None or dec is None):
            try:
                catinfo = self._get_cataloginfo_by_objname(
                    objname=objname, catalog_type=catalog_type, fov_ra=fov_ra, fov_dec=fov_dec
                )
                self.ra = catinfo["ra"][0]
                self.dec = catinfo["dec"][0]
                self.fov_ra = catinfo["fov_ra"][0]
                self.fov_dec = catinfo["fov_dec"][0]
                self.filename = catinfo["filename"][0]
                self.save_date = catinfo["save_date"][0]
            except:
                try:
                    coord = self._query_coord_from_objname(objname=objname)
                    self.ra = coord.ra.deg
                    self.dec = coord.dec.deg
                except:
                    raise ValueError(f"Failed to query coordinates for {objname}")

        if objname is None and ra is not None and dec is not None:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            try:
                catinfo = self._get_cataloginfo_by_coord(
                    coord=coord, fov_ra=fov_ra, fov_dec=fov_dec, overlapped_fraction=self.overlapped_fraction
                )
                self.objname = catinfo["objname"][0]
                self.ra = catinfo["ra"][0]
                self.dec = catinfo["dec"][0]
                self.fov_ra = catinfo["fov_ra"][0]
                self.fov_dec = catinfo["fov_dec"][0]
                self.filename = catinfo["filename"][0]
                self.save_date = catinfo["save_date"][0]
            except:
                ra_hms = coord.ra.hms
                dec_dms = coord.dec.dms
                self.objname = f"J{int(ra_hms.h):02}{int(ra_hms.m):02}{ra_hms.s:05.2f}{int(dec_dms.d):+03}{int(abs(dec_dms.m)):02}{abs(dec_dms.s):04.1f}"

        if (self.objname is None) or (self.ra is None) or (self.dec is None):
            raise ValueError("objname, ra, and dec must be provided")

        self._update_history()

    def _query_coord_from_objname(self, objname) -> SkyCoord:
        from astroquery.simbad import Simbad

        custom_simbad = Simbad()
        custom_simbad.add_votable_fields("ra", "dec")
        result_table = custom_simbad.query_object(objname)

        if result_table is not None:
            ra = result_table["ra"][0]
            dec = result_table["dec"][0]
            return SkyCoord(ra, dec, unit=(u.deg, u.deg))
        raise ValueError("Object not found in SIMBAD.")


if __name__ == "__main__":
    ra = 10.68458
    dec = -41.26917
    catalog = SkyCatalog(ra=ra, dec=dec, catalog_type="GAIAXP", fov_ra=1.3, fov_dec=0.9)
