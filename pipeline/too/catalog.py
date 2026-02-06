import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

class SkyCatalogHistory:
    HISTORY_FIELDS = ["objname", "ra", "dec", "fov_ra", "fov_dec", "cat_type"]

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
        self.verbose = verbose
        self.objname = objname
        self.ra = ra
        self.dec = dec
        self.fov_ra = fov_ra
        self.fov_dec = fov_dec
        self.catalog_type = catalog_type
        self.overlapped_fraction = overlapped_fraction
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
        try:
            data = self._query(catalog_name=catalog_name, verbose=verbose)
        except Exception:
            raise ValueError(f"{self.objname} does not exist in {catalog_name} catalog")

        self.data = None
        if data:
            formatted_data = format_func(data)
            self.data = self._filter_sources_in_fov(formatted_data)

    def _get_GAIAXP(self, objname=None, ra=None, dec=None, fov_ra=1.3, fov_dec=0.9, verbose=False):
        raise ValueError(
            "GAIAXP catalog must be queried from external source. Use GAIA, APASS, PS1, SDSS, or SMSS instead."
        )

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

    def _update_history(self):
        self.history = SkyCatalogHistory(
            objname=self.objname,
            ra=self.ra,
            dec=self.dec,
            fov_ra=self.fov_ra,
            fov_dec=self.fov_dec,
            cat_type=self.catalog_type,
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
                coord = self._query_coord_from_objname(objname=objname)
                self.ra = coord.ra.deg
                self.dec = coord.dec.deg
            except:
                raise ValueError(f"Failed to query coordinates for {objname}")

        if objname is None and ra is not None and dec is not None:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
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


def _get_ra_dec_columns(table):
    """Return (ra_col, dec_col) for a table, or (None, None) if not found."""
    ra_col = dec_col = None
    for col in table.colnames:
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
    return ra_col, dec_col


def query_catalogs(sky_coord, size=30, unit=u.arcsec, catalogs_to_query=["GAIA"], catalog_mag_range=(10, 20)):
    catalog_sources_list = []

    fov = (size * unit).to(u.deg).value

    for cat_type in catalogs_to_query:
        try:
            sky_catalog = SkyCatalog(
                ra=sky_coord.ra.deg,
                dec=sky_coord.dec.deg,
                fov_ra=fov,
                fov_dec=fov,
                catalog_type=cat_type,
            )

            ref_sources, _ = sky_catalog.get_reference_sources(
                mag_lower=catalog_mag_range[0], mag_upper=catalog_mag_range[1]
            )

            if len(ref_sources) > 0:
                catalog_sources_list.append({"catalog": cat_type, "sources": ref_sources})
        except Exception as e:
            pass

    # Combine and deduplicate sources from all catalogs
    # Match sources by position (within 1 arcsec tolerance)
    if len(catalog_sources_list) > 0:
        from astropy.table import vstack

        # Collect all sources with their RA/Dec
        all_sources_coords = []
        all_sources_tables = []

        for cat_info in catalog_sources_list:
            catalog_sources = cat_info["sources"]
            ra_col, dec_col = _get_ra_dec_columns(catalog_sources)
            if ra_col is not None and dec_col is not None:
                coords = SkyCoord(ra=catalog_sources[ra_col], dec=catalog_sources[dec_col], unit="deg")
                all_sources_coords.append(coords)
                all_sources_tables.append(catalog_sources)

        if len(all_sources_coords) > 0:
            # Combine all coordinates
            # Convert to arrays and combine
            all_ra = []
            all_dec = []
            for coords in all_sources_coords:
                all_ra.extend(coords.ra.deg)
                all_dec.extend(coords.dec.deg)
            combined_coords = SkyCoord(ra=all_ra, dec=all_dec, unit="deg")
            combined_table = vstack(all_sources_tables)

            # Deduplicate sources within 1 arcsec tolerance
            tolerance = 1.0 * u.arcsec
            unique_mask = np.ones(len(combined_coords), dtype=bool)

            for i in range(len(combined_coords)):
                if unique_mask[i]:
                    # Find all sources within tolerance of this source
                    sep = combined_coords[i].separation(combined_coords)
                    matches = sep < tolerance
                    matches[i] = False  # Exclude self
                    if np.any(matches):
                        # Mark duplicates as False (keep first occurrence)
                        unique_mask[matches] = False

            unique_sources = combined_table[unique_mask]
            # Replace catalog_sources_list with a single unified list
            catalog_sources_list = [{"catalog": "UNIFIED", "sources": unique_sources}]

    return catalog_sources_list
