from typing import Iterable, Tuple, Sequence, Any
import numpy as np
from astropy.coordinates import SkyCoord, Distance, search_around_sky
from astropy.time import Time
from astropy.table import Table, hstack, MaskedColumn
import astropy.units as u
from astropy.units import Quantity
import operator


def match_two_catalogs(
    sci_tbl: Table,
    ref_tbl: Table,
    *,
    x0: str = "ALPHA_J2000",
    y0: str = "DELTA_J2000",
    x1: str | None = None,
    y1: str | None = None,
    radius: float | Quantity = 1,
    join: str = "inner",
    correct_pm: bool = False,
    obs_time: Time | None = None,
    pm_keys: dict = dict(pmra="pmra", pmdec="pmdec", parallax="parallax", ref_epoch=2016.0),
) -> Table:
    """
    Cross-match two catalogues on the sky and (optionally) apply proper-motion
    correction to the *reference* catalogue before matching.
    The coordinates are assumed to be Equatorial (RA, Dec) in degrees.

    Parameters
    ----------
    sci_tbl, ref_tbl
        `astropy.table.Table` objects to be matched.
    x0, y0, x1, y1
        Column names that contain right ascension and declination *in degrees*.
        If `x1` / `y1` are omitted they default to `x0` / `y0`.
    radius
        Maximum separation for a match.  A bare ``float`` is interpreted
        in **arcseconds**; a `~astropy.units.Quantity` may carry any angle unit.
    how : {'inner', 'left', 'outer'}, optional
        Join strategy:

        * ``'inner'`` - return only matched rows (default)
        * ``'left'``  - return every row of *sci_tbl* and mask unmatched
                        entries from ref_tbl
        * ``'right'``  - return every row of *ref_tbl* and mask unmatched
                        entries from sci_tbl
        * ``'outer'`` - return all matched rows *and* all unmatched from both
                        catalogues
    correct_pm
        If *True*, propagate stars in the *reference* catalogue from their
        catalogued epoch (`pm_info['ref_epoch']`, default 2016.0 TDB) to
        `obs_time` using their proper motion and parallax.
    obs_time
        Observation time of the *science* catalogue as an `~astropy.time.Time`
        instance.  Required when ``correct_pm=True``.
    pm_keys
        Mapping that defines column names for the kinematic quantities and the
        reference epoch::

            dict(pmra="pmra", pmdec="pmdec",
                 parallax="parallax", ref_epoch=2016.0)

        You can pass additional keys (e.g. ``"rv"``) without hurting anything.

    Returns
    -------
    Table
        A merged `~astropy.table.Table` with one row per successful positional
        match.  Columns from the smaller (“driver”) catalogue keep their
        original names; duplicate names coming from the second catalogue receive
        the suffix ``"_ref"`` or ``"_sci"`` depending on which side they came
        from.

    Notes
    -----
    * Rows whose kinematic information is incomplete are **kept** but **not**
      epoch-propagated.  They will still match if their uncorrected position
      lies within *radius*.
    * Negative or zero parallaxes are converted into ``NaN`` distances via
      ``Distance(parallax, allow_negative=True)``; these rows are likewise
      retained but un-moved.
    """

    if join not in {"inner", "left", "right", "outer"}:
        raise ValueError("join must be 'inner', 'left' , 'right',  or 'outer'")

    if x1 is None:
        x1 = x0
    if y1 is None:
        y1 = y0

    coord_sci = SkyCoord(sci_tbl[x0], sci_tbl[y0], unit="deg", copy=False)
    coord_ref = SkyCoord(ref_tbl[x1], ref_tbl[y1], unit="deg", copy=False)

    # update coord_ref if correct_pm is True
    if correct_pm:
        from astropy.coordinates import Distance

        if obs_time is None:
            raise ValueError("obs_time must be provided if correct_pm is True")

        # Vectorised columns with units
        pm_ra = ref_tbl[pm_keys["pmra"]] * u.mas / u.yr
        pm_dec = ref_tbl[pm_keys["pmdec"]] * u.mas / u.yr
        good = np.isfinite(pm_ra) & np.isfinite(pm_dec)

        if pm_keys.get("parallax") is not None:
            parallax = ref_tbl[pm_keys["parallax"]] * u.mas
            dist = Distance(parallax=parallax, allow_negative=True)
            good &= np.isfinite(dist)
        else:
            dist = None  # let SkyCoord use its default (no distance)

        if np.any(good):
            kwargs = dict(
                ra=ref_tbl[x1][good] * u.deg,
                dec=ref_tbl[y1][good] * u.deg,
                pm_ra_cosdec=pm_ra[good],
                pm_dec=pm_dec[good],
                obstime=Time(pm_keys["ref_epoch"], format="jyear"),
            )
            if dist is not None:
                kwargs["distance"] = dist[good]
            moved = SkyCoord(**kwargs).apply_space_motion(new_obstime=obs_time)

            coord_ref.ra[good] = moved.ra
            coord_ref.dec[good] = moved.dec
            coord_ref._sky_coord_frame.cache.clear()

    if join == "left":
        coord0, coord1 = coord_sci, coord_ref
        tbl0, tbl1 = sci_tbl, ref_tbl
        tag1 = "ref"
    elif join == "right":
        coord0, coord1 = coord_ref, coord_sci
        tbl0, tbl1 = ref_tbl, sci_tbl
        tag1 = "sci"
    elif join == "inner":
        if len(coord_sci) > len(coord_ref):
            coord0, coord1 = coord_ref, coord_sci
            tbl0, tbl1 = ref_tbl, sci_tbl
            tag1 = "sci"
        else:
            coord0, coord1 = coord_sci, coord_ref
            tbl0, tbl1 = sci_tbl, ref_tbl
            tag1 = "ref"
    else:
        pass

    if join in {"inner", "left", "right"}:
        idx, sep2d, _ = coord0.match_to_catalog_sky(coord1)
        rtol = radius * u.arcsec if not isinstance(radius, Quantity) else radius
        matched = sep2d < rtol

        if join == "left" or join == "right":
            out = tbl0.copy()
            sliced = tbl1[idx]

            dupes = set(out.colnames) & set(sliced.colnames)
            for name in dupes:
                sliced.rename_column(name, f"{name}_{tag1}")

            for name in sliced.colnames:
                out[name] = MaskedColumn(sliced[name].data, mask=~matched)

            out["separation"] = MaskedColumn(sep2d.arcsec, mask=~matched)
            return out

        else:
            m0 = tbl0[matched]
            m1 = tbl1[idx[matched]]

            dupes = set(m0.colnames) & set(m1.colnames)
            for name in dupes:
                m1.rename_column(name, f"{name}_{tag1}")

            merged = hstack([m0, m1], join_type="exact")
            merged["separation"] = sep2d[matched].arcsec
            return merged

    elif join == "outer":
        sep_rad = radius * u.arcsec if not isinstance(radius, Quantity) else radius
        idx_s, idx_r, seps, _ = search_around_sky(coord_sci, coord_ref, sep_rad)

        order = np.lexsort([seps.arcsec, idx_s])
        idx_s_s, idx_r_s = idx_s[order], idx_r[order]
        first = np.unique(idx_s_s, return_index=True)[1]
        sci_idx = idx_s_s[first]
        ref_idx = idx_r_s[first]

        used_s = set(sci_idx)
        used_r = set(ref_idx)

        rows = []
        for isci, iref in zip(sci_idx, ref_idx):
            row = {}
            for n in sci_tbl.colnames:
                row[f"sci_{n}"] = sci_tbl[n][isci]
            for n in ref_tbl.colnames:
                row[f"ref_{n}"] = ref_tbl[n][iref]
            row["separation"] = seps[(idx_s == isci) & (idx_r == iref)][0].arcsec
            rows.append(row)

        for isci in range(len(sci_tbl)):
            if isci not in used_s:
                row = {}
                for n in sci_tbl.colnames:
                    row[f"sci_{n}"] = sci_tbl[n][isci]
                for n in ref_tbl.colnames:
                    row[f"ref_{n}"] = np.ma.masked
                row["separation"] = np.ma.masked
                rows.append(row)

        for iref in range(len(ref_tbl)):
            if iref not in used_r:
                row = {}
                for n in sci_tbl.colnames:
                    row[f"sci_{n}"] = np.ma.masked
                for n in ref_tbl.colnames:
                    row[f"ref_{n}"] = ref_tbl[n][iref]
                row["separation"] = np.ma.masked
                rows.append(row)

        return Table(rows)

    else:
        raise ValueError(f"Unknown join method: {join}")


Condition = Tuple[str, Any, str]  # (column, value, method)

_OPS = {
    # greater-than
    "lower": operator.gt,  ">":  operator.gt,
    # greater-or-equal
    ">=":   operator.ge,
    # less-than
    "upper": operator.lt,  "<":  operator.lt,
    # less-or-equal
    "<=":   operator.le,
    # equal
    "equal": operator.eq,  "==": operator.eq,  "=": operator.eq,
}  # fmt: skip


def build_condition_mask(table, conditions: Iterable[Condition]) -> np.ndarray:
    """
    Return a boolean mask that is True only for rows satisfying *all* conditions.

    Parameters
    ----------
    table : Table | DataFrame | structured ndarray
        Object supporting ``table[col]`` column access.
    conditions : iterable of (key, method, value)
        method can be any alias listed in _METHOD_MAP (case-insensitive).

    Returns
    -------
    numpy.ndarray[bool]  shape (len(table),)
    """

    # later incorporate numexpr

    conditions = _parse_conditions(conditions)
    mask = np.ones(len(table), dtype=bool)

    for key, method, value in conditions:
        m = method.strip().lower()
        if m not in _OPS:
            raise ValueError(f"Unknown method '{method}'. Allowed: {', '.join(_OPS)}")
        mask &= _OPS[m](table[key], value)

    return mask


def _parse_conditions(conditions: Sequence[Any]) -> Iterable[Condition]:
    """
    Turn *raw* into an iterable of (key, op, value) tuples.

    *raw* may be:
      • already an iterable of 3-tuples
      • a flat 1-D list/array whose length is a multiple of 3
    """
    # Understand something like [(k, op, v), …]
    if conditions and isinstance(conditions[0], (tuple, list)) and len(conditions[0]) == 3:
        return conditions  # type: ignore[arg-type]

    # If not, try the “flat” form
    if len(conditions) % 3 != 0:
        raise ValueError("Flat conditions must have length divisible by 3 " "(key, op, value repeated).")
    it = iter(conditions)
    return list(zip(it, it, it))


def filter_table(table: Table, conditions: Iterable[Condition]) -> Table:
    mask = build_condition_mask(table, conditions)
    return table[mask]


def add_id_column(table: Table) -> Table:
    """
    Add an 'id' column as the first column of an Astropy Table.
    The 'id' values are 0-based row indices.

    Parameters
    ----------
    table : astropy.table.Table
        The input table.

    Returns
    -------
    astropy.table.Table
        A new table with 'id' as the first column.
    """
    # Create an index array
    ids = np.arange(len(table))

    # Insert the column at position 0
    table.add_column(ids, name="id", index=0)

    return table
