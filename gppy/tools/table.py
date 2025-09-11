from __future__ import annotations
from typing import Iterable, Tuple, Sequence, Any, List, Dict, Optional
import numpy as np
from astropy.coordinates import SkyCoord, Distance, search_around_sky
from astropy.time import Time
from astropy.table import Table, hstack, MaskedColumn
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord, Angle
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
        idx_s, idx_r, seps, _ = search_around_sky(coord_sci, coord_ref, sep_rad)  # avoid match_to_catalog_sky for outer

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


###############################################################################


def match_multi_catalogs(
    cats: Sequence[Table],
    *,
    ra_keys: Sequence[str] = ("ALPHA_J2000",),
    dec_keys: Sequence[str] = ("DELTA_J2000",),
    radius: float | u.Quantity = 1.0,  # arcsec if float
    join: str = "inner",  # {'inner','left','outer'}
    cat_names: Optional[Sequence[str]] = None,  # nice labels for suffixing
    pivot: int | None = 0,  # index of pivot catalog used for tie-breaks and separations
) -> Table:
    """
    Joint sky crossmatch of N catalogs using a friends-of-friends graph (single-linkage)
    within *radius*. Returns one row per matched group (component), with at most one
    row selected from each catalog.

    Strategy
    --------
    1) Build SkyCoord for each catalog.
    2) Add edges for all cross-catalog pairs within *radius* (search_around_sky).
    3) Take connected components; for each component select ≤1 row per catalog:
       - choose a pivot row (prefer pivot catalog if present, else the lowest-index present);
       - in each other catalog, pick the row closest to the pivot.
    4) Emit rows per component according to *join*:
       - 'inner': components that include all catalogs
       - 'left' : components that include the pivot catalog (default: cats[0])
       - 'outer': any component with ≥1 catalog

    Notes
    -----
    * *radius* is in arcsec if float; any angle Quantity works.
    * Duplicate column names are suffixed with f"_{name}" where *name* is taken
      from *cat_names* (or f"cat{i}" if omitted), matching your 2-way behavior.
    * Adds separation columns (to the pivot) named 'sep_arcsec_<name>'.
    """
    # --- normalize inputs ---
    if isinstance(radius, (int, float)):
        radius = float(radius) * u.arcsec
    if cat_names is None:
        cat_names = [f"cat{i}" for i in range(len(cats))]
    if len(cats) != len(cat_names):
        raise ValueError("len(cat_names) must match number of catalogs")
    if pivot is None:
        pivot = 0
    if not (0 <= pivot < len(cats)):
        raise ValueError("pivot must be a valid catalog index")

    # Resolve RA/Dec keys for each catalog (allow a single key applied to all)
    def key_for(keys, i):
        return keys[i] if i < len(keys) else keys[-1]

    ra_cols = [key_for(ra_keys, i) for i in range(len(cats))]
    dec_cols = [key_for(dec_keys, i) for i in range(len(cats))]

    # --- SkyCoord per catalog ---
    coords = []
    for i, (tbl, ra, dec) in enumerate(zip(cats, ra_cols, dec_cols)):
        if ra not in tbl.colnames or dec not in tbl.colnames:
            raise KeyError(f"Catalog {i} missing RA/Dec columns: {ra}, {dec}")
        coords.append(SkyCoord(ra=tbl[ra] * u.deg, dec=tbl[dec] * u.deg, frame="icrs"))

    # --- Union-Find (Disjoint Set) over all rows of all catalogs ---
    # Global node id = (catalog_index, row_index) -> map to integer id
    offsets = [0]
    for t in cats:
        offsets.append(offsets[-1] + len(t))
    total = offsets[-1]

    parent = list(range(total))
    rank = [0] * total

    def gid(cat_idx: int, row_idx: int) -> int:
        return offsets[cat_idx] + row_idx

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[rb] < rank[ra]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # --- Add edges for every cross-catalog pair within radius ---
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            idx_i, idx_j, sep, _ = coords[i].search_around_sky(coords[j], radius)
            # idx_i are indices into coords[j]; idx_j into coords[i] (yes, astropy returns (j,i) order)
            # To keep it intuitive, re-map correctly:
            # search_around_sky(A, B) returns indices (idxA, idxB) such that sep(A[idxA], B[idxB]) < radius
            # We called with coords[j] as first arg and coords[i] as second.
            a_idx = idx_i  # in catalog j
            b_idx = idx_j  # in catalog i
            for rj, ri in zip(a_idx, b_idx):
                union(gid(j, int(rj)), gid(i, int(ri)))

    # --- Collect components -> {root: {cat: [row_indices...]}} ---
    comps: Dict[int, Dict[int, list]] = {}
    for cat_i, tbl in enumerate(cats):
        for row_i in range(len(tbl)):
            node = gid(cat_i, row_i)
            root = find(node)
            d = comps.setdefault(root, {})
            d.setdefault(cat_i, []).append(row_i)

    # --- Helper: choose one row per catalog in a component ---
    def choose_representatives(comp: Dict[int, list]) -> Tuple[int, Dict[int, int]]:
        """Return (pivot_cat, {cat: row_idx}) for a component."""
        cats_present = sorted(comp.keys())
        # choose pivot catalog (prefer requested pivot if present, else smallest present)
        piv_cat = pivot if pivot in comp else cats_present[0]
        # choose a pivot row (if multiple in piv_cat, pick the one with highest local density / or arbitrary closest to median)
        # Simpler heuristic: if multiple, choose the row with minimal average separation to all other catalogs' nearest
        piv_candidates = comp[piv_cat]
        if len(piv_candidates) == 1 or len(cats_present) == 1:
            piv_row = piv_candidates[0]
        else:
            # score each candidate by sum of nearest separations to other catalogs present
            best_score, piv_row = None, piv_candidates[0]
            for pr in piv_candidates:
                c_pr = coords[piv_cat][pr]
                score = 0.0
                for other in cats_present:
                    if other == piv_cat:
                        continue
                    # nearest in 'other'
                    sep = c_pr.separation(coords[other][comp[other]]).arcsec
                    score += float(sep.min()) if len(sep) else 1e9
                if best_score is None or score < best_score:
                    best_score, piv_row = score, pr

        chosen = {piv_cat: piv_row}
        # now pick closest per other catalog
        c_piv = coords[piv_cat][piv_row]
        for other in cats_present:
            if other == piv_cat:
                continue
            cand_rows = comp[other]
            seps = c_piv.separation(coords[other][cand_rows]).arcsec
            best_j = int(seps.argmin())
            chosen[other] = cand_rows[best_j]
        return piv_cat, chosen

    # --- Build output rows according to join policy ---
    rows_per_component: List[Dict[Tuple[int, str], Any]] = []
    sep_names = []  # to retain order for later column creation
    for root, comp in comps.items():
        cats_present = set(comp.keys())

        if join == "inner":
            if len(cats_present) < len(cats):
                continue
        elif join == "left":
            if pivot not in cats_present:
                continue
        elif join == "outer":
            pass
        else:
            raise ValueError("join must be one of {'inner','left','outer'}")

        piv_cat, chosen = choose_representatives(comp)

        # Construct a dict mapping (cat_i, colname) -> value
        row_dict: Dict[Tuple[int, str], Any] = {}
        # Also keep separations to pivot
        for cat_i, tbl in enumerate(cats):
            name = cat_names[cat_i]
            if cat_i in chosen:
                ri = chosen[cat_i]
                for col in tbl.colnames:
                    row_dict[(cat_i, col)] = tbl[col][ri]
            else:
                # unmatched: create masked entries later
                for col in tbl.colnames:
                    row_dict[(cat_i, col)] = None  # will become masked if column supports it

        # separations to pivot (arcsec)
        c_piv = coords[piv_cat][chosen[piv_cat]]
        for cat_i in range(len(cats)):
            label = f"sep_arcsec_{cat_names[cat_i]}"
            if cat_i in chosen:
                c_i = coords[cat_i][chosen[cat_i]]
                sep_arcsec = c_piv.separation(c_i).arcsec
                row_dict[(-1, label)] = sep_arcsec
            else:
                row_dict[(-1, label)] = None
            if label not in sep_names:
                sep_names.append(label)

        rows_per_component.append(row_dict)

    # --- Assemble merged Table with suffixed columns per catalog ---
    # Determine output columns order & masks
    out_cols: List[Tuple[Tuple[int, str], str]] = []  # ((cat_i, colname), out_colname)
    # build per-catalog column mapping with suffix if duplicate
    seen = {}
    for cat_i, tbl in enumerate(cats):
        name = cat_names[cat_i]
        for col in tbl.colnames:
            base = col
            out = base if (base not in seen) else f"{base}_{name}"
            seen[out] = True
            out_cols.append(((cat_i, col), out))
    # add separations at the end
    out_cols.extend([((-1, sn), sn) for sn in sep_names])

    # create astropy Table
    out = Table()
    for key, out_name in out_cols:
        data = [row.get(key, None) for row in rows_per_component]
        # make masked column if there are Nones
        if any(v is None for v in data):
            col = MaskedColumn(data=data, name=out_name, mask=[v is None for v in data])
        else:
            col = data
        out[out_name] = col

    return out


###############################################################################

###############################################################################

###############################################################################


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
