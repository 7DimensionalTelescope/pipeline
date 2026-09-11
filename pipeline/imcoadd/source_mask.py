"""Calibrated per-source mask-radius law.

For every source the flux branch solves ``FLUX_AUTO * profile(R) = skysig / K_THRESH``.
For an extended source the final semi-major radius is::

    max(R_flux, GALAXY_SAFETY_MARGIN * f(x1, x2) * max(KRON_RADIUS, 1) * A_IMAGE)

where ``f`` is the fitted two-ratio dilation returned by :func:`galaxy_dilation`.
Point sources use ``R_flux`` alone.  The fitted constants below come from the external calibration
artifacts named beside each block; keep their provenance here with the executable formula.
"""

import threading

import numpy as np

# double Moffat profile fit for point sources; coeffs predetermined from external calibration.
ALPHA_SLOPE = 1.5067571888356017
ALPHA_INTERCEPT = 0.7765939512685198
BETA_CORE = 2.0380495854589302
BETA_WING = 0.75
F_WING = 0.14267918102994284
K_ALPHA = 1.0
RNORM_PX = 1000.0

K_THRESH = 32  # 64  # TODO: tune

# The clips are the predictors' 1st and 99th percentiles.  The safety margin is exp(-p10) of
# ln(R_fit / R_opt), so the Kron branch reaches or exceeds the measured R_opt for 90% of the
# calibration galaxies rather than being merely median-unbiased.
DILATION_COEFFS = (
    1.2863216529093424,
    -0.9227011965816579,
    0.5494693078776561,
    0.0688760733182969,
    0.0351545653431481,
)
CONCENTRATION_CLIP = (1.035181160986305, 2.8625749365907764)
OUTER_RATIO_CLIP = (1.3463635627688317, 1.9722462032777797)
GALAXY_SAFETY_MARGIN = 1.47

RMIN, RMAX, NGRID = 0.02, 2e4, 8192
N_ALPHA, ALPHA_GRID = 2048, (0.5, 500.0)

FLUX_BRANCH_COLUMNS = ("AWIN_IMAGE", "FLUX_AUTO")
DILATION_COLUMNS = ("FWHM_IMAGE", "FLUX_RADIUS_50", "FLUX_RADIUS_80")

_tabulation = None
_tabulation_lock = threading.Lock()


def _component_integral(alpha, beta):
    """Flux of (1+(r/alpha)^2)^-beta inside RNORM_PX."""
    alpha = np.asarray(alpha, float)
    x = np.log1p((RNORM_PX / alpha) ** 2)
    return np.pi * alpha**2 * (-np.expm1((1.0 - beta) * x)) / (beta - 1.0)


def _component(r, alpha, beta):
    return (1.0 + (np.asarray(r, float) / alpha) ** 2) ** (-beta) / _component_integral(alpha, beta)


def profile(r, alpha_core):
    """Double-Moffat surface brightness at radius r, in ADU/px per unit FLUX_AUTO."""
    core = _component(r, alpha_core, BETA_CORE)
    wing = _component(r, K_ALPHA * np.asarray(alpha_core, float), BETA_WING)
    return (1.0 - F_WING) * core + F_WING * wing


def core_width(awin):
    """Moffat core width from AWIN_IMAGE, floored at the tabulation's lowest node."""
    return np.maximum(ALPHA_SLOPE * np.asarray(awin, float) + ALPHA_INTERCEPT, ALPHA_GRID[0])


def tabulate():
    """(rgrid, agrid, pgrid) for the radius inversion, built once per process (128 MiB, ~0.6 s)."""
    global _tabulation
    if _tabulation is None:
        with _tabulation_lock:
            if _tabulation is None:
                rgrid = np.geomspace(RMIN, RMAX, NGRID)
                agrid = np.geomspace(*ALPHA_GRID, N_ALPHA)
                pgrid = np.array([profile(rgrid, a) for a in agrid])
                if not np.all(np.diff(pgrid, axis=1) < 0):
                    raise ValueError("p(r) must be monotonic in r for the radius inversion")
                _tabulation = (rgrid, agrid, pgrid)
    return _tabulation


def _invert(target, alpha):
    """Radius where profile(R; alpha) = target, log-interpolated between the two bracketing alpha nodes."""
    rgrid, agrid, pgrid = tabulate()
    target = np.atleast_1d(np.asarray(target, float))
    alpha = np.broadcast_to(np.atleast_1d(np.asarray(alpha, float)), target.shape)
    hi = np.clip(np.searchsorted(agrid, alpha), 1, len(agrid) - 1)
    lo = hi - 1
    log_nodes = np.log(agrid)
    weight = np.clip((np.log(alpha) - log_nodes[lo]) / (log_nodes[hi] - log_nodes[lo]), 0.0, 1.0)

    def at(node):
        out = np.empty_like(target)
        for j in np.unique(node):
            selected = node == j
            # p decreases with r, so both tabulations are reversed for np.interp
            out[selected] = np.interp(target[selected], pgrid[j][::-1], rgrid[::-1], left=rgrid[-1], right=0.0)
        return out

    return (1.0 - weight) * at(lo) + weight * at(hi)


def optimized_radius_for_point_sources_with_known_profile(flux, awin, skysig):
    """Semi-major axis where the source's profile falls to the supplied sky noise / K_THRESH."""
    flux = np.maximum(np.asarray(flux, float), 1.0)
    return _invert(float(skysig) / K_THRESH / flux, core_width(awin))


def galaxy_dilation(cat, fallback):
    """Fitted (pre-safety-margin) dilation on the MAG_AUTO semi-major axis."""
    r50 = np.asarray(cat["FLUX_RADIUS_50"], float)
    usable = np.isfinite(r50) & (r50 > 0)
    r50 = np.where(usable, r50, 1.0)
    x1 = np.clip(np.asarray(cat["FWHM_IMAGE"], float) / r50, *CONCENTRATION_CLIP)
    x2 = np.clip(np.asarray(cat["FLUX_RADIUS_80"], float) / r50, *OUTER_RATIO_CLIP)
    a, b1, b2, c1, c2 = DILATION_COEFFS
    f = np.exp(a + b1 * x1 + b2 * x2 + c1 * x1**2 + c2 * x2**2)
    return np.where(usable & np.isfinite(f), f, fallback)


def dilated_kron_based_radius_for_galaxies(kron_axis, dilation):
    """Safety-margined extended-source radius from the MAG_AUTO axis and fitted dilation."""
    return GALAXY_SAFETY_MARGIN * np.asarray(dilation, float) * np.asarray(kron_axis, float)


def has_columns(cat, columns):
    return all(name in cat.colnames for name in columns)


def mask_semi_major(cat, skysig, star_scale, galaxy_scale, class_star_cut, logger=None):
    """Final mask semi-major axis in the catalog frame's pixels.

    Point: ``R_flux``.  Extended: ``max(R_flux, margin * f * Kron axis)``.  Without a usable
    flux branch, the point-source radius falls back to ``star_scale * Kron axis`` while the
    extended-source radius remains the safety-margined fitted (or ``galaxy_scale`` fallback)
    Kron branch.
    """
    kron_axis = np.clip(np.asarray(cat["KRON_RADIUS"], float), 1.0, None) * np.asarray(cat["A_IMAGE"], float)
    is_extended = np.asarray(cat["CLASS_STAR"], float) < class_star_cut

    if has_columns(cat, DILATION_COLUMNS):
        dilation = galaxy_dilation(cat, galaxy_scale)
    else:
        dilation = np.full(len(cat), galaxy_scale, dtype=float)
        if logger is not None:
            logger.debug(
                f"Catalog lacks {DILATION_COLUMNS}; extended sources use "
                f"the safety-margined constant x{galaxy_scale * GALAXY_SAFETY_MARGIN:g}"
            )

    dilated_kron_radius = dilated_kron_based_radius_for_galaxies(kron_axis, dilation)

    if skysig and has_columns(cat, FLUX_BRANCH_COLUMNS):
        r_opt = optimized_radius_for_point_sources_with_known_profile(cat["FLUX_AUTO"], cat["AWIN_IMAGE"], skysig)
        return np.where(is_extended, np.maximum(r_opt, dilated_kron_radius), r_opt)

    if logger is not None:
        logger.debug(f"No sky-noise value or {FLUX_BRANCH_COLUMNS}; mask radius from the Kron ellipse alone")
    return np.where(is_extended, dilated_kron_radius, star_scale * kron_axis)
