"""Shifted-overscan defect detection.

A "shifted overscan" defect is a band of columns whose pixel values sit at
the camera bias level (~`BIAS_LEVEL` ADU), produced when the readout starts
shifted by ~20-30 columns and the overscan region ends up imaged inside the
frame instead of at the edge. It can affect science, dark and flat frames.

This file holds three generations of the metric:

* v1 (``shifted_overscan_score``): plateau-dip + paired-cliff heuristics on
  a running-median baseline. Currently in production. Returns negative ->
  shift detected.

* v2 (``shifted_overscan_score_v2``): bias-anchored window-min over column
  elevation; cheaper than v1 and zero false positives in our tests, but
  bails on shallow darks.

* v3 (``shifted_overscan_score_v3``): bias-anchored matched filter / LRT.
  ~Same compute as v2, returns POSITIVE -> shift detected (sign matches the
  log-likelihood ratio convention). Robust to shallow darks.
"""

from typing import Literal
import numpy as np

from ..const import BIAS_LEVEL


# --------------------------------------------------------------------------
# v1: plateau-dip + paired-cliff (currently in production, "lower is worse")
# --------------------------------------------------------------------------

SHIFTED_SCORE_THRESHOLD = -5.0


def combined_shifted_score(scores: list[float]) -> float:
    """Worst (lowest) of a list of v1 scores. Used for coadded master frames
    where the individual frames' scores are no longer separately accessible."""
    return float(min(scores)) if scores else 0.0


def shifted_overscan_score(
    image: np.ndarray,
    row_frac: float = 0.6,
    profile_max_rows: int = 384,
    win_baseline: int = 801,
    win_smooth: int = 8,
    plateau_win: int = 16,
    plateau_range_max: float = 5.0,
    plateau_qlo: float = 0.1,
    plateau_qhi: float = 0.9,
    plateau_min_run: int = 5,
    pair_dmin: int = 5,
    pair_dmax: int = 50,
    pair_sigma_min: float = 8.0,
    dip_abs_min: float = 10.0,
    noise_floor: float = 1.0,
) -> float:
    from numpy.lib.stride_tricks import sliding_window_view

    # Use the middle row_frac of rows to reduce influence of row-edge artifacts
    # and source concentrations at detector edges on the column-mean profile.
    h = image.shape[0]
    r0 = int(h * (1.0 - row_frac) / 2.0)
    r1 = h - r0
    band = image[r0:r1]
    nr = band.shape[0]
    step = max(1, (nr + profile_max_rows - 1) // profile_max_rows)
    sample = band[::step]
    # Per-column mean is biased high in crowded fields; median tracks the sky
    # floor. SExtractor's mode proxy (2.5*median - 1.5*mean) uses both, with no
    # extra passes beyond mean+median on the decimated band. Clip to [lo, hi]
    # with lo=min(mean,median), hi=max(mean,median) so heavily skewed columns
    # do not produce negative or extrapolated values.
    col_mean = sample.mean(axis=0).astype(np.float32, copy=False)
    col_med = np.median(sample, axis=0).astype(np.float32, copy=False)
    mode = 2.5 * col_med - 1.5 * col_mean
    lo = np.minimum(col_mean, col_med)
    hi = np.maximum(col_mean, col_med)
    col = np.clip(mode, lo, hi).astype(np.float32, copy=False)
    bulk_med = float(np.median(col))

    # Reflect padding so an edge-anchored defect is compared to interior columns
    # rather than to a replication of itself.
    pb = win_baseline // 2
    baseline = np.median(sliding_window_view(np.pad(col, pb, mode="reflect"), win_baseline), axis=-1)
    resid = col - baseline
    mad = float(np.median(np.abs(resid - np.median(resid))))
    # 1 ADU floor: col is a mean over O(100-400) rows so real column-to-column
    # scatter is already <=1 ADU even on noisy frames, while genuine shifted-
    # overscan defects are tens of ADU. Without it, quantization-dominated
    # short exposures collapse sigma to ~0 and inflate the score by 1e6.
    col_step = float(np.median(np.abs(np.diff(col))))
    sigma = max(1.4826 * mad, col_step, noise_floor)

    # Plateau-dip path: catches edge-flush and wide interior dips where
    # col-mean stays flat to within plateau_range_max.
    pp = plateau_win // 2
    win = sliding_window_view(np.pad(col, (pp, plateau_win - 1 - pp), mode="edge"), plateau_win)
    local_range = np.quantile(win, plateau_qhi, axis=-1) - np.quantile(win, plateau_qlo, axis=-1)
    is_plateau = local_range < plateau_range_max
    if is_plateau.any():
        p_int = is_plateau.astype(np.int8)
        d_run = np.diff(np.concatenate([[0], p_int, [0]]))
        starts = np.where(d_run == 1)[0]
        ends = np.where(d_run == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) < plateau_min_run:
                is_plateau[s:e] = False
    if is_plateau.sum() >= plateau_min_run:
        dip = np.where(is_plateau, -resid, 0.0).astype(np.float32, copy=False)
        ps = win_smooth // 2
        smooth = sliding_window_view(np.pad(dip, (ps, win_smooth - 1 - ps), mode="edge"), win_smooth).mean(axis=-1)
        plateau_score = -float(smooth.max() / sigma)
    else:
        plateau_score = 0.0

    # Paired-cliff path: catches narrow interior dips by requiring a sharp
    # drop followed by a sharp rise within pair_dmax cols, both > pair_sigma_min
    # x median|diff|, with the plateau between sitting > dip_abs_min below the
    # global median (kills hot-column pair artifacts).
    dcol = np.diff(col)
    med_abs = max(float(np.median(np.abs(dcol))), noise_floor)
    cliff_thr = pair_sigma_min * med_abs
    bulk_limit = bulk_med - dip_abs_min
    col_right = col[1:]
    best_pair = 0.0
    for dd in range(pair_dmin, pair_dmax + 1):
        fd = dcol[:-dd]
        tr = dcol[dd:]
        right_after_drop = col_right[:-dd]
        right_before_rise = col[dd:-1]
        msk = (fd < -cliff_thr) & (tr > cliff_thr) & (right_after_drop < bulk_limit) & (right_before_rise < bulk_limit)
        if msk.any():
            mag = (-fd) * tr
            v = float(mag[msk].max())
            if v > best_pair:
                best_pair = v
    cliff_score = -float(np.sqrt(best_pair) / med_abs) if best_pair > 0 else 0.0

    return min(plateau_score, cliff_score)


def check_shifted_overscan(
    image: np.ndarray,
    threshold: float = SHIFTED_SCORE_THRESHOLD,
    **kwargs,
) -> tuple[bool, float]:
    score = shifted_overscan_score(image, **kwargs)
    return (score < threshold), score


def explain_shifted_overscan_score(
    image: np.ndarray,
    row_frac: float = 0.6,
    profile_max_rows: int = 384,
    win_baseline: int = 801,
    win_smooth: int = 8,
    plateau_win: int = 16,
    plateau_range_max: float = 5.0,
    plateau_qlo: float = 0.1,
    plateau_qhi: float = 0.9,
    plateau_min_run: int = 5,
    pair_dmin: int = 5,
    pair_dmax: int = 50,
    pair_sigma_min: float = 8.0,
    dip_abs_min: float = 10.0,
    noise_floor: float = 1.0,
) -> dict:
    # Bit-equivalent replay of `shifted_overscan_score` returning every
    # intermediate array (col profile, baseline, plateau-dip and paired-cliff
    # path internals, score).
    from numpy.lib.stride_tricks import sliding_window_view

    h = image.shape[0]
    r0 = int(h * (1.0 - row_frac) / 2.0)
    r1 = h - r0
    band = image[r0:r1]
    nr = band.shape[0]
    step = max(1, (nr + profile_max_rows - 1) // profile_max_rows)
    sample = band[::step]
    col_mean = sample.mean(axis=0).astype(np.float32, copy=False)
    col_med = np.median(sample, axis=0).astype(np.float32, copy=False)
    mode = 2.5 * col_med - 1.5 * col_mean
    lo = np.minimum(col_mean, col_med)
    hi = np.maximum(col_mean, col_med)
    col = np.clip(mode, lo, hi).astype(np.float32, copy=False)
    bulk_med = float(np.median(col))

    pb = win_baseline // 2
    baseline = np.median(sliding_window_view(np.pad(col, pb, mode="reflect"), win_baseline), axis=-1)
    resid = col - baseline
    mad = float(np.median(np.abs(resid - np.median(resid))))
    col_step = float(np.median(np.abs(np.diff(col))))
    sigma = max(1.4826 * mad, col_step, noise_floor)

    # Plateau-dip path
    pp = plateau_win // 2
    win = sliding_window_view(np.pad(col, (pp, plateau_win - 1 - pp), mode="edge"), plateau_win)
    local_range = np.quantile(win, plateau_qhi, axis=-1) - np.quantile(win, plateau_qlo, axis=-1)
    is_plateau = local_range < plateau_range_max
    if is_plateau.any():
        p_int = is_plateau.astype(np.int8)
        d_run = np.diff(np.concatenate([[0], p_int, [0]]))
        starts = np.where(d_run == 1)[0]
        ends = np.where(d_run == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) < plateau_min_run:
                is_plateau[s:e] = False
    plateau_argmax_col = -1
    if is_plateau.sum() >= plateau_min_run:
        dip = np.where(is_plateau, -resid, 0.0).astype(np.float32, copy=False)
        ps = win_smooth // 2
        smooth = sliding_window_view(np.pad(dip, (ps, win_smooth - 1 - ps), mode="edge"), win_smooth).mean(axis=-1)
        plateau_score = -float(smooth.max() / sigma)
        plateau_argmax_col = int(np.argmax(smooth))
    else:
        dip = np.zeros_like(resid, dtype=np.float32)
        smooth = np.zeros_like(resid, dtype=np.float32)
        plateau_score = 0.0

    # Paired-cliff path
    dcol = np.diff(col)
    med_abs = max(float(np.median(np.abs(dcol))), noise_floor)
    cliff_thr = pair_sigma_min * med_abs
    bulk_limit = bulk_med - dip_abs_min
    col_right = col[1:]
    best_pair = 0.0
    best_dd = -1
    best_drop_idx = -1
    for dd in range(pair_dmin, pair_dmax + 1):
        fd = dcol[:-dd]
        tr = dcol[dd:]
        right_after_drop = col_right[:-dd]
        right_before_rise = col[dd:-1]
        msk = (fd < -cliff_thr) & (tr > cliff_thr) & (right_after_drop < bulk_limit) & (right_before_rise < bulk_limit)
        if msk.any():
            mag = (-fd) * tr
            mag_masked = np.where(msk, mag, 0.0)
            i_local = int(np.argmax(mag_masked))
            v = float(mag_masked[i_local])
            if v > best_pair:
                best_pair = v
                best_dd = dd
                best_drop_idx = i_local
    cliff_score = -float(np.sqrt(best_pair) / med_abs) if best_pair > 0 else 0.0
    if best_dd >= 0:
        cliff_drop_col = best_drop_idx + 1
        cliff_rise_col = best_drop_idx + best_dd
    else:
        cliff_drop_col = -1
        cliff_rise_col = -1

    return {
        "row_band": (r0, r1),
        "row_step": int(step),
        "col_mean": col_mean,
        "col_med": col_med,
        "col": col,
        "bulk_med": bulk_med,
        "baseline": baseline.astype(np.float32, copy=False),
        "resid": resid.astype(np.float32, copy=False),
        "sigma": float(sigma),
        "plateau_mask": is_plateau,
        "dip": dip,
        "smooth_dip": smooth.astype(np.float32, copy=False),
        "plateau_score": float(plateau_score),
        "plateau_argmax_col": int(plateau_argmax_col),
        "dcol": dcol.astype(np.float32, copy=False),
        "med_abs": float(med_abs),
        "cliff_thr": float(cliff_thr),
        "bulk_limit": float(bulk_limit),
        "best_pair_mag": float(best_pair),
        "best_pair_dd": int(best_dd),
        "cliff_drop_col": int(cliff_drop_col),
        "cliff_rise_col": int(cliff_rise_col),
        "cliff_score": float(cliff_score),
        "score": float(min(plateau_score, cliff_score)),
    }


def plot_shifted_overscan_explanation(
    image: np.ndarray,
    figsize: tuple = (12, 12),
    bin_factor: int = 4,
    zoom: int | None = 200,
    title_prefix: str = "",
    **kwargs,
):
    # 4-panel diagnostic for v1: image with column markers, col profile +
    # baseline, plateau-dip arrays, paired-cliff arrays.
    import matplotlib.pyplot as plt

    from .calc import bin_image  # local import to avoid circular dependency

    info = explain_shifted_overscan_score(image, **kwargs)
    score = info["score"]
    plateau_s = info["plateau_score"]
    cliff_s = info["cliff_score"]
    if plateau_s == 0.0 and cliff_s == 0.0:
        winner = "(neither path triggered)"
    elif plateau_s <= cliff_s:
        winner = "plateau-dip"
    else:
        winner = "paired-cliff"
    W = info["col"].shape[0]
    H = image.shape[0]

    fig, axes = plt.subplots(4, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 2, 2, 2]}, sharex=False)
    ax_img, ax_col, ax_pl, ax_cl = axes

    bf = max(1, int(bin_factor))
    img_disp = bin_image(image, bf, bf, method="mean")
    finite = np.isfinite(img_disp)
    if finite.any():
        med = float(np.median(img_disp[finite]))
        mad = float(np.median(np.abs(img_disp[finite] - med)))
        vmin = med - 2.0 * 1.4826 * mad
        vmax = med + 5.0 * 1.4826 * mad
    else:
        vmin, vmax = 0.0, 1.0
    ax_img.imshow(
        img_disp,
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
        extent=[0, W, H, 0],
        aspect="auto",
        interpolation="nearest",
    )
    r0, r1 = info["row_band"]
    ax_img.axhspan(r0, r1, alpha=0.12, color="cyan", label=f"profile rows [{r0},{r1}]")
    if info["plateau_argmax_col"] >= 0 and plateau_s < 0:
        ax_img.axvline(
            info["plateau_argmax_col"],
            color="orange",
            lw=1.0,
            label=f"plateau-dip peak col={info['plateau_argmax_col']}",
        )
    if info["cliff_drop_col"] >= 0:
        ax_img.axvline(
            info["cliff_drop_col"],
            color="red",
            lw=1.0,
            ls="--",
            label=f"cliff drop col={info['cliff_drop_col']}",
        )
        ax_img.axvline(
            info["cliff_rise_col"],
            color="red",
            lw=1.0,
            ls=":",
            label=f"cliff rise col={info['cliff_rise_col']}",
        )
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.set_ylabel("row")
    ax_img.legend(loc="upper right", fontsize=8)
    ax_img.set_title(
        f"{title_prefix}SHFTSCR={score:.2f}   "
        f"plateau-dip={plateau_s:.2f}, paired-cliff={cliff_s:.2f}   winner: {winner}"
    )

    candidate_cols = [c for c in [info["plateau_argmax_col"], info["cliff_drop_col"]] if c >= 0]
    if zoom is not None and candidate_cols:
        c0 = max(0, min(candidate_cols) - zoom)
        c1 = min(W, max(candidate_cols) + zoom)
    else:
        c0, c1 = 0, W

    x = np.arange(W)

    ax_col.plot(x, info["col"], lw=0.7, color="k", label="col (mode-clip)")
    ax_col.plot(
        x,
        info["baseline"],
        lw=0.7,
        color="C0",
        label=f"baseline (running median, win={kwargs.get('win_baseline', 801)})",
    )
    ax_col.axhline(info["bulk_med"], color="green", lw=0.5, ls="--", label=f"bulk_med={info['bulk_med']:.1f}")
    ax_col.axhline(info["bulk_limit"], color="red", lw=0.5, ls="--", label=f"bulk_limit={info['bulk_limit']:.1f}")
    if info["plateau_argmax_col"] >= 0 and plateau_s < 0:
        ax_col.axvline(info["plateau_argmax_col"], color="orange", lw=0.7, alpha=0.7)
    if info["cliff_drop_col"] >= 0:
        ax_col.axvline(info["cliff_drop_col"], color="red", lw=0.7, ls="--", alpha=0.7)
        ax_col.axvline(info["cliff_rise_col"], color="red", lw=0.7, ls=":", alpha=0.7)
    ax_col.set_xlim(c0, c1)
    ax_col.set_ylabel("ADU")
    ax_col.legend(loc="upper right", fontsize=8)
    ax_col.set_title("col profile")

    ymax = float(max(info["dip"].max(), info["smooth_dip"].max(), 1.0))
    ax_pl.fill_between(
        x,
        0,
        info["plateau_mask"].astype(float) * ymax,
        step="mid",
        alpha=0.12,
        color="orange",
        label="plateau mask",
    )
    ax_pl.plot(x, info["dip"], lw=0.5, color="gray", label="dip = -(col-baseline) on plateau")
    ax_pl.plot(x, info["smooth_dip"], lw=0.9, color="C3", label=f"smoothed dip (win={kwargs.get('win_smooth', 8)})")
    ax_pl.axhline(info["sigma"], color="C0", lw=0.5, ls=":", label=f"sigma={info['sigma']:.2f}")
    if info["plateau_argmax_col"] >= 0 and plateau_s < 0:
        ax_pl.axvline(
            info["plateau_argmax_col"],
            color="C3",
            lw=1.0,
            ls="--",
            label=f"argmax @ col {info['plateau_argmax_col']}, dip/sigma={-plateau_s:.2f}",
        )
    ax_pl.set_xlim(c0, c1)
    ax_pl.set_ylabel("ADU")
    ax_pl.legend(loc="upper right", fontsize=8)
    ax_pl.set_title(f"plateau-dip path  ->  -smoothed_dip.max()/sigma = {plateau_s:.3f}")

    xd = x[:-1] + 0.5
    ax_cl.plot(xd, info["dcol"], lw=0.5, color="gray", label="dcol = diff(col)")
    ax_cl.axhline(
        +info["cliff_thr"],
        color="red",
        lw=0.5,
        ls=":",
        label=f"+/- cliff_thr = {info['cliff_thr']:.2f} = 8*med|dcol|",
    )
    ax_cl.axhline(-info["cliff_thr"], color="red", lw=0.5, ls=":")
    if info["cliff_drop_col"] >= 0:
        ax_cl.axvline(
            info["cliff_drop_col"] - 0.5,
            color="red",
            lw=1.0,
            ls="--",
            label=f"drop @ col {info['cliff_drop_col']}",
        )
        ax_cl.axvline(
            info["cliff_rise_col"] + 0.5,
            color="red",
            lw=1.0,
            ls=":",
            label=f"rise @ col {info['cliff_rise_col']} (dd={info['best_pair_dd']})",
        )
    ax_cl.set_xlim(c0, c1)
    ax_cl.set_ylabel("dcol (ADU)")
    ax_cl.set_xlabel("column")
    ax_cl.legend(loc="upper right", fontsize=8)
    ax_cl.set_title(f"paired-cliff path  ->  -sqrt(drop*rise)/med|dcol| = {cliff_s:.3f}")

    fig.tight_layout()
    return fig, info


# --------------------------------------------------------------------------
# v2: bias-anchored window-min ("lower is worse")
# --------------------------------------------------------------------------

SHIFTED_SCORE_THRESHOLD_V2 = -5.0


def _shifted_v2_qualifying_threshold(bulk_elev: float, bias_tol: float, bulk_frac: float) -> float:
    # window-mean must satisfy both the absolute "near bias" cap and the
    # relative "closer to bias than to bulk" cap; the latter prevents bulk
    # noise from looking bias-like when bulk_elev is itself ~bias_tol.
    return min(bias_tol, bulk_frac * bulk_elev)


def shifted_overscan_score_v2(
    image: np.ndarray,
    bias_level: float = BIAS_LEVEL,
    row_frac: float = 0.6,
    profile_max_rows: int = 384,
    width_min: int = 5,
    width_max: int = 50,
    bias_tol_sigma: float = 5.0,
    bias_tol_floor: float = 20.0,
    bulk_frac: float = 0.5,
    bulk_elev_min: float = 10.0,
    noise_floor: float = 1.0,
) -> float:
    # Score: -(bulk_elev - min_mean_col_elev) / sigma over qualifying windows.
    # Uses GLOBAL bulk_elev = median(col - bias_level); fast but assumes the
    # bulk level is roughly constant across the image. Returns 0 when
    # bulk_elev < bulk_elev_min (cannot test bias-like frames at all).
    h = image.shape[0]
    r0 = int(h * (1.0 - row_frac) / 2.0)
    r1 = h - r0
    band = image[r0:r1]
    nr = band.shape[0]
    step = max(1, (nr + profile_max_rows - 1) // profile_max_rows)
    sample = band[::step]
    col_mean = sample.mean(axis=0).astype(np.float32, copy=False)
    col_med = np.median(sample, axis=0).astype(np.float32, copy=False)
    mode = 2.5 * col_med - 1.5 * col_mean
    lo = np.minimum(col_mean, col_med)
    hi = np.maximum(col_mean, col_med)
    col = np.clip(mode, lo, hi).astype(np.float32, copy=False)

    elev = col - float(bias_level)
    bulk_elev = float(np.median(elev))
    if bulk_elev < bulk_elev_min:
        return 0.0

    dcol = np.diff(col)
    sigma = max(float(np.median(np.abs(dcol))), noise_floor)
    bias_tol = max(bias_tol_floor, bias_tol_sigma * sigma)
    qual_thr = _shifted_v2_qualifying_threshold(bulk_elev, bias_tol, bulk_frac)

    csum = np.empty(elev.shape[0] + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(elev, dtype=np.float64, out=csum[1:])
    best_score = 0.0
    for w in range(width_min, width_max + 1):
        means = (csum[w:] - csum[:-w]) / w
        msk = means < qual_thr
        if not msk.any():
            continue
        s = -float((bulk_elev - means[msk].min()) / sigma)
        if s < best_score:
            best_score = s
    return best_score


def check_shifted_overscan_v2(
    image: np.ndarray,
    threshold: float = SHIFTED_SCORE_THRESHOLD_V2,
    **kwargs,
) -> tuple[bool, float]:
    score = shifted_overscan_score_v2(image, **kwargs)
    return (score < threshold), score


def explain_shifted_overscan_score_v2(
    image: np.ndarray,
    bias_level: float = BIAS_LEVEL,
    row_frac: float = 0.6,
    profile_max_rows: int = 384,
    width_min: int = 5,
    width_max: int = 50,
    bias_tol_sigma: float = 5.0,
    bias_tol_floor: float = 20.0,
    bulk_frac: float = 0.5,
    bulk_elev_min: float = 10.0,
    noise_floor: float = 1.0,
) -> dict:
    # Returns col, elev, bulk_elev, bias_tol, qual_thr, sigma, bailed_out,
    # plus per-width lower envelope, the (best_w, best_c, best_mean) that
    # produced the score, the mf_curve at width_min, and the longest run of
    # qualifying width_min windows (the dip extent), and `score`.
    h = image.shape[0]
    r0 = int(h * (1.0 - row_frac) / 2.0)
    r1 = h - r0
    band = image[r0:r1]
    nr = band.shape[0]
    step = max(1, (nr + profile_max_rows - 1) // profile_max_rows)
    sample = band[::step]
    col_mean = sample.mean(axis=0).astype(np.float32, copy=False)
    col_med = np.median(sample, axis=0).astype(np.float32, copy=False)
    mode = 2.5 * col_med - 1.5 * col_mean
    lo = np.minimum(col_mean, col_med)
    hi = np.maximum(col_mean, col_med)
    col = np.clip(mode, lo, hi).astype(np.float32, copy=False)

    elev = col - float(bias_level)
    bulk_elev = float(np.median(elev))

    dcol = np.diff(col)
    sigma = max(float(np.median(np.abs(dcol))), noise_floor)
    bias_tol = max(bias_tol_floor, bias_tol_sigma * sigma)
    qual_thr = _shifted_v2_qualifying_threshold(bulk_elev, bias_tol, bulk_frac)

    bailed_out = bulk_elev < bulk_elev_min
    csum = np.empty(elev.shape[0] + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(elev, dtype=np.float64, out=csum[1:])

    width_grid = np.arange(width_min, width_max + 1, dtype=np.int32)
    n_w = width_grid.shape[0]
    width_min_means = np.full(n_w, np.inf, dtype=np.float64)
    width_min_means_col = np.full(n_w, -1, dtype=np.int64)
    best_score = 0.0
    best_w = -1
    best_c = -1
    best_mean = float("nan")
    mf_curve_at_w_min = None
    run_start = -1
    run_end = -1
    run_length = 0
    run_dip_left = -1
    run_dip_right = -1
    run_mean_elev = float("nan")

    if not bailed_out:
        for i, w in enumerate(width_grid):
            w = int(w)
            means = (csum[w:] - csum[:-w]) / w
            min_idx = int(np.argmin(means))
            min_val = float(means[min_idx])
            width_min_means[i] = min_val
            width_min_means_col[i] = min_idx
            if min_val < qual_thr:
                s = -float((bulk_elev - min_val) / sigma)
                if s < best_score:
                    best_score = s
                    best_w = w
                    best_c = min_idx
                    best_mean = min_val

        # Longest run of qualifying width_min windows (the dip extent).
        w0 = int(width_min)
        mf_curve_at_w_min = (csum[w0:] - csum[:-w0]) / w0
        bias_like = mf_curve_at_w_min < qual_thr
        if bias_like.any():
            d_run = np.diff(np.concatenate([[0], bias_like.astype(np.int8), [0]]))
            starts = np.where(d_run == 1)[0]
            ends = np.where(d_run == -1)[0]
            lens = ends - starts
            longest = int(np.argmax(lens))
            run_start = int(starts[longest])
            run_end = int(ends[longest])
            run_length = int(lens[longest])
            run_dip_left = run_start
            run_dip_right = (run_end - 1) + (w0 - 1)
            seg_lo = run_dip_left
            seg_hi = run_dip_right + 1
            run_mean_elev = float((csum[seg_hi] - csum[seg_lo]) / (seg_hi - seg_lo))

    return {
        "row_band": (r0, r1),
        "row_step": int(step),
        "col": col,
        "elev": elev,
        "bulk_elev": bulk_elev,
        "bias_level": float(bias_level),
        "bias_tol": float(bias_tol),
        "qual_thr": float(qual_thr),
        "sigma": float(sigma),
        "bulk_elev_min": float(bulk_elev_min),
        "bailed_out": bool(bailed_out),
        "width_grid": width_grid,
        "width_min_means": width_min_means,
        "width_min_means_col": width_min_means_col,
        "best_w": int(best_w),
        "best_c": int(best_c),
        "best_mean": float(best_mean),
        "mf_curve_at_w_min": (
            mf_curve_at_w_min.astype(np.float32, copy=False) if mf_curve_at_w_min is not None else None
        ),
        "run_start": int(run_start),
        "run_end": int(run_end),
        "run_length": int(run_length),
        "run_dip_left": int(run_dip_left),
        "run_dip_right": int(run_dip_right),
        "run_mean_elev": float(run_mean_elev),
        "score": float(best_score),
    }


def plot_shifted_overscan_explanation_v2(
    image: np.ndarray,
    figsize: tuple = (12, 12),
    bin_factor: int = 4,
    zoom: int | None = 200,
    title_prefix: str = "",
    **kwargs,
):
    # 4-panel diagnostic for v2: image with dip span shaded, col profile with
    # bias and bulk_elev refs, elevation curve with bias_tol band, matched
    # filter curves (per-width min envelope + running-mean at winning width).
    import matplotlib.pyplot as plt

    from .calc import bin_image

    info = explain_shifted_overscan_score_v2(image, **kwargs)
    score = info["score"]
    W = info["col"].shape[0]
    H = image.shape[0]
    has_dip = info["run_length"] > 0
    dip_lo = info["run_dip_left"]
    dip_hi = info["run_dip_right"]

    fig, axes = plt.subplots(4, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 2, 2, 2]}, sharex=False)
    ax_img, ax_col, ax_el, ax_mf = axes

    bf = max(1, int(bin_factor))
    img_disp = bin_image(image, bf, bf, method="mean")
    finite = np.isfinite(img_disp)
    if finite.any():
        med = float(np.median(img_disp[finite]))
        mad = float(np.median(np.abs(img_disp[finite] - med)))
        vmin = med - 2.0 * 1.4826 * mad
        vmax = med + 5.0 * 1.4826 * mad
    else:
        vmin, vmax = 0.0, 1.0
    ax_img.imshow(
        img_disp,
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
        extent=[0, W, H, 0],
        aspect="auto",
        interpolation="nearest",
    )
    r0, r1 = info["row_band"]
    ax_img.axhspan(r0, r1, alpha=0.12, color="cyan", label=f"profile rows [{r0},{r1}]")
    if has_dip:
        ax_img.axvspan(
            dip_lo,
            dip_hi + 1,
            alpha=0.30,
            color="red",
            label=f"dip span [{dip_lo},{dip_hi}] w={dip_hi-dip_lo+1}",
        )
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.set_ylabel("row")
    ax_img.legend(loc="upper right", fontsize=8)
    bail_str = "  (bailed: bulk too close to bias)" if info["bailed_out"] else ""
    ax_img.set_title(f"{title_prefix}SHFTSCR_v2 = {score:.2f}{bail_str}")

    if zoom is not None and has_dip:
        center = (dip_lo + dip_hi) // 2
        c0 = max(0, center - zoom)
        c1 = min(W, center + zoom + 1)
    else:
        c0, c1 = 0, W
    x = np.arange(W)

    ax_col.plot(x, info["col"], lw=0.7, color="k", label="col (mode-clip)")
    ax_col.axhline(info["bias_level"], color="red", lw=0.7, ls="--", label=f"bias_level={info['bias_level']:.0f}")
    ax_col.axhline(
        info["bias_level"] + info["bulk_elev"],
        color="green",
        lw=0.5,
        ls=":",
        label=f"bulk = bias + {info['bulk_elev']:.1f}",
    )
    ax_col.axhline(
        info["bias_level"] + info["qual_thr"],
        color="orange",
        lw=0.5,
        ls=":",
        label=f"qual_thr = bias + {info['qual_thr']:.1f}",
    )
    ax_col.fill_between(
        x,
        info["bias_level"] - info["bias_tol"],
        info["bias_level"] + info["bias_tol"],
        alpha=0.10,
        color="red",
        label=f"bias_tol band (+/- {info['bias_tol']:.1f})",
    )
    if has_dip:
        ax_col.axvspan(dip_lo, dip_hi + 1, alpha=0.20, color="red")
    ax_col.set_xlim(c0, c1)
    ax_col.set_ylabel("ADU")
    ax_col.legend(loc="upper right", fontsize=8)
    ax_col.set_title("col profile")

    ax_el.plot(x, info["elev"], lw=0.6, color="C0", label="elev = col - bias_level")
    ax_el.axhline(0.0, color="red", lw=0.7, ls="--", label="bias_level (elev=0)")
    ax_el.axhspan(
        -info["bias_tol"],
        info["bias_tol"],
        alpha=0.08,
        color="red",
        label=f"+/- bias_tol = {info['bias_tol']:.1f}",
    )
    ax_el.axhline(
        info["qual_thr"],
        color="orange",
        lw=0.7,
        ls="--",
        label=f"qual_thr = {info['qual_thr']:.1f}  (= min(bias_tol, bulk_frac*bulk))",
    )
    ax_el.axhline(info["bulk_elev"], color="green", lw=0.5, ls=":", label=f"bulk_elev = {info['bulk_elev']:.1f}")
    if has_dip:
        ax_el.axvspan(dip_lo, dip_hi + 1, alpha=0.20, color="red", label=f"dip mean elev = {info['run_mean_elev']:.2f}")
    ax_el.set_xlim(c0, c1)
    ax_el.set_ylabel("ADU above bias")
    ax_el.legend(loc="upper right", fontsize=8)
    ax_el.set_title("elevation above bias_level")

    if info["mf_curve_at_w_min"] is not None:
        w0 = int(kwargs.get("width_min", 5))
        mf = info["mf_curve_at_w_min"]
        xc = np.arange(mf.shape[0]) + (w0 - 1) / 2.0
        ax_mf.plot(xc, mf, lw=0.7, color="C3", label=f"running mean elev (w={w0})")
        ax_mf.axhline(info["qual_thr"], color="orange", lw=0.7, ls="--", label=f"qual_thr = {info['qual_thr']:.1f}")
        ax_mf.axhline(info["bulk_elev"], color="green", lw=0.5, ls=":", label=f"bulk_elev = {info['bulk_elev']:.1f}")
        if has_dip:
            ax_mf.axvspan(
                dip_lo,
                dip_hi + 1,
                alpha=0.20,
                color="red",
                label=f"longest qualifying run = {info['run_length']} window starts",
            )
            ax_mf.axvline(
                info["best_c"] + (w0 - 1) / 2.0,
                color="C3",
                lw=1.0,
                ls="--",
                label=f"score min @ col {info['best_c']}, mean={info['best_mean']:.2f}",
            )
        ax_mf.set_xlim(c0, c1)
        ax_mf.set_ylabel("running mean elev")
        ax_mf.set_xlabel("column")
        ax_mf.legend(loc="upper right", fontsize=8)
        ax_mf.set_title(f"matched filter (w={w0}); longest run -> dip width")
    else:
        ax_mf.text(
            0.5,
            0.5,
            "no window's running mean reached qual_thr\n=> score = 0",
            transform=ax_mf.transAxes,
            ha="center",
            va="center",
        )
        ax_mf.set_xlabel("column")
        widths = info["width_grid"]
        wmm = info["width_min_means"]
        ax_mf2 = ax_mf.twinx()
        ax_mf2.plot(widths, wmm, lw=0.9, color="C3", label="min running-mean elev across image")
        ax_mf2.axhline(info["qual_thr"], color="orange", lw=0.5, ls="--", label=f"qual_thr = {info['qual_thr']:.1f}")
        ax_mf2.set_ylabel("min running-mean elev")
        ax_mf2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig, info


# --------------------------------------------------------------------------
# v3: bias-anchored matched filter / LRT (HIGHER is worse; positive = signal)
# --------------------------------------------------------------------------
#
# Score (per window of width w starting at column c0):
#
#                  sum_c bg_elev(c) * ( bg_elev(c) - 2 elev(c) )
#   score(c0, w) = --------------------------------------------
#                          2 sigma sqrt( sum_c bg_elev(c)^2 )
#
#   bg_elev(c) = max(bg_smooth(c) - bias_level, 0)     (kernel)
#   elev(c)    = col(c) - bias_level                   (data)
#   sigma      = max( median |diff(col)|, noise_floor )
#
# Reported score = max over (c0, w) of score(c0,w). Positive => H1 (full
# dip-to-bias) favoured over H0 (no signal); zero => half-amplitude dip;
# negative => col follows bg (no signal). Statistical interpretation: this
# is the LRT for a known-amplitude rectangular dip from bg(c) to bias_level,
# normalised to unit variance under H0. Equivalent to the textbook x.s >
# 0.5 |s|^2 decision rule with x = (col - bg) (Gaussian noise under H0) and
# s = (bias_level - bg) (the dip signal); see the README of this module for
# the full derivation.

SHIFTED_SCORE_THRESHOLD_V3 = +5.0


def _shifted_v3_smoothed_bg(col: np.ndarray, bin_size: int = 200) -> np.ndarray:
    # Coarse-binned median + linear interp. Robust to localised dips that
    # span <~25% of a bin (median stays at the bulk) and to wide gradients
    # (vignetting, bias tilt). O(W) cost, far cheaper than a wide running
    # median.
    W = int(col.shape[0])
    bs = max(8, int(bin_size))
    n_bins = max(1, (W + bs - 1) // bs)
    pad_w = n_bins * bs - W
    col_pad = np.pad(col, (0, pad_w), mode="edge")
    bin_meds = np.median(col_pad.reshape(n_bins, bs), axis=1).astype(np.float64, copy=False)
    bin_centers = (np.arange(n_bins) + 0.5) * bs - 0.5
    return np.interp(np.arange(W, dtype=np.float64), bin_centers, bin_meds).astype(np.float32, copy=False)


def _compress_columns_to_1d(
    image: np.ndarray,
    profile_max_rows: int = 384,
    method: Literal["median", "sextractor"] = "sextractor",
) -> np.ndarray:
    nr = image.shape[0]
    step = max(1, (nr + profile_max_rows - 1) // profile_max_rows)
    sample = image[::step]
    col_med = np.median(sample, axis=0).astype(np.float32, copy=False)
    if method == "median":
        return col_med
    elif method == "sextractor":
        col_mean = sample.mean(axis=0).astype(np.float32, copy=False)
        mode = 2.5 * col_med - 1.5 * col_mean
        lo = np.minimum(col_mean, col_med)
        hi = np.maximum(col_mean, col_med)
        col = np.clip(mode, lo, hi).astype(np.float32, copy=False)
        return col


def shifted_overscan_score_v3(
    image: np.ndarray,
    bias_level: float = BIAS_LEVEL,
    row_frac: float = 0.6,
    profile_max_rows: int = 384,
    bg_bin_size: int = 200,
    width_min: int = 1,  # minimum pixel width to consider
    width_max: int = 50,  # overscan width ~24
    noise_floor: float = 1.0,  # ADU
    min_kernel_norm: float = 4.0,
    background_method: Literal["median", "sextractor"] = "median",
) -> float:
    """matched filter based detection. z-score like score"""
    h0, w0 = image.shape
    r0 = int(h0 * (1.0 - row_frac) / 2.0)
    strip = _compress_columns_to_1d(image[r0 : h0 - r0], profile_max_rows, method=background_method)

    background = _shifted_v3_smoothed_bg(strip, bg_bin_size)
    # Clip at 0: only positive-elevation bg can host a dip *to* bias_level.
    background_elevation = np.maximum(background - float(bias_level), 0.0).astype(np.float64, copy=False)
    observed_elevation = (strip - float(bias_level)).astype(np.float64, copy=False)

    # cheap proxy for noise (lower bounded by noise_floor)
    sigma = max(float(np.median(np.abs(np.diff(strip)))), noise_floor)
    inv_s2 = 1.0 / (sigma * sigma)

    csum_bb = np.empty(w0 + 1, dtype=np.float64)
    csum_eb = np.empty(w0 + 1, dtype=np.float64)
    csum_bb[0] = 0.0
    csum_eb[0] = 0.0
    np.cumsum(background_elevation * background_elevation, dtype=np.float64, out=csum_bb[1:])
    np.cumsum(observed_elevation * background_elevation, dtype=np.float64, out=csum_eb[1:])

    best_score = 0.0
    for w in range(int(width_min), int(width_max) + 1):
        # sliding window sums with width w
        sum_bb_w = csum_bb[w:] - csum_bb[:-w]
        sum_eb_w = csum_eb[w:] - csum_eb[:-w]
        # Reject windows with too-weak kernel
        # (e.g. bias frames where background_elevation ~ 0 everywhere, division by sum_bb blows up).
        valid = sum_bb_w * inv_s2 > min_kernel_norm
        if not valid.any():
            continue
        pseudo_z_score = (sum_bb_w - 2.0 * sum_eb_w) / (2.0 * sigma * np.sqrt(sum_bb_w))
        score_w = np.where(valid, pseudo_z_score, 0.0)  # pseudo_z_score where valid, 0 otherwise
        s = float(score_w.max())
        if s > best_score:
            best_score = s
    return best_score


def check_shifted_overscan_v3(
    image: np.ndarray,
    threshold: float = SHIFTED_SCORE_THRESHOLD_V3,
    **kwargs,
) -> tuple[bool, float]:
    score = shifted_overscan_score_v3(image, **kwargs)
    return (score > threshold), score


def explain_shifted_overscan_score_v3(
    image: np.ndarray,
    bias_level: float = BIAS_LEVEL,
    row_frac: float = 0.6,
    profile_max_rows: int = 384,
    bg_bin_size: int = 200,
    width_min: int = 5,
    width_max: int = 50,
    noise_floor: float = 1.0,
    min_kernel_norm: float = 4.0,
    background_method: Literal["median", "sextractor"] = "sextractor",
) -> dict:
    """replay shifted_overscan_score_v3 with intermediates for plotting"""
    h0, w0 = image.shape
    r0 = int(h0 * (1.0 - row_frac) / 2.0)
    strip = _compress_columns_to_1d(image[r0 : h0 - r0], profile_max_rows, method=background_method)

    background = _shifted_v3_smoothed_bg(strip, bg_bin_size)
    background_elevation = np.maximum(background - float(bias_level), 0.0).astype(np.float64, copy=False)
    observed_elevation = (strip - float(bias_level)).astype(np.float64, copy=False)

    sigma = max(float(np.median(np.abs(np.diff(strip)))), noise_floor)
    inv_s2 = 1.0 / (sigma * sigma)

    csum_bb = np.empty(w0 + 1, dtype=np.float64)
    csum_eb = np.empty(w0 + 1, dtype=np.float64)
    csum_bb[0] = 0.0
    csum_eb[0] = 0.0
    np.cumsum(background_elevation * background_elevation, dtype=np.float64, out=csum_bb[1:])
    np.cumsum(observed_elevation * background_elevation, dtype=np.float64, out=csum_eb[1:])

    width_grid = np.arange(width_min, width_max + 1, dtype=np.int32)
    width_best_score = np.zeros(width_grid.shape[0], dtype=np.float64)

    best_score = 0.0
    best_w = -1
    best_c = -1
    best_score_curve = None

    for i, w in enumerate(width_grid):
        w = int(w)
        sum_bb_w = csum_bb[w:] - csum_bb[:-w]
        sum_eb_w = csum_eb[w:] - csum_eb[:-w]
        valid = sum_bb_w * inv_s2 > min_kernel_norm
        # guard against sum_bb_w = 0 entries; np.where masks them anyway
        pseudo_z_score = (sum_bb_w - 2.0 * sum_eb_w) / (2.0 * sigma * np.sqrt(np.maximum(sum_bb_w, 1e-24)))
        score_w = np.where(valid, pseudo_z_score, 0.0)
        if valid.any():
            j = int(np.argmax(score_w))
            v = float(score_w[j])
            width_best_score[i] = v
            if v > best_score:
                best_score = v
                best_w = w
                best_c = j
                best_score_curve = score_w.astype(np.float32, copy=False)

    # Contiguous "H1 favoured" (score > 0) run at the winning width containing best_c.
    run_length = 0
    run_dip_left = -1
    run_dip_right = -1
    run_mean_observed_elevation = float("nan")
    if best_w > 0 and best_score_curve is not None and best_c >= 0:
        favoured = best_score_curve > 0
        if favoured[best_c]:
            lo_j = best_c
            while lo_j > 0 and favoured[lo_j - 1]:
                lo_j -= 1
            hi_j = best_c
            n = favoured.shape[0]
            while hi_j + 1 < n and favoured[hi_j + 1]:
                hi_j += 1
            run_length = (hi_j + 1) - lo_j
            # window-start coords: [lo_j, hi_j]; image-col coords: [lo_j, hi_j+w-1]
            run_dip_left = lo_j
            run_dip_right = hi_j + (best_w - 1)
            run_mean_observed_elevation = float(observed_elevation[run_dip_left : run_dip_right + 1].mean())

    return {
        "row_band": (r0, h0 - r0),
        "strip": strip,
        "background": background,
        "background_elevation": background_elevation,
        "observed_elevation": observed_elevation,
        "sigma": float(sigma),
        "width_grid": width_grid,
        "width_best_score": width_best_score,
        "best_w": int(best_w),
        "best_c": int(best_c),
        "best_score_curve": best_score_curve,
        "run_length": int(run_length),
        "run_dip_left": int(run_dip_left),
        "run_dip_right": int(run_dip_right),
        "run_mean_observed_elevation": float(run_mean_observed_elevation),
        "score": float(best_score),
    }


def plot_shifted_overscan_explanation_v3(
    image: np.ndarray,
    figsize: tuple = (12, 12),
    bin_factor: int = 4,
    zoom: int | None = 200,
    title_prefix: str = "",
    bias_level: float = BIAS_LEVEL,
    **kwargs,
):
    """4-panel diagnostic: image + dip span, strip vs background, observed_elevation
    + background_elevation kernel, matched-filter score curve + per-width envelope."""
    import matplotlib.pyplot as plt

    from .calc import bin_image

    info = explain_shifted_overscan_score_v3(image, bias_level=bias_level, **kwargs)
    score = info["score"]
    h0, w0 = image.shape
    has_dip = info["run_length"] > 0
    dip_lo = info["run_dip_left"]
    dip_hi = info["run_dip_right"]
    best_w = info["best_w"]
    best_c = info["best_c"]

    fig, axes = plt.subplots(4, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 2, 2, 2]}, sharex=False)
    ax_img, ax_strip, ax_elev, ax_mf = axes

    bf = max(1, int(bin_factor))
    img_disp = bin_image(image, bf, bf, method="mean")
    finite = np.isfinite(img_disp)
    if finite.any():
        med = float(np.median(img_disp[finite]))
        mad = float(np.median(np.abs(img_disp[finite] - med)))
        vmin = med - 2.0 * 1.4826 * mad
        vmax = med + 5.0 * 1.4826 * mad
    else:
        vmin, vmax = 0.0, 1.0
    ax_img.imshow(
        img_disp,
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
        extent=[0, w0, h0, 0],
        aspect="auto",
        interpolation="nearest",
    )
    rb0, rb1 = info["row_band"]
    ax_img.axhspan(rb0, rb1, alpha=0.12, color="cyan", label=f"profile rows [{rb0},{rb1}]")
    if has_dip:
        ax_img.axvspan(
            dip_lo,
            dip_hi + 1,
            alpha=0.30,
            color="red",
            label=f"dip span [{dip_lo},{dip_hi}] w={dip_hi-dip_lo+1}",
        )
    ax_img.set_xlim(0, w0)
    ax_img.set_ylim(h0, 0)
    ax_img.set_ylabel("row")
    ax_img.set_title(
        f"{title_prefix}v3 score = {score:.3f}  "
        f"(best_w={best_w}, best_c={best_c}, "
        f"run_mean_observed_elevation={info['run_mean_observed_elevation']:.2f})"
    )
    ax_img.legend(loc="upper right", fontsize=8)

    if zoom is None or not has_dip:
        c0, c1 = 0, w0
    else:
        cmid = (dip_lo + dip_hi) // 2
        c0 = max(0, cmid - zoom)
        c1 = min(w0, cmid + zoom)

    x = np.arange(w0)
    ax_strip.plot(x, info["strip"], lw=0.6, color="C0", label="strip (column profile)")
    ax_strip.plot(x, info["background"], lw=0.9, color="C2", label="background (coarse-bin median)")
    ax_strip.axhline(bias_level, color="red", lw=0.7, ls="--", label=f"bias_level = {bias_level:.0f}")
    if has_dip:
        ax_strip.axvspan(dip_lo, dip_hi + 1, alpha=0.20, color="red")
    ax_strip.set_xlim(c0, c1)
    ax_strip.set_ylabel("ADU")
    ax_strip.legend(loc="upper right", fontsize=8)
    ax_strip.set_title("strip vs background")

    ax_elev.plot(x, info["observed_elevation"], lw=0.6, color="C0", label="observed_elevation = strip - bias_level")
    ax_elev.plot(
        x,
        info["background_elevation"],
        lw=0.9,
        color="C2",
        label="background_elevation = max(background - bias_level, 0)  (kernel)",
    )
    ax_elev.axhline(0.0, color="red", lw=0.7, ls="--", label="bias_level (elevation = 0)")
    if has_dip:
        ax_elev.axvspan(dip_lo, dip_hi + 1, alpha=0.20, color="red")
    ax_elev.set_xlim(c0, c1)
    ax_elev.set_ylabel("ADU above bias")
    ax_elev.legend(loc="upper right", fontsize=8)
    ax_elev.set_title("elevation curves and matched-filter kernel")

    if info["best_score_curve"] is None:
        ax_mf.text(0.5, 0.5, "no qualifying signal", ha="center", va="center", transform=ax_mf.transAxes)
        ax_mf.set_axis_off()
    else:
        score_curve = info["best_score_curve"]
        ax_mf.plot(
            np.arange(score_curve.shape[0]), score_curve, lw=0.6, color="C0", label=f"pseudo_z_score at w={best_w}"
        )
        ax_mf.axhline(0.0, color="red", lw=0.7, ls="--", label="0 (no preference)")
        ax_mf.axhline(
            SHIFTED_SCORE_THRESHOLD_V3,
            color="orange",
            lw=0.5,
            ls="--",
            label=f"thresh = {SHIFTED_SCORE_THRESHOLD_V3:.0f}",
        )
        if best_c >= 0:
            ax_mf.axvline(best_c, color="C3", lw=0.6, ls=":")
        if has_dip:
            # score curve has length w0-w+1; run on it is [dip_lo, dip_hi-w+1].
            ax_mf.axvspan(dip_lo, dip_hi - best_w + 2, alpha=0.20, color="red")
        ax_mf.set_xlim(c0, c1)
        ax_mf.set_ylabel(f"pseudo_z_score at w={best_w}")
        ax_mf.legend(loc="upper right", fontsize=8)
        ax_mf.set_title("matched-filter score (positive = H1 favoured)")
        ax_mf.set_xlabel("column")
        ax_mf2 = ax_mf.twinx()
        ax_mf2.plot(info["width_grid"], info["width_best_score"], lw=0.9, color="C3", label="best score per width")
        ax_mf2.set_ylabel("best score across image")
        ax_mf2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig, info
