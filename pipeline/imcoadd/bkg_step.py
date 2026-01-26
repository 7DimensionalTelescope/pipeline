import numpy as np


def _moving_average(x, w):
    w = int(w)
    w = max(3, min(w, len(x) - 1))
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, k, mode="same")


def _rle(values):
    """Run-length encoding: returns (run_values, run_lengths, run_starts)."""
    v = np.asarray(values)
    if v.size == 0:
        return v, np.array([], int), np.array([], int)
    change = np.flatnonzero(v[1:] != v[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, v.size]
    lengths = ends - starts
    run_vals = v[starts]
    return run_vals, lengths, starts


def step_background_check(
    stripe,
    detrend_win=801,  # should follow slow gradient (hundreds–thousands px)
    q_adu=1.0,  # quantization bin in ADU; usually 1.0
    min_run=25,  # plateau must be at least this long to “count”
    min_steps=3,  # require at least this many steps
    min_plateau_frac=0.35,  # fraction of samples in long plateaus
    max_unique_levels=200,  # after quantization, too many levels => not staircase
    min_step_adu=1.0,  # typical step height; adjust if your steps are larger
):
    """
    Returns: (is_stepy, info)
    is_stepy=True => likely quantized/staircase background.
    """
    x = np.asarray(stripe, dtype=np.float64)
    n = x.size
    if n < 10:
        return False, {"reason": "too_short"}

    # 1) detrend
    trend = _moving_average(x, detrend_win)
    resid = x - trend

    # 2) quantize residual to ADU bins
    q = np.round(resid / q_adu).astype(np.int32)

    # quick “few-levels” sanity check (staircase often has limited levels)
    uniq = np.unique(q).size

    # 3) RLE to get plateaus + step locations
    run_vals, run_lens, run_starts = _rle(q)
    long_mask = run_lens >= min_run

    plateau_frac = float(run_lens[long_mask].sum()) / n if n else 0.0
    n_long_runs = int(long_mask.sum())

    # step sizes between consecutive runs
    step_sizes = np.abs(np.diff(run_vals)).astype(np.int32)
    step_positions = run_starts[1:]  # indices where a new run begins

    # count “meaningful” steps (ignore tiny +/-1 jitter if needed)
    meaningful = step_sizes >= int(np.round(min_step_adu / q_adu))
    n_steps = int(meaningful.sum())

    # 4) score & decision
    # Score favors: lots of long plateaus + multiple meaningful steps + not too many levels.
    score = (
        2.0 * plateau_frac
        + 0.15 * min(n_steps, 50) / 50.0
        + 0.5 * (1.0 - min(uniq, max_unique_levels) / max_unique_levels)
    )

    is_steppy = (plateau_frac >= min_plateau_frac) and (n_steps >= min_steps) and (uniq <= max_unique_levels)

    info = {
        "n": n,
        "detrend_win": int(detrend_win),
        "q_adu": float(q_adu),
        "unique_levels": int(uniq),
        "plateau_frac_long": plateau_frac,
        "n_long_plateaus": n_long_runs,
        "n_steps_meaningful": n_steps,
        "score": float(score),
        "step_positions": step_positions[meaningful][:30],  # preview
        "step_sizes_bins": step_sizes[meaningful][:30],
    }
    return is_steppy, info
