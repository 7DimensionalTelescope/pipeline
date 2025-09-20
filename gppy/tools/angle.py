import numpy as np


def pa_alignment(theta_deg, weights=None):
    """
    Computes the degree of alignment of a set of directionless angles like the
    position angle (PA) of ellipses.

    theta_deg: array-like of angles in degrees, e.g. [-90, 90] range is fine
    weights:   optional nonnegative weights (same shape as theta_deg)

    Returns:
      R               alignment score in [0,1]
      theta_axis_deg  mean axis (directionless), degrees in [-90, 90)
      V               circular variance on the axis (1-R)
      sd_deg          circular stdev on the axis (approx), degrees
      Q               2x2 nematic tensor (for diagnostics)
    """
    theta = np.deg2rad(np.asarray(theta_deg))
    if weights is None:
        w = np.ones_like(theta, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    wsum = w.sum()
    assert wsum > 0, "Sum of weights must be > 0"

    # double-angle trick
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)

    C = np.sum(w * c2) / wsum
    S = np.sum(w * s2) / wsum

    R = np.hypot(C, S)  # alignment in [0,1]
    theta_axis = 0.5 * np.arctan2(S, C)  # mean axis (radians)
    theta_axis_deg = np.rad2deg(theta_axis)

    V = 1.0 - R

    # Circular stdev (approx): std_phi = sqrt(-2 ln R); divide by 2 to map back to theta
    sd_rad = 0.5 * np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12))))
    sd_deg = np.rad2deg(sd_rad)

    # Nematic tensor (equivalent to doubling trick)
    # Q = <u u^T> - 1/2 I, with u=(cosθ, sinθ)
    # In 2D this reduces to the matrix below; its principal eigenvector is the axis.
    Q = 0.5 * np.array([[C, S], [S, -C]])

    return R, theta_axis_deg, V, sd_deg, Q
