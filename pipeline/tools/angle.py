import numpy as np


def pa_alignment(theta_deg, weights=None, normalize=True):
    """
    Computes the degree of alignment of a set of directionless angles like the
    position angle (PA) of ellipses.

    theta_deg: array-like of angles in degrees, e.g. [-90, 90] range is fine
    weights: optional nonnegative weights (same shape as theta_deg). e.g., elongation = a/b >= 1

    if normalize=False: the sum of weighted directionless vectors are divided by their number, not the sum of weights

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
    if normalize:
        wsum = w.sum()
    else:
        wsum = len(w)
    assert wsum > 0, "Sum of weights must be > 0"

    # double-angle trick
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)

    # Q_stokes ~ R * cos(2 * theta_axis)
    # U_stokes ~ R * sin(2 * theta_axis)
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


def azimuth_deg_from_center(x, y, x_center, y_center):
    """
    Compute azimuthal angle (in degrees) of positions (x, y) with respect
    to an image center (x_center, y_center).

    phi_deg counts counterclockwise from the positive x-axis

    x, y, x_center, y_center can be scalars or arrays; broadcasting rules apply.

    Returns:
        phi_deg : array-like of azimuths in degrees, range (-180, 180]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x_center
    dy = y - y_center

    phi = np.arctan2(dy, dx)  # radians, (-π, π]
    phi_deg = np.rad2deg(phi)  # degrees
    return phi_deg


def pa_quadrupole_alignment(theta_deg, phi_deg, weights=None, normalize=True):
    """
    Quadrupolar (position-dependent) nematic alignment.

    theta_deg : array-like
        Directionless position angles of the stars, e.g. [-90, 90] degrees.
    phi_deg   : array-like
        Azimuthal angles of the star positions (e.g. from azimuth_deg_from_center),
        in degrees, same shape as theta_deg.
    weights   : optional nonnegative weights, same shape as theta_deg

    Returns:
      Q_amp             quadrupolar alignment score in [0, 1]
      rel_axis_deg      mean *relative* axis (theta - phi), degrees in [-90, 90)
      V                 circular variance for the relative axis (1 - Q_amp)
      sd_deg            circular stdev of relative axis (approx), degrees
      Q_rel_tensor      2x2 nematic tensor in the relative frame (diagnostics)
    """
    theta = np.deg2rad(np.asarray(theta_deg))
    phi = np.deg2rad(np.asarray(phi_deg))
    assert theta.shape == phi.shape, "theta_deg and phi_deg must have same shape"

    if weights is None:
        w = np.ones_like(theta, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    if normalize:
        wsum = w.sum()
    else:
        wsum = len(w)
    assert wsum > 0, "Sum of weights must be > 0"

    # Relative angle in the local (position-dependent) frame
    alpha = theta - phi

    # Spin-2 trick on the relative angle
    c2 = np.cos(2 * alpha)
    s2 = np.sin(2 * alpha)

    C = np.sum(w * c2) / wsum
    S = np.sum(w * s2) / wsum

    Q_amp = np.hypot(C, S)  # quadrupole order parameter in [0,1]
    rel_axis = 0.5 * np.arctan2(S, C)  # mean *relative* axis (radians)
    rel_axis_deg = np.rad2deg(rel_axis)  # in [-90, 90)

    V = 1.0 - Q_amp

    # Same approximation for circular stdev, now applied to Q_amp
    sd_rad = 0.5 * np.sqrt(max(0.0, -2.0 * np.log(max(Q_amp, 1e-12))))
    sd_deg = np.rad2deg(sd_rad)

    # Nematic tensor in the relative frame
    Q_rel_tensor = 0.5 * np.array([[C, S], [S, -C]])

    return Q_amp, rel_axis_deg, V, sd_deg, Q_rel_tensor
