import numpy as np


def lupton_asinh(img: np.array, sky: float, noise: float, hi_clip=None, k_soft=3.0, vmin_sigma=-1.0):
    """
    img: raw image
    sky, noise: from estimate_sky_noise_annulus
    hi_clip: value to clip the top end (e.g. 99.7th percentile). If None, compute.
    k_soft: softening in σ; smaller -> more compression of bright core
    vmin_sigma: map sky + vmin_sigma*noise to display 0
    """
    if hi_clip is None:
        hi_clip = np.percentile(img, 99.7)

    soft = k_soft * noise  # the “Q*σ” softening scale
    # shift by sky and clamp low values a bit below sky
    shifted = img - sky
    shifted = np.maximum(shifted, vmin_sigma * noise)
    shifted = np.minimum(shifted, hi_clip - sky)

    # Lupton-style scaling to [0,1]
    num = np.arcsinh(shifted / soft)
    den = np.arcsinh((hi_clip - sky) / soft)
    scaled = num / den
    return np.clip(scaled, 0, 1)
