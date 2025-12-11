from astropy.io import fits
from numba import njit, prange
from ..utils import add_suffix
import numpy as np

# Try to import C++ implementation, fallback to Numba if not available
_HAS_CPP = False
try:
    import importlib.util
    from pathlib import Path
    import glob

    # The C++ module is in the weight/ subdirectory
    weight_dir = Path(__file__).parent / "weight"
    # Search for weight_cpp*.so or weight_cpp*.pyd (handles platform-specific suffixes)
    pattern = str(weight_dir / "weight_cpp*.so")
    matches = glob.glob(pattern)
    if not matches:
        pattern = str(weight_dir / "weight_cpp*.pyd")
        matches = glob.glob(pattern)

    if matches:
        cpp_module_path = Path(matches[0])
        spec = importlib.util.spec_from_file_location("weight_cpp", cpp_module_path)
        weight_cpp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(weight_cpp)
        _compute_sig_r_cpp = weight_cpp.compute_sig_r
        _compute_sig_rp_cpp = weight_cpp.compute_sig_rp
        _compute_final_weight_cpp = weight_cpp.compute_final_weight
        _compute_weight_combined_cpp = weight_cpp.compute_weight_combined
        _HAS_CPP = True
except (ImportError, FileNotFoundError, AttributeError, ValueError):
    _HAS_CPP = False


def _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file):
    """
    Load calibration arrays and metadata shared by CPU and GPU paths.
    """
    mfg_data = []
    n_images = []
    egain = None
    for file in [sig_z_file, d_m_file, f_m_file]:
        data, header = fits.getdata(file, header=True)
        mfg_data.append(data.astype(np.float32))
        n_images.append(np.float32(header["NFRAMES"]))
        if file == d_m_file:
            egain = np.float32(header["EGAIN"])

    if egain is None:
        raise ValueError(f"Gain information missing from {d_m_file}")

    sig_z, d_m, f_m = mfg_data
    p_z, p_d, p_f = n_images
    sig_f = fits.getdata(sig_f_file).astype(np.float32)

    return sig_z, d_m, f_m, sig_f, p_z, p_d, p_f, egain


def calc_weight_with_gpu(images, d_m_file, f_m_file, sig_z_file, sig_f_file, device_id=0, weight=True):
    """
    Execute the CuPy/CUDA-based weight-map calculation in-process.
    """
    from ..cuda.weight_map import calc_weight as gpu_calc_weight

    sig_z, d_m, f_m, sig_f, p_z, p_d, p_f, egain = _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file)

    gpu_calc_weight(
        images,
        d_m,
        f_m,
        sig_z,
        sig_f,
        p_d,
        p_z,
        p_f,
        egain,
        weight=weight,
        device=device_id,
    )


def calc_weight_with_cpu(images, d_m_file, f_m_file, sig_z_file, sig_f_file, weight=True, **kwargs):
    sig_z, d_m, f_m, sig_f, p_z, p_d, p_f, egain = _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file)
    sig_b_squared = np.zeros_like(d_m, dtype=np.float32)

    sig_zm = sig_z / np.sqrt(p_z)
    sig_dm_sq = (d_m / egain + (1 + 1 / p_z) * sig_z**2) / p_d
    sig_fm = sig_f / np.sqrt(p_f)

    sig_r_sq = np.empty_like(d_m, dtype=np.float32)
    sig_rp_sq = np.empty_like(d_m, dtype=np.float32)

    out_names = add_suffix(images, suffix="weight")

    for fname, outname in zip(images, out_names):
        r_p = fits.getdata(fname).astype(np.float32)

        # Use optimized C++ combined function if available (single pass, most efficient)
        if _HAS_CPP:
            result = np.empty_like(d_m, dtype=np.float32)
            _compute_weight_combined_cpp(
                r_p, f_m, d_m, egain, sig_z, sig_zm, sig_dm_sq, f_m, sig_fm, sig_b_squared, weight, result
            )
        else:
            # Fall back to step-by-step computation with Numba
            _compute_sig_r(r_p, f_m, d_m, egain, sig_z, sig_r_sq)
            _compute_sig_rp(sig_r_sq, sig_zm, sig_dm_sq, f_m, r_p, sig_fm, sig_rp_sq)
            if weight:
                result = 1.0 / (sig_rp_sq + sig_b_squared)
            else:
                result = np.sqrt(sig_rp_sq + sig_b_squared)

        fits.writeto(outname, result.astype(np.float32), overwrite=True)


@njit(parallel=True)
def _compute_sig_r(sci, flat, dark, gain, sig_z, out):
    """
    out[i,j] = (max(sci[i,j] * flat[i,j] + dark[i,j], 0) / gain) + sig_z[i,j]**2
    """
    h, w = sci.shape
    for i in prange(h):
        for j in range(w):
            poisson_component = sci[i, j] * flat[i, j] + dark[i, j]
            clipped = poisson_component if poisson_component > 0.0 else 0.0
            out[i, j] = clipped / gain + sig_z[i, j] * sig_z[i, j]


@njit(parallel=True)
def _compute_sig_rp(sig_r_squared, sig_zm, sig_dm_sq, f_m, r_p, sig_fm, out):
    """
    out[i,j] = (sig_r_squared + sig_zm**2 + sig_dm_sq) / f_m**2
              + (r_p**2)*(sig_fm**2)/f_m**2
    """
    h, w = sig_r_squared.shape
    for i in prange(h):
        for j in range(w):
            fm_sq = f_m[i, j] * f_m[i, j]
            sig_zm_sq = sig_zm[i, j] * sig_zm[i, j]
            sig_fm_sq = sig_fm[i, j] * sig_fm[i, j]
            term1 = (sig_r_squared[i, j] + sig_zm_sq + sig_dm_sq[i, j]) / fm_sq
            term2 = (r_p[i, j] * r_p[i, j]) * sig_fm_sq / fm_sq
            out[i, j] = term1 + term2
