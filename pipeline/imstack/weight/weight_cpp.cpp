#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void compute_sig_r(
    py::array_t<float> sci,
    py::array_t<float> flat,
    py::array_t<float> dark,
    float gain,
    py::array_t<float> sig_z,
    py::array_t<float> out
) {
    auto sci_buf = sci.unchecked<2>();
    auto flat_buf = flat.unchecked<2>();
    auto dark_buf = dark.unchecked<2>();
    auto sig_z_buf = sig_z.unchecked<2>();
    auto out_buf = out.mutable_unchecked<2>();

    const int h = sci.shape(0);
    const int w = sci.shape(1);

    #pragma omp parallel for
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float poisson_component = sci_buf(i, j) * flat_buf(i, j) + dark_buf(i, j);
            float clipped = std::max(poisson_component, 0.0f);
            float sig_z_val = sig_z_buf(i, j);
            out_buf(i, j) = clipped / gain + sig_z_val * sig_z_val;
        }
    }
}

void compute_sig_rp(
    py::array_t<float> sig_r_squared,
    py::array_t<float> sig_zm,
    py::array_t<float> sig_dm_sq,
    py::array_t<float> f_m,
    py::array_t<float> r_p,
    py::array_t<float> sig_fm,
    py::array_t<float> out
) {
    auto sig_r_sq_buf = sig_r_squared.unchecked<2>();
    auto sig_zm_buf = sig_zm.unchecked<2>();
    auto sig_dm_sq_buf = sig_dm_sq.unchecked<2>();
    auto f_m_buf = f_m.unchecked<2>();
    auto r_p_buf = r_p.unchecked<2>();
    auto sig_fm_buf = sig_fm.unchecked<2>();
    auto out_buf = out.mutable_unchecked<2>();

    const int h = sig_r_squared.shape(0);
    const int w = sig_r_squared.shape(1);

    #pragma omp parallel for
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float fm_val = f_m_buf(i, j);
            float fm_sq = fm_val * fm_val;
            float sig_zm_val = sig_zm_buf(i, j);
            float sig_zm_sq = sig_zm_val * sig_zm_val;
            float sig_fm_val = sig_fm_buf(i, j);
            float sig_fm_sq = sig_fm_val * sig_fm_val;
            float r_p_val = r_p_buf(i, j);
            
            float term1 = (sig_r_sq_buf(i, j) + sig_zm_sq + sig_dm_sq_buf(i, j)) / fm_sq;
            float term2 = (r_p_val * r_p_val) * sig_fm_sq / fm_sq;
            out_buf(i, j) = term1 + term2;
        }
    }
}

void compute_final_weight(
    py::array_t<float> sig_rp_sq,
    py::array_t<float> sig_b_squared,
    bool weight,
    py::array_t<float> out
) {
    auto sig_rp_sq_buf = sig_rp_sq.unchecked<2>();
    auto sig_b_sq_buf = sig_b_squared.unchecked<2>();
    auto out_buf = out.mutable_unchecked<2>();

    const int h = sig_rp_sq.shape(0);
    const int w = sig_rp_sq.shape(1);

    #pragma omp parallel for
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float sum = sig_rp_sq_buf(i, j) + sig_b_sq_buf(i, j);
            if (weight) {
                out_buf(i, j) = 1.0f / sum;
            } else {
                out_buf(i, j) = std::sqrt(sum);
            }
        }
    }
}

// Combined function that does everything in one pass (most efficient)
void compute_weight_combined(
    py::array_t<float> sci,
    py::array_t<float> flat,
    py::array_t<float> dark,
    float gain,
    py::array_t<float> sig_z,
    py::array_t<float> sig_zm,
    py::array_t<float> sig_dm_sq,
    py::array_t<float> f_m,
    py::array_t<float> sig_fm,
    py::array_t<float> sig_b_squared,
    bool weight,
    py::array_t<float> out
) {
    auto sci_buf = sci.unchecked<2>();
    auto flat_buf = flat.unchecked<2>();
    auto dark_buf = dark.unchecked<2>();
    auto sig_z_buf = sig_z.unchecked<2>();
    auto sig_zm_buf = sig_zm.unchecked<2>();
    auto sig_dm_sq_buf = sig_dm_sq.unchecked<2>();
    auto f_m_buf = f_m.unchecked<2>();
    auto sig_fm_buf = sig_fm.unchecked<2>();
    auto sig_b_sq_buf = sig_b_squared.unchecked<2>();
    auto out_buf = out.mutable_unchecked<2>();

    const int h = sci.shape(0);
    const int w = sci.shape(1);

    #pragma omp parallel for
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            // Step 1: compute sig_r_squared
            float poisson_component = sci_buf(i, j) * flat_buf(i, j) + dark_buf(i, j);
            float clipped = std::max(poisson_component, 0.0f);
            float sig_z_val = sig_z_buf(i, j);
            float sig_r_sq = clipped / gain + sig_z_val * sig_z_val;
            
            // Step 2: compute sig_rp_squared
            float fm_val = f_m_buf(i, j);
            float fm_sq = fm_val * fm_val;
            float sig_zm_val = sig_zm_buf(i, j);
            float sig_zm_sq = sig_zm_val * sig_zm_val;
            float sig_fm_val = sig_fm_buf(i, j);
            float sig_fm_sq = sig_fm_val * sig_fm_val;
            float r_p_val = sci_buf(i, j);
            
            float term1 = (sig_r_sq + sig_zm_sq + sig_dm_sq_buf(i, j)) / fm_sq;
            float term2 = (r_p_val * r_p_val) * sig_fm_sq / fm_sq;
            float sig_rp_sq = term1 + term2;
            
            // Step 3: compute final weight
            float sum = sig_rp_sq + sig_b_sq_buf(i, j);
            if (weight) {
                out_buf(i, j) = 1.0f / sum;
            } else {
                out_buf(i, j) = std::sqrt(sum);
            }
        }
    }
}

PYBIND11_MODULE(weight_cpp, m) {
    m.doc() = "Fast C++ implementation of weightmap calculation";
    m.def("compute_sig_r", &compute_sig_r, 
          "Compute sigma_r squared",
          py::arg("sci"), py::arg("flat"), py::arg("dark"), 
          py::arg("gain"), py::arg("sig_z"), py::arg("out"));
    m.def("compute_sig_rp", &compute_sig_rp,
          "Compute sigma_rp squared",
          py::arg("sig_r_squared"), py::arg("sig_zm"), py::arg("sig_dm_sq"),
          py::arg("f_m"), py::arg("r_p"), py::arg("sig_fm"), py::arg("out"));
    m.def("compute_final_weight", &compute_final_weight,
          "Compute final weight or sigma from sig_rp_squared",
          py::arg("sig_rp_sq"), py::arg("sig_b_squared"), py::arg("weight"), py::arg("out"));
    m.def("compute_weight_combined", &compute_weight_combined,
          "Compute complete weightmap in one pass (most efficient)",
          py::arg("sci"), py::arg("flat"), py::arg("dark"), py::arg("gain"),
          py::arg("sig_z"), py::arg("sig_zm"), py::arg("sig_dm_sq"),
          py::arg("f_m"), py::arg("sig_fm"), py::arg("sig_b_squared"),
          py::arg("weight"), py::arg("out"));
}
