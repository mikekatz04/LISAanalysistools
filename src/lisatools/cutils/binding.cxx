#include "Detector.hpp"
#include "PSD.hpp"
#include "galactic_response.hpp"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding.hpp"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#endif

namespace py = pybind11;

// ============================================================================
// OrbitsWrap method implementations (unchanged)
// ============================================================================

void OrbitsWrap::get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num)
{
    orbits->get_light_travel_time_arr(
        return_pointer_and_check_length(ltt,  "ltt",  num, 1),
        return_pointer_and_check_length(t,    "t",    num, 1),
        return_pointer_and_check_length(link, "sc",   num, 1),
        num
    );
}

void OrbitsWrap::get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num)
{
    orbits->get_pos_arr(
        return_pointer_and_check_length(pos_x, "pos_x", num, 1),
        return_pointer_and_check_length(pos_y, "pos_y", num, 1),
        return_pointer_and_check_length(pos_z, "pos_z", num, 1),
        return_pointer_and_check_length(t,     "t",     num, 1),
        return_pointer_and_check_length(sc,    "sc",    num, 1),
        num
    );
}

void OrbitsWrap::get_normal_unit_vec_wrap(array_type<double> normal_unit_vec_x, array_type<double> normal_unit_vec_y, array_type<double> normal_unit_vec_z, array_type<double> t, array_type<int> link, int num)
{
    orbits->get_normal_unit_vec_arr(
        return_pointer_and_check_length(normal_unit_vec_x, "n_arr_x", num, 1),
        return_pointer_and_check_length(normal_unit_vec_y, "n_arr_y", num, 1),
        return_pointer_and_check_length(normal_unit_vec_z, "n_arr_z", num, 1),
        return_pointer_and_check_length(t,    "t",    num, 1),
        return_pointer_and_check_length(link, "link", num, 1),
        num
    );
}

// ============================================================================
// XYZSensitivityMatrixWrap method implementations
// ============================================================================

void XYZSensitivityMatrixWrap::get_noise_tfs_wrap(
    array_type<double> freqs,
    array_type<double> oms_xx, array_type<std::complex<double>> oms_xy, array_type<std::complex<double>> oms_xz,
    array_type<double> oms_yy, array_type<std::complex<double>> oms_yz, array_type<double> oms_zz,
    array_type<double> tm_xx,  array_type<std::complex<double>> tm_xy,  array_type<std::complex<double>> tm_xz,
    array_type<double> tm_yy,  array_type<std::complex<double>> tm_yz,  array_type<double> tm_zz,
    int num_freqs, int num_times,
    array_type<int> time_indices)
{
    sensitivity_matrix->get_noise_tfs_arr(
        return_pointer_and_check_length(freqs,  "freqs",  num_freqs,              1),
        return_pointer_and_check_length(oms_xx, "oms_xx", num_freqs * num_times,  1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_xy, "oms_xy", num_freqs * num_times, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_xz, "oms_xz", num_freqs * num_times, 1)),
        return_pointer_and_check_length(oms_yy, "oms_yy", num_freqs * num_times,  1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_yz, "oms_yz", num_freqs * num_times, 1)),
        return_pointer_and_check_length(oms_zz, "oms_zz", num_freqs * num_times,  1),
        return_pointer_and_check_length(tm_xx,  "tm_xx",  num_freqs * num_times,  1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_xy,  "tm_xy",  num_freqs * num_times, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_xz,  "tm_xz",  num_freqs * num_times, 1)),
        return_pointer_and_check_length(tm_yy,  "tm_yy",  num_freqs * num_times,  1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_yz,  "tm_yz",  num_freqs * num_times, 1)),
        return_pointer_and_check_length(tm_zz,  "tm_zz",  num_freqs * num_times,  1),
        num_freqs, num_times,
        return_pointer_and_check_length(time_indices, "time_indices", num_times, 1)
    );
}

void XYZSensitivityMatrixWrap::psd_likelihood_wrap(
    array_type<double> like_contrib_final, array_type<double> f_arr,
    array_type<std::complex<double>> data,
    array_type<int> data_index_all, array_type<int> time_index_all,
    array_type<double> Soms_d_in_all, array_type<double> Sa_a_in_all,
    array_type<double> Amp_all, array_type<double> alpha_all,
    array_type<double> f_1_all, array_type<double> f_knee_all, array_type<double> f_2_all,
    array_type<double> spline_in_isi_oms_all, array_type<double> spline_in_testmass_all,
    double differential_component, int num_freqs, int num_times,
    array_type<bool> dips_mask, int num_psds)
{
    int total_tf_pairs = num_times * num_freqs;
    sensitivity_matrix->psd_likelihood_wrap(
        return_pointer_and_check_length(like_contrib_final, "like_contrib_final", num_psds,                  1),
        return_pointer_and_check_length(f_arr,              "f_arr",              num_freqs,                 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(data, "data", num_psds * 3 * total_tf_pairs, 1)),
        return_pointer_and_check_length(data_index_all,     "data_index_all",     num_psds,                  1),
        return_pointer_and_check_length(time_index_all,     "time_index_all",     num_times,                 1),
        return_pointer_and_check_length(Soms_d_in_all,      "Soms_d_in_all",      num_psds,                  1),
        return_pointer_and_check_length(Sa_a_in_all,        "Sa_a_in_all",        num_psds,                  1),
        return_pointer_and_check_length(Amp_all,            "Amp_all",            num_psds,                  1),
        return_pointer_and_check_length(alpha_all,          "alpha_all",          num_psds,                  1),
        return_pointer_and_check_length(f_1_all,            "f_1_all",            num_psds,                  1),
        return_pointer_and_check_length(f_knee_all,         "f_knee_all",         num_psds,                  1),
        return_pointer_and_check_length(f_2_all,            "f_2_all",            num_psds,                  1),
        return_pointer_and_check_length(spline_in_isi_oms_all,  "spline_in_isi_oms_all",  num_psds * num_freqs, 1),
        return_pointer_and_check_length(spline_in_testmass_all, "spline_in_testmass_all", num_psds * num_freqs, 1),
        differential_component,
        num_freqs, num_times,
        return_pointer_and_check_length(dips_mask, "dips_mask", num_times * num_freqs, 1),
        num_psds
    );
}

void XYZSensitivityMatrixWrap::get_noise_covariance_wrap(
    array_type<double> freqs, array_type<int> time_indices,
    double Soms_d_in, double Sa_a_in,
    double Amp, double alpha, double f_1, double f_knee, double f_2,
    array_type<double> spline_in_isi_oms_arr, array_type<double> spline_in_testmass_arr,
    array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
    array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
    int num_freqs, int num_times)
{
    int total_size = num_freqs * num_times;
    sensitivity_matrix->get_noise_covariance_arr(
        return_pointer_and_check_length(freqs,        "freqs",        num_freqs, 1),
        return_pointer_and_check_length(time_indices, "time_indices", num_times, 1),
        Soms_d_in, Sa_a_in,
        Amp, alpha, f_1, f_knee, f_2,
        return_pointer_and_check_length(spline_in_isi_oms_arr,  "spline_in_isi_oms_arr",  num_freqs, 1),
        return_pointer_and_check_length(spline_in_testmass_arr, "spline_in_testmass_arr", num_freqs, 1),
        return_pointer_and_check_length(c00_arr, "c00_arr", total_size, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c01_arr, "c01_arr", total_size, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c02_arr, "c02_arr", total_size, 1)),
        return_pointer_and_check_length(c11_arr, "c11_arr", total_size, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c12_arr, "c12_arr", total_size, 1)),
        return_pointer_and_check_length(c22_arr, "c22_arr", total_size, 1),
        num_freqs, num_times
    );
}

void XYZSensitivityMatrixWrap::get_inverse_det_wrap(
    array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
    array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
    array_type<double> i00_arr, array_type<std::complex<double>> i01_arr, array_type<std::complex<double>> i02_arr,
    array_type<double> i11_arr, array_type<std::complex<double>> i12_arr, array_type<double> i22_arr,
    array_type<double> det_arr,
    int num)
{
    sensitivity_matrix->get_inverse_det_arr(
        return_pointer_and_check_length(c00_arr, "c00_arr", num, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c01_arr, "c01_arr", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c02_arr, "c02_arr", num, 1)),
        return_pointer_and_check_length(c11_arr, "c11_arr", num, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c12_arr, "c12_arr", num, 1)),
        return_pointer_and_check_length(c22_arr, "c22_arr", num, 1),
        return_pointer_and_check_length(i00_arr, "i00_arr", num, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(i01_arr, "i01_arr", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(i02_arr, "i02_arr", num, 1)),
        return_pointer_and_check_length(i11_arr, "i11_arr", num, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(i12_arr, "i12_arr", num, 1)),
        return_pointer_and_check_length(i22_arr, "i22_arr", num, 1),
        return_pointer_and_check_length(det_arr, "det_arr", num, 1),
        num
    );
}

// ============================================================================
// Legacy free-function wrappers (unchanged)
// ============================================================================

template<typename T>
T* return_ptr(array_type<T> input1, std::string name, int N, int multiplier)
{
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    T *ptr1 = static_cast<T *>(input1.get_compatible_typed_pointer());
#else
    py::buffer_info buf1 = input1.request();
    if (buf1.size != N * multiplier)
    {
        std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(N * multiplier) + ". It's length is " + std::to_string(buf1.size) + ".";
        throw std::invalid_argument(err_out);
    }
    T* ptr1 = static_cast<T *>(buf1.ptr);
#endif
    return ptr1;
};

void psd_likelihood_legacy_wrap(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                         array_type<int> data_index_all, array_type<double> Soms_d_in_all, array_type<double> Sa_a_in_all, array_type<double> E_Soms_d_in_all, array_type<double> E_Sa_a_in_all, 
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> f_1_all, array_type<double> kn_all, array_type<double> f_2_all, double df, int data_length, int num_data, int num_psds)
{
    psd_likelihood_wrap(
        return_ptr(like_contrib_final, "like_contrib_final", num_psds, 1),
        return_ptr(f_arr,              "f_arr",              data_length, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_ptr(data, "data", data_length * num_data, 1)),
        return_ptr(data_index_all,  "data_index_all",  num_psds, 1),
        return_ptr(Soms_d_in_all,   "Soms_d_in_all",   num_psds, 1),
        return_ptr(Sa_a_in_all,     "Sa_a_in_all",     num_psds, 1),
        return_ptr(E_Soms_d_in_all, "E_Soms_d_in_all", num_psds, 1),
        return_ptr(E_Sa_a_in_all,   "E_Sa_a_in_all",   num_psds, 1),
        return_ptr(Amp_all,         "Amp_all",          num_psds, 1),
        return_ptr(alpha_all,       "alpha_all",        num_psds, 1),
        return_ptr(f_1_all,         "f_1_all",          num_psds, 1),
        return_ptr(kn_all,          "kn_all",           num_psds, 1),
        return_ptr(f_2_all,         "f_2_all",          num_psds, 1),
        df, data_length, num_data, num_psds
    );
}

void get_psd_val_legacy_wrap(array_type<double> Sn_A_out, array_type<double> Sn_E_out, array_type<double> f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double f_1, double kn, double f_2, int num_f)
{
    get_psd_val_wrap(
        return_ptr(Sn_A_out, "Sn_A_out", num_f, 1),
        return_ptr(Sn_E_out, "Sn_E_out", num_f, 1),
        return_ptr(f_arr,    "f_arr",    num_f, 1),
        A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in,
        Amp, alpha, f_1, kn, f_2, num_f
    );
}

void psd_likelihood_binding(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                         array_type<int> data_index_all, array_type<double> Soms_d_in_all, array_type<double> Sa_a_in_all, array_type<double> E_Soms_d_in_all, array_type<double> E_Sa_a_in_all, 
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> f_1_all, array_type<double> kn_all, array_type<double> f_2_all, double df, int data_length, int num_data, int num_psds)
{
    psd_likelihood_wrap(
        return_ptr(like_contrib_final, "like_contrib_final", num_psds, 1),
        return_ptr(f_arr,              "f_arr",              data_length, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_ptr(data, "data", data_length * num_data, 1)),
        return_ptr(data_index_all,  "data_index_all",  num_psds, 1),
        return_ptr(Soms_d_in_all,   "Soms_d_in_all",   num_psds, 1),
        return_ptr(Sa_a_in_all,     "Sa_a_in_all",     num_psds, 1),
        return_ptr(E_Soms_d_in_all, "E_Soms_d_in_all", num_psds, 1),
        return_ptr(E_Sa_a_in_all,   "E_Sa_a_in_all",   num_psds, 1),
        return_ptr(Amp_all,         "Amp_all",          num_psds, 1),
        return_ptr(alpha_all,       "alpha_all",        num_psds, 1),
        return_ptr(f_1_all,         "f_1_all",          num_psds, 1),
        return_ptr(kn_all,          "kn_all",           num_psds, 1),
        return_ptr(f_2_all,         "f_2_all",          num_psds, 1),
        df, data_length, num_data, num_psds
    );
}

void compute_logpdf_binding(array_type<double> logpdf_out, array_type<int> component_index, array_type<double> points,
                    array_type<double> weights, array_type<double> mins, array_type<double> maxs, 
                    array_type<double> means, array_type<double> invcovs, array_type<double> dets, array_type<double> log_Js, 
                    int num_points, array_type<int> start_index, int num_components, int ndim)
{
    compute_logpdf_wrap(
        return_ptr(logpdf_out,        "logpdf_out",   num_points,                  1),
        return_ptr(component_index,   "component_index", num_points,               1),
        return_ptr(points,            "points",       num_points * ndim,           1),
        return_ptr(weights,           "weights",      num_components,              1),
        return_ptr(mins,              "mins",         num_components * ndim,       1),
        return_ptr(maxs,              "maxs",         num_components * ndim,       1),
        return_ptr(means,             "means",        num_components * ndim,       1),
        return_ptr(invcovs,           "invcovs",      num_components * ndim * ndim, 1),
        return_ptr(dets,              "dets",         num_components,              1),
        return_ptr(log_Js,            "log_Js",       num_components,              1),
        num_points,
        return_ptr(start_index, "start_index", num_components + 1, 1),
        num_components, ndim
    );
}

// ============================================================================
// Misc helpers (unchanged)
// ============================================================================

void check_12()
{
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    printf("CHECK 12 GOOD\n");
#else
    printf("CHECK 12 BAD\n");
#endif
}

void check_orbits(Orbits *orbits)
{
    printf("%e\n", orbits->x_arr[0]);
}

std::string get_module_path() {
    py::gil_scoped_acquire acquire;
    py::object module = py::module::import("pycppdetector");
    try {
        return module.attr("__file__").cast<std::string>();
    } catch (const py::error_already_set& e) {
        std::cerr << "Error getting __file__ attribute: " << e.what() << std::endl;
        return "";
    }
}

// ============================================================================
// pybind11 module
// ============================================================================

void detector_part(py::module &m) {

    // ---- OrbitsWrap ----
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<OrbitsWrap>(m, "OrbitsWrapGPU")
#else
    py::class_<OrbitsWrap>(m, "OrbitsWrapCPU")
#endif
    .def(py::init<double, double, int, double, double, int,
                  array_type<double>, array_type<double>, array_type<double>,
                  array_type<int>, array_type<int>, array_type<int>, double>(),
         py::arg("sc_t0"), py::arg("sc_dt"), py::arg("sc_N"),
         py::arg("ltt_t0"), py::arg("ltt_dt"), py::arg("ltt_N"),
         py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"),
         py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))
    .def("get_light_travel_time_wrap", &OrbitsWrap::get_light_travel_time_wrap)
    .def("get_pos_wrap",               &OrbitsWrap::get_pos_wrap)
    .def("get_normal_unit_vec_wrap",   &OrbitsWrap::get_normal_unit_vec_wrap)
    .def_readwrite("orbits", &OrbitsWrap::orbits)
    .def("__copy__",     [](const OrbitsWrap &self) { return OrbitsWrap(self); })
    .def("__deepcopy__", [](const OrbitsWrap &self, py::dict) { return OrbitsWrap(self); });

    // ---- Orbits ----
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<Orbits>(m, "OrbitsGPU")
#else
    py::class_<Orbits>(m, "OrbitsCPU")
#endif
    .def(py::init<double, double, int, double, double, int,
                  double *, double *, double *, int *, int *, int *, double>(),
         py::arg("sc_t0"), py::arg("sc_dt"), py::arg("sc_N"),
         py::arg("ltt_t0"), py::arg("ltt_dt"), py::arg("ltt_N"),
         py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"),
         py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"));

    // ---- GalacticGridSetup ----
    py::class_<GalacticGridSetup>(m, "GalacticGridSetup", py::module_local())
    .def(py::init<>())
    .def("compute", &GalacticGridSetup::compute,
         py::arg("N_lambda") = 90, py::arg("N_beta") = 60)
    .def("print_summary", &GalacticGridSetup::print_summary)
    .def_readonly("N_lambda", &GalacticGridSetup::N_lambda)
    .def_readonly("N_beta",   &GalacticGridSetup::N_beta)
    .def_readonly("N_sky",    &GalacticGridSetup::N_sky)
    .def_readonly("N_quad",   &GalacticGridSetup::N_quad)
    .def_property_readonly("lam_ecl", [](GalacticGridSetup &s) {
        return py::array_t<double>(s.lam_ecl.size(), s.lam_ecl.data());
    })
    .def_property_readonly("beta_ecl", [](GalacticGridSetup &s) {
        return py::array_t<double>(s.beta_ecl.size(), s.beta_ecl.data());
    })
    .def_property_readonly("cos_beta_ecl", [](GalacticGridSetup &s) {
        return py::array_t<double>(s.cos_beta_ecl.size(), s.cos_beta_ecl.data());
    })
    .def_property_readonly("quad_weights", [](GalacticGridSetup &s) {
        return py::array_t<double>(s.quad_weights.size(), s.quad_weights.data());
    })
    .def_property_readonly("R_vals_quad", [](GalacticGridSetup &s) {
        return py::array_t<double>(s.R_vals_quad.size(), s.R_vals_quad.data());
    })
    .def_property_readonly("z_vals_quad", [](GalacticGridSetup &s) {
        return py::array_t<double>(s.z_vals_quad.size(), s.z_vals_quad.data());
    });

    // ---- GalacticGridWrap ----
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<GalacticGridWrap>(m, "GalacticGridWrapGPU")
#else
    py::class_<GalacticGridWrap>(m, "GalacticGridWrapCPU")
#endif
    .def(py::init<
             array_type<double>, array_type<double>,  // R_vals_quad, z_vals_quad
             array_type<double>, array_type<double>,  // quad_weights, cos_beta_ecl
             array_type<double>, array_type<double>,  // lam_ecl, beta_ecl
             int, int,                                // N_quad, N_sky
             double, double,                          // alpha0, beta0
             int, int>(),                             // N_times, N_freqs
         py::arg("R_vals_quad"), py::arg("z_vals_quad"),
         py::arg("quad_weights"), py::arg("cos_beta_ecl"),
         py::arg("lam_ecl"), py::arg("beta_ecl"),
         py::arg("N_quad"), py::arg("N_sky"),
         py::arg("alpha0"), py::arg("beta0"),
         py::arg("N_times"), py::arg("N_freqs"),
         "Construct and allocate galactic grid on device.\n"
         "Call initialize_wrap() once before inference starts.")
    .def("initialize_wrap", &GalacticGridWrap::initialize_wrap,
         py::arg("times"), py::arg("R_d"), py::arg("z_d"), py::arg("N_times"),
         "Compute fixed sky weights and R_avg from disk parameters and orbit times.\n"
         "Must be called once before passing this object to set_galactic_grid().")
    .def("compute_gal_covariance_wrap", &GalacticGridWrap::compute_gal_covariance_wrap,
         py::arg("freqs"),
         py::arg("Amp"), py::arg("alpha"),
            py::arg("f_1"), py::arg("f_knee"), py::arg("f_2"),
         py::arg("avg_d"), py::arg("N_freqs"), py::arg("N_times"),
         "Diagnostic: compute R_gal_arr = R_avg * S_gal(f) for a single parameter set.\n"
         "Not used on the inference hot path.");

    // ---- XYZSensitivityMatrixWrap ----
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<XYZSensitivityMatrixWrap>(m, "XYZSensitivityMatrixWrapGPU")
#else
    py::class_<XYZSensitivityMatrixWrap>(m, "XYZSensitivityMatrixWrapCPU")
#endif
    .def(py::init<array_type<double>, array_type<double>, int, double, int, bool>(),
         py::arg("averaged_ltts_arr"), py::arg("delta_ltts_arr"),
         py::arg("n_times"), py::arg("armlength"), py::arg("generation"), py::arg("spline_noise"))
    .def("set_galactic_grid", &XYZSensitivityMatrixWrap::set_galactic_grid,
         py::arg("gal_wrap"),
         "Attach a GalacticGridWrap (already initialized) to include the galactic\n"
         "foreground in the likelihood.  Pass None to disable.")
    .def("disable_galactic_grid", &XYZSensitivityMatrixWrap::disable_galactic_grid,
         "Detach galactic grid (equivalent to set_galactic_grid(None)).")
    .def("get_noise_tfs_wrap",        &XYZSensitivityMatrixWrap::get_noise_tfs_wrap)
    .def("psd_likelihood_wrap",       &XYZSensitivityMatrixWrap::psd_likelihood_wrap)
    .def("get_noise_covariance_wrap", &XYZSensitivityMatrixWrap::get_noise_covariance_wrap)
    .def("get_inverse_det_wrap",      &XYZSensitivityMatrixWrap::get_inverse_det_wrap)
    .def_readwrite("sensitivity_matrix", &XYZSensitivityMatrixWrap::sensitivity_matrix)
    .def("__copy__",     [](const XYZSensitivityMatrixWrap &self) { return XYZSensitivityMatrixWrap(self); })
    .def("__deepcopy__", [](const XYZSensitivityMatrixWrap &self, py::dict) { return XYZSensitivityMatrixWrap(self); });

    // ---- XYZSensitivityMatrix (raw) ----
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<XYZSensitivityMatrix>(m, "XYZSensitivityMatrixGPU")
#else
    py::class_<XYZSensitivityMatrix>(m, "XYZSensitivityMatrixCPU")
#endif
    .def(py::init<double *, double *, int, double, int, bool>(),
         py::arg("averaged_ltts_arr"), py::arg("delta_ltts_arr"),
         py::arg("n_times"), py::arg("armlength"), py::arg("generation"), py::arg("spline_noise"));

    // ---- Legacy free functions ----
    m.def("psd_likelihood_legacy_wrap", &psd_likelihood_legacy_wrap);
    m.def("get_psd_val_legacy_wrap",    &get_psd_val_legacy_wrap);
    m.def("psd_likelihood",             &psd_likelihood_binding);
    m.def("compute_logpdf",             &compute_logpdf_binding);
}


PYBIND11_MODULE(pycppdetector, m) {
    m.doc() = "Orbits/Detector C++ plug-in";

    detector_part(m);
    m.def("check_orbits",       &check_orbits);
    m.def("get_module_path_cpp", &get_module_path);
    m.def("check_12",           &check_12);

    try {
        std::string path_at_init = m.attr("__file__").cast<std::string>();
        m.attr("module_dir") = py::cast(path_at_init.substr(0, path_at_init.find_last_of("/\\")));
    } catch (py::error_already_set &e) {
        std::cerr << "Could not capture __file__ at init time." << std::endl;
        e.restore();
        PyErr_Clear();
    }
}
