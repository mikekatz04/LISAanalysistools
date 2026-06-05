#include "Detector.hpp"
#include "PSD.hpp"
#include "wdm_domain.hh"             // TDI_XYZ / TDI_AET / TDI_AE macros
#include <string>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "binding.hpp"

// Phase 3J: this binding TU is the SOLE registration site for the shared
// wrapper classes (OrbitsWrap, LISAResponseWrap, TDIConfigWrap,
// CubicSplineWrap_responselisa, ...). Setting the toggle to 1 BEFORE
// including lisatools_header_abi.hpp marks this file as the owner; every
// downstream TU (lisa-on-gpu/binding_tof.cxx, future GBGPU/BBHx bindings)
// must leave the toggle at its default 0 and adds a static_assert(!toggle, ...)
// that fires at compile time if they ever try to claim ownership.
#define LISATOOLS_IS_WRAPPER_OWNER 1
#include "lisatools_header_abi.hpp"

// Phase 3J.B: ensure OrbitsView's POD layout stays byte-equal to class Orbits.
// OrbitsView is the cross-wheel POD interface downstream packages consume in
// place of the typed Orbits* pointer (see plan section "POD-view side-channel").
// Asserting layout equivalence at every LAT build means a future change to
// either struct that introduces drift fails at compile time, instead of
// producing silently-wrong cross-wheel field reads at runtime.
#include "orbits_view.hpp"
#include <cstddef>
static_assert(sizeof(Orbits) == sizeof(OrbitsView),
    "OrbitsView size drift vs class Orbits -- bump LISATOOLS_HEADER_ABI_VERSION");
static_assert(offsetof(Orbits, sc_t0)       == offsetof(OrbitsView, sc_t0),       "OrbitsView.sc_t0 layout drift");
static_assert(offsetof(Orbits, sc_dt)       == offsetof(OrbitsView, sc_dt),       "OrbitsView.sc_dt layout drift");
static_assert(offsetof(Orbits, sc_N)        == offsetof(OrbitsView, sc_N),        "OrbitsView.sc_N layout drift");
static_assert(offsetof(Orbits, ltt_t0)      == offsetof(OrbitsView, ltt_t0),      "OrbitsView.ltt_t0 layout drift");
static_assert(offsetof(Orbits, ltt_dt)      == offsetof(OrbitsView, ltt_dt),      "OrbitsView.ltt_dt layout drift");
static_assert(offsetof(Orbits, ltt_N)       == offsetof(OrbitsView, ltt_N),       "OrbitsView.ltt_N layout drift");
static_assert(offsetof(Orbits, n_arr)       == offsetof(OrbitsView, n_arr),       "OrbitsView.n_arr layout drift");
static_assert(offsetof(Orbits, ltt_arr)     == offsetof(OrbitsView, ltt_arr),     "OrbitsView.ltt_arr layout drift");
static_assert(offsetof(Orbits, x_arr)       == offsetof(OrbitsView, x_arr),       "OrbitsView.x_arr layout drift");
static_assert(offsetof(Orbits, nlinks)      == offsetof(OrbitsView, nlinks),      "OrbitsView.nlinks layout drift");
static_assert(offsetof(Orbits, nspacecraft) == offsetof(OrbitsView, nspacecraft), "OrbitsView.nspacecraft layout drift");
static_assert(offsetof(Orbits, armlength)   == offsetof(OrbitsView, armlength),   "OrbitsView.armlength layout drift");
static_assert(offsetof(Orbits, links)       == offsetof(OrbitsView, links),       "OrbitsView.links layout drift");
static_assert(offsetof(Orbits, sc_r)        == offsetof(OrbitsView, sc_r),        "OrbitsView.sc_r layout drift");
static_assert(offsetof(Orbits, sc_e)        == offsetof(OrbitsView, sc_e),        "OrbitsView.sc_e layout drift");

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#endif

namespace nb = nanobind;

// Phase 3E (2026-06-02): LISAResponse-related pybind11 bindings absorbed
// from lisa-on-gpu. The actual class definitions and `response_part(nb::module&)`
// implementation live in binding_flr.cxx, which is compiled into the
// pycppdetector pybind11 module alongside this file.
void response_part(nb::module_ &m);

void OrbitsWrap::get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num)
{
    orbits->get_light_travel_time_arr(
        return_pointer_and_check_length(ltt, "ltt", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(link, "link", num, 1),
        num
    );
}


void OrbitsWrap::get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num)
{
    orbits->get_pos_arr(
        return_pointer_and_check_length(pos_x, "pos_x", num, 1),
        return_pointer_and_check_length(pos_y, "pos_y", num, 1),
        return_pointer_and_check_length(pos_z, "pos_z", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(sc, "sc", num, 1),
        num
    );
}


void OrbitsWrap::get_normal_unit_vec_wrap(array_type<double>normal_unit_vec_x, array_type<double>normal_unit_vec_y, array_type<double>normal_unit_vec_z, array_type<double>t, array_type<int>link, int num)
{
    
// #ifdef __CUDACC__
    orbits->get_normal_unit_vec_arr(
        return_pointer_and_check_length(normal_unit_vec_x, "n_arr_x", num, 1),
        return_pointer_and_check_length(normal_unit_vec_y, "n_arr_y", num, 1),
        return_pointer_and_check_length(normal_unit_vec_z, "n_arr_z", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(link, "link", num, 1),
        num
    );
}


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

#if 0  // === XYZBackend disabled (symbol issues on Linux): XYZSensitivityMatrixWrap impls ===
void XYZSensitivityMatrixWrap::get_noise_tfs_wrap(array_type<double> freqs,
                          array_type<double> oms_xx, array_type<std::complex<double>> oms_xy, array_type<std::complex<double>> oms_xz, array_type<double> oms_yy, array_type<std::complex<double>> oms_yz, array_type<double> oms_zz,
                          array_type<double> tm_xx, array_type<std::complex<double>> tm_xy, array_type<std::complex<double>> tm_xz, array_type<double> tm_yy, array_type<std::complex<double>> tm_yz, array_type<double> tm_zz,
                          int num_freqs, int num_times,
                          array_type<int> time_indices)
{
    sensitivity_matrix->get_noise_tfs_arr(
        return_pointer_and_check_length(freqs, "freqs", num_freqs, 1),
        return_pointer_and_check_length(oms_xx, "oms_xx", num_freqs * num_times, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_xy, "oms_xy", num_freqs * num_times, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_xz, "oms_xz", num_freqs * num_times, 1)),
        return_pointer_and_check_length(oms_yy, "oms_yy", num_freqs * num_times, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_yz, "oms_yz", num_freqs * num_times, 1)),
        return_pointer_and_check_length(oms_zz, "oms_zz", num_freqs * num_times, 1),
        return_pointer_and_check_length(tm_xx, "tm_xx", num_freqs * num_times, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_xy, "tm_xy", num_freqs * num_times, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_xz, "tm_xz", num_freqs * num_times, 1)),
        return_pointer_and_check_length(tm_yy, "tm_yy", num_freqs * num_times, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_yz, "tm_yz", num_freqs * num_times, 1)),
        return_pointer_and_check_length(tm_zz, "tm_zz", num_freqs * num_times, 1),
        num_freqs,
        num_times,
        return_pointer_and_check_length(time_indices, "time_indices", num_times, 1)
    );
}

void XYZSensitivityMatrixWrap::psd_likelihood_wrap(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data,
                          array_type<int> data_index_all, array_type<int> time_index_all,
                          array_type<double> Soms_d_in_all, array_type<double> Sa_a_in_all,
                          array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> slope_1_all, array_type<double> f_knee_all, array_type<double> slope_2_all,
                          array_type<double> spline_in_isi_oms_all, array_type<double> spline_in_testmass_all,
                          double differential_component, int num_freqs, int num_times,
                          array_type<bool> dips_mask, int num_psds)
{
    int total_tf_pairs = num_times * num_freqs;
    sensitivity_matrix->psd_likelihood_wrap(
        return_pointer_and_check_length(like_contrib_final, "like_contrib_final", num_psds, 1),
        return_pointer_and_check_length(f_arr, "f_arr", num_freqs, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(data, "data", num_psds * 3 * total_tf_pairs, 1)),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_psds, 1),
        return_pointer_and_check_length(time_index_all, "time_index_all", num_times, 1),
        return_pointer_and_check_length(Soms_d_in_all, "Soms_d_in_all", num_psds, 1),
        return_pointer_and_check_length(Sa_a_in_all, "Sa_a_in_all", num_psds, 1),
        return_pointer_and_check_length(Amp_all, "Amp_all", num_psds, 1),
        return_pointer_and_check_length(alpha_all, "alpha_all", num_psds, 1),
        return_pointer_and_check_length(slope_1_all, "slope_1_all", num_psds, 1),
        return_pointer_and_check_length(f_knee_all, "f_knee_all", num_psds, 1),
        return_pointer_and_check_length(slope_2_all, "slope_2_all", num_psds, 1),
        return_pointer_and_check_length(spline_in_isi_oms_all, "spline_in_isi_oms_all", num_psds * num_freqs, 1),
        return_pointer_and_check_length(spline_in_testmass_all, "spline_in_testmass_all", num_psds * num_freqs, 1),
        differential_component,
        num_freqs,
        num_times,
        return_pointer_and_check_length(dips_mask, "dips_mask", num_times * num_freqs, 1),
        num_psds
    );
}

void XYZSensitivityMatrixWrap::get_noise_covariance_wrap(
    array_type<double> freqs, array_type<int> time_indices,
    double Soms_d_in, double Sa_a_in,
    double Amp, double alpha, double slope_1, double f_knee, double slope_2,
    array_type<double> spline_in_isi_oms_arr, array_type<double> spline_in_testmass_arr,
    array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
    array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
    int num_freqs, int num_times)
{
    int total_size = num_freqs * num_times;
    sensitivity_matrix->get_noise_covariance_arr(
        return_pointer_and_check_length(freqs, "freqs", num_freqs, 1),
        return_pointer_and_check_length(time_indices, "time_indices", num_times, 1),
        Soms_d_in, Sa_a_in,
        Amp, alpha, slope_1, f_knee, slope_2,
        return_pointer_and_check_length(spline_in_isi_oms_arr, "spline_in_isi_oms_arr", num_freqs, 1),
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
#endif  // === end XYZSensitivityMatrixWrap impls ===


std::string get_module_path() {
    // Acquire the GIL if it's not already held (safe to call multiple times)
    nb::gil_scoped_acquire acquire;

    // Import the module by its name
    // Note: The module name here ("pycppdetector") must match the name used in NB_MODULE
    nb::object module = nb::module_::import_("pycppdetector");

    // Access the __file__ attribute and cast it to a C++ string
    try {
        std::string path = nb::cast<std::string>(module.attr("__file__"));
        return path;
    } catch (const nb::python_error& e) {
        // Handle the error if __file__ attribute is missing (e.g., if module is a namespace package)
        std::cerr << "Error getting __file__ attribute: " << e.what() << std::endl;
        return "";
    }
}

// NB_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt



template<typename T>
T* return_ptr(array_type<T> input1, std::string name, int N, int multiplier)
{
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = input1.data();
        
#else
        if (input1.size() != static_cast<size_t>(N) * static_cast<size_t>(multiplier))
        {
            std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(static_cast<size_t>(N) * static_cast<size_t>(multiplier)) + ". It's length is " + std::to_string(input1.size()) + ".";
            throw std::invalid_argument(err_out);
        }
        T* ptr1 = input1.data();
#endif
        return ptr1;
};

// void psd_likelihood_legacy_wrap(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
//                          array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all, 
//                          array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> sl1_all, array_type<double> kn_all, array_type<double> sl2_all, double df, int data_length, int num_data, int num_psds)
// {
//     psd_likelihood_wrap(
//         return_ptr(like_contrib_final, "like_contrib_final", num_psds, 1),
//         return_ptr(f_arr, "f_arr", data_length, 1),
//         reinterpret_cast<gcmplx::complex<double>*>(return_ptr(data, "data", data_length * num_data, 1)),
//         return_ptr(data_index_all, "data_index_all", num_psds, 1),
//         return_ptr(Soms_d_in_all, "Soms_d_in_all", num_psds, 1),
//         return_ptr(Sa_a_in_all, "Sa_a_in_all", num_psds, 1),
//         return_ptr(E_Soms_d_in_all, "E_Soms_d_in_all", num_psds, 1),
//         return_ptr(E_Sa_a_in_all, "E_Sa_a_in_all", num_psds, 1),
//         return_ptr(Amp_all, "Amp_all", num_psds, 1),
//         return_ptr(alpha_all, "alpha_all", num_psds, 1),
//         return_ptr(sl1_all, "sl1_all", num_psds, 1),
//         return_ptr(kn_all, "kn_all", num_psds, 1),
//         return_ptr(sl2_all, "sl2_all", num_psds, 1),
//         df, data_length, num_data, num_psds
//     );

// }

// void get_psd_val_legacy_wrap(array_type<double> Sn_A_out, array_type<double> Sn_E_out, array_type<double> f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
//                                double Amp, double alpha, double sl1, double kn, double sl2, int num_f)
// {
//     get_psd_val_wrap(
//         return_ptr(Sn_A_out, "Sn_A_out", num_f, 1),
//         return_ptr(Sn_E_out, "Sn_E_out", num_f, 1),
//         return_ptr(f_arr, "f_arr", num_f, 1),
//         A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in,
//         Amp, alpha, sl1, kn, sl2, num_f
//     );
// }

// void psd_likelihood_binding(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
//                          array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all, 
//                          array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> sl1_all, array_type<double> kn_all, array_type<double> sl2_all, double df, int data_length, int num_data, int num_psds)
// {
//     psd_likelihood_wrap(
//         return_ptr(like_contrib_final, "like_contrib_final", num_psds, 1),
//         return_ptr(f_arr, "f_arr", data_length, 1),
//         reinterpret_cast<gcmplx::complex<double>*>(return_ptr(data, "data", data_length * num_data, 1)),
//         return_ptr(data_index_all, "data_index_all", num_psds, 1),
//         return_ptr(Soms_d_in_all, "Soms_d_in_all", num_psds, 1),
//         return_ptr(Sa_a_in_all, "Sa_a_in_all", num_psds, 1),
//         return_ptr(E_Soms_d_in_all, "E_Soms_d_in_all", num_psds, 1),
//         return_ptr(E_Sa_a_in_all, "E_Sa_a_in_all", num_psds, 1),
//         return_ptr(Amp_all, "Amp_all", num_psds, 1),
//         return_ptr(alpha_all, "alpha_all", num_psds, 1),
//         return_ptr(sl1_all, "sl1_all", num_psds, 1),
//         return_ptr(kn_all, "kn_all", num_psds, 1),
//         return_ptr(sl2_all, "sl2_all", num_psds, 1),
//         df, data_length, num_data, num_psds
//     );
// }

// void compute_logpdf_binding(array_type<double> logpdf_out, array_type<int> component_index, array_type<double> points,
//                     array_type<double> weights, array_type<double> mins, array_type<double> maxs, 
//                     array_type<double> means, array_type<double> invcovs, array_type<double> dets, array_type<double> log_Js, 
//                     int num_points, array_type<int> start_index, int num_components, int ndim)
// {
//     compute_logpdf_wrap(
//         return_ptr(logpdf_out, "logpdf_out", num_points, 1),
//         return_ptr(component_index, "component_index", num_points, 1),
//         return_ptr(points, "points", num_points * ndim, 1),
//         return_ptr(weights, "weights", num_components, 1),
//         return_ptr(mins, "mins", num_components * ndim, 1),
//         return_ptr(maxs, "maxs", num_components * ndim, 1),
//         return_ptr(means, "means", num_components * ndim, 1),
//         return_ptr(invcovs, "invcovs", num_components * ndim * ndim, 1),
//         return_ptr(dets, "dets", num_components, 1),
//         return_ptr(log_Js, "log_Js", num_components, 1),
//         num_points,
//         return_ptr(start_index, "start_index", num_components + 1, 1),
//         num_components,
//         ndim
//     );
// }

void detector_part(nb::module_ &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<OrbitsWrap>(m, "OrbitsWrapGPU")
#else
    nb::class_<OrbitsWrap>(m, "OrbitsWrapCPU")
#endif 

    // Bind the constructor
    .def(nb::init<double, double, int, double, double, int, array_type<double>, array_type<double>, array_type<double>, array_type<int>, array_type<int>, array_type<int>, double>(), 
         nb::arg("sc_t0"), nb::arg("sc_dt"), nb::arg("sc_N"), nb::arg("ltt_t0"), nb::arg("ltt_dt"), nb::arg("ltt_N"), nb::arg("n_arr"), nb::arg("ltt_arr"), nb::arg("x_arr"), nb::arg("links"), nb::arg("sc_r"), nb::arg("sc_e"), nb::arg("armlength"))
    // Bind member functions
    .def("get_light_travel_time_wrap", &OrbitsWrap::get_light_travel_time_wrap, "Get the light travel time.")
    .def("get_pos_wrap", &OrbitsWrap::get_pos_wrap, "Get spacecraft position.")
    .def("get_normal_unit_vec_wrap", &OrbitsWrap::get_normal_unit_vec_wrap, "Get link normal vector.")
    // You can also expose public data members directly using def_rw
    .def_rw("orbits", &OrbitsWrap::orbits)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    .def("__copy__",  [](const OrbitsWrap &self) {
        return OrbitsWrap(self);
    })
    .def("__deepcopy__", [](const OrbitsWrap &self, nb::dict) {
        return OrbitsWrap(self);
    });


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<Orbits>(m, "OrbitsGPU")
#else
    nb::class_<Orbits>(m, "OrbitsCPU")
#endif

    // Bind the constructor
    .def(nb::init<double, double, int, double, double, int, double *, double *, double *, int *, int *, int *, double>(),
         nb::arg("sc_t0"), nb::arg("sc_dt"), nb::arg("sc_N"), nb::arg("ltt_t0"), nb::arg("ltt_dt"), nb::arg("ltt_N"), nb::arg("n_arr"), nb::arg("ltt_arr"), nb::arg("x_arr"), nb::arg("links"), nb::arg("sc_r"), nb::arg("sc_e"), nb::arg("armlength"))

    ;

#if 0  // === XYZBackend disabled (symbol issues on Linux): pybind11 class bindings ===
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<XYZSensitivityMatrixWrap>(m, "XYZSensitivityMatrixWrapGPU")
#else
    nb::class_<XYZSensitivityMatrixWrap>(m, "XYZSensitivityMatrixWrapCPU")
#endif
    .def(nb::init<array_type<double>, array_type<double>, int, double, int, bool, double>(),
            nb::arg("averaged_ltts_arr"), nb::arg("delta_ltts_arr"), nb::arg("n_times"), nb::arg("armlength"), nb::arg("generation"), nb::arg("spline_noise"), nb::arg("window_factor") = 1.0)
    .def("get_noise_tfs_wrap", &XYZSensitivityMatrixWrap::get_noise_tfs_wrap, "Get noise transfer functions.")
    .def("psd_likelihood_wrap", &XYZSensitivityMatrixWrap::psd_likelihood_wrap, "Compute PSD likelihood.")
    .def("get_noise_covariance_wrap", &XYZSensitivityMatrixWrap::get_noise_covariance_wrap, "Compute noise covariance matrix.")
    .def("get_inverse_det_wrap", &XYZSensitivityMatrixWrap::get_inverse_det_wrap, "Batch invert 3x3 Hermitian matrices and compute determinants.")
    .def_rw("sensitivity_matrix", &XYZSensitivityMatrixWrap::sensitivity_matrix)
    .def("__copy__",  [](const XYZSensitivityMatrixWrap &self) {
        return XYZSensitivityMatrixWrap(self);
    })
    .def("__deepcopy__", [](const XYZSensitivityMatrixWrap &self, nb::dict) {
        return XYZSensitivityMatrixWrap(self);
    });

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<XYZSensitivityMatrix>(m, "XYZSensitivityMatrixGPU")
#else
    nb::class_<XYZSensitivityMatrix>(m, "XYZSensitivityMatrixCPU")
#endif
    .def(nb::init<double *, double *, int, double, int, bool, double>(),
            nb::arg("averaged_ltts_arr"), nb::arg("delta_ltts_arr"), nb::arg("n_times"), nb::arg("armlength"), nb::arg("generation"), nb::arg("spline_noise"), nb::arg("window_factor") = 1.0)
    ;
#endif  // === end pybind11 class bindings ===
}


NB_MODULE(pycppdetector, m) {
    m.doc() = "Orbits/Detector/Response C++ plug-in"; // Optional module docstring

    // Phase 3L.7k (2026-06-04): TDI_XYZ / TDI_AET / TDI_AE module-level
    // attrs migrated from the (now-retiring) fastlisaresponse_backend_*.
    // tdionthefly module so LISAToolsBackendMethods.TDITypeDict can be
    // populated from lisatools_backend_*.pycppdetector directly.
    m.attr("TDI_XYZ") = TDI_XYZ;
    m.attr("TDI_AET") = TDI_AET;
    m.attr("TDI_AE")  = TDI_AE;

    // Call initialization functions from other files
    detector_part(m);
    // Phase 3E: LISA-response wrappers (LISAResponseWrap, TDIConfigWrap,
    // OrbitsWrap_responselisa, CubicSplineWrap_responselisa). Defined in
    // binding_flr.cxx, absorbed from lisa-on-gpu.
    response_part(m);
    m.def("check_orbits", &check_orbits, "Make sure that we can insert orbits properly.");

    m.def("get_module_path_cpp", &get_module_path, "Returns the file path of the module");
    m.def("check_12", &check_12, "Check12");

    try {
        std::string path_at_init = nb::cast<std::string>(m.attr("__file__"));
        // std::cout << "Module loaded from: " << path_at_init << std::endl;
        m.attr("module_dir") = nb::cast(path_at_init.substr(0, path_at_init.find_last_of("/\\")));
    } catch (nb::python_error &e) {
         // Handle potential error here, e.g., by logging or setting a default value
        std::cerr << "Could not capture __file__ at init time." << std::endl;
        e.restore(); // Restore exception state for proper Python handling
        PyErr_Clear();
    }
}