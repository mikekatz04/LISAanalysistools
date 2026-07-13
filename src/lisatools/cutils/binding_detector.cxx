// binding_detector.cxx -- nanobind registrations for the pycppdetector module
// (renamed from binding.cxx at the 2026-06 stft_tof merge to match the
// binding_flr / binding_wdm_* naming convention).
//
// Contents: OrbitsWrap + Orbits, GalacticGridSetup + GalacticGridWrap,
// XYZSensitivityMatrixWrap + XYZSensitivityMatrix, legacy Sangria PSD/GMM
// functions, and the NB_MODULE(pycppdetector) entry point.

#include "Detector.hpp"
#include "PSD.hpp"
#include "galactic_response.hpp"
#include "domains.hpp"               // domain classes + TDI_XYZ / TDI_AET / TDI_AE macros
#include <string>
#include <cstring>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "binding_detector.hpp"
#include "binding_domains.hpp"       // STFTDomainWrap, FDDomainForStftWrap, STFTFresnelWrap

// Phase 3J: this binding TU is the SOLE registration site for the shared
// wrapper classes (OrbitsWrap, LISAResponseWrap, TDIConfigWrap, ...).
// (CubicSplineWrap is GBT-owned as of 2026-06-10 and registered in
// GBT's `interp` module, not here.) Setting the toggle to 1 BEFORE
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

namespace nb = nanobind;

// Phase 3E (2026-06-02): LISAResponse-related bindings absorbed
// from lisa-on-gpu. The actual class definitions and `response_part(nb::module&)`
// implementation live in binding_flr.cxx, which is compiled into the
// pycppdetector nanobind module alongside this file.
void response_part(nb::module_ &m);

// ============================================================================
// OrbitsWrap method implementations
// ============================================================================

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

// ============================================================================
// XYZSensitivityMatrixWrap method implementations
//
// Reactivated at the 2026-06 stft_tof merge (the "symbol issues on Linux"
// were the missing CPU/GPU class-name aliases, now present in PSD.hpp /
// binding_detector.hpp). Bodies carry the stft_tof-side evolution: the
// sl* -> f_* spectral-parameter renames, the run_async flag, and the FD
// time-averaged transfer-function attachment.
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

void XYZSensitivityMatrixWrap::psd_likelihood_wrap(
    array_type<double> like_contrib_final, array_type<double> f_arr,
    array_type<std::complex<double>> data,
    array_type<int> data_index_all, array_type<int> time_index_all,
    array_type<double> Soms_d_in_all, array_type<double> Sa_a_in_all,
    array_type<double> Amp_all, array_type<double> alpha_all,
    array_type<double> f_1_all, array_type<double> f_knee_all, array_type<double> f_2_all,
    array_type<double> spline_in_isi_oms_all, array_type<double> spline_in_testmass_all,
    double differential_component, int num_freqs, int num_times,
    array_type<bool> dips_mask, int num_psds, bool run_async)
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
        num_freqs,
        num_times,
        return_pointer_and_check_length(dips_mask, "dips_mask", num_times * num_freqs, 1),
        num_psds,
        run_async
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

void XYZSensitivityMatrixWrap::set_averaged_tfs_wrap(
    array_type<double> oms_xx, array_type<std::complex<double>> oms_xy, array_type<std::complex<double>> oms_xz,
    array_type<double> oms_yy, array_type<std::complex<double>> oms_yz, array_type<double> oms_zz,
    array_type<double> tm_xx,  array_type<std::complex<double>> tm_xy,  array_type<std::complex<double>> tm_xz,
    array_type<double> tm_yy,  array_type<std::complex<double>> tm_yz,  array_type<double> tm_zz, int nf)
{
    sensitivity_matrix->set_averaged_tfs(
        return_pointer_and_check_length(oms_xx, "oms_xx_avg", nf, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_xy, "oms_xy_avg", nf, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_xz, "oms_xz_avg", nf, 1)),
        return_pointer_and_check_length(oms_yy, "oms_yy_avg", nf, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_yz, "oms_yz_avg", nf, 1)),
        return_pointer_and_check_length(oms_zz, "oms_zz_avg", nf, 1),
        return_pointer_and_check_length(tm_xx, "tm_xx_avg", nf, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_xy, "tm_xy_avg", nf, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_xz, "tm_xz_avg", nf, 1)),
        return_pointer_and_check_length(tm_yy, "tm_yy_avg", nf, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_yz, "tm_yz_avg", nf, 1)),
        return_pointer_and_check_length(tm_zz, "tm_zz_avg", nf, 1),
        nf);
}

void XYZSensitivityMatrixWrap::disable_averaged_tfs_wrap() { sensitivity_matrix->disable_averaged_tfs(); }

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

// ============================================================================
// Legacy Sangria-era functions (reactivated at the 2026-06 stft_tof merge
// with the sl* -> f_* parameter renames).
// ============================================================================

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

// Copy a host std::vector<double> into a NumPy-owned array (nanobind port of
// the pybind11 py::array_t copy-return used by the GalacticGridSetup props).
static nb::ndarray<nb::numpy, double> vec_to_numpy(const std::vector<double> &v)
{
    double *out = new double[v.size()];
    std::memcpy(out, v.data(), v.size() * sizeof(double));
    nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return nb::ndarray<nb::numpy, double>(out, {v.size()}, owner);
}

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

    // ---- GalacticGridSetup (host-side quadrature/sky-grid builder) ----
    // The C++ type is per-backend aliased (GalacticGridSetupGPU/CPU) so the
    // CPU and GPU plugin .so's register distinct typeids; the Python-facing
    // name stays unsuffixed in each module (it is a host-only helper).
    nb::class_<GalacticGridSetup>(m, "GalacticGridSetup")
    .def(nb::init<>())
    .def("compute", &GalacticGridSetup::compute,
         nb::arg("N_lambda") = 90, nb::arg("N_beta") = 60)
    .def("print_summary", &GalacticGridSetup::print_summary)
    .def_ro("N_lambda", &GalacticGridSetup::N_lambda)
    .def_ro("N_beta",   &GalacticGridSetup::N_beta)
    .def_ro("N_sky",    &GalacticGridSetup::N_sky)
    .def_ro("N_quad",   &GalacticGridSetup::N_quad)
    .def_prop_ro("lam_ecl", [](GalacticGridSetup &s) {
        return vec_to_numpy(s.lam_ecl);
    })
    .def_prop_ro("beta_ecl", [](GalacticGridSetup &s) {
        return vec_to_numpy(s.beta_ecl);
    })
    .def_prop_ro("cos_beta_ecl", [](GalacticGridSetup &s) {
        return vec_to_numpy(s.cos_beta_ecl);
    })
    .def_prop_ro("quad_weights", [](GalacticGridSetup &s) {
        return vec_to_numpy(s.quad_weights);
    })
    .def_prop_ro("R_vals_quad", [](GalacticGridSetup &s) {
        return vec_to_numpy(s.R_vals_quad);
    })
    .def_prop_ro("z_vals_quad", [](GalacticGridSetup &s) {
        return vec_to_numpy(s.z_vals_quad);
    });

    // ---- GalacticGridWrap ----
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<GalacticGridWrap>(m, "GalacticGridWrapGPU")
#else
    nb::class_<GalacticGridWrap>(m, "GalacticGridWrapCPU")
#endif
    .def(nb::init<
             array_type<double>, array_type<double>,  // R_vals_quad, z_vals_quad
             array_type<double>, array_type<double>,  // quad_weights, cos_beta_ecl
             array_type<double>, array_type<double>,  // lam_ecl, beta_ecl
             int, int,                                // N_quad, N_sky
             double, double, double,                  // alpha0, beta0, t0
             int, int>(),                             // N_times, N_freqs
         nb::arg("R_vals_quad"), nb::arg("z_vals_quad"),
         nb::arg("quad_weights"), nb::arg("cos_beta_ecl"),
         nb::arg("lam_ecl"), nb::arg("beta_ecl"),
         nb::arg("N_quad"), nb::arg("N_sky"),
         nb::arg("alpha0"), nb::arg("beta0"), nb::arg("t0"),
         nb::arg("N_times"), nb::arg("N_freqs"),
         "Construct and allocate galactic grid on device.\n"
         "Call initialize_wrap() once before inference starts.")
    .def("initialize_wrap", &GalacticGridWrap::initialize_wrap,
         nb::arg("times"), nb::arg("R_d"), nb::arg("z_d"), nb::arg("N_times"),
         "Compute fixed sky weights and R_avg from disk parameters and orbit times.\n"
         "Must be called once before passing this object to set_galactic_grid().")
    .def("compute_gal_covariance_wrap", &GalacticGridWrap::compute_gal_covariance_wrap,
         nb::arg("freqs"),
         nb::arg("Amp"), nb::arg("alpha"),
         nb::arg("f_1"), nb::arg("f_knee"), nb::arg("f_2"),
         nb::arg("avg_d"), nb::arg("N_freqs"), nb::arg("N_times"),
         "Diagnostic: compute R_gal_arr = R_avg * S_gal(f) for a single parameter set.\n"
         "Not used on the inference hot path.");

    // ---- XYZSensitivityMatrixWrap ----
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<XYZSensitivityMatrixWrap>(m, "XYZSensitivityMatrixWrapGPU")
#else
    nb::class_<XYZSensitivityMatrixWrap>(m, "XYZSensitivityMatrixWrapCPU")
#endif
    .def(nb::init<array_type<double>, array_type<double>, int, double, int, bool, double>(),
         nb::arg("averaged_ltts_arr"), nb::arg("delta_ltts_arr"),
         nb::arg("n_times"), nb::arg("armlength"), nb::arg("generation"), nb::arg("spline_noise"), nb::arg("window_factor"))
    .def("set_galactic_grid", &XYZSensitivityMatrixWrap::set_galactic_grid,
         nb::arg("gal_wrap").none(),
         "Attach a GalacticGridWrap (already initialized) to include the galactic\n"
         "foreground in the likelihood.  Pass None to disable.")
    .def("disable_galactic_grid", &XYZSensitivityMatrixWrap::disable_galactic_grid,
         "Detach galactic grid (equivalent to set_galactic_grid(None)).")
    .def("get_noise_tfs_wrap",        &XYZSensitivityMatrixWrap::get_noise_tfs_wrap,
         nb::call_guard<nb::gil_scoped_release>(), "Get noise transfer functions.")
    .def("psd_likelihood_wrap", &XYZSensitivityMatrixWrap::psd_likelihood_wrap,
         nb::arg("like_contrib_final"), nb::arg("f_arr"), nb::arg("data"),
         nb::arg("data_index_all"), nb::arg("time_index_all"),
         nb::arg("Soms_d_in_all"), nb::arg("Sa_a_in_all"),
         nb::arg("Amp_all"), nb::arg("alpha_all"), nb::arg("f_1_all"), nb::arg("f_knee_all"), nb::arg("f_2_all"),
         nb::arg("spline_in_isi_oms_all"), nb::arg("spline_in_testmass_all"),
         nb::arg("differential_component"), nb::arg("num_freqs"), nb::arg("num_times"),
         nb::arg("dips_mask"), nb::arg("num_psds"),
         nb::arg("run_async") = false,
         nb::call_guard<nb::gil_scoped_release>(),
         "Compute PSD likelihood.")
    .def("get_noise_covariance_wrap", &XYZSensitivityMatrixWrap::get_noise_covariance_wrap,
         nb::call_guard<nb::gil_scoped_release>(), "Compute noise covariance matrix.")
    .def("set_averaged_tfs_wrap",     &XYZSensitivityMatrixWrap::set_averaged_tfs_wrap, "Attach FD time-averaged transfer functions.")
    .def("disable_averaged_tfs_wrap", &XYZSensitivityMatrixWrap::disable_averaged_tfs_wrap, "Detach FD time-averaged transfer functions.")
    .def("get_inverse_det_wrap",      &XYZSensitivityMatrixWrap::get_inverse_det_wrap,
         nb::call_guard<nb::gil_scoped_release>(), "Batch invert 3x3 Hermitian matrices and compute determinants.")
    .def_rw("sensitivity_matrix", &XYZSensitivityMatrixWrap::sensitivity_matrix)
    .def("__copy__",  [](const XYZSensitivityMatrixWrap &self) {
        return XYZSensitivityMatrixWrap(self);
    })
    .def("__deepcopy__", [](const XYZSensitivityMatrixWrap &self, nb::dict) {
        return XYZSensitivityMatrixWrap(self);
    });

    // ---- XYZSensitivityMatrix (raw) ----
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<XYZSensitivityMatrix>(m, "XYZSensitivityMatrixGPU")
#else
    nb::class_<XYZSensitivityMatrix>(m, "XYZSensitivityMatrixCPU")
#endif
    .def(nb::init<double *, double *, int, double, int, bool, double>(),
            nb::arg("averaged_ltts_arr"), nb::arg("delta_ltts_arr"), nb::arg("n_times"), nb::arg("armlength"), nb::arg("generation"), nb::arg("spline_noise"), nb::arg("window_factor") = 1.0)
    ;

    m.def("psd_likelihood_legacy_wrap", &psd_likelihood_legacy_wrap,
          nb::call_guard<nb::gil_scoped_release>(), "Legacy PSD likelihood wrapping");
    m.def("get_psd_val_legacy_wrap", &get_psd_val_legacy_wrap,
          nb::call_guard<nb::gil_scoped_release>(), "Legacy PSD val wrapping");
    m.def("psd_likelihood", &psd_likelihood_binding,
          nb::call_guard<nb::gil_scoped_release>(), "PSD likelihood computation");
    m.def("compute_logpdf", &compute_logpdf_binding,
          nb::call_guard<nb::gil_scoped_release>(), "Compute log PDF from GMM");
}


// ============================================================================
// STFT / FD domain wrap registrations (2026-06 domains consolidation).
//
// Nanobind port of the incoming stft_tof branch's pybind11 domains_part()
// (pre-merge binding.cxx:388-444). The wrap classes live in
// binding_domains.hpp; the underlying domain classes in domains.{hpp,cu}.
// The incoming FDDomainWrap is registered as FDDomainForStftWrap{CPU,GPU}
// because the Phase-3L.1 chunked-het FDDomainWrap (binding_flr.cxx) owns
// the FDDomainWrap py-name. The TDI_XYZ/TDI_AET/TDI_AE module attrs the
// original set here are already exported in NB_MODULE below (canonical
// 1/2/3 values).
// ============================================================================

void domains_part(nb::module_ &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<STFTDomainWrap>(m, "STFTDomainWrapGPU")
#else
    nb::class_<STFTDomainWrap>(m, "STFTDomainWrapCPU")
#endif
    .def(nb::init<int, int, int, double, double, double, double, double,
                  array_type<std::complex<double>>, array_type<std::complex<double>>,
                  int, int, int>(),
         nb::arg("num_times"), nb::arg("num_freqs"), nb::arg("num_channels"),
         nb::arg("t0"), nb::arg("f_min"), nb::arg("f_max"),
         nb::arg("dt"), nb::arg("df"),
         nb::arg("data"), nb::arg("invC"),
         nb::arg("num_data"), nb::arg("num_noise"), nb::arg("tdi_type"))
    .def("compute_likelihood_terms", &STFTDomainWrap::compute_likelihood_terms,
         nb::arg("d_h_out"), nb::arg("h_h_out"), nb::arg("template_vals"),
         nb::arg("start_times"), nb::arg("start_freqs"), nb::arg("num_binaries"),
         nb::arg("data_index"), nb::arg("noise_index"),
         nb::arg("n_t_template"), nb::arg("n_f_template"),
         nb::arg("run_async") = false,
         nb::call_guard<nb::gil_scoped_release>(),
         "Compute (d|h) and (h|h) likelihood terms for a batch of binaries.");

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<FDDomainForStftWrap>(m, "FDDomainForStftWrapGPU")
#else
    nb::class_<FDDomainForStftWrap>(m, "FDDomainForStftWrapCPU")
#endif
    .def(nb::init<int, int, double, double, double,
                  array_type<std::complex<double>>, array_type<std::complex<double>>,
                  int, int, int>(),
         nb::arg("num_freqs"), nb::arg("num_channels"),
         nb::arg("f_min"), nb::arg("f_max"), nb::arg("df"),
         nb::arg("data"), nb::arg("invC"),
         nb::arg("num_data"), nb::arg("num_noise"), nb::arg("tdi_type"))
    .def("compute_likelihood_terms", &FDDomainForStftWrap::compute_likelihood_terms,
         nb::arg("d_h_out"), nb::arg("h_h_out"), nb::arg("template_vals"),
         nb::arg("start_freqs"), nb::arg("num_binaries"),
         nb::arg("data_index"), nb::arg("noise_index"),
         nb::arg("n_f_template"),
         nb::arg("run_async") = false,
         nb::call_guard<nb::gil_scoped_release>(),
         "Compute (d|h) and (h|h) likelihood terms for a batch of binaries (FD).");

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<STFTFresnelWrap>(m, "STFTFresnelWrapGPU")
#else
    nb::class_<STFTFresnelWrap>(m, "STFTFresnelWrapCPU")
#endif
    .def(nb::init<int, int, int, double, double, double, double, double, double, bool, bool>(),
         nb::arg("num_times"), nb::arg("num_freqs"), nb::arg("num_channels"),
         nb::arg("t0"), nb::arg("f_min"), nb::arg("f_max"),
         nb::arg("dt"), nb::arg("df"), nb::arg("window_alpha") = 0.0,
         nb::arg("use_midpoint") = false, nb::arg("linear_envelope") = false)
    .def("compute_fourier_values", &STFTFresnelWrap::compute_fourier_values,
         nb::arg("output"), nb::arg("amps"), nb::arg("phase0s"),
         nb::arg("f0s"), nb::arg("fdot0s"), nb::arg("t0s"),
         nb::arg("freqs"), nb::arg("window_factor"),
         nb::arg("num_binaries"), nb::arg("num_freqs"),
         nb::call_guard<nb::gil_scoped_release>(),
         "Compute Fresnel-based Fourier values for a batch of binaries.\n\n"
         "linear_envelope has NO effect here (by design): this API receives\n"
         "no response stencil, hence no per-segment amplitude slope; the\n"
         "envelope correction acts only on the likelihood/fill paths where\n"
         "FresnelColumn::setup derives the slope from the +-D TDI samples.");
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
    // Phase 3E: LISA-response wrappers (LISAResponseWrap, TDIConfigWrap).
    // Defined in binding_flr.cxx, absorbed from lisa-on-gpu.
    // (The legacy OrbitsWrap_responselisa was deleted at Phase 3L.7p
    // 2026-06-04 in favor of the canonical OrbitsWrap; CubicSplineWrap
    // moved to GBT's `interp` module 2026-06-10.)
    response_part(m);
    // 2026-06 domains consolidation: STFT/FD domain wraps (STFTDomainWrap,
    // FDDomainForStftWrap, STFTFresnelWrap) from binding_domains.hpp.
    domains_part(m);
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
