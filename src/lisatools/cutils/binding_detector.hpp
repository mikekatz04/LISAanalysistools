// binding_detector.hpp -- wrapper-class declarations + shared nanobind array
// helpers for the pycppdetector module (renamed from binding.hpp at the
// 2026-06 stft_tof merge to match the binding_flr / binding_wdm_* naming
// convention).
#ifndef __LAT_BINDING_DETECTOR_HPP__
#define __LAT_BINDING_DETECTOR_HPP__

#include "Detector.hpp"
#include "PSD.hpp"
#include "galactic_response.hpp"
#include "domains.hpp"
#include <string>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
template<typename T>
using array_type = nb::ndarray<T, nb::device::cuda>;
#else
template<typename T>
using array_type = nb::ndarray<T, nb::device::cpu>;
#endif


// Legacy Sangria-era PSD functions (kept for compatibility; reactivated at the
// 2026-06 stft_tof merge with the sl* -> f_* parameter renames).
void psd_likelihood_legacy_wrap(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data,
                         array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all,
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> f_1_all, array_type<double> kn_all, array_type<double> f_2_all, double df, int data_length, int num_data, int num_psds);

void get_psd_val_legacy_wrap(array_type<double> Sn_A_out, array_type<double> Sn_E_out, array_type<double> f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double f_1, double kn, double f_2, int num_f);

// Wrapper for psd_likelihood (same as legacy, exposed with different name for consistency)
void psd_likelihood_binding(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data,
                         array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all,
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> f_1_all, array_type<double> kn_all, array_type<double> f_2_all, double df, int data_length, int num_data, int num_psds);

// Wrapper for compute_logpdf
void compute_logpdf_binding(array_type<double> logpdf_out, array_type<int> component_index, array_type<double> points,
                    array_type<double> weights, array_type<double> mins, array_type<double> maxs,
                    array_type<double> means, array_type<double> invcovs, array_type<double> dets, array_type<double> log_Js,
                    int num_points, array_type<int> start_index, int num_components, int ndim);

template<typename T>
T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
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

// stft_tof merge: pointer extraction without a length check, for arrays whose
// length is not knowable at the call site (nanobind port of the incoming
// pybind11 helper).
template<typename T>
T* return_pointer_no_check(array_type<T> input1)
{
    return input1.data();
};


// now add the OrbitsWrap class
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define OrbitsWrap OrbitsWrapGPU
#else
#define OrbitsWrap OrbitsWrapCPU
#endif

class OrbitsWrap {
  public:
    Orbits *orbits;
    OrbitsWrap(double sc_t0_, double sc_dt_, int sc_N_, double ltt_t0_, double ltt_dt_, int ltt_N_, array_type<double> n_arr_, array_type<double> ltt_arr_, array_type<double> x_arr_, array_type<int> links_, array_type<int> sc_r_, array_type<int> sc_e_, double armlength_)
    {

        double *_n_arr = return_pointer_and_check_length(n_arr_, "n_arr", sc_N_, 6 * 3);
        double *_ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", ltt_N_, 6);
        double *_x_arr = return_pointer_and_check_length(x_arr_, "x_arr", sc_N_, 3 * 3);

        int *_sc_r = return_pointer_and_check_length(sc_r_, "sc_r", 6, 1);
        int *_sc_e = return_pointer_and_check_length(sc_e_, "sc_e", 6, 1);
        int *_links = return_pointer_and_check_length(links_, "links", 6, 1);

        orbits = new Orbits(sc_t0_, sc_dt_, sc_N_, ltt_t0_, ltt_dt_, ltt_N_, _n_arr, _ltt_arr, _x_arr, _links,  _sc_r, _sc_e, armlength_);
    };
    ~OrbitsWrap(){
        delete orbits;
    };

    OrbitsWrap(const OrbitsWrap& other) {
        orbits = new Orbits(*other.orbits);
    }

    void get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num);
    void get_normal_unit_vec_wrap(array_type<double>normal_unit_vec_x, array_type<double>normal_unit_vec_y, array_type<double>normal_unit_vec_z, array_type<double>t, array_type<int>link, int num);
    void get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num);
    template<typename T>
    T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
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

};


// ============================================================================
// GalacticGridWrap
//
// Wraps GalacticGrid for Python (stft_tof merge, nanobind port).  Lifecycle:
//   1. construct  — allocate_and_setup
//   2. initialize — compute sky weights + R_avg (called once before inference)
//   3. compute_gal_covariance — optional diagnostic
// ============================================================================

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define GalacticGridWrap GalacticGridWrapGPU
#else
#define GalacticGridWrap GalacticGridWrapCPU
#endif

class GalacticGridWrap {
public:
    GalacticGrid *gal_grid;

    /**
     * @param R_vals_quad  (N_quad * N_sky,)
     * @param z_vals_quad  (N_quad * N_sky,)
     * @param quad_weights (N_quad,)
     * @param cos_beta_ecl (N_sky,)
     * @param lam_ecl      (N_sky,)
     * @param beta_ecl     (N_sky,)
     * @param N_quad       number of quadrature nodes (16)
     * @param N_sky        number of sky pixels
     * @param alpha0       LISA orbit initial phase (rad)
     * @param beta0        LISA orbit inclination (rad)
     * @param t0           LISA orbit reference time for alpha and beta (s)
     * @param N_times      number of time segments (max)
     * @param N_freqs      number of frequency bins (max)
     */
    GalacticGridWrap(
        array_type<double> R_vals_quad, array_type<double> z_vals_quad,
        array_type<double> quad_weights, array_type<double> cos_beta_ecl,
        array_type<double> lam_ecl, array_type<double> beta_ecl,
        int N_quad, int N_sky,
        double alpha0, double beta0, double t0,
        int N_times, int N_freqs)
    {
        gal_grid = new GalacticGrid();
        gal_grid->allocate_and_setup(
            return_pointer_and_check_length(R_vals_quad,  std::string("R_vals_quad"),  N_quad * N_sky, 1),
            return_pointer_and_check_length(z_vals_quad,  std::string("z_vals_quad"),  N_quad * N_sky, 1),
            return_pointer_and_check_length(quad_weights, std::string("quad_weights"), N_quad,         1),
            return_pointer_and_check_length(cos_beta_ecl, std::string("cos_beta_ecl"), N_sky,          1),
            return_pointer_and_check_length(lam_ecl,      std::string("lam_ecl"),      N_sky,          1),
            return_pointer_and_check_length(beta_ecl,     std::string("beta_ecl"),     N_sky,          1),
            N_quad, N_sky, alpha0, beta0, t0, N_times, N_freqs
        );
    }

    ~GalacticGridWrap() {
        gal_grid->free_gpu();
        delete gal_grid;
    }

    /**
     * Compute fixed sky weights and R_avg.
     * Call once before inference starts.
     *
     * @param times   (N_times,) segment centre times [s]
     * @param R_d     disk radial scale length [kpc]
     * @param z_d     disk vertical scale height [kpc]
     * @param N_times number of time segments
     */
    void initialize_wrap(array_type<double> times, double R_d, double z_d, int N_times)
    {
        gal_grid->initialize(
            R_d, z_d,
            return_pointer_and_check_length(times, std::string("times"), N_times, 1),
            N_times
        );
    }

    /**
     * Diagnostic: compute R_gal_arr = R_avg * S_gal(f) for a single set of
     * spectral params.  Not called on the hot path.
     */
    void compute_gal_covariance_wrap(
        array_type<double> freqs,
        double Amp, double alpha,
        double f_1, double f_knee, double f_2,
        double avg_d,
        int N_freqs, int N_times)
    {
        gal_grid->compute_gal_covariance(
            return_pointer_and_check_length(freqs, std::string("freqs"), N_freqs, 1),
            N_freqs, N_times,
            Amp, alpha, f_1, f_knee, f_2, avg_d
        );
    }

    /** Return device pointer to R_avg (N_times * 6 doubles). */
    double* get_R_avg_ptr() { return gal_grid->R_avg; }

    /** Return device pointer to R_gal_arr ((N_times * N_freqs * 6) doubles). */
    double* get_R_gal_ptr() { return gal_grid->R_gal_arr; }
};


// ============================================================================
// XYZSensitivityMatrixWrap
//
// Reactivated at the 2026-06 stft_tof merge: the former "symbol issues on
// Linux" were the missing CPU/GPU class-name aliases, now in place here and
// in PSD.hpp.
// ============================================================================

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define XYZSensitivityMatrixWrap XYZSensitivityMatrixWrapGPU
#else
#define XYZSensitivityMatrixWrap XYZSensitivityMatrixWrapCPU
#endif

class XYZSensitivityMatrixWrap {
public:
    XYZSensitivityMatrix *sensitivity_matrix;

    XYZSensitivityMatrixWrap(array_type<double> averaged_ltts_arr_, array_type<double> delta_ltts_arr_, int n_times_, double armlength_, int generation_, bool spline_noise_, double window_factor_)
    {
        double *_averaged_ltts_arr = return_pointer_and_check_length(averaged_ltts_arr_, std::string("averaged_ltts_arr"), n_times_, 6);
        double *_delta_ltts_arr = return_pointer_and_check_length(delta_ltts_arr_, std::string("delta_ltts_arr"), n_times_, 6);

        sensitivity_matrix = new XYZSensitivityMatrix(_averaged_ltts_arr, _delta_ltts_arr, n_times_, armlength_, generation_, spline_noise_, window_factor_);
    }

    ~XYZSensitivityMatrixWrap() {
        delete sensitivity_matrix;
    };

    XYZSensitivityMatrixWrap(const XYZSensitivityMatrixWrap& other) {
        sensitivity_matrix = new XYZSensitivityMatrix(*other.sensitivity_matrix);
    }

    /**
     * Attach a GalacticGrid so the likelihood kernel includes the galactic
     * foreground.  Pass None (nullptr) to disable.
     *
     * The GalacticGridWrap must outlive this object.
     * initialize_wrap() must have been called on it before inference starts.
     */
    void set_galactic_grid(GalacticGridWrap *gal_wrap)
    {
        if (gal_wrap == nullptr)
            sensitivity_matrix->set_galactic_grid(nullptr);
        else
            sensitivity_matrix->set_galactic_grid(gal_wrap->gal_grid->R_avg);
    }

    /** Disable galactic foreground (equivalent to set_galactic_grid(None)). */
    void disable_galactic_grid()
    {
        sensitivity_matrix->set_galactic_grid(nullptr);
    }

    void get_noise_tfs_wrap(
        array_type<double> freqs,
        array_type<double> oms_xx, array_type<std::complex<double>> oms_xy, array_type<std::complex<double>> oms_xz,
        array_type<double> oms_yy, array_type<std::complex<double>> oms_yz, array_type<double> oms_zz,
        array_type<double> tm_xx,  array_type<std::complex<double>> tm_xy,  array_type<std::complex<double>> tm_xz,
        array_type<double> tm_yy,  array_type<std::complex<double>> tm_yz,  array_type<double> tm_zz,
        int num_freqs, int num_times,
        array_type<int> time_indices);

    void psd_likelihood_wrap(
        array_type<double> like_contrib_final, array_type<double> f_arr,
        array_type<std::complex<double>> data,
        array_type<int> data_index_all, array_type<int> time_index_all,
        array_type<double> Soms_d_in_all, array_type<double> Sa_a_in_all,
        array_type<double> Amp_all, array_type<double> alpha_all,
        array_type<double> f_1_all, array_type<double> f_knee_all, array_type<double> f_2_all,
        array_type<double> spline_in_isi_oms_all, array_type<double> spline_in_testmass_all,
        double differential_component, int num_freqs, int num_times,
        array_type<bool> dips_mask, int num_psds, bool run_async = false);

    void get_noise_covariance_wrap(
        array_type<double> freqs, array_type<int> time_indices,
        double Soms_d_in, double Sa_a_in,
        double Amp, double alpha, double f_1, double f_knee, double f_2,
        array_type<double> spline_in_isi_oms_arr, array_type<double> spline_in_testmass_arr,
        array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
        array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
        int num_freqs, int num_times);

    void set_averaged_tfs_wrap(
        array_type<double> oms_xx, array_type<std::complex<double>> oms_xy, array_type<std::complex<double>> oms_xz,
        array_type<double> oms_yy, array_type<std::complex<double>> oms_yz, array_type<double> oms_zz,
        array_type<double> tm_xx,  array_type<std::complex<double>> tm_xy,  array_type<std::complex<double>> tm_xz,
        array_type<double> tm_yy,  array_type<std::complex<double>> tm_yz,  array_type<double> tm_zz, int nf);
    void disable_averaged_tfs_wrap();

    void get_inverse_det_wrap(
        array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
        array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
        array_type<double> i00_arr, array_type<std::complex<double>> i01_arr, array_type<std::complex<double>> i02_arr,
        array_type<double> i11_arr, array_type<std::complex<double>> i12_arr, array_type<double> i22_arr,
        array_type<double> det_arr,
        int num);
};


// ============================================================================
// STFT / FD domain wraps (STFTDomainWrap, FDDomainWrap, STFTFresnelWrap)
//
// NOT yet bound here. The incoming stft_tof domain machinery is being
// consolidated into domains.{hpp,cu} together with the existing WDM/FD
// domain classes (wdm_settings.hh / wdm_domain.hh / fd_domain.hh) -- see the
// domains-consolidation work item. Binding surface lands there; note the
// incoming FDDomainWrap name collides with the Phase-3L.1 FDDomainWrap in
// binding_fd_domain.hpp and must be reconciled, not double-registered.
// ============================================================================


#endif // __LAT_BINDING_DETECTOR_HPP__
