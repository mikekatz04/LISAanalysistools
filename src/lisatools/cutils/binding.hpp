#ifndef __BINDING_HPP__
#define __BINDING_HPP__

#include "Detector.hpp"
#include "PSD.hpp"
#include "galactic_response.hpp"
#include "domains.hpp"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
template<typename T>
using array_type = cai::cuda_array_t<T>;
#else
template<typename T>
using array_type = py::array_t<T>;
#endif


void psd_likelihood_legacy_wrap(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                         array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all, 
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> f_1_all, array_type<double> kn_all, array_type<double> f_2_all, double df, int data_length, int num_data, int num_psds);

void get_psd_val_legacy_wrap(array_type<double> Sn_A_out, array_type<double> Sn_E_out, array_type<double> f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double f_1, double kn, double f_2, int num_f);

void psd_likelihood_binding(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                         array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all, 
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> f_1_all, array_type<double> kn_all, array_type<double> f_2_all, double df, int data_length, int num_data, int num_psds);

void compute_logpdf_binding(array_type<double> logpdf_out, array_type<int> component_index, array_type<double> points,
                    array_type<double> weights, array_type<double> mins, array_type<double> maxs, 
                    array_type<double> means, array_type<double> invcovs, array_type<double> dets, array_type<double> log_Js, 
                    int num_points, array_type<int> start_index, int num_components, int ndim);

template<typename T>
T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
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


// ============================================================================
// OrbitsWrap
// ============================================================================

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
        double *_n_arr   = return_pointer_and_check_length(n_arr_,   "n_arr",   sc_N_,  6 * 3);
        double *_ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", ltt_N_, 6);
        double *_x_arr   = return_pointer_and_check_length(x_arr_,   "x_arr",   sc_N_,  3 * 3);
        int *_sc_r       = return_pointer_and_check_length(sc_r_,    "sc_r",    6, 1);
        int *_sc_e       = return_pointer_and_check_length(sc_e_,    "sc_e",    6, 1);
        int *_links      = return_pointer_and_check_length(links_,   "links",   6, 1);
        orbits = new Orbits(sc_t0_, sc_dt_, sc_N_, ltt_t0_, ltt_dt_, ltt_N_, _n_arr, _ltt_arr, _x_arr, _links, _sc_r, _sc_e, armlength_);
    };
    ~OrbitsWrap() { delete orbits; };
    OrbitsWrap(const OrbitsWrap& other) { orbits = new Orbits(*other.orbits); }

    void get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num);
    void get_normal_unit_vec_wrap(array_type<double> normal_unit_vec_x, array_type<double> normal_unit_vec_y, array_type<double> normal_unit_vec_z, array_type<double> t, array_type<int> link, int num);
    void get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num);

    template<typename T>
    T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
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
};

template<typename T>
T* return_pointer_no_check(array_type<T> input1)
{
    #if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        return static_cast<T *>(input1.get_compatible_typed_pointer());
#else
        py::buffer_info buf1 = input1.request();
        return static_cast<T *>(buf1.ptr);
#endif
};


// ============================================================================
// GalacticGridWrap
//
// Wraps GalacticGrid for Python.  Lifecycle:
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
     * @param N_times      number of time segments (max)
     * @param N_freqs      number of frequency bins (max)
     */
    GalacticGridWrap(
        array_type<double> R_vals_quad, array_type<double> z_vals_quad,
        array_type<double> quad_weights, array_type<double> cos_beta_ecl,
        array_type<double> lam_ecl, array_type<double> beta_ecl,
        int N_quad, int N_sky,
        double alpha0, double beta0,
        int N_times, int N_freqs)
    {
        gal_grid = new GalacticGrid();
        gal_grid->allocate_and_setup(
            return_pointer_and_check_length(R_vals_quad,  "R_vals_quad",  N_quad * N_sky, 1),
            return_pointer_and_check_length(z_vals_quad,  "z_vals_quad",  N_quad * N_sky, 1),
            return_pointer_and_check_length(quad_weights, "quad_weights", N_quad,         1),
            return_pointer_and_check_length(cos_beta_ecl, "cos_beta_ecl", N_sky,          1),
            return_pointer_and_check_length(lam_ecl,      "lam_ecl",      N_sky,          1),
            return_pointer_and_check_length(beta_ecl,     "beta_ecl",     N_sky,          1),
            N_quad, N_sky, alpha0, beta0, N_times, N_freqs
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
            return_pointer_and_check_length(times, "times", N_times, 1),
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
            return_pointer_and_check_length(freqs, "freqs", N_freqs, 1),
            N_freqs, N_times,
            Amp, alpha, f_1, f_knee, f_2, avg_d
        );
    }

    /** Return device pointer to R_avg (N_times * 6 doubles). */
    double* get_R_avg_ptr() { return gal_grid->R_avg; }

    /** Return device pointer to R_gal_arr ((N_times * N_freqs * 6) doubles). */
    double* get_R_gal_ptr() { return gal_grid->R_gal_arr; }

    template<typename T>
    T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
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
};


// ============================================================================
// XYZSensitivityMatrixWrap
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
        double *_averaged_ltts_arr = return_pointer_and_check_length(averaged_ltts_arr_, "averaged_ltts_arr", n_times_, 6);
        double *_delta_ltts_arr = return_pointer_and_check_length(delta_ltts_arr_, "delta_ltts_arr", n_times_, 6);

        sensitivity_matrix = new XYZSensitivityMatrix(_averaged_ltts_arr, _delta_ltts_arr, n_times_, armlength_, generation_, spline_noise_, window_factor_);
    }

    ~XYZSensitivityMatrixWrap() { delete sensitivity_matrix; }

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

    void get_inverse_det_wrap(
        array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
        array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
        array_type<double> i00_arr, array_type<std::complex<double>> i01_arr, array_type<std::complex<double>> i02_arr,
        array_type<double> i11_arr, array_type<std::complex<double>> i12_arr, array_type<double> i22_arr,
        array_type<double> det_arr,
        int num);

    template<typename T>
    T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
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
};


// STFTDomain / FDDomain Python wrappers
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define STFTDomainWrap STFTDomainWrapGPU
#define FDDomainWrap FDDomainWrapGPU
#define STFTFresnelWrap STFTFresnelWrapGPU
#else
#define STFTDomainWrap STFTDomainWrapCPU
#define FDDomainWrap FDDomainWrapCPU
#define STFTFresnelWrap STFTFresnelWrapCPU
#endif

class STFTDomainWrap {
public:
    STFTDomain *domain;

    STFTDomainWrap(int num_times, int num_freqs, int num_channels,
                   double t0, double f_min, double f_max,
                   double dt, double df,
                   array_type<std::complex<double>> data_arr,
                   array_type<std::complex<double>> invC_arr,
                   int num_data, int num_noise, int tdi_type)
    {
        cmplx *data_ptr = reinterpret_cast<cmplx*>(
            return_pointer_no_check(data_arr));
        cmplx *invC_ptr = reinterpret_cast<cmplx*>(
            return_pointer_no_check(invC_arr));

        domain = new STFTDomain(num_times, num_freqs, num_channels,
                                t0, f_min, f_max, dt, df,
                                data_ptr, invC_ptr,
                                num_data, num_noise, tdi_type);
    }

    ~STFTDomainWrap() {
        delete domain;
    }

    void compute_likelihood_terms(
        array_type<std::complex<double>> d_h_out,
        array_type<std::complex<double>> h_h_out,
        array_type<std::complex<double>> template_vals,
        array_type<double> start_times,
        array_type<double> start_freqs,
        int num_binaries,
        array_type<int> data_index,
        array_type<int> noise_index,
        int n_t_template,
        int n_f_template,
        bool run_async = false)
    {
        cmplx *d_h_ptr = reinterpret_cast<cmplx*>(
            return_pointer_and_check_length(d_h_out, "d_h_out", num_binaries, 1));
        cmplx *h_h_ptr = reinterpret_cast<cmplx*>(
            return_pointer_and_check_length(h_h_out, "h_h_out", num_binaries, 1));
        cmplx *tmpl_ptr = reinterpret_cast<cmplx*>(
            return_pointer_no_check(template_vals));
        double *st_ptr = return_pointer_and_check_length(start_times, "start_times", num_binaries, 1);
        double *sf_ptr = return_pointer_and_check_length(start_freqs, "start_freqs", num_binaries, 1);
        int *di_ptr = return_pointer_and_check_length(data_index, "data_index", num_binaries, 1);
        int *ni_ptr = return_pointer_and_check_length(noise_index, "noise_index", num_binaries, 1);

        domain->compute_likelihood_terms_wrap(
            d_h_ptr, h_h_ptr, tmpl_ptr,
            st_ptr, sf_ptr,
            num_binaries,
            di_ptr, ni_ptr,
            n_t_template, n_f_template, run_async);
    }
};

class FDDomainWrap {
public:
    FDDomain *domain;

    FDDomainWrap(int num_freqs, int num_channels,
                 double f_min, double f_max, double df,
                 array_type<std::complex<double>> data_arr,
                 array_type<std::complex<double>> invC_arr,
                 int num_data, int num_noise, int tdi_type)
    {
        cmplx *data_ptr = reinterpret_cast<cmplx*>(
            return_pointer_no_check(data_arr));
        cmplx *invC_ptr = reinterpret_cast<cmplx*>(
            return_pointer_no_check(invC_arr));

        domain = new FDDomain(num_freqs, num_channels,
                              f_min, f_max, df,
                              data_ptr, invC_ptr,
                              num_data, num_noise, tdi_type);
    }

    ~FDDomainWrap() {
        delete domain;
    }

    void compute_likelihood_terms(
        array_type<std::complex<double>> d_h_out,
        array_type<std::complex<double>> h_h_out,
        array_type<std::complex<double>> template_vals,
        array_type<double> start_freqs,
        int num_binaries,
        array_type<int> data_index,
        array_type<int> noise_index,
        int n_f_template,
        bool run_async = false)
    {
        cmplx *d_h_ptr = reinterpret_cast<cmplx*>(
            return_pointer_and_check_length(d_h_out, "d_h_out", num_binaries, 1));
        cmplx *h_h_ptr = reinterpret_cast<cmplx*>(
            return_pointer_and_check_length(h_h_out, "h_h_out", num_binaries, 1));
        cmplx *tmpl_ptr = reinterpret_cast<cmplx*>(
            return_pointer_no_check(template_vals));
        double *sf_ptr = return_pointer_and_check_length(start_freqs, "start_freqs", num_binaries, 1);
        int *di_ptr = return_pointer_and_check_length(data_index, "data_index", num_binaries, 1);
        int *ni_ptr = return_pointer_and_check_length(noise_index, "noise_index", num_binaries, 1);

        domain->compute_likelihood_terms_wrap(
            d_h_ptr, h_h_ptr, tmpl_ptr,
            sf_ptr,
            num_binaries,
            di_ptr, ni_ptr,
            n_f_template, run_async);
    }
};

class STFTFresnelWrap {
public:
    STFTFresnel *fresnel;

    STFTFresnelWrap(int num_times, int num_freqs, int num_channels,
                    double t0, double f_min, double f_max,
                    double dt, double df, double window_alpha = 0.0)
    {
        fresnel = new STFTFresnel(num_times, num_freqs, num_channels,
                                  t0, f_min, f_max, dt, df, window_alpha);
    }

    ~STFTFresnelWrap() { delete fresnel; }

    void compute_fourier_values(
        array_type<std::complex<double>> output,
        array_type<double> amps,
        array_type<double> phase0s,
        array_type<double> f0s,
        array_type<double> fdot0s,
        array_type<double> t0s,
        array_type<double> freqs,
        double window_factor,
        int num_binaries,
        int num_freqs)
    {
        cmplx *out_ptr = reinterpret_cast<cmplx*>(
            return_pointer_and_check_length(output, "output", num_binaries * num_freqs, 1));
        double *amp_ptr = return_pointer_and_check_length(amps, "amps", num_binaries, 1);
        double *ph_ptr = return_pointer_and_check_length(phase0s, "phase0s", num_binaries, 1);
        double *f0_ptr = return_pointer_and_check_length(f0s, "f0s", num_binaries, 1);
        double *fd_ptr = return_pointer_and_check_length(fdot0s, "fdot0s", num_binaries, 1);
        double *t0_ptr = return_pointer_and_check_length(t0s, "t0s", num_binaries, 1);
        double *freq_ptr = return_pointer_and_check_length(freqs, "freqs", num_binaries * num_freqs, 1);

        fresnel->compute_fourier_values_wrap(
            out_ptr, amp_ptr, ph_ptr, f0_ptr, fd_ptr, t0_ptr,
            freq_ptr, window_factor, num_binaries, num_freqs);
    }
};

#endif // __BINDING_HPP__