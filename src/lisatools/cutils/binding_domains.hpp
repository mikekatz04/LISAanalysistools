#ifndef __BINDING_DOMAINS_HPP__
#define __BINDING_DOMAINS_HPP__

// binding_domains.hpp -- nanobind wrapper classes for the STFT/FD domain
// machinery in domains.{hpp,cu} (STFTDomainWrap, FDDomainForStftWrap,
// STFTFresnelWrap).
//
// 2026-06 domains consolidation: ported to nanobind from the incoming
// stft_tof branch's pybind11 wrap classes (pre-merge binding.hpp:208-373).
// The incoming wrap around the FD specialisation was named FDDomainWrap;
// it is renamed FDDomainForStftWrap here because the Phase-3L.1 chunked-het
// FDDomainWrap (binding_fd_domain.hpp, registered in binding_flr.cxx) owns
// the FDDomainWrap py-name. TODO(unify): merge the two FD domain classes +
// wraps in a follow-up.
//
// Registration happens in binding_detector.cxx's domains_part(m) -- LAT's
// pycppdetector is the single registrant (Phase 3J rule).

#include "domains.hpp"
#include "binding_detector.hpp"  // array_type<T>, return_pointer_and_check_length,
                                 // return_pointer_no_check

// Per-backend CPU/GPU wrap-class alias block (sprint-wide rule): these wrap
// classes are compiled into both the CPU and GPU shared objects, so each
// build must emit a distinct C++ type name. Both branches must carry the
// same entry set.
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define STFTDomainWrap STFTDomainWrapGPU
#define FDDomainForStftWrap FDDomainForStftWrapGPU
#define STFTFresnelWrap STFTFresnelWrapGPU
#else
#define STFTDomainWrap STFTDomainWrapCPU
#define FDDomainForStftWrap FDDomainForStftWrapCPU
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

class FDDomainForStftWrap {
public:
    FDDomainForStft *domain;

    FDDomainForStftWrap(int num_freqs, int num_channels,
                        double f_min, double f_max, double df,
                        array_type<std::complex<double>> data_arr,
                        array_type<std::complex<double>> invC_arr,
                        int num_data, int num_noise, int tdi_type)
    {
        cmplx *data_ptr = reinterpret_cast<cmplx*>(
            return_pointer_no_check(data_arr));
        cmplx *invC_ptr = reinterpret_cast<cmplx*>(
            return_pointer_no_check(invC_arr));

        domain = new FDDomainForStft(num_freqs, num_channels,
                                     f_min, f_max, df,
                                     data_ptr, invC_ptr,
                                     num_data, num_noise, tdi_type);
    }

    ~FDDomainForStftWrap() {
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
                    double dt, double df, double window_alpha = 0.0,
                    bool use_midpoint = false)
    {
        fresnel = new STFTFresnel(num_times, num_freqs, num_channels,
                                  t0, f_min, f_max, dt, df, window_alpha,
                                  use_midpoint);
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

#endif // __BINDING_DOMAINS_HPP__
