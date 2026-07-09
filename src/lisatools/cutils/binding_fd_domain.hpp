#ifndef __BINDING_FD_DOMAIN_HPP__
#define __BINDING_FD_DOMAIN_HPP__

// FDDomainWrap -- pybind11 wrapper for FDDomain.
// Phase 3L (2026-06-02): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.hpp:281-299
// to LISAanalysistools.

#include "fd_domain.hh"
#include "binding_flr.hpp"  // ReturnPointerBase, array_type<T>, return_pointer*

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define FDDomainWrap FDDomainWrapGPU
#else
#define FDDomainWrap FDDomainWrapCPU
#endif

class FDDomainWrap : public ReturnPointerBase {
  public:
    FDDomain *fd;
    FDDomainWrap(
        array_type<std::complex<double>> fd_data_,
        array_type<double>               fd_invC_,
        int n_rfft_, int num_channel_, int num_data_, int num_noise_,
        int ind_min_, int ind_max_, double df_)
    {
        fd = new FDDomain(
            (cmplx*) return_pointer_and_check_length(
                fd_data_, "fd_data",
                n_rfft_ * num_channel_ * num_data_, 1),
            return_pointer(fd_invC_, "fd_invC"),
            n_rfft_, num_channel_, num_data_, num_noise_,
            ind_min_, ind_max_, df_);
    };
    // Windowed-row layout: row i of fd_data / fd_invC stores absolute bins
    // [start_inds[i], start_inds[i] + n_rfft) of the global rfft grid (the
    // sub-band buffer's per-cell ``min_freq_inds``). start_inds must stay
    // alive (and may be updated in place) for the lifetime of this wrap.
    FDDomainWrap(
        array_type<std::complex<double>> fd_data_,
        array_type<double>               fd_invC_,
        int n_rfft_, int num_channel_, int num_data_, int num_noise_,
        int ind_min_, int ind_max_, double df_,
        array_type<int>                  start_inds_)
    {
        fd = new FDDomain(
            (cmplx*) return_pointer_and_check_length(
                fd_data_, "fd_data",
                n_rfft_ * num_channel_ * num_data_, 1),
            return_pointer(fd_invC_, "fd_invC"),
            n_rfft_, num_channel_, num_data_, num_noise_,
            ind_min_, ind_max_, df_,
            return_pointer_and_check_length(
                start_inds_, "start_inds", num_data_, 1));
    };
    ~FDDomainWrap(){ delete fd; };
};

#endif // __BINDING_FD_DOMAIN_HPP__
