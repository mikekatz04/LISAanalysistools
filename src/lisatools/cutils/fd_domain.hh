#ifndef __FD_DOMAIN_HH__
#define __FD_DOMAIN_HH__

// FDDomain -- frequency-domain data container + inverse-noise descriptor.
// Used by the chunked-heterodyne and signal-heterodyne kernels to evaluate
// per-bin <d|h> and <h|h>.
//
// Phase 3L (2026-06-02): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.hh:561-610
// to LISAanalysistools as the first installment of the C++ TDIonTheFly
// carve-out. Fully header-inline (no out-of-line method bodies in .cu),
// so no .cu file companion needed.

#include "gbt_global.h"
#include <cstddef>

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define FDDomain FDDomainGPU
#else
#define FDDomain FDDomainCPU
#endif

class FDDomain {
  public:
    cmplx  *fd_data;   // (num_data, num_channel, n_rfft) complex
    double *fd_invC;   // tdi_type=TDI_XYZ: (num_noise, num_channel, num_channel, n_rfft)
                       // tdi_type=TDI_AET/AE: (num_noise, num_channel, n_rfft)
    int    n_rfft;
    int    num_channel;
    int    num_data;
    int    num_noise;
    int    ind_min;    // inclusive
    int    ind_max;    // inclusive
    double df;
    double Tobs;       // = 1/df, kept for convenience

    CUDA_CALLABLE_MEMBER
    FDDomain(cmplx *fd_data_, double *fd_invC_, int n_rfft_,
             int num_channel_, int num_data_, int num_noise_,
             int ind_min_, int ind_max_, double df_)
    {
        fd_data     = fd_data_;
        fd_invC     = fd_invC_;
        n_rfft      = n_rfft_;
        num_channel = num_channel_;
        num_data    = num_data_;
        num_noise   = num_noise_;
        ind_min     = ind_min_;
        ind_max     = ind_max_;
        df          = df_;
        Tobs        = 1.0 / df_;
    };
    CUDA_DEVICE inline cmplx get_data(int k, int channel, int data_index) const
    {
        return fd_data[(size_t) data_index * num_channel * n_rfft
                       + (size_t) channel * n_rfft + k];
    }
    CUDA_DEVICE inline double get_invC_diag(int k, int channel, int noise_index) const
    {
        return fd_invC[(size_t) noise_index * num_channel * n_rfft
                       + (size_t) channel * n_rfft + k];
    }
    CUDA_DEVICE inline double get_invC_cross(int k, int c1, int c2, int noise_index) const
    {
        return fd_invC[(((size_t) noise_index * num_channel + c1)
                        * num_channel + c2) * n_rfft + k];
    }
    CUDA_DEVICE inline bool in_band(int k) const
    {
        return (k >= ind_min) && (k <= ind_max);
    }
};

#endif // __FD_DOMAIN_HH__
