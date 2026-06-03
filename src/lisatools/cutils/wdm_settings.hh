#ifndef __WDM_SETTINGS_HH__
#define __WDM_SETTINGS_HH__

// WDMSettings -- POD config describing the WDM (Wilson Daubechies Meyer)
// time-frequency grid and the active (m, n) band of interest.
//
// Phase 3L (2026-06-02): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.hh:459-488
// to LISAanalysistools. Fully header-inline (no .cu counterpart).
//
// WDMDomain (still in lisa-on-gpu, planned to move) and
// WaveletLookupTable (still in lisa-on-gpu, planned to move) both
// inherit from WDMSettings. The `#define` alias below ensures the
// inherited class names continue to resolve through the
// per-build-backend type aliasing.

#include "gbt_global.h"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define WDMSettings WDMSettingsGPU
#else
#define WDMSettings WDMSettingsCPU
#endif

class WDMSettings{
  public:
    int Nt;
    int Nf;
    int num_channel;
    double layer_df;
    double layer_dt;
    int ind_min_t;
    int ind_max_t;
    int ind_min_f;
    int ind_max_f;
    int Nf_active;
    int Nt_active;

    CUDA_CALLABLE_MEMBER
    WDMSettings(double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_){
        Nf = Nf_;
        Nt = Nt_;
        num_channel = num_channel_;
        layer_df = layer_df_;
        layer_dt = layer_dt_;
        ind_min_t = ind_min_t_;
        ind_max_t = ind_max_t_;
        ind_min_f = ind_min_f_;
        ind_max_f = ind_max_f_;
        Nf_active = ind_max_f - ind_min_f + 1; // inclusive
        Nt_active = ind_max_t - ind_min_t + 1; // inclusive
    };
};

#endif // __WDM_SETTINGS_HH__
