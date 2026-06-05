#ifndef __BINDING_WDM_SETTINGS_HPP__
#define __BINDING_WDM_SETTINGS_HPP__

// WDMSettingsWrap -- pybind11 wrapper for WDMSettings.
// Phase 3L (2026-06-02): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.hpp:227-244
// to LISAanalysistools.
//
// WDMDomainWrap (still in lisa-on-gpu) inherits from WDMSettingsWrap;
// the `#define` alias keeps the inherited class name resolution working
// for both backends after this move.

#include "wdm_settings.hh"
#include "binding_flr.hpp"  // ReturnPointerBase

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define WDMSettingsWrap WDMSettingsWrapGPU
#else
#define WDMSettingsWrap WDMSettingsWrapCPU
#endif

class WDMSettingsWrap : public ReturnPointerBase {
  public:
    WDMSettings *wdm_settings;

    WDMSettingsWrap(double layer_df_, double layer_dt_, int Nf_, int Nt_,
                    int num_channel_, int ind_min_t_, int ind_max_t_,
                    int ind_min_f_, int ind_max_f_)
    {
        wdm_settings = new WDMSettings(
            layer_df_, layer_dt_, Nf_, Nt_, num_channel_,
            ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_
        );
    };
    ~WDMSettingsWrap(){
        delete wdm_settings;
    };

};

#endif // __BINDING_WDM_SETTINGS_HPP__
