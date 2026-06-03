#ifndef __BINDING_WDM_DOMAIN_HPP__
#define __BINDING_WDM_DOMAIN_HPP__

// WDMDomainWrap -- pybind11 wrapper for WDMDomain.
// Phase 3L (2026-06-02): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.hpp:234-264
// to LISAanalysistools.
//
// WDMDomainWrap inherits from WDMSettingsWrap (also in LAT at Phase 3L);
// the `#define` alias below keeps the inherited class-name resolution
// working for both backends after this move.

#include "wdm_domain.hh"
#include "binding_wdm_settings.hpp"  // WDMSettingsWrap + ReturnPointerBase

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define WDMDomainWrap WDMDomainWrapGPU
#else
#define WDMDomainWrap WDMDomainWrapCPU
#endif

class WDMDomainWrap : public WDMSettingsWrap {
  public:
    WDMDomain *wdm;

    WDMDomainWrap(array_type<double> wdm_data_, array_type<double> wdm_noise_,
                  double layer_df_, double layer_dt_, int Nf_, int Nt_,
                  int num_channel_, int ind_min_t_, int ind_max_t_,
                  int ind_min_f_, int ind_max_f_,
                  int num_data_, int num_noise_)
      : WDMSettingsWrap(layer_df_, layer_dt_, Nf_, Nt_, num_channel_,
                        ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_)
    {
        // TODO: adjust noise length check to TDI setups
        int Nt_active = ind_max_t_ - ind_min_t_ + 1;
        int Nf_active = ind_max_f_ - ind_min_f_ + 1;
        wdm = new WDMDomain(
            return_pointer_and_check_length(wdm_data_, "wdm_data", Nt_active * Nf_active * num_channel_ * num_data_, 1),
            return_pointer(wdm_noise_, "wdm_noise"),  // return_pointer_and_check_length(wdm_noise_, "wdm_noise", Nt_ * Nf_ * num_channel_ * num_noise_, 1),
            layer_df_, layer_dt_, Nf_, Nt_, num_channel_,
            ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_,
            num_data_, num_noise_
        );
    };
    ~WDMDomainWrap(){
        delete wdm;
        // base dtor handles wdm_settings
    };
};

#endif // __BINDING_WDM_DOMAIN_HPP__
