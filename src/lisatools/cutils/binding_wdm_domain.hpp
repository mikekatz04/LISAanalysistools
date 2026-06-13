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

    // Batched (d|h)/(h|h) likelihood terms on the WDM grid — WDM counterpart
    // of STFTDomainWrap::compute_likelihood_terms (2026-06 merge follow-up).
    // Outputs are real doubles; template sub-grids are addressed by integer
    // (m, n) start indices on the full WDM grid. The Python wrap layer
    // (lisatools.domaincomputation.WDMComputationGroup) converts physical
    // start times/frequencies to indices and validates active-band coverage
    // before calling (the GPU kernel cannot throw).
    void compute_likelihood_terms(
        array_type<double> d_h_out,
        array_type<double> h_h_out,
        array_type<double> template_vals,
        array_type<int> start_layer_m,
        array_type<int> start_time_n,
        int num_binaries,
        array_type<int> data_index,
        array_type<int> noise_index,
        int n_m_template,
        int n_n_template,
        int tdi_type,
        bool run_async = false)
    {
        double *d_h_ptr = return_pointer_and_check_length(d_h_out, "d_h_out", num_binaries, 1);
        double *h_h_ptr = return_pointer_and_check_length(h_h_out, "h_h_out", num_binaries, 1);
        double *tmpl_ptr = return_pointer_and_check_length(
            template_vals, "template_vals",
            num_binaries * wdm->num_channel * n_m_template * n_n_template, 1);
        int *sm_ptr = return_pointer_and_check_length(start_layer_m, "start_layer_m", num_binaries, 1);
        int *sn_ptr = return_pointer_and_check_length(start_time_n, "start_time_n", num_binaries, 1);
        int *di_ptr = return_pointer_and_check_length(data_index, "data_index", num_binaries, 1);
        int *ni_ptr = return_pointer_and_check_length(noise_index, "noise_index", num_binaries, 1);

        wdm->compute_likelihood_terms_wrap(
            d_h_ptr, h_h_ptr, tmpl_ptr,
            sm_ptr, sn_ptr,
            num_binaries,
            di_ptr, ni_ptr,
            n_m_template, n_n_template, tdi_type, run_async);
    }
};

#endif // __BINDING_WDM_DOMAIN_HPP__
