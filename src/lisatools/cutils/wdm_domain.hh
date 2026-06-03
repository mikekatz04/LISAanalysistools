#ifndef __WDM_DOMAIN_HH__
#define __WDM_DOMAIN_HH__

// WDMDomain -- WDM (Wilson Daubechies Meyer) time-frequency-domain data
// container + inverse-noise descriptor. Inherits from WDMSettings (already
// in LAT) to share grid metadata; adds wdm_data + wdm_noise pointers and
// the per-pixel inner-product / chain-rule helpers used by the chunked-
// heterodyne (and v2 signal-heterodyne) kernels.
//
// Phase 3L (2026-06-02): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.hh:466-525
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu:381-846
// to LISAanalysistools. All 12 method bodies are now header-inline (they
// are all CUDA_DEVICE-only and small enough to inline), so there is no
// .cu companion file.

#include "gbt_global.h"
#include "wdm_settings.hh"
#include <stdexcept>

#ifndef TDI_XYZ
#define TDI_XYZ 1
#endif
#ifndef TDI_AET
#define TDI_AET 2
#endif
#ifndef TDI_AE
#define TDI_AE 3
#endif

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define WDMDomain WDMDomainGPU
#else
#define WDMDomain WDMDomainCPU
#endif

class WDMDomain : public WDMSettings{
  public:

    double *wdm_data;
    double *wdm_noise;
    int num_data;
    int num_noise;

    CUDA_CALLABLE_MEMBER
    WDMDomain(double *wdm_data_, double *wdm_noise_, double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_, int num_data_, int num_noise_):
    WDMSettings(layer_df_, layer_dt_, Nf_, Nt_, num_channel_, ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_)
    {
        wdm_data = wdm_data_;
        wdm_noise = wdm_noise_;
        num_data = num_data_;
        num_noise = num_noise_;
    };

    CUDA_DEVICE inline
    int get_pixel_index(int m, int n, int channel, int data_index)
    {
        if (data_index >= num_data)
        {
#ifdef __CUDACC__
#else
            throw std::invalid_argument("data_index is larger than available data instances.");
#endif
        }
        return ((data_index * num_channel + channel) * Nf_active + (m - ind_min_f)) * Nt_active + (n - ind_min_t);
    }

    CUDA_DEVICE inline
    int get_pixel_index_noise(int m, int n, int channel, int noise_index)
    {
        if (noise_index >= num_noise)
        {
#ifdef __CUDACC__
#else
            throw std::invalid_argument("noise_index is larger than available noise instances.");
#endif
        }
        return ((noise_index * num_channel + channel) * Nf_active + (m - ind_min_f)) * Nt_active + (n - ind_min_t);
    }

    CUDA_DEVICE inline
    int get_pixel_index_noise_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index)
    {
        return (((noise_index * num_channel + channel_i) * num_channel + channel_j) * Nf_active + (m - ind_min_f)) * Nt_active + (n - ind_min_t);
    }

    CUDA_DEVICE inline
    double get_pixel_data_value(int m, int n, int channel, int data_index)
    {
        return wdm_data[get_pixel_index(m, n, channel, data_index)];
    }

    CUDA_DEVICE inline
    double get_pixel_noise_value(int m, int n, int channel, int noise_index)
    {
        return wdm_noise[get_pixel_index_noise(m, n, channel, noise_index)];
    }

    CUDA_DEVICE inline
    double get_pixel_noise_value_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index)
    {
        return wdm_noise[get_pixel_index_noise_cross_channel(m, n, channel_i, channel_j, noise_index)];
    }

    CUDA_DEVICE inline
    void get_inner_product_value(double *d_h, double *h_h, double wdm_template_nm, int m, int n, int channel, int data_index, int noise_index)
    {
        double wdm_data_nm = get_pixel_data_value(m, n, channel, data_index);
        double wdm_noise_nm = get_pixel_noise_value(m, n, channel, noise_index);
        double val_d_h = wdm_data_nm * wdm_template_nm * wdm_noise_nm * 0.25;
        double val_h_h = wdm_template_nm * wdm_template_nm * wdm_noise_nm * 0.25;

        *d_h = val_d_h;
        *h_h = val_h_h;
    }

    CUDA_DEVICE inline
    void get_inner_product_value_cross_channel(double *d_h, double *h_h, double wdm_template_nm_i, double wdm_template_nm_j, int m, int n, int channel_i, int channel_j, int data_index, int noise_index)
    {
        // assume data is channel_i, template is channel_j
        double wdm_data_nm_i = get_pixel_data_value(m, n, channel_i, data_index);
        double wdm_noise_nm_ij = get_pixel_noise_value_cross_channel(m, n, channel_i, channel_j, noise_index);

        // 0.25 factor is needed. Check python code
        double val_d_h = wdm_data_nm_i * wdm_template_nm_j * wdm_noise_nm_ij * 0.25;
        double val_h_h = wdm_template_nm_i * wdm_template_nm_j * wdm_noise_nm_ij * 0.25;

        *d_h = val_d_h;
        *h_h = val_h_h;
    }

    CUDA_DEVICE inline
    void add_ip_contrib(double *d_h_tmp, double *h_h_tmp, double *w_mn, int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
#ifdef __CUDACC__
        int tid = threadIdx.x;
#else
        int tid = 0;
#endif

        double d_h_val = 0.0;
        double h_h_val = 0.0;
        if (tdi_type == TDI_XYZ)
        {
            for (int channel_i = 0; channel_i < 3; channel_i += 1)
            {
                for (int channel_j = 0; channel_j < 3; channel_j += 1)
                {

                    // TODO: change from 9 to 6 calculations?
                    get_inner_product_value_cross_channel(&d_h_val, &h_h_val, w_mn[channel_i], w_mn[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                    d_h_tmp[tid] += d_h_val;
                    h_h_tmp[tid] += h_h_val;

                }
            }
        }
        else if (tdi_type == TDI_AET)
        {
            // AET: three orthogonal channels, diagonal noise. The caller is
            // responsible for providing AET-projected data/template values and
            // a diagonal-only noise buffer; both the CPU and CUDA builds run
            // the same loop.
            for (int channel_i = 0; channel_i < 3; channel_i += 1)
            {
                get_inner_product_value(&d_h_val, &h_h_val, w_mn[channel_i], layer_m, n, channel_i, data_index, noise_index);
                d_h_tmp[tid] += d_h_val;
                h_h_tmp[tid] += h_h_val;
            }
        }
        else if (tdi_type == TDI_AE)
        {
            // AE: two orthogonal channels (T dropped). Same loop body as AET
            // but truncated to channels {0,1}; the caller must pre-project.
            for (int channel_i = 0; channel_i < 2; channel_i += 1)
            {
                get_inner_product_value(&d_h_val, &h_h_val, w_mn[channel_i], layer_m, n, channel_i, data_index, noise_index);
                d_h_tmp[tid] += d_h_val;
                h_h_tmp[tid] += h_h_val;
            }
        }
    }

    CUDA_DEVICE inline
    void add_ip_swap_contrib(double *d_h_add_acc, double *d_h_remove_acc, double *add_add_acc, double *remove_remove_acc, double *add_remove_acc, double *w_mn_add, double *w_mn_remove, int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
        // Accumulators are per-thread scalars (register-resident in the caller). We
        // sum into local temporaries here and write them back at the end, so the
        // hot channel loop touches no shared/global memory and the previous
        // 5xNUM_THREADS_HERE shared staging buffer is gone.
        double d_h_add_local = 0.0;
        double d_h_remove_local = 0.0;
        double add_add_local = 0.0;
        double remove_remove_local = 0.0;
        double add_remove_local = 0.0;

        double d_h_val = 0.0;
        double hh_val = 0.0;

        int nchannels = 3;
        if (tdi_type == TDI_AE) nchannels = 2;

        if (tdi_type == TDI_XYZ)
        {
            for (int channel_i = 0; channel_i < 3; channel_i += 1)
            {
                for (int channel_j = 0; channel_j < 3; channel_j += 1)
                {
                    get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_add[channel_i], w_mn_add[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                    d_h_add_local += d_h_val;
                    add_add_local += hh_val;

                    get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_remove[channel_i], w_mn_remove[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                    d_h_remove_local += d_h_val;
                    remove_remove_local += hh_val;

                    // <h_add|h_remove>: only hh_val (= add_i * remove_j * noise_ij) is needed.
                    get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_add[channel_i], w_mn_remove[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                    add_remove_local += hh_val;
                }
            }
        }
        else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
        {
            // AET/AE: orthogonal channels, diagonal per-pixel noise. AET keeps
            // all three channels, AE drops T via nchannels=2. Caller must
            // supply data/template/noise in the projected basis. Same loop on
            // CPU and CUDA.
            for (int channel_i = 0; channel_i < nchannels; channel_i += 1)
            {
                get_inner_product_value(&d_h_val, &hh_val, w_mn_add[channel_i], layer_m, n, channel_i, data_index, noise_index);
                d_h_add_local += d_h_val;
                add_add_local += hh_val;

                get_inner_product_value(&d_h_val, &hh_val, w_mn_remove[channel_i], layer_m, n, channel_i, data_index, noise_index);
                d_h_remove_local += d_h_val;
                remove_remove_local += hh_val;

                get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_add[channel_i], w_mn_remove[channel_i], layer_m, n, channel_i, channel_i, data_index, noise_index);
                add_remove_local += hh_val;
            }
        }
        else
        {
#ifdef __CUDACC__
#else
            throw std::invalid_argument("Incorrect TDI type.");
#endif
        }

        *d_h_add_acc += d_h_add_local;
        *d_h_remove_acc += d_h_remove_local;
        *add_add_acc += add_add_local;
        *remove_remove_acc += remove_remove_local;
        *add_remove_acc += add_remove_local;
    }

    // Per-pixel chain-rule contribution:
    //   grad_acc_k += sum_{c,c'} (w_d - w_h)_c * (dw_h/dtheta_k)_{c'} * N^{-1}_{cc'} * 0.25
    // (XYZ cross-channel; the AET / AE branches use the diagonal noise).
    CUDA_DEVICE inline
    void add_grad_contrib(double *grad_acc_k, const double *w_mn, const double *dw_mn_dk,
                          int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
        double local_acc = 0.0;
        if (tdi_type == TDI_XYZ)
        {
            for (int ci = 0; ci < 3; ci += 1)
            {
                double w_d_i = get_pixel_data_value(layer_m, n, ci, data_index);
                double r_i = w_d_i - w_mn[ci];
                for (int cj = 0; cj < 3; cj += 1)
                {
                    double N_ij = get_pixel_noise_value_cross_channel(layer_m, n, ci, cj, noise_index);
                    local_acc += r_i * dw_mn_dk[cj] * N_ij * 0.25;
                }
            }
        }
        else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
        {
            int nchannels = (tdi_type == TDI_AE) ? 2 : 3;
            for (int c = 0; c < nchannels; c += 1)
            {
                double w_d = get_pixel_data_value(layer_m, n, c, data_index);
                double N_c = get_pixel_noise_value(layer_m, n, c, noise_index);
                local_acc += (w_d - w_mn[c]) * dw_mn_dk[c] * N_c * 0.25;
            }
        }
        *grad_acc_k += local_acc;
    }

    // Swap variant: accumulates +/- r_after * dw * N^{-1}, where
    //   r_after = w_d - w_add_center + w_rem_center.
    // `sign` selects between the add side (+1, dw = dw_add) and the remove
    // side (-1, dw = dw_rem); the helper is called once per parameter and
    // once per side.
    CUDA_DEVICE inline
    void add_swap_grad_contrib_one_side(
        double *grad_acc_k, double sign,
        const double *w_mn_add, const double *w_mn_rem, const double *dw_mn_dk,
        int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
        double local_acc = 0.0;
        if (tdi_type == TDI_XYZ)
        {
            for (int ci = 0; ci < 3; ci += 1)
            {
                double w_d_i = get_pixel_data_value(layer_m, n, ci, data_index);
                double r_i = w_d_i - w_mn_add[ci] + w_mn_rem[ci];
                for (int cj = 0; cj < 3; cj += 1)
                {
                    double N_ij = get_pixel_noise_value_cross_channel(layer_m, n, ci, cj, noise_index);
                    local_acc += sign * r_i * dw_mn_dk[cj] * N_ij * 0.25;
                }
            }
        }
        else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
        {
            int nchannels = (tdi_type == TDI_AE) ? 2 : 3;
            for (int c = 0; c < nchannels; c += 1)
            {
                double w_d = get_pixel_data_value(layer_m, n, c, data_index);
                double N_c = get_pixel_noise_value(layer_m, n, c, noise_index);
                double r_c = w_d - w_mn_add[c] + w_mn_rem[c];
                local_acc += sign * r_c * dw_mn_dk[c] * N_c * 0.25;
            }
        }
        *grad_acc_k += local_acc;
    }
};

#endif // __WDM_DOMAIN_HH__
