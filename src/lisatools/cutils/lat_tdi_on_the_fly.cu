// === LISATDIonTheFly out-of-line method bodies ===
// Phase 3L.5 (2026-06-03): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu:
//   - destructor: lines 31-35
//   - first block: lines 6414-7629
//   - run_wave_tdi: lines 7741-7809

#include "lat_tdi_on_the_fly.hh"
#include "gbt_global.h"
#include "Interpolate.hh"
#include "LISAResponse.hh"
#include "Detector.hpp"
#include "fd_domain.hh"
#include "wdm_settings.hh"
#include "wdm_domain.hh"
#include <stdexcept>
#include <cmath>

#ifdef __CUDACC__
#define NUM_THREADS_HERE 128
#else
#define NUM_THREADS_HERE 1
#endif

#ifndef N_PARAMS_MAX
#define N_PARAMS_MAX 20
#endif

#ifndef NLINKS
#define NLINKS 6
#endif

// ---------------------------------------------------------------
// OrbitsSplineCache evaluation helpers
// (was TDIonTheFly.cu:1679-1731 -- the producer side
// populate_orbit_spline_cache stays in lisa-on-gpu until Phase 3L.7
// alongside the GB chunked-het kernels that drive it.)
// ---------------------------------------------------------------

// Inline helper: locate the segment index for an arbitrary t in the cache's
// uniform grid, clamped to [0, N_cp - 2].
CUDA_DEVICE
inline int _orbit_cache_seg(const OrbitsSplineCache *c, double t)
{
    int seg = (int) ((t - c->t_cp0) / c->dt_cp);
    if (seg < 0)             seg = 0;
    if (seg > c->N_cp - 2)   seg = c->N_cp - 2;
    return seg;
}

CUDA_DEVICE
inline int _orbit_cache_link_index(int link)
{
    // Mirrors Orbits::get_link_ind. Returns -1 on bad link (caller bug).
    switch (link) {
        case 12: return 0;
        case 23: return 1;
        case 31: return 2;
        case 13: return 3;
        case 32: return 4;
        case 21: return 5;
        default: return -1;
    }
}

CUDA_DEVICE
inline double cache_get_light_travel_time(const OrbitsSplineCache *c,
                                           double t, int link)
{
    const int link_i = _orbit_cache_link_index(link);
    const int seg    = _orbit_cache_seg(c, t);
    const int p      = link_i * c->N_cp + seg;
    const double dx  = t - c->t_cp[seg];
    return c->ltt_y[p]
         + c->ltt_c1[p] * dx
         + c->ltt_c2[p] * dx * dx
         + c->ltt_c3[p] * dx * dx * dx;
}

CUDA_DEVICE
inline Vec cache_get_pos(const OrbitsSplineCache *c, double t, int sc)
{
    const int seg = _orbit_cache_seg(c, t);
    const double dx = t - c->t_cp[seg];
    const int base = (sc - 1) * 3;   // sc in 1..3
    double v[3];
    for (int xyz = 0; xyz < 3; ++xyz) {
        const int p = (base + xyz) * c->N_cp + seg;
        v[xyz] = c->pos_y[p]
              + c->pos_c1[p] * dx
              + c->pos_c2[p] * dx * dx
              + c->pos_c3[p] * dx * dx * dx;
    }
    return Vec(v[0], v[1], v[2]);
}

// ---------------------------------------------------------------
// Destructor (was TDIonTheFly.cu:31-35)
// ---------------------------------------------------------------
CUDA_DEVICE
LISATDIonTheFly::~LISATDIonTheFly()
{
    return;
}

// ---------------------------------------------------------------
// First block: get_sky_vectors through get_tdi_heterodyned
// (was TDIonTheFly.cu:6414-7629)
// ---------------------------------------------------------------
void LISATDIonTheFly::get_sky_vectors(Vec *k, Vec *u, Vec *v, double *params)
{

    double beta = params[beta_index];
    double lam = params[lam_index];
    double cosbeta = cos(beta);
    double sinbeta = sin(beta);

    double coslam = cos(lam);
    double sinlam = sin(lam);
    v->x = -sinbeta * coslam;
    v->y = -sinbeta * sinlam;
    v->z = cosbeta;
    u->x = sinlam;
    u->y = -coslam;
    u->z = 0.0;
    k->x = -cosbeta * coslam;
    k->y = -cosbeta * sinlam;
    k->z = -sinbeta;
    
}

CUDA_DEVICE
void LISATDIonTheFly::xi_projections(double *xi_p, double *xi_c, Vec u, Vec v, Vec n)
{
    double u_dot_n = u.dot(n);
    double v_dot_n = v.dot(n);

    *xi_p = 0.5 * ((u_dot_n * u_dot_n) - (v_dot_n * v_dot_n));
    *xi_c = u_dot_n * v_dot_n;
}

CUDA_DEVICE
void LISATDIonTheFly::fill_link_arrays(int *link_Space_craft_rec, int *link_Space_craft_em)
{
    for (int i = THREAD_START_X; i < NLINKS; i += BLOCK_INCR_X)
    {
        link_Space_craft_rec[i] = orbits->sc_r[i];
        link_Space_craft_em[i] = orbits->sc_e[i];
        // links[i] = orbits->links[i];
        // if (threadIdx.x == 1)
        // printf("%d %d %d %d\n", orbits->sc_r[i], orbits->sc_e[i], link_Space_craft_em[i], link_Space_craft_rec[i]);
    }
    CUDA_SYNC_THREADS;
}
CUDA_DEVICE
void LISATDIonTheFly::get_tdi_Xf(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i, int *link_Space_craft_rec, int *link_Space_craft_em, Vec k, Vec u, Vec v)
{
    double t;
    cmplx tdi_channel_tmp[3];
    for (int i = THREAD_START_X; i < N; i += BLOCK_INCR_X)
    {
        t = t_data[i];
        get_tdi_Xf_single(&tdi_channel_tmp[0], t, params, k, u, v, link_Space_craft_rec, link_Space_craft_em, bin_i);
        
        for (int channel = 0; channel < tdi_config->num_channels; channel += 1)
        {
            tdi_channels_arr[channel * N + i] = tdi_channel_tmp[channel];
        }
    }
}

// void LISATDIonTheFly::get_tdi_Xf_single_with_f_fdot(cmplx *tdi_channel, double *f, double *fdot, double t, double *params, Vec *k, Vec *u, Vec *v, int *link_Space_craft_rec, int *link_Space_craft_em)
// {
//     get_tdi_Xf_single(tdi_channel, t, k, u, v, link_Space_craft_rec, link_Space_craft_em);

//     cmplx tdi_channels_up[3];
//     cmplx tdi_channels_down[3];
//     double eps_rel = 1e-9
//     double t_up = t * (1. + eps_rel);
//     double t_down = t * (1. - eps_rel);
//     double h = t_up - t;

//     get_tdi_Xf_single(tdi_channels_up, t_up, k, u, v, link_Space_craft_rec, link_Space_craft_em);
//     get_tdi_Xf_single(tdi_channels_down, t_down, k, u, v, link_Space_craft_rec, link_Space_craft_em);

//     double phase_mid, phase_up, phase_down;

//     for (int i = 0; i < 3; i += 1)
//     {
//         phase_down = gcmplx::arg(tdi_channels_down[i]);
//         phase_mid = gcmplx::arg(tdi_channels[i]);
//         phase_up = gcmplx::arg(tdi_channels_up[i]);

//         if (phase_up - phase_down) > M_PI
//         {
            
//         }

//         f[i] = (phase_up - phase_down) / (2 * h);
//         fdot[i] = (phase_up - 2 * phase_mid + phase_up) / (h * h);
//     }
// }

void LISATDIonTheFly::get_tdi_Xf_single(cmplx *tdi_channel, double t, double *params, Vec k, Vec u, Vec v, int *link_Space_craft_rec, int *link_Space_craft_em, int bin_i)
{
    Vec x_rec;
    Vec x_em;
    Vec n;
    double delay_rec, phase_change;
    double delay_em;
    double xi_p;
    double xi_c;
    double k_dot_n, k_dot_x_rec, k_dot_x_em;
    double L;
    double hp_del_rec, hp_del_em, hc_del_rec, hc_del_em;
    cmplx I(0.0, 1.0);
    double pre_factor, large_factor_real, large_factor_imag;

    tdi_channel[0] = 0.0;
    tdi_channel[1] = 0.0;
    tdi_channel[2] = 0.0;
    int sc_r, sc_e;
    double total_delay;
    double time_eval, time_rec, time_em;
    double norm;
    cmplx tmp_channel_output[3];
    tmp_channel_output[0] = 0.0;
    tmp_channel_output[1] = 0.0;
    tmp_channel_output[2] = 0.0;
    int window = 0;
    bool is_okay = true;
    for (int unit_i = 0; unit_i < tdi_config->num_units; unit_i += 1)
    {
        int unit_start = tdi_config->unit_starts[unit_i];
        int unit_length = tdi_config->unit_lengths[unit_i];
        int base_link = tdi_config->tdi_base_link[unit_i];
        int base_link_index = orbits->get_link_ind(base_link);
        int channel = tdi_config->channels[unit_i];
        double sign = tdi_config->tdi_signs_in[unit_i];

        total_delay = 0.0;
        for (int sub_i = 0; sub_i < unit_length; sub_i += 1)
        {
            
            int combination_index = unit_start + sub_i;
            int combination_link = tdi_config->tdi_link_combinations[combination_index];
            // int combination_link_index;
            // if (combination_link == -11)
            // {
            //     combination_link_index = -1;
            // }
            // else
            // {
            //     combination_link_index = orbits->get_link_ind(combination_link);
            // }

            if (combination_link != -11)
            {
                total_delay += orbits->get_light_travel_time(t, combination_link);
            }
        }
        
        time_eval = t - total_delay;
        time_rec = time_eval;

        window = orbits->get_window(time_rec, orbits->ltt_t0, orbits->ltt_dt, orbits->ltt_N);
        if (window == -1)
        {
            // out of bounds
            is_okay = false;
            break;
        }
        window = orbits->get_window(time_eval, orbits->ltt_t0, orbits->ltt_dt, orbits->ltt_N);
        if (window == -1)
        {
            // out of bounds
            is_okay = false;
            break;
        }
        window = orbits->get_window(time_rec, orbits->sc_t0, orbits->sc_dt, orbits->sc_N);
        if (window == -1)
        {
            // out of bounds
            is_okay = false;
            break;
        }
        window = orbits->get_window(time_eval, orbits->sc_t0, orbits->sc_dt, orbits->sc_N);
        if (window == -1)
        {
            // out of bounds
            is_okay = false;
            break;
        }
        
        L = orbits->get_light_travel_time(time_rec, base_link);
        time_em = time_rec - L;

        sc_r = link_Space_craft_rec[base_link_index];
        sc_e = link_Space_craft_em[base_link_index];

        
        x_rec = orbits->get_pos(time_rec, sc_r);
        x_em = orbits->get_pos(time_em, sc_e);
        n = x_rec - x_em; // # TODO: check if this right
        norm = sqrt(n.dot(n));
        n = n / norm;

        k_dot_n = k.dot(n);
        k_dot_x_rec = k.dot(x_rec); // receiver
        k_dot_x_em = k.dot(x_em); // emitter

        // Guard the LISA arm-response singularity: when the wave propagation
        // direction k is parallel to the arm n, (1-k.n) -> 0 while xi_p, xi_c
        // -> 0 simultaneously, producing 0 * Inf = NaN. Skip the contribution
        // on the singular line (limit is well-defined and ~0 for sources not
        // sitting exactly on the arm axis).
        {
            double _denom = 1. - k_dot_n;
            if (fabs(_denom) < 1.0e-12) continue;
            pre_factor = 1. / _denom;
        }

        delay_rec = time_rec - k_dot_x_rec * C_inv;
        delay_em = time_em - k_dot_x_em * C_inv;

        xi_projections(&xi_p, &xi_c, u, v, n);

        phase_change = 0.0; // the real part
        get_hp_hc(&hp_del_rec, &hc_del_rec, delay_rec, params, phase_change, bin_i);
        get_hp_hc(&hp_del_em, &hc_del_em, delay_em, params, phase_change, bin_i);
        
        large_factor_real = (hp_del_em - hp_del_rec) * xi_p + (hc_del_em - hc_del_rec) * xi_c;
        
        phase_change = M_PI / 2.0; // the real part
        get_hp_hc(&hp_del_rec, &hc_del_rec, delay_rec, params, phase_change, bin_i);
        get_hp_hc(&hp_del_em, &hc_del_em, delay_em, params, phase_change, bin_i);
        
        large_factor_imag = (hp_del_em - hp_del_rec) * xi_p + (hc_del_em - hc_del_rec) * xi_c;
        tmp_channel_output[channel] += sign * pre_factor * (large_factor_real + I * large_factor_imag);
    }
    if (is_okay)
    {
        for (int channel = 0; channel < 3; channel += 1)
        {
            tdi_channel[channel] = tmp_channel_output[channel];
        }
    }
}

// FLY: 100 1.556340000000e+06 0 3 13 9.221909979440e-23 1.556340000000e+06
// FLY: 100 1.556340000000e+06 0 2 31 9.120999116182e-23 1.556331661075e+06

// BASE: 155634 1.556340000000e+06 0 3 13 9.221909189436e-23 1.556340000000e+06
// BASE: 155634 1.556340000000e+06 0 2 31 8.403496674567e-23 1.556331661075e+06


// ============================================================================
// Orbit-cache variants of get_tdi_Xf_single / get_tdi_Xf / get_tdi /
// get_tdi_heterodyned. Mirror the originals but swap raw orbit lookups
// (orbits->get_*) for cached cubic-spline evaluations (cache_get_*).
// ============================================================================
CUDA_DEVICE
void LISATDIonTheFly::get_tdi_Xf_single_cached(
    cmplx *tdi_channel, double t, double *params,
    Vec k, Vec u, Vec v,
    int *link_Space_craft_rec, int *link_Space_craft_em, int bin_i,
    OrbitsSplineCache *cache)
{
    Vec x_rec, x_em, n;
    double delay_rec, phase_change, delay_em;
    double xi_p, xi_c;
    double k_dot_n, k_dot_x_rec, k_dot_x_em;
    double L;
    double hp_del_rec, hp_del_em, hc_del_rec, hc_del_em;
    cmplx I(0.0, 1.0);
    double pre_factor, large_factor_real, large_factor_imag;

    tdi_channel[0] = 0.0;
    tdi_channel[1] = 0.0;
    tdi_channel[2] = 0.0;
    int sc_r, sc_e;
    double total_delay;
    double time_eval, time_rec, time_em;
    double norm;
    cmplx tmp_channel_output[3];
    tmp_channel_output[0] = 0.0;
    tmp_channel_output[1] = 0.0;
    tmp_channel_output[2] = 0.0;
    for (int unit_i = 0; unit_i < tdi_config->num_units; unit_i += 1)
    {
        int unit_start  = tdi_config->unit_starts[unit_i];
        int unit_length = tdi_config->unit_lengths[unit_i];
        int base_link   = tdi_config->tdi_base_link[unit_i];
        int base_link_index = _orbit_cache_link_index(base_link);
        int channel     = tdi_config->channels[unit_i];
        double sign     = tdi_config->tdi_signs_in[unit_i];

        total_delay = 0.0;
        for (int sub_i = 0; sub_i < unit_length; sub_i += 1)
        {
            int combination_index = unit_start + sub_i;
            int combination_link  = tdi_config->tdi_link_combinations[combination_index];
            if (combination_link != -11)
            {
                total_delay += cache_get_light_travel_time(cache, t, combination_link);
            }
        }

        time_eval = t - total_delay;
        time_rec  = time_eval;

        // Bounds checks against orbits' raw global tables are not needed:
        // by construction the cache is built from t-values inside the
        // source's valid orbit window, so any t inside the chunk is in
        // bounds. Compare to the original which sets is_okay=false on
        // get_window == -1.

        L       = cache_get_light_travel_time(cache, time_rec, base_link);
        time_em = time_rec - L;

        sc_r = link_Space_craft_rec[base_link_index];
        sc_e = link_Space_craft_em[base_link_index];

        x_rec = cache_get_pos(cache, time_rec, sc_r);
        x_em  = cache_get_pos(cache, time_em, sc_e);
        n     = x_rec - x_em;
        norm  = sqrt(n.dot(n));
        n     = n / norm;

        k_dot_n     = k.dot(n);
        k_dot_x_rec = k.dot(x_rec);
        k_dot_x_em  = k.dot(x_em);

        // Guard the arm-response singularity (see get_tdi_Xf_single).
        {
            double _denom = 1.0 - k_dot_n;
            if (fabs(_denom) < 1.0e-12) continue;
            pre_factor = 1.0 / _denom;
        }
        delay_rec  = time_rec - k_dot_x_rec * C_inv;
        delay_em   = time_em  - k_dot_x_em  * C_inv;

        xi_projections(&xi_p, &xi_c, u, v, n);

        phase_change = 0.0;
        get_hp_hc(&hp_del_rec, &hc_del_rec, delay_rec, params, phase_change, bin_i);
        get_hp_hc(&hp_del_em,  &hc_del_em,  delay_em,  params, phase_change, bin_i);
        large_factor_real = (hp_del_em - hp_del_rec) * xi_p + (hc_del_em - hc_del_rec) * xi_c;

        phase_change = M_PI / 2.0;
        get_hp_hc(&hp_del_rec, &hc_del_rec, delay_rec, params, phase_change, bin_i);
        get_hp_hc(&hp_del_em,  &hc_del_em,  delay_em,  params, phase_change, bin_i);
        large_factor_imag = (hp_del_em - hp_del_rec) * xi_p + (hc_del_em - hc_del_rec) * xi_c;

        tmp_channel_output[channel] += sign * pre_factor * (large_factor_real + I * large_factor_imag);
    }
    for (int channel = 0; channel < 3; channel += 1)
    {
        tdi_channel[channel] = tmp_channel_output[channel];
    }
}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi_Xf_cached(
    cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i,
    int *link_Space_craft_rec, int *link_Space_craft_em,
    Vec k, Vec u, Vec v, OrbitsSplineCache *cache)
{
    double t;
    cmplx tdi_channel_tmp[3];
    for (int i = THREAD_START_X; i < N; i += BLOCK_INCR_X)
    {
        t = t_data[i];
        get_tdi_Xf_single_cached(&tdi_channel_tmp[0], t, params, k, u, v,
                                  link_Space_craft_rec, link_Space_craft_em,
                                  bin_i, cache);
        for (int channel = 0; channel < tdi_config->num_channels; channel += 1)
        {
            tdi_channels_arr[channel * N + i] = tdi_channel_tmp[channel];
        }
    }
}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi_cached(
    void *buffer, int buffer_length,
    cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int bin_i, int nchannels,
    OrbitsSplineCache *cache)
{
#ifdef __CUDACC__
#else
    if (buffer_length < 2 * N * sizeof(double) + 1 * N * sizeof(int) + 1 * N * sizeof(bool))
    {
        throw std::invalid_argument("Buffer length not long enough.");
    }
#endif

    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];

    fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params);

    get_tdi_Xf_cached(tdi_channels_arr, params, t_arr, N, bin_i,
                       link_Space_craft_rec, link_Space_craft_em, k, u, v, cache);
    CUDA_SYNC_THREADS;

    // Phase-extract scratch carved out of the caller-allocated buffer.
    double *flip      = (double *) buffer;
    double *pjump     = &flip[N];
    int    *count     = (int *)  &pjump[N];
    bool   *fix_count = (bool *) &count[N];
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    for (int i = start; i < N; i += incr)
    {
        phi_ref[i] = get_phase_ref(t_arr[i], params, bin_i);
    }
    CUDA_SYNC_THREADS;
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N,
                                     &tdi_amp[0],     &tdi_phase[0],
                                     &tdi_channels_arr[0], &phi_ref[0]);
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N,
                                     &tdi_amp[N],     &tdi_phase[N],
                                     &tdi_channels_arr[N], &phi_ref[0]);
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N,
                                     &tdi_amp[2 * N], &tdi_phase[2 * N],
                                     &tdi_channels_arr[2 * N], &phi_ref[0]);

    double *ph_correct_buffer = &flip[0];
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[0]);
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[N]);
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[2 * N]);
}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi_heterodyned_cached(
    void *buffer, int buffer_length,
    cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref_het,
    double *params, double *t_arr, int N, int bin_i, int nchannels,
    double f0_grid, OrbitsSplineCache *cache)
{
    get_tdi_cached(buffer, buffer_length, tdi_channels_arr,
                    tdi_amp, tdi_phase, phi_ref_het,
                    params, t_arr, N, bin_i, nchannels, cache);
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    const double two_pi_f0 = 2.0 * M_PI * f0_grid;
    for (int i = start; i < N; i += incr)
    {
        phi_ref_het[i] -= two_pi_f0 * t_arr[i];
    }
    CUDA_SYNC_THREADS;
}


// Raw TDI evaluators: fill ``tdi_channels_arr`` (nchannels * N raw complex
// samples) and ``phi_ref`` (N points, UN-HETERODYNED -- i.e. just
// ``get_phase_ref(t_i)`` straight from the source, NO carrier
// subtraction). Skip the per-channel amplitude/phase extract + unwrap that
// get_tdi[_cached] performs.
//
// IMPORTANT -- why we emit un-het phi_ref here rather than phi_ref_het:
// the downstream ``new_extract_amplitude_and_phase`` consumes phiR via
// ``remainder(phiR, 2*pi)``, which is NOT invariant under shifts by
// 2*pi*f0_grid*t (the carrier offset is not a multiple of 2*pi). If we
// pre-subtracted the carrier here, every per-channel extract would see a
// shifted phiR and produce a Dphi that differs from the OLD direct-path
// get_tdi convention by a per-sample non-2*pi amount. That residual would
// NOT cancel against the downstream ``+ dphi_ref + phi0_chunk`` term --
// it would offset the slow-signal phase, shoving the FFTed energy off the
// snapped chunk-FD bin. Caller is responsible for the carrier subtraction
// when forming dphi_ref for the spline fit (see
// fast_wdm_inner_heterodyne_spline).
//
// Used by the chunked-het spline path so the caller can extract + unwrap
// one channel at a time into single-channel coefficient buffers
// (~6 KB / kernel shared-mem reduction).
CUDA_DEVICE
void LISATDIonTheFly::get_tdi_raw(
    cmplx *tdi_channels_arr, double *phi_ref,
    double *params, double *t_arr, int N, int bin_i, int nchannels)
{
    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];

    fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params);
    get_tdi_Xf(tdi_channels_arr, params, t_arr, N, bin_i,
                link_Space_craft_rec, link_Space_craft_em, k, u, v);
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    for (int i = start; i < N; i += incr)
    {
        phi_ref[i] = get_phase_ref(t_arr[i], params, bin_i);
    }
    CUDA_SYNC_THREADS;
}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi_raw_cached(
    cmplx *tdi_channels_arr, double *phi_ref,
    double *params, double *t_arr, int N, int bin_i, int nchannels,
    OrbitsSplineCache *cache)
{
    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];

    fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params);
    get_tdi_Xf_cached(tdi_channels_arr, params, t_arr, N, bin_i,
                       link_Space_craft_rec, link_Space_craft_em,
                       k, u, v, cache);
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    for (int i = start; i < N; i += incr)
    {
        phi_ref[i] = get_phase_ref(t_arr[i], params, bin_i);
    }
    CUDA_SYNC_THREADS;
}


CUDA_DEVICE
double LISATDIonTheFly::get_amp(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
#ifdef __CUDACC__
#else
    throw std::invalid_argument("Not implemented.");
#endif

}

CUDA_DEVICE
double LISATDIonTheFly::get_f(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
#ifdef __CUDACC__
#else
    throw std::invalid_argument("Not implemented.");
#endif

}

CUDA_DEVICE
double LISATDIonTheFly::get_fdot(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
#ifdef __CUDACC__
#else
    throw std::invalid_argument("Not implemented.");
#endif

}

CUDA_DEVICE
double LISATDIonTheFly::get_phase(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
#ifdef __CUDACC__
#else
    throw std::invalid_argument("Not implemented.");
#endif
}

CUDA_DEVICE
void LISATDIonTheFly::get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i)
{
    double amp = get_amp(t, params, bin_i);
    double phase = get_phase(t, params, bin_i);
    double psi = params[psi_index];
    double inc = params[inc_index];
    
    double inc_p = (1. + cos(inc) * cos(inc)) / 2.;
    double inc_c = cos(inc);
    
    // *hp = amp * (inc_p * cos(2. * psi) * cos(phase + phase_change) - inc_c * sin(2. * psi) * sin(phase + phase_change));
    // *hc = amp * (-inc_p * sin(2. * psi) * cos(phase + phase_change) - inc_c * cos(2. * psi) * sin(phase + phase_change));  
    double cos2psi = cos(2.0 * psi);
    double sin2psi = sin(2.0 * psi);
    double cosiota = cos(inc);

    double hSp = -cos(phase + phase_change) * amp * (1.0 + cosiota * cosiota);
    double hSc = -sin(phase + phase_change) * 2.0 * amp * cosiota;

    *hp = hSp * cos2psi - hSc * sin2psi;
    *hc = hSp * sin2psi + hSc * cos2psi;
    // printf("FLYIN: %.12e %.12e %.12e %.12e\n", amp, phase, inc, psi);
            

}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels)
{   

#ifdef __CUDACC__
#else
    if (buffer_length < 2 * N * sizeof(double) + 1 * N * sizeof(int) + 1 * N * sizeof(bool))
    {
        throw std::invalid_argument("Buffer length not long enough.");
    }
#endif

    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];
    // CUDA_SHARED int links[NLINKS];
    
    fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params);
    get_tdi_Xf(tdi_channels_arr, params, t_arr, N, bin_i, link_Space_craft_rec, link_Space_craft_em, k, u, v);
    CUDA_SYNC_THREADS;
    
    // will get reset inside function
    double *flip = (double*)buffer;
    double *pjump = &flip[N];
    int *count = (int *)&pjump[N];
    bool *fix_count = (bool *)&count[N];

    CUDA_SYNC_THREADS;
#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    for (int i = start; i < N; i += incr)
    {
        phi_ref[i] = get_phase_ref(t_arr[i], params, bin_i);
    }
    CUDA_SYNC_THREADS;
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N, &tdi_amp[0], &tdi_phase[0], &tdi_channels_arr[0], &phi_ref[0]);
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N, &tdi_amp[N], &tdi_phase[N], &tdi_channels_arr[N], &phi_ref[0]);
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N, &tdi_amp[2 * N], &tdi_phase[2 * N], &tdi_channels_arr[2 * N], &phi_ref[0]);
    
    // //  FILE *fp1 = fopen("check_phase_before_unwrap.txt", "w");
    // // for (int n = 0; n < N; n += 1)
    // // {
    // //     fprintf(fp1, "%.12e, %.12e, %.12e, %.12e, %.12e, %.12e\n", t_arr[n], Xamp[n], Xphase[n], M[n], Mf[n], phi_ref[n]);
    // //     fflush(fp1);
    // // }
    // // fclose(fp1);
    
    double *ph_correct_buffer = &flip[0];
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[0]);
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[N]);
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[2 * N]);

    // cmplx I(0.0, 1.0);
    // for (int i = 0; i < N; i += 1)
    // {
    //     X[i] = Xamp[i] * gcmplx::exp(-I * Xphase[i]);
    //     Y[i] = Yamp[i] * gcmplx::exp(-I * Yphase[i]);
    //     Z[i] = Zamp[i] * gcmplx::exp(-I * Zphase[i]);
    // }
    // CUDA_SYNC_THREADS;


    // for (int i = 0; i < N; i += 1)
    // {
    //     printf("WEEET2: %d %.12e %.12e %.12e %.12e %.12e\n", i, Xamp[i], Xphase[i], X[i].real(), X[i].imag(), phi_ref[i]);
    // }

    // FILE *fp = fopen("temp_check_amp_phase_22.txt", "w");
    // for (int n = 0; n < N; n += 1)
    // {
    //     fprintf(fp, "%.12e, %.12e, %.12e, %.12e\n", t_arr[n], Xamp[n], Xphase[n], phi_ref[n]);
    //     fflush(fp);
    // }
    // fclose(fp);

    // new_extract_phase(X, phi_ref, N, t_arr);
    // new_extract_phase(Y, phi_ref, N, t_arr);
    // new_extract_phase(Z, phi_ref, N, t_arr);

    // for (int i = 0; i < N; i += 1)
    // {
    //     printf("WEEET3: %d %.12e %.12e %.12e %.12e %.12e\n", i, Xamp[i], Xphase[i], X[i].real(), X[i].imag(), phi_ref[i]);
    // }


    // extract_amplitude_and_phase(flip, pjump, N, Yamp, Yphase, Y, Yf, phi_ref);
    // unwrap_phase(N, Yphase);

    // extract_amplitude_and_phase(flip, pjump, N, Zamp, Zphase, Z, Zf, phi_ref);
    // unwrap_phase(N, Zphase);
}


// CUDA_DEVICE
// void LISATDIonTheFly::get_tdi_Xf(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi, int bin_i)
// {

// #ifdef __CUDACC__
//     int start = threadIdx.x;
//     int incr = blockDim.x;
// #else // __CUDACC__
//     int start = 0;
//     int incr = 1;
// #endif // __CUDACC__
//     for (int i = start; i < N; i += incr)
//     {
//         get_tdi_n(X, Y, Z, phi_ref, params, t_arr[i], i, N, costh, phi, cosi, psi, bin_i);
//     }
//     CUDA_SYNC_THREADS;
// }

// CUDA_DEVICE
// void LISATDIonTheFly::get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i)
// {
//     printf("Not Implemented. TODO: best way to do this?");
// }

// CUDA_DEVICE
// double LISATDIonTheFly::get_phase_ref(double t, double *params, int bin_i)
// {
//     printf("Not Implemented. TODO: best way to do this?");
// }


CUDA_DEVICE
void LISATDIonTheFly::unwrap_phase(int N, double *phase)
{
    double u, v, q;
    int i;
    
    // std::cout << "start phase[0]: " << phase[0] << std::endl;
    v = phase[0];
    for(i=0; i<N ;i++)
    {
        u = phase[i];

        // std::cout << "bef u: " << u << " v: " << v << " phase[i]: " << phase[i] << std::endl;
        q = rint(fabs(u-v)/(2. * M_PI));
        if(q > 0.0)
        {
           if(v > u) u += q*2. * M_PI;
           else      u -= q*2. * M_PI;
        }

        v = u;
        phase[i] = u;

        // std::cout << "aft u: " << u << " v: " << v << " q: " << q << " phase[i]: " << phase[i] << std::endl;
        
    }
    // for(i=0; i<N ;i++)
    // {
    //     printf("%d %.12e\n", i, phase[i]);
    // }
}


template<typename T>
CUDA_DEVICE void cumsum(T *sdata, int N)
{
    // cumsum
#ifdef __CUDACC__
    // Specialize BlockScan for a 1D block of 128 threads of type int
    using BlockScan = cub::BlockScan<T, NUM_THREADS_HERE>;

    // Allocate shared memory for BlockScan
    CUDA_SHARED typename BlockScan::TempStorage temp_storage;

    // Obtain input item for each thread
    int tid = threadIdx.x;
    int total_run = 0;
    int index;
    T thread_data;
    while (total_run < N)
    {
        index = total_run + threadIdx.x;
        if (index < N)
        {
            thread_data = sdata[index];
        }
        else
        {
            thread_data = 0.0;
        }

        CUDA_SYNC_THREADS;
        // Collectively compute the block-wide exclusive prefix sum
        // This is the cummulative sum over the width of the block 
        // (sometimes the array is longer which we adjust for with total_run)
        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
        CUDA_SYNC_THREADS;
    //         // Perform the parallel prefix sum (Blelloch algorithm)
    //     __syncthreads();    
    //     for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) 
    //     {
    //         if ((tid >= stride) && (total_run + tid < N)) 
    //         {
    //             sdata[total_run + tid] += sdata[total_run + tid - stride];
    //         }
    //         __syncthreads(); // Synchronize threads within the block
    //     }
    //     __syncthreads();
        CUDA_SYNC_THREADS;
        if (index < N)
        {
            sdata[index] = thread_data;
        }
        CUDA_SYNC_THREADS;
        // -1 is here because the first element of the next step will be the last element of the previous step
        // this means the first element of the new step is the cummulative sum of the previous step.
        total_run += (NUM_THREADS_HERE - 1);
    }
    // __syncthreads();
    // //     
    // // }
    // // CUDA_SYNC_THREADS;

    /*
    extern __shared__ float temp[];
    // allocated on invocation int thid = threadIdx.x; int offset = 1;

    // build sum in place up the tree
    for (int d = n >> 1; d > 0; d >> = 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0)
    {
        temp[n - 1] = 0;
    } // clear the last element
    __syncthreads();
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >> = 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
__syncthreads();
    */

    // if (threadIdx.x == 0)
    // {
    //     for (int i = 1; i < N; i += 1)
    //     {
    //         sdata[i] += sdata[i - 1];
    //     }
    // }
    // CUDA_SYNC_THREADS;
#else
    for (int i = 1; i < N; i += 1)
    {
        sdata[i] += sdata[i - 1];
    }
#endif
}


CUDA_DEVICE
void LISATDIonTheFly::new_unwrap_phase(double *ph_correct_buffer, int N, double *phase)
{
    double dd, ddmod;
    double period = 2. * M_PI;
    double interval_high =  period / 2.;
    double interval_low = -interval_high;
    double ph_tmp;
    double discont = period / 2.;
#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__

    for (int i = start; i < N; i += incr)
    {
        ph_correct_buffer[i] = 0.0;
    }

    CUDA_SYNC_THREADS;
    double tmp_remainder;
    // std::cout << "start phase[0]: " << phase[0] << std::endl;
    for(int i= start + 1; i<N ; i += incr)
    {
        dd = phase[i] - phase[i - 1]; 
        tmp_remainder = remainder(dd - interval_low, period);
        while (tmp_remainder < 0.0){tmp_remainder += period;}
        ddmod = tmp_remainder + interval_low;

        if ((ddmod == interval_low) && (dd > 0))
        {
            ddmod = interval_high;
        }
        ph_tmp = ddmod - dd;

        if (abs(dd) < discont)
        {
            ph_tmp = 0.0;
        }
        ph_correct_buffer[i] = ph_tmp;
        // printf("PHASE CORR: %d %e %e %e %e %e\n", i, dd, ddmod, ph_correct_buffer[i], remainder(dd - interval_low, period), interval_low);
    }
    CUDA_SYNC_THREADS;

    cumsum(ph_correct_buffer, N);
    CUDA_SYNC_THREADS;

    double tmp;
    for (int i = start + 1; i < N; i += incr)
    {
        tmp = phase[i] + ph_correct_buffer[i];
        // printf("CHANGE: %d %e %e %e \n", i, phase[i], ph_correct_buffer[i], tmp);
        phase[i] = tmp;

    }
    CUDA_SYNC_THREADS;
//     CHANGE 135 -3.613320762606 6.283185307179587 2.669864544573587
// CHANGE 136 2.66749491221 1.7763568394002505e-15 2.667494912210002
// CHANGE 137 2.662627049648 1.7763568394002505e-15 2.662627049648002
// CHANGE 138 -3.627703057173 6.283185307179588 2.655482250006588
}

CUDA_DEVICE
void LISATDIonTheFly::new_extract_amplitude_and_phase(int *count, bool *fix_count, double *flip, double *pjump, int Ns, double *As, double *Dphi, cmplx *M, double *phiR)
{
    bool is_min;
    double dA1, dA2, dA3, test1, test2;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    for (int i = start; i < Ns; i += incr)
    {
        count[i] = 0;
        pjump[i] = 0.0;
        flip[i] = 1.0;
        fix_count[i] = false;
        As[i] = gcmplx::abs(M[i]);
    }
    CUDA_SYNC_THREADS;
    for (int i = (start + 1); i < Ns - 1; i += incr)
    {   
        is_min = (As[i] < As[i - 1]) && (As[i] < As[i + 1]);

        // printf("CHECKIT2 %d %e %d\n", i, As[i], is_min);
        if (is_min)
        {
            dA1 =  As[i + 1] + As[i - 1] - 2.0*As[i];  //regular second derivative
            dA2 = -As[i + 1] + As[i - 1] - 2.0*As[i];  //second derivative if i+1 first negative value
            dA3 = -As[i + 1] + As[i - 1] + 2.0*As[i];  //second derivative if i first negative value
            test1 = (abs(dA2/dA1) < 0.1);
            test2 = (abs(dA3/dA1) < 0.1);
            // TODO: check this. 
            if (test1)
            {
                // NEED TO BE CAREFUL HERE
                count[i + 1] = 1;
            }
            else if (test2)
            {
                count[i] = 1;
            }
        }
    }

    CUDA_SYNC_THREADS;

    // cumsum
    cumsum(count, Ns);
    CUDA_SYNC_THREADS;

    // Cooperative stride (was ``i += 1`` -- a bug that made every thread
    // re-do the whole length [start, Ns-1) and race on the same shared
    // addresses; harmless on CPU where incr == 1 but huge wasted work on
    // GPU and a memory-consistency risk).
    for (int i = start; i < Ns - 1; i += incr)
    {
        flip[i] = pow(-1., count[i]);
        pjump[i] = count[i] * M_PI;
    }
    CUDA_SYNC_THREADS;

    if (THREAD_ZERO)
    {
        flip[Ns-1]  = flip[Ns-2];
        pjump[Ns-1] = pjump[Ns-2];
    }
    CUDA_SYNC_THREADS;

    double v;
    for(int i=start; i<Ns ; i += incr)
    {
        As[i] = flip[i]*As[i];
        // printf("HUH: %e %e\n", flip[i], As[i]);
        v = remainder(phiR[i], 2 * M_PI);
        Dphi[i] = -atan2(M[i].imag(),M[i].real())+pjump[i]-v;
        // if ((i > 11670))
        // printf("INIT new: %d %e %e %e %e %e %e\n", i, -atan2(Mf[i],M[i]), flip[i], pjump[i], As[i], Dphi[i], v);
    
    }
    CUDA_SYNC_THREADS;
}


CUDA_DEVICE
void LISATDIonTheFly::extract_amplitude_and_phase(double *flip, double *pjump, int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR)
{

    int i;
    double v;
    double dA1, dA2, dA3;
    
    // This catches sign flips in the amplitude. Can't catch flips at either end of array
    flip[0]  = 1.0;
    pjump[0] = 0.0;

    i = 1;
    do
    {
        flip[i] = flip[i-1];
        pjump[i] = pjump[i-1];
        
        //local min
        if((As[i] < As[i-1]) && (As[i] < As[i+1]))
        {
            dA1 =  As[i+1] + As[i-1] - 2.0*As[i];  // regular second derivative
            dA2 = -As[i+1] + As[i-1] - 2.0*As[i];  // second derivative if i+1 first negative value
            dA3 = -As[i+1] + As[i-1] + 2.0*As[i];  // second derivative if i first negative value

            if(fabs(dA2/dA1) < 0.1)
            {
                flip[i+1]  = -1.0*flip[i];
                pjump[i+1] = pjump[i]+M_PI;
                i++; // skip an extra place since i+1 already dealt with
            }
            if(fabs(dA3/dA1) < 0.1)
            {
                flip[i]  = -1.0*flip[i-1];
                pjump[i] = pjump[i-1]+M_PI;
            }
        }
        
        i++;
        
    }while(i < Ns-1);
    
    flip[Ns-1]  = flip[Ns-2];
    pjump[Ns-1] = pjump[Ns-2];
    
    
    for(i=0; i<Ns ;i++)
    {
        As[i] = flip[i]*As[i];
        // printf("HUH: %e %e\n", flip[i], As[i]);
        v = remainder(phiR[i], 2 * M_PI);
        Dphi[i] = -atan2(Mf[i],M[i])+pjump[i]-v;
        // if ((i > 11670))
        // printf("INIT: %d %e %e %e %e %e %e\n", i, -atan2(Mf[i],M[i]), flip[i], pjump[i], As[i], Dphi[i], v);
    
    }
    
}



CUDA_DEVICE
double LISATDIonTheFly::get_phase_ref(double t, double *params, int bin_i)
{   
    // TD is based on t_sc rather than t (t_ssb)
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    
    get_sky_vectors(&k, &u, &v, params);
    // reference phase is at spacecraft 1
    Vec x_rec = orbits->get_pos(t, 1);
    double k_dot_x_rec = k.dot(x_rec);
    double t_sc = t - k_dot_x_rec * C_inv;
    double phase_ref = get_phase(t_sc, params, bin_i);
    return phase_ref;
}

// CUDA_DEVICE
// double GBTDIonTheFly::get_phase_ref(double t, double *params, int bin_i)
// {   
//     double f0    = params[f0_index];
// //     if (N_store == NULL)
// //     {
// // #ifdef __CUDACC__
// // #else
// //         throw std::invalid_argument("N_store not set yet.\n");
// // #endif
// //     }
//     double t_diff = t - t_ref;
//     return 2.0 * M_PI * (int(f0 * T) / T) * t_diff;
// }


CUDA_DEVICE
void LISATDIonTheFly::new_extract_phase(cmplx *M, double *phiR, int N, double *t_arr)
{
    cmplx I(0.0, 1.0);

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    // FILE *fp = fopen("temp_check.txt", "w");

    for (int n = start; n < N; n += incr)
    {
        // TODO: do we want to do this. We take conj to match N/T
        M[n] = gcmplx::conj(M[n]);
        // fprintf(fp, "%.12e, %.12e, %.12e, %.12e\n", t_arr[n], M[n].real(), M[n].imag(), phiR[n]);
        M[n] *= gcmplx::exp(-I * phiR[n]);
    }
    CUDA_SYNC_THREADS;

    // fclose(fp);
}

int LISATDIonTheFly::get_tdi_buffer_size(int N)
{
    return 2 * N * sizeof(double) + 1 * N * sizeof(bool) + 1 * N * sizeof(int);
}


// Heterodyned-phi_ref variant. See header for semantics; the only
// difference from ``get_tdi`` is the final per-sample subtraction
// ``phi_ref_het[i] = phi_ref[i] - 2*pi*f0_grid*t_arr[i]`` applied
// after unwrap. ``tdi_amp`` and ``tdi_phase`` are unchanged.
CUDA_DEVICE
void LISATDIonTheFly::get_tdi_heterodyned(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double *phi_ref_het, double *params, double *t_arr, int N, int bin_i, int nchannels, double f0_grid)
{
    get_tdi(buffer, buffer_length,
            tdi_channels_arr, tdi_amp, tdi_phase, phi_ref_het,
            params, t_arr, N, bin_i, nchannels);
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    const double two_pi_f0 = 2.0 * M_PI * f0_grid;
    for (int i = start; i < N; i += incr)
    {
        phi_ref_het[i] -= two_pi_f0 * t_arr[i];
    }
    CUDA_SYNC_THREADS;
}

// ---------------------------------------------------------------
// run_wave_tdi (was TDIonTheFly.cu:7741-7809)
// ---------------------------------------------------------------
void LISATDIonTheFly::run_wave_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    N_store = N;
    // printf("orbits inside: %e", orbits->armlength);
    if (orbits == NULL)
    {
#ifdef __CUDACC__
#else
        throw std::invalid_argument("Need to add orbital information.\n");
#endif
    }

    if (this->tdi_config == NULL)
    {
#ifdef __CUDACC__
#else
        throw std::invalid_argument("Need to add tdi config2.\n");
#endif
    }

#ifdef __CUDACC__
    int start = blockIdx.x;
    int increment = gridDim.x;

    int start2 = threadIdx.x;
    int increment2 = blockDim.x;
#else
    int start = 0;
    int increment = 1;

    int start2 = 0;
    int increment2 = 1;
#endif

// TODO: make this better?
#ifdef __CUDACC__
#else
    if (n_params > N_PARAMS_MAX)
    {
        throw std::invalid_argument("n_params is too long, need to recompile and increase N_PARAMS_MAX.");
    }
#endif
    CUDA_SHARED double params_here[N_PARAMS_MAX];  // TODO: maybe shared? only if registers are filled up

     // TODO: CHECK THIS!!
    for (int bin_i = start; bin_i < num_bin; bin_i += increment)
    {
        CUDA_SYNC_THREADS;
        // read params into faster memory for gpu / cpu does not matter really
        for (int i = start2; i < n_params; i += increment2)
        {
            params_here[i] = params[bin_i * n_params + i];
        }
        CUDA_SYNC_THREADS;

        double *t_here = &t_arr[bin_i * N];
        
        get_tdi(
            buffer, buffer_length,
            &tdi_channels_arr[bin_i * nchannels * N], 
            &tdi_amp[bin_i * nchannels * N], &tdi_phase[bin_i * nchannels * N],
            &phi_ref[bin_i * N],
            params_here, t_here, N, bin_i, nchannels);
        CUDA_SYNC_THREADS;   
    }
    return;
}
