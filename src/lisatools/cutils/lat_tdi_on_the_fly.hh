#ifndef __LAT_TDI_ON_THE_FLY_HH__
#define __LAT_TDI_ON_THE_FLY_HH__

// LISATDIonTheFly -- base class for all on-the-fly TDI evaluators.
// Builds smooth (Aplus, Across) -> X/Y/Z (or A/E/T) per-sample channels
// for arbitrary source models, with optional per-block orbit-spline
// caches to amortize the orbit evaluations across binaries inside a
// shared-memory chunked kernel.
//
// Source-class specializations (GBTDIonTheFly, SOBBHTDIonTheFly,
// FDSplineTDIWaveform, TDSplineTDIWaveform) override the virtual
// get_amp / get_phase / get_f / get_fdot accessors to plug in the
// source-specific intrinsic waveform.
//
// Phase 3L.5 (2026-06-03): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.hh:47-204
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu (~26 method
//   bodies, ~1300 lines total spanning lines 31-35, 6414-7629, 7741-7809)
// to LISAanalysistools. lat_tdi_on_the_fly.cu compiles the method bodies;
// downstream consumers (lisa-on-gpu's tdionthefly module) recompile the
// .cu in-place via the same pattern already in use for LISAResponse.cu
// since Phase 3E.
//
// Inheritance hierarchy after this move:
//   LISATDIonTheFly        (LAT)
//     |-- GBTDIonTheFly    (lisa-on-gpu; planned move to GBGPU at 3L.7)
//     |-- SOBBHTDIonTheFly (lisa-on-gpu; planned move to BBHx  at 3L.8)
//     |-- FDSplineTDIWaveform (lisa-on-gpu; planned move to LAT at 3L.6)
//     `-- TDSplineTDIWaveform (lisa-on-gpu; planned move to LAT at 3L.6)
//
// CPU/GPU class-name alias (sprint-wide rule): LISATDIonTheFly +
// OrbitsSplineCache are aliased to backend-specific symbols below so the
// CPU and GPU plugin wheels emit distinct mangled names and remain
// side-loadable in the same process. Derived classes that stay in
// lisa-on-gpu (GBTDIonTheFly etc.) anchor their vtables to the
// per-backend base symbol via this alias.

#include "gbt_global.h"
#include "Detector.hpp"        // Orbits, Vec
#include "LISAResponse.hh"     // TDIConfig
#include "Interpolate.hh"      // CubicSpline + NLINKS
#include "fd_domain.hh"
#include "wdm_settings.hh"
#include "wdm_domain.hh"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define LISATDIonTheFly  LISATDIonTheFlyGPU
#define OrbitsSplineCache OrbitsSplineCacheGPU
#else
#define LISATDIonTheFly  LISATDIonTheFlyCPU
#define OrbitsSplineCache OrbitsSplineCacheCPU
#endif

// In-kernel orbit spline cache. Holds cubic-spline coefficients for the
// 6 link LTTs and 9 spacecraft-xyz positions, sampled at N_cp uniform
// times within a chunk. Populated once per chunk per block; reused
// across all binaries.
//
// Cached evaluation replaces ``orbits->get_light_travel_time`` /
// ``orbits->get_pos`` global-mem lookups (~32-64 per TDI sample per
// binary) with cheap shared-mem cubic evals. The producer side
// (populate_orbit_spline_cache) stays in lisa-on-gpu for now; it will
// move alongside the GB chunked-het kernels at Phase 3L.7.
struct OrbitsSplineCache
{
    double  t_cp0;           // chunk_t_start (absolute s)
    double  dt_cp;           // uniform cp spacing (s)
    int     N_cp;
    double *t_cp;            // [N_cp]
    double *ltt_y;           // [6 * N_cp]  per-link LTT y0
    double *ltt_c1;          // [6 * N_cp]
    double *ltt_c2;          // [6 * N_cp]
    double *ltt_c3;          // [6 * N_cp]
    double *pos_y;           // [9 * N_cp]  (sc, xyz) row-major: pos_y[(sc*3+xyz)*N_cp + i]
    double *pos_c1;
    double *pos_c2;
    double *pos_c3;
};

class LISATDIonTheFly{
    public:
        Orbits *orbits;
        TDIConfig *tdi_config;
        int inc_index;
        int psi_index;
        int lam_index;
        int beta_index;
        int N_store;

        CUDA_DEVICE
        void run_wave_tdi(
            void *buffer, int buffer_length, cmplx *tdi_channels_arr,
            double *tdi_amp, double *tdi_phase, double *phi_ref,
            double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
        );
        CUDA_CALLABLE_MEMBER
        LISATDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, int inc_index_, int psi_index_, int lam_index_, int beta_index_)
        {
            orbits = orbits_;
            tdi_config = tdi_config_;
            inc_index = inc_index_;
            psi_index = psi_index_;
            lam_index = lam_index_;
            beta_index = beta_index_;
        };
        CUDA_CALLABLE_MEMBER
        ~LISATDIonTheFly();
        CUDA_DEVICE
        void print_orbits_tdi();
        CUDA_DEVICE
        void fill_link_arrays(int *link_space_craft_rec, int *link_space_craft_em);
        CUDA_DEVICE
        void get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);

        // Heterodyned-phi_ref variant of get_tdi. Returns the same
        // (tdi_amp, tdi_phase) outputs but with phi_ref replaced by
        // ``phi_ref_het[i] = phi_ref[i] - 2*pi*f0_grid*t_arr[i]``, where
        // ``f0_grid`` is the snapped chunk-rfft carrier. The residual is
        // slow over the chunk (O(1 rad)) so it can be unwrapped/splined
        // robustly even at coarse N -- this is the prerequisite for the
        // within-chunk source-signal spline cache.
        //
        // Only meaningful for sources with a well-defined constant
        // carrier (GB / SOBBH). Sources without a constant ``f0``
        // (MBHB chirp, EMRI) should call ``get_tdi`` instead -- carrier
        // subtraction is undefined there.
        CUDA_DEVICE
        void get_tdi_heterodyned(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref_het, double *params, double *t_arr, int N, int bin_i, int nchannels, double f0_grid);

        // ---- Orbit-cache variants ------------------------------------
        // Same outputs as their non-cached siblings, but the orbit
        // lookups (``orbits->get_pos`` and ``orbits->get_light_travel_time``)
        // are redirected to ``cache`` -- a cubic-spline cache built once
        // per chunk per block via ``populate_orbit_spline_cache``. Use
        // these when the chunk + the source's get_tdi inner-loop delays
        // are all within the cache window (true by construction in the
        // chunked-het kernels).
        //
        // Window checks against the orbits' raw global tables are
        // skipped -- by construction the cache is built from t-values
        // inside the source's valid orbit window, so any t inside the
        // chunk is in-bounds.
        CUDA_DEVICE
        void get_tdi_Xf_single_cached(cmplx *tdi_channel, double t, double *params, Vec k, Vec u, Vec v, int *link_Space_craft_rec, int *link_Space_craft_em, int bin_i, OrbitsSplineCache *cache);

        CUDA_DEVICE
        void get_tdi_Xf_cached(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i, int *link_Space_craft_rec, int *link_Space_craft_em, Vec k, Vec u, Vec v, OrbitsSplineCache *cache);

        CUDA_DEVICE
        void get_tdi_cached(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double *phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels, OrbitsSplineCache *cache);

        CUDA_DEVICE
        void get_tdi_heterodyned_cached(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double *phi_ref_het, double *params, double *t_arr, int N, int bin_i, int nchannels, double f0_grid, OrbitsSplineCache *cache);

        // Raw variants of get_tdi[_cached]: fill tdi_channels_arr
        // (nchannels * N raw complex samples) and phi_ref (N, UN-heterodyned
        // -- get_phase_ref(t_i) straight from the source, NO carrier
        // subtraction), but skip the per-channel amplitude/phase extract
        // + unwrap.
        CUDA_DEVICE
        void get_tdi_raw(cmplx *tdi_channels_arr, double *phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);
        CUDA_DEVICE
        void get_tdi_raw_cached(cmplx *tdi_channels_arr, double *phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels, OrbitsSplineCache *cache);
        CUDA_DEVICE
        void get_tdi_Xf(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i, int *link_space_craft_rec, int *link_space_craft_em, Vec k, Vec u, Vec v);
        CUDA_DEVICE
        void get_tdi_Xf_single(cmplx *tdi_channel, double t, double *params, Vec k, Vec u, Vec v, int *link_space_craft_rec, int *link_space_craft_em, int bin_i);
        CUDA_DEVICE
        void extract_amplitude_and_phase(double *flip, double *pjump, int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR);
        CUDA_DEVICE
        void new_extract_amplitude_and_phase(int *count, bool *fix_count, double *flip, double *pjump, int Ns, double *As, double *Dphi, cmplx *M, double *phiR);
        CUDA_CALLABLE_MEMBER
        int get_tdi_buffer_size(int N);
        CUDA_DEVICE
        void unwrap_phase(int N, double *phase);
        CUDA_DEVICE
        void new_unwrap_phase(double *ph_correct_buffer, int N, double *phase);
        CUDA_DEVICE
        void new_extract_phase(cmplx *M, double *phiR, int N, double *t_arr);
        CUDA_DEVICE
        double get_phase_ref(double t, double *params, int bin_i);
        CUDA_DEVICE
        void get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i);
        CUDA_DEVICE
        void get_sky_vectors(Vec *k, Vec *u, Vec *v, double *params);
        CUDA_DEVICE
        void xi_projections(double *xi_p, double *xi_c, Vec u, Vec v, Vec n);
        CUDA_DEVICE
        virtual double get_amp(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_phase(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_f(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_fdot(double t, double *params, int bin_i);
};

#endif // __LAT_TDI_ON_THE_FLY_HH__
