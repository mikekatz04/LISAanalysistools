#ifndef LAT_CHUNKED_HET_KERNELS_HH
#define LAT_CHUNKED_HET_KERNELS_HH

// ============================================================================
// Chunked-heterodyne kernel primitives: ABI constants, shared-memory PODs,
// FFT machinery dependency, and source-agnostic helpers used by the
// `wdm_het_*` templated kernel family.
//
// **Status (Phase 3L.7a slices 1 + 2a + 2b, 2026-06-04):**
// - Slice 1: FAST_WDM_* macros + WDMHet*Bufs PODs + WDM_HET_PATH_*
//   arena macros (top of file).
// - Slice 2a: FFT machinery moved to `lat_wdm_fft.hh` (included below).
// - Slice 2b: source-agnostic chunked-het helpers
//   (`wdm_fit_cubic_spline`, `populate_orbit_spline_cache`,
//    `fast_wdm_inner_heterodyne_spline`, `fast_wdm_inner_heterodyne`,
//    `fast_wdm_inner_heterodyne_direct`, `gb_chunk_fd_to_wdm`)
//   appended below the macros.
// - Slice 3-prep: `NUM_THREADS_HERE` block-size knob +
//   `FAST_WDM_K_PER_THREAD_MAX` register-array sizer (below the
//   include block).
// - Slice 3 (pending): the 4 templated chunked-het kernel bodies
//   (`wdm_het_{fill_global,get_ll,swap_ll,get_fstat_ll}_kernel`) and
//   their `*_impl<SourceT>` host launchers.
//
// Once Slice 3 lands, GBGPU and BBHx can instantiate
// `wdm_het_*_impl<GBTDIonTheFly>` / `wdm_het_*_impl<SOBBHTDIonTheFly>`
// against this header in their own pybind11 binding TUs without
// depending on lisa-on-gpu (which retires once Phase 3L.7 + 3L.8
// land).
//
// **Sprint rule -- no aliasing needed.** `WDMHetDirectBufs` and
// `WDMHetSplineBufs` are file-static shared-memory layouts that never
// escape a translation unit (no pybind11 binding, no exported
// symbol). See LAT CLAUDE.md "CPU/GPU class-name aliasing" rule,
// point 5.
//
// **Dependencies pulled in below**:
//   - `global.hpp` -> `gbt_global.h` -> `cmplx` typedef + CUDA macros
//   - `lat_wdm_fft.hh` -> `wdm_spline_radix2_fft`, `wdm_fft_dispatch`
//   - `lat_tdi_on_the_fly.hh` -> `LISATDIonTheFly`, `OrbitsSplineCache`,
//     `Orbits`, `Vec`, `NLINKS`, `fit_cubic_spline_pcr`,
//     `fit_cubic_spline_thomas`, `CUBIC_SPLINE_LINEAR_SPACING`
//
// `NUM_THREADS_HERE` + `FAST_WDM_K_PER_THREAD_MAX` are owned here so
// downstream waveform packages instantiating the chunked-het kernels
// (GBGPU + BBHx after Phase 3L.7 / 3L.8) get a consistent block-size
// knob without redefining it.
// ============================================================================

#include "global.hpp"           // -> gbt_global.h -> cmplx + CUDA macros
#include "lat_wdm_fft.hh"       // wdm_spline_radix2_fft, wdm_fft_dispatch
#include "lat_tdi_on_the_fly.hh" // LISATDIonTheFly, OrbitsSplineCache, Orbits


// ----------------------------------------------------------------------------
// NUM_THREADS_HERE = blockDim.x for the chunked-heterodyne kernel family.
// Should be a power of 2 and a multiple of 32 (warp size). At
// `Nt_sub=256` this also sets the per-thread iteration count for stride
// loops:
//   K_PER_THREAD = ceil(Nt_sub / NUM_THREADS_HERE)
//   (64 -> 4 iters/thread, 128 -> 2 iters/thread, 256 -> 1 iter/thread)
// Larger `NUM_THREADS_HERE` gives the FFT/iFFT more cooperative
// parallelism per transform at the cost of per-SM block count (each
// block holds the same shared mem regardless of thread count). 128 is
// the empirical sweet spot on A100 for our shared-mem footprint.
//
// On CPU this collapses to 1 (the `THREAD_START_X` / `BLOCK_INCR_X`
// stubs degenerate to a single virtual thread iterating fully).
//
// Defined here -- not in the host source file -- so downstream waveform
// packages instantiating the chunked-het kernels (GBGPU + BBHx after
// Phase 3L.7 / 3L.8) get a consistent block-size knob without having
// to redefine it.
// ----------------------------------------------------------------------------
#ifdef __CUDACC__
#define NUM_THREADS_HERE 128
#else
#define NUM_THREADS_HERE 1
#endif


// ----------------------------------------------------------------------------
// Upper bound on the number of thread-strided iterations any per-thread
// register array sees when sweeping `[0, Nt_sub)` at
// `blockDim.x = NUM_THREADS_HERE`. Compile-time so it can size
// `constexpr` arrays.
//   GPU: ceil(256 / 64) = 4   -> arrays stay in registers
//   CPU: ceil(4096 / 1) = 4096 (CPU has one virtual thread iterating fully)
//
// The previous formula `FAST_WDM_NT_SUB_MAX / FAST_WDM_NCHANNELS_MAX`
// produced 85 on GPU which spilled the register arrays to local memory
// (the divisor should be the thread stride, not channel count).
// ----------------------------------------------------------------------------
#define FAST_WDM_K_PER_THREAD_MAX \
    ((FAST_WDM_NT_SUB_MAX + NUM_THREADS_HERE - 1) / NUM_THREADS_HERE)


// ----------------------------------------------------------------------------
// Tukey window selectors. Per Test G in
// `check_shortened_wdm.py`, `N_sparse=1024, tukey_alpha=0.0` reproduces
// mm5/mm2 ~ 5e-13 (matches the dense TD->FD->WDM floor); `N_sparse=64,
// tukey_alpha=0.05` reaches mm5/mm2 ~ 1e-7 -- the small Tukey collapses
// the heterodyne-band requirement.
// ----------------------------------------------------------------------------
#define FAST_WDM_TUKEY_ALPHA_TD          0.02   // TD-based chunked stitch
#define FAST_WDM_TUKEY_ALPHA_HET_WIDE    0.01   // FD heterodyne, N_sparse >= 512
#define FAST_WDM_TUKEY_ALPHA_HET_NARROW  0.05   // FD heterodyne, N_sparse  < 512
#define FAST_WDM_TUKEY_ALPHA_AUTO       -1.0    // sentinel: auto-pick


// ----------------------------------------------------------------------------
// Sparse-grid + channel + Nt_sub caps -- shared-memory budget for the
// chunked-heterodyne kernels (see lisa-on-gpu's TDIonTheFly.cu for the
// full per-block budget calc; ~40 KB at the validated baseline).
//
// GPU caps are tuned to fit A100's ~99 KB per-block shared-mem cap;
// CPU caps are raised since `CUDA_SHARED` collapses to stack/heap.
// ----------------------------------------------------------------------------
#ifdef __CUDACC__
#define FAST_WDM_N_SPARSE_MAX  512
#else
#define FAST_WDM_N_SPARSE_MAX  4096
#endif

#define FAST_WDM_NCHANNELS_MAX 3

#ifdef __CUDACC__
#define FAST_WDM_NT_SUB_MAX  1024
#else
#define FAST_WDM_NT_SUB_MAX  4096
#endif


// ----------------------------------------------------------------------------
// Source-signal spline cache. Selected at RUNTIME per kernel call via
// the `N_cp_sig` parameter:
//   N_cp_sig <= 0  -> direct path: source->get_tdi at all N_sparse points.
//   N_cp_sig >  0  -> spline cache: source->get_tdi_heterodyned at N_cp_sig
//                     points, cubic-spline-interpolate to N_sparse.
//
// Per the density study at the half-day-wavelet baseline,
// N_cp_sig=48 -> mm < 4e-11 (GB) / 4e-9 (SOBBH) vs lisatools.
// See CHUNKED_HET_DESIGN_NOTES.md.
// ----------------------------------------------------------------------------
#ifdef __CUDACC__
#define FAST_WDM_N_CP_SIG_MAX 48
#else
#define FAST_WDM_N_CP_SIG_MAX 2048
#endif


// ----------------------------------------------------------------------------
// Default cap on gridDim.x for the chunked-het kernels (binaries axis).
// Each (x, z) block keeps its own per-(chunk, binary) heap scratch slot
// (`chunk_fd` / `chunk_wdm` / `tdi_channels` / `get_tdi_scratch`), so
// total heap scratch scales as `gd_x * n_chunks * per_slot_size`.
// With Nf=4096 / Nt_sub=256 / nch=3 / N_sparse=256 a single slot is
// ~24 MB for `w_chunk`, so the default of 4 keeps total at ~1.5 GB for
// n_chunks=16.
// ----------------------------------------------------------------------------
#define FAST_WDM_HET_GRID_DIM_X_DEFAULT 4


// ----------------------------------------------------------------------------
// Orbit spline-cache density. The orbits don't depend on the source, so
// this cache is shared across all binaries within a chunk.
// ----------------------------------------------------------------------------
#define FAST_WDM_N_CP_ORBIT_MAX 48


// ============================================================================
// Shared-memory layout for the chunked-het kernels.
//
// The direct-path and spline-path buffer sets are MUTUALLY EXCLUSIVE per
// (chunk, binary) invocation -- `use_spline_cache` picks exactly one --
// so they share the same physical shared memory via the `WDM_HET_PATH_*`
// arena overlay. The amp/phase coefficient stacks inside the spline
// struct are single-channel (the spline path fits + evaluates one
// channel at a time inside `fast_wdm_inner_heterodyne_spline`).
//
// `reinterpret_cast` over a raw `__shared__ char` arena -- rather than a
// C++ union -- because the `cmplx` field in `WDMHetSplineBufs` has a
// user-defined constructor, which historically makes NVCC mis-handle
// `__shared__ union` of those types. `__shared__` memory is
// uninitialised at runtime (no constructors run), so the
// `reinterpret_cast` view is well-defined: every kernel branch is the
// FIRST writer to its own subset of bytes.
// ============================================================================
struct WDMHetDirectBufs {
    double t_sparse_buf  [FAST_WDM_N_SPARSE_MAX];
    double tdi_amp_buf   [FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    double tdi_phase_buf [FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    double phi_ref_buf   [FAST_WDM_N_SPARSE_MAX];
};

struct WDMHetSplineBufs {
    double t_cp_buf            [FAST_WDM_N_CP_SIG_MAX];
    double amp_y_buf           [FAST_WDM_N_CP_SIG_MAX];
    double amp_c1_buf          [FAST_WDM_N_CP_SIG_MAX];
    double amp_c2_buf          [FAST_WDM_N_CP_SIG_MAX];
    double amp_c3_buf          [FAST_WDM_N_CP_SIG_MAX];
    double phase_y_buf         [FAST_WDM_N_CP_SIG_MAX];
    double phase_c1_buf        [FAST_WDM_N_CP_SIG_MAX];
    double phase_c2_buf        [FAST_WDM_N_CP_SIG_MAX];
    double phase_c3_buf        [FAST_WDM_N_CP_SIG_MAX];
    double dphi_ref_y_buf      [FAST_WDM_N_CP_SIG_MAX];
    double dphi_ref_c1_buf     [FAST_WDM_N_CP_SIG_MAX];
    double dphi_ref_c2_buf     [FAST_WDM_N_CP_SIG_MAX];
    double dphi_ref_c3_buf     [FAST_WDM_N_CP_SIG_MAX];
    double B_buf               [FAST_WDM_N_CP_SIG_MAX];
    double pcr_scratch         [8 * FAST_WDM_N_CP_SIG_MAX];
    // Un-het `phi_ref` scratch -- filled by `get_tdi_raw[_cached]` and
    // read by per-channel `new_extract_amplitude_and_phase` (which needs
    // the un-heterodyned `phi_ref` to keep its `remainder(., 2*pi)`
    // unwrap decisions consistent with the old `get_tdi` convention).
    // `dphi_ref_y_buf` holds the carrier-subtracted version that feeds
    // the `dphi_ref` spline.
    double phi_ref_un_het_buf  [FAST_WDM_N_CP_SIG_MAX];
    cmplx  tdi_channels_cp_buf [FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_CP_SIG_MAX];
    char   extract_scratch_buf [21 * FAST_WDM_N_CP_SIG_MAX + 16];
};

// Compile-time size + alignment for the arena overlay.
//
// Why 16: `cmplx` is two doubles (16 B) and the strictest alignment of
// any member across both structs is doubles' 8 -- 16 keeps us safely
// aligned for `cmplx` loads/stores and gives nice 128-bit boundaries
// for the FFT inner loops. CUDA allocates `__shared__` to its declared
// alignment.
#define WDM_HET_PATH_BYTES \
    ((sizeof(WDMHetDirectBufs) > sizeof(WDMHetSplineBufs)) \
     ? sizeof(WDMHetDirectBufs) : sizeof(WDMHetSplineBufs))
#define WDM_HET_PATH_ALIGN 16


// ============================================================================
// Slice 2b helpers (2026-06-04) -- source-agnostic chunked-het primitives.
// ============================================================================


// ----------------------------------------------------------------------------
// CPU/GPU dispatching wrapper for the cubic-spline fitter from
// GPUBackendTools/Interpolate.hh. GPU uses the cooperative PCR solver
// (requires `pcr_scratch` of size `8*N` doubles); CPU uses the
// sequential Thomas algorithm (`pcr_scratch` unused).
// ----------------------------------------------------------------------------
CUDA_DEVICE
inline void wdm_fit_cubic_spline(double *x, double *y,
                                  double *c1, double *c2, double *c3,
                                  double *B, double *pcr_scratch,
                                  int N, int spline_type)
{
#ifdef __CUDACC__
    fit_cubic_spline_pcr(x, y, c1, c2, c3, B, pcr_scratch, N, spline_type);
#else
    (void) pcr_scratch;
    fit_cubic_spline_thomas(x, y, c1, c2, c3, B, N, spline_type);
#endif
}


// ===========================================================================
// OrbitsSplineCache  --  in-kernel cubic-spline cache for LISA orbit data
// ---------------------------------------------------------------------------
// Built once at chunk entry: sample `orbits->get_pos` (3 spacecraft x 3 xyz)
// and `orbits->get_light_travel_time` (6 links) at a sparse uniform t-grid
// inside the chunk, then PCR-fit cubic splines through each scalar series.
// Stored in shared mem; reused across all binaries in the chunk (the orbits
// don't depend on the source). Replaces what would otherwise be ~num_bin x
// N_sparse x (32-64) global-mem orbit lookups per chunk with one cooperative
// fit + cheap shared-mem cubic evals.
//
// Storage (caller allocates):
//   t_cp[N_cp]                              -- uniform time grid, shared
//   ltt_y[6 * N_cp]  + 3 coefs (4 arrays)   -- per-link LTT splines
//   pos_y[9 * N_cp]  + 3 coefs (4 arrays)   -- per-(sc, xyz) position splines
//   B_buf[N_cp]                             -- tridiagonal RHS scratch
//   pcr_scratch[8 * N_cp]                   -- PCR ping-pong scratch (GPU only)
// Total persistent: (1 + 4*6 + 4*9) * N_cp = 61 * N_cp doubles.
// For N_cp=32 (the density-study baseline at 30-day chunks): ~15.6 KB.
//
// The normal-link vector `n_link` does NOT need its own cache -- it can be
// derived in-loop from the cached positions: n = (x_em - x_rec) / L.
//
// The link indexing (0..5) follows `Orbits::get_link_ind`:
//   0 = 12, 1 = 23, 2 = 31, 3 = 13, 4 = 32, 5 = 21.
//
// `OrbitsSplineCache` struct is declared in `lat_tdi_on_the_fly.hh` so
// that `LISATDIonTheFly`'s cached member functions can take it as a
// parameter.
// ===========================================================================

// Populate the cache. Called ONCE per chunk per block (before the binary
// loop). All threads in the block cooperate -- spline fits use the PCR
// solver on GPU, Thomas on CPU. Caller pre-allocates all buffers + scratch.
CUDA_DEVICE
inline void populate_orbit_spline_cache(
    OrbitsSplineCache *cache,
    Orbits *orbits,
    double chunk_t_start, double T_chunk,
    int N_cp,
    double *t_cp_buf,
    double *ltt_y_buf, double *ltt_c1_buf, double *ltt_c2_buf, double *ltt_c3_buf,
    double *pos_y_buf, double *pos_c1_buf, double *pos_c2_buf, double *pos_c3_buf,
    double *B_buf, double *pcr_scratch)
{
    // Link order matches Orbits::get_link_ind: 12, 23, 31, 13, 32, 21
    static const int LINKS[6] = {12, 23, 31, 13, 32, 21};
    const double dt_cp = T_chunk / (double) (N_cp - 1);

    cache->t_cp0   = chunk_t_start;
    cache->dt_cp   = dt_cp;
    cache->N_cp    = N_cp;
    cache->t_cp    = t_cp_buf;
    cache->ltt_y   = ltt_y_buf;
    cache->ltt_c1  = ltt_c1_buf;
    cache->ltt_c2  = ltt_c2_buf;
    cache->ltt_c3  = ltt_c3_buf;
    cache->pos_y   = pos_y_buf;
    cache->pos_c1  = pos_c1_buf;
    cache->pos_c2  = pos_c2_buf;
    cache->pos_c3  = pos_c3_buf;

    // 1) Build t_cp grid and sample raw orbits at the cp times.
    for (int i = THREAD_START_X; i < N_cp; i += BLOCK_INCR_X) {
        t_cp_buf[i] = chunk_t_start + (double) i * dt_cp;
    }
    CUDA_SYNC_THREADS;
    // Sample LTT per link, per cp time.
    for (int idx = THREAD_START_X; idx < 6 * N_cp; idx += BLOCK_INCR_X) {
        const int link_i = idx / N_cp;
        const int i      = idx - link_i * N_cp;
        ltt_y_buf[link_i * N_cp + i] =
            orbits->get_light_travel_time(t_cp_buf[i], LINKS[link_i]);
    }
    // Sample positions per (sc, xyz), per cp time.
    for (int idx = THREAD_START_X; idx < 9 * N_cp; idx += BLOCK_INCR_X) {
        const int sx = idx / N_cp;        // 0..8 (= sc * 3 + xyz)
        const int i  = idx - sx * N_cp;
        const int sc  = sx / 3 + 1;       // spacecraft index 1..3
        const int xyz = sx - (sx / 3) * 3;
        Vec p = orbits->get_pos(t_cp_buf[i], sc);
        double v;
        if      (xyz == 0) v = p.x;
        else if (xyz == 1) v = p.y;
        else               v = p.z;
        pos_y_buf[sx * N_cp + i] = v;
    }
    CUDA_SYNC_THREADS;

    // 2) Fit cubic splines (uniform grid -> LINEAR_SPACING).
    for (int link_i = 0; link_i < 6; ++link_i) {
        wdm_fit_cubic_spline(t_cp_buf,
                              &ltt_y_buf  [link_i * N_cp],
                              &ltt_c1_buf [link_i * N_cp],
                              &ltt_c2_buf [link_i * N_cp],
                              &ltt_c3_buf [link_i * N_cp],
                              B_buf, pcr_scratch,
                              N_cp, CUBIC_SPLINE_LINEAR_SPACING);
        CUDA_SYNC_THREADS;
    }
    for (int sx = 0; sx < 9; ++sx) {
        wdm_fit_cubic_spline(t_cp_buf,
                              &pos_y_buf  [sx * N_cp],
                              &pos_c1_buf [sx * N_cp],
                              &pos_c2_buf [sx * N_cp],
                              &pos_c3_buf [sx * N_cp],
                              B_buf, pcr_scratch,
                              N_cp, CUBIC_SPLINE_LINEAR_SPACING);
        CUDA_SYNC_THREADS;
    }
}


// ============================================================================
// fast_wdm_inner_heterodyne_spline  --  source-signal spline-cache variant
// ----------------------------------------------------------------------------
//
// Replaces the `N_sparse` get_tdi calls per (chunk, binary) with only
// `N_cp_sig` `get_tdi_heterodyned` calls + an in-kernel cubic-spline
// fit + dense evaluation. The heterodyned `phi_ref` makes the unwrap
// robust at sparse `N_cp_sig` sampling (carrier removed in-kernel).
//
// Algorithm (per-channel pipeline -- amp/phase coefficient buffers are
// single-channel and reused across the channel loop, dropping ~6 KB of
// static shared per kernel vs. the old all-channels-at-once layout):
//   1. Build uniform `t_cp[N_cp_sig]` grid over the chunk.
//   2. Call `source->get_tdi_heterodyned_raw[_cached](... f0_grid)` at
//      the cp times. Fills `tdi_channels_cp_buf[nchannels * N_cp_sig]`
//      (raw complex TDI) and `dphi_ref_y_buf[N_cp_sig]` (single-channel
//      heterodyned `phi_ref`). No per-channel extract/unwrap yet.
//   3. Fit the `dphi_ref` cubic spline once (it is per-source, not per
//      channel).
//   4. For each channel c:
//      (a) `new_extract_amplitude_and_phase` into single-channel
//          `amp_y_buf[N_cp_sig]`, `phase_y_buf[N_cp_sig]`.
//      (b) `new_unwrap_phase` on `phase_y_buf`.
//      (c) Fit amp + phase splines (reusing `B_buf` / `pcr_scratch`).
//      (d) Evaluate amp(t), phase(t), dphi_ref(t) at the `N_sparse`
//          t-grid and write
//          `slow_buf[c * N_sparse + i] = amp * exp(i * phase)`, with
//          Tukey taper. Barrier before reusing the single-channel
//          coefficient buffers for the next channel.
//      Slow phase folds in the `chunk_t_start` carrier offset because
//      we splined the heterodyned-against-t_abs `phi_ref`:
//         phase_total = tdi_phase + phi_ref - 2*pi*f0_grid*tau
//                     = tdi_phase + (dphi_ref + 2*pi*f0_grid*t)
//                                  - 2*pi*f0_grid*tau
//                     = tdi_phase + dphi_ref + 2*pi*f0_grid*chunk_t_start
//      since t = chunk_t_start + tau.
//   5. FFT + place into `chunk_fd_out` (identical to direct path).
//
// Per-(chunk, binary) get_tdi cost: ~`N_cp_sig/N_sparse` = 48/256 = 5x cheaper.
// Per the density study: GB mm ~ 4e-11, SOBBH mm ~ 4e-9 at the half-day
// wavelet baseline. Both clear the science threshold.
//
// All workspaces (`t_cp`, single-channel amp/phase y0+c1+c2+c3, single
// `dphi_ref` y0+c1+c2+c3, PCR/B scratch, raw `tdi_channels_cp` scratch,
// extract+unwrap scratch) are caller-allocated.
// ============================================================================
CUDA_DEVICE
inline void fast_wdm_inner_heterodyne_spline(
    cmplx *chunk_fd_out,            // (nchannels * n_rfft_chunk); caller zero-inits
    LISATDIonTheFly *source,
    double *params,
    int bin_i, int carrier_index,
    double chunk_t_start, double T_chunk,
    int N_sparse, int log2_N_sparse, int N_cp_sig,
    int n_rfft_chunk, int nchannels, double tukey_alpha,
    // Spline workspaces (amp/phase buffers are SINGLE-CHANNEL after the
    // per-channel-pipeline refactor -- reused across the c-loop):
    double *t_cp_buf,               // (N_cp_sig,)
    double *amp_y_buf,              // (N_cp_sig,)          single channel, reused
    double *amp_c1_buf,             // (N_cp_sig,)
    double *amp_c2_buf,             // (N_cp_sig,)
    double *amp_c3_buf,             // (N_cp_sig,)
    double *phase_y_buf,            // (N_cp_sig,)          single channel, reused
    double *phase_c1_buf, double *phase_c2_buf, double *phase_c3_buf,
    double *dphi_ref_y_buf,         // (N_cp_sig,)  carrier-subtracted; spline target
    double *dphi_ref_c1_buf, double *dphi_ref_c2_buf, double *dphi_ref_c3_buf,
    double *B_buf,                  // (N_cp_sig,) tridiagonal RHS scratch
    double *pcr_scratch,            // (8 * N_cp_sig,) GPU-only scratch
    double *phi_ref_un_het_buf,     // (N_cp_sig,) un-het phi_ref for extract
    cmplx  *tdi_channels_cp_buf,    // (nchannels * N_cp_sig) -- raw TDI scratch
    cmplx  *slow_buf,               // (nchannels * N_sparse) -- FFT in/out
    void   *extract_scratch,        // >= 21*N_cp_sig bytes
                                    //   layout: flip[N_cp] | pjump[N_cp]
                                    //         | count[N_cp] | fix_count[N_cp]
    int     extract_scratch_len,
    OrbitsSplineCache *orbit_cache)  // nullptr -> direct orbit lookups
{
    const double dt_sparse  = T_chunk / (double) N_sparse;
    const double dt_cp      = T_chunk / (double) (N_cp_sig - 1);
    const double f0         = params[carrier_index];
    const double df_chunk   = 1.0 / T_chunk;
    const int    k_f0       = (int) round(f0 / df_chunk);
    const double f0_grid    = (double) k_f0 * df_chunk;
    const int    half_Nsp   = N_sparse / 2;
    const double scale_X    = 0.5 * dt_sparse;
    const double phi0_chunk = 2.0 * M_PI * f0_grid * chunk_t_start;
    const double two_pi_f0  = 2.0 * M_PI * f0_grid;

    double alpha_eff = tukey_alpha;
    if (alpha_eff == FAST_WDM_TUKEY_ALPHA_AUTO) {
        alpha_eff = (N_sparse >= 512)
            ? FAST_WDM_TUKEY_ALPHA_HET_WIDE
            : FAST_WDM_TUKEY_ALPHA_HET_NARROW;
    }

    // ---- 1) cp time grid (uniform) ----------------------------------------
    for (int i = THREAD_START_X; i < N_cp_sig; i += BLOCK_INCR_X) {
        t_cp_buf[i] = chunk_t_start + (double) i * dt_cp;
    }
    CUDA_SYNC_THREADS;

    // ---- 2) raw TDI evaluation at cp times --------------------------------
    // Fills tdi_channels_cp_buf[nchannels * N_cp_sig] (raw complex TDI)
    // and phi_ref_un_het_buf[N_cp_sig] (un-heterodyned phi_ref). The
    // per-channel amp/phase extract+unwrap is deferred to the c-loop below
    // so we only need single-channel coefficient storage.
    (void) extract_scratch_len;
    if (orbit_cache != nullptr) {
        source->get_tdi_raw_cached(
            tdi_channels_cp_buf, phi_ref_un_het_buf,
            params, t_cp_buf, N_cp_sig, bin_i, nchannels,
            orbit_cache);
    } else {
        source->get_tdi_raw(
            tdi_channels_cp_buf, phi_ref_un_het_buf,
            params, t_cp_buf, N_cp_sig, bin_i, nchannels);
    }
    CUDA_SYNC_THREADS;


    // ---- 3) heterodyne-subtract phi_ref into dphi_ref_y_buf, then fit ----
    // dphi_ref_y_buf[i] = phi_ref(t_cp[i]) - 2*pi*f0_grid*t_cp[i].
    // phi_ref_un_het_buf stays intact for use by the per-channel extract
    // below. The dphi_ref spline (fit here, evaluated in step 4d) is the
    // OLD get_tdi_heterodyned convention -- preserves bitwise math match
    // against the direct path.
    for (int i = THREAD_START_X; i < N_cp_sig; i += BLOCK_INCR_X) {
        dphi_ref_y_buf[i] = phi_ref_un_het_buf[i] - two_pi_f0 * t_cp_buf[i];
    }
    CUDA_SYNC_THREADS;

    wdm_fit_cubic_spline(t_cp_buf, dphi_ref_y_buf,
                          dphi_ref_c1_buf, dphi_ref_c2_buf, dphi_ref_c3_buf,
                          B_buf, pcr_scratch,
                          N_cp_sig, CUBIC_SPLINE_LINEAR_SPACING);
    CUDA_SYNC_THREADS;

    // ---- 4) per-channel: extract + unwrap + fit (amp, phase) + evaluate ---
    // Carve extract+unwrap scratch out of extract_scratch (>= 21*N_cp_sig B).
    // ``flip`` doubles as the unwrap correction buffer (same convention as
    // new_extract_amplitude_and_phase + new_unwrap_phase share inside
    // get_tdi).
    double *flip      = (double *) extract_scratch;
    double *pjump     = &flip[N_cp_sig];
    int    *count     = (int *)  &pjump[N_cp_sig];
    bool   *fix_count = (bool *) &count[N_cp_sig];

    const cmplx  I_c       = cmplx(0.0, 1.0);
    const double n_taper   = 0.5 * alpha_eff * (double) (N_sparse - 1);
    const int    N_cp_last = N_cp_sig - 1;

    for (int c = 0; c < nchannels; ++c) {
        // (a) extract |M_c| -> amp_y_buf, arg(M_c) - phi_ref -> phase_y_buf.
        //     phiR MUST be un-heterodyned (see get_tdi_raw doc): the
        //     remainder(phiR, 2*pi) inside extract is not invariant under
        //     shifts by 2*pi*f0*t, and any per-sample drift it would
        //     introduce does NOT cancel against the downstream dphi_ref
        //     spline eval.
        source->new_extract_amplitude_and_phase(
            count, fix_count, flip, pjump, N_cp_sig,
            amp_y_buf, phase_y_buf,
            &tdi_channels_cp_buf[c * N_cp_sig],
            phi_ref_un_het_buf);
        CUDA_SYNC_THREADS;

        // (b) unwrap phase_y_buf in place; flip is reused as the cumulative
        //     correction buffer (size N_cp_sig).
        source->new_unwrap_phase(flip, N_cp_sig, phase_y_buf);
        CUDA_SYNC_THREADS;

        // (c) fit amp + phase splines into the single-channel coefficient
        //     buffers (B_buf / pcr_scratch reused across the fits).
        wdm_fit_cubic_spline(t_cp_buf, amp_y_buf,
                              amp_c1_buf, amp_c2_buf, amp_c3_buf,
                              B_buf, pcr_scratch,
                              N_cp_sig, CUBIC_SPLINE_LINEAR_SPACING);
        CUDA_SYNC_THREADS;
        wdm_fit_cubic_spline(t_cp_buf, phase_y_buf,
                              phase_c1_buf, phase_c2_buf, phase_c3_buf,
                              B_buf, pcr_scratch,
                              N_cp_sig, CUBIC_SPLINE_LINEAR_SPACING);
        CUDA_SYNC_THREADS;

        // (d) evaluate amp, phase, dphi_ref on the N_sparse t-grid and
        //     write the windowed slow signal for this channel into slow_buf.
        for (int i = THREAD_START_X; i < N_sparse; i += BLOCK_INCR_X) {
            const double tau = (double) i * dt_sparse;
            const double t   = chunk_t_start + tau;

            // Segment lookup (uniform t_cp): seg = floor((t - cp[0]) / dt_cp).
            int seg = (int) ((t - chunk_t_start) / dt_cp);
            if (seg < 0)              seg = 0;
            if (seg > N_cp_last - 1)  seg = N_cp_last - 1;
            const double dx = t - t_cp_buf[seg];

            const double amp =
                amp_y_buf [seg]
              + amp_c1_buf[seg] * dx
              + amp_c2_buf[seg] * dx * dx
              + amp_c3_buf[seg] * dx * dx * dx;
            const double tdi_phase =
                phase_y_buf [seg]
              + phase_c1_buf[seg] * dx
              + phase_c2_buf[seg] * dx * dx
              + phase_c3_buf[seg] * dx * dx * dx;
            const double dphi_ref =
                dphi_ref_y_buf [seg]
              + dphi_ref_c1_buf[seg] * dx
              + dphi_ref_c2_buf[seg] * dx * dx
              + dphi_ref_c3_buf[seg] * dx * dx * dx;

            const double phase_total = tdi_phase + dphi_ref + phi0_chunk;
            cmplx s = (cmplx)(amp) * gcmplx::exp(I_c * phase_total);

            // Tukey window: same formula as the direct path (scipy convention).
            if (alpha_eff > 0.0 && n_taper > 0.0) {
                double w = 1.0;
                const double di = (double) i;
                const double dlast = (double) (N_sparse - 1);
                if (di < n_taper) {
                    const double xn = di / n_taper;
                    w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                } else if (di > dlast - n_taper) {
                    const double xn = (dlast - di) / n_taper;
                    w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                }
                s = cmplx(s.real() * w, s.imag() * w);
            }
            slow_buf[c * N_sparse + i] = s;
        }
        // Critical: barrier before the next channel reuses amp_y_buf,
        // phase_y_buf, and the c1/c2/c3 stacks.
        CUDA_SYNC_THREADS;
    }


    // ---- 5) FFT slow_buf in place, per channel ----------------------------
    for (int c = 0; c < nchannels; ++c) {
        wdm_spline_radix2_fft(&slow_buf[c * N_sparse],
                              N_sparse, log2_N_sparse, /*inverse=*/false);
        CUDA_SYNC_THREADS;
    }


    // ---- 6) Place into chunk_fd_out (identical to direct path) -----------
    for (int c = 0; c < nchannels; ++c) {
        for (int m_idx = THREAD_START_X; m_idx < N_sparse; m_idx += BLOCK_INCR_X) {
            const int m = (m_idx < half_Nsp) ? m_idx : (m_idx - N_sparse);
            const int kbin = k_f0 + m;
            if (kbin >= 0 && kbin < n_rfft_chunk) {
                const cmplx v = slow_buf[c * N_sparse + m_idx];
                chunk_fd_out[c * n_rfft_chunk + kbin] =
                    cmplx(v.real() * scale_X, v.imag() * scale_X);
            }
        }
    }
    CUDA_SYNC_THREADS;

}


// ----------------------------------------------------------------------------
// Source-class-agnostic variant. `source` is a pointer to any
// :class:`LISATDIonTheFly` subclass (`GBTDIonTheFly`, `SOBBHTDIonTheFly`,
// future variants); `carrier_index` selects which entry of `params` is
// the heterodyne carrier (1 for GB's f0, 5 for SOBBH's f_low).
// ----------------------------------------------------------------------------
CUDA_DEVICE
inline void fast_wdm_inner_heterodyne(
    cmplx *chunk_fd_out,            // (nchannels * n_rfft_chunk); caller zero-inits
    LISATDIonTheFly *source,
    double *params,                 // source-class-specific params at t_ref
    int bin_i,
    int carrier_index,              // 1 for GB f0, 5 for SOBBH f_low
    double chunk_t_start,           // absolute start time of this chunk (s)
    double T_chunk,                 // chunk duration (s) = N_chunk_td * dt
    int N_sparse, int log2_N_sparse,
    int n_rfft_chunk,               // = N_chunk_td / 2 + 1
    int nchannels,                  // 3 for XYZ
    double tukey_alpha,             // 0 = rect, AUTO = recommended (see #defines)
    // workspace --------------------------------------------------------------
    double *t_sparse_buf,           // (N_sparse,)
    double *tdi_amp_buf,            // (nchannels * N_sparse)
    double *tdi_phase_buf,          // (nchannels * N_sparse)
    double *phi_ref_buf,            // (N_sparse,)
    cmplx  *tdi_channels_buf,       // (nchannels * N_sparse), used by gb->get_tdi
    cmplx  *slow_buf,               // (nchannels * N_sparse), reused as FFT input/output
    void   *get_tdi_scratch,        // get_tdi internal scratch
    int     get_tdi_scratch_len,
    OrbitsSplineCache *orbit_cache)  // nullptr -> direct orbit lookups
{
    const double dt_sparse  = T_chunk / (double) N_sparse;
    const double f0         = params[carrier_index];
    const double df_chunk   = 1.0 / T_chunk;
    const int    k_f0       = (int) round(f0 / df_chunk);
    const double f0_grid    = (double) k_f0 * df_chunk;
    const int    half_Nsp   = N_sparse / 2;
    const double scale_X    = 0.5 * dt_sparse;

    // Resolve Tukey alpha (sentinel -> auto-pick per N_sparse).
    double alpha_eff = tukey_alpha;
    if (alpha_eff == FAST_WDM_TUKEY_ALPHA_AUTO) {
        alpha_eff = (N_sparse >= 512)
            ? FAST_WDM_TUKEY_ALPHA_HET_WIDE
            : FAST_WDM_TUKEY_ALPHA_HET_NARROW;
    }

    // ---- 1) sparse time grid for this chunk -------------------------------
    for (int i = THREAD_START_X; i < N_sparse; i += BLOCK_INCR_X) {
        t_sparse_buf[i] = chunk_t_start + (double) i * dt_sparse;
    }
    CUDA_SYNC_THREADS;

    // ---- 2) sparse TDI evaluation: tdi_amp, tdi_phase, phase_ref ----------
    if (orbit_cache != nullptr) {
        source->get_tdi_cached(get_tdi_scratch, get_tdi_scratch_len,
                                tdi_channels_buf,
                                tdi_amp_buf, tdi_phase_buf, phi_ref_buf,
                                params, t_sparse_buf, N_sparse, bin_i, nchannels,
                                orbit_cache);
    } else {
        source->get_tdi(get_tdi_scratch, get_tdi_scratch_len,
                        tdi_channels_buf,
                        tdi_amp_buf, tdi_phase_buf, phi_ref_buf,
                        params, t_sparse_buf, N_sparse, bin_i, nchannels);
    }
    CUDA_SYNC_THREADS;


    // ---- 3) slow signal + optional Tukey window ---------------------------
    const cmplx I_c(0.0, 1.0);
    // Tukey denominator: alpha*(N-1)/2, matching scipy.signal.windows.tukey
    // (NOT alpha*N/2 -- using N gives a ~1% offset that shifts ~0.1-0.3% of
    // spectral leakage into adjacent bins vs the Python reference).
    const double n_taper = 0.5 * alpha_eff * (double) (N_sparse - 1);
    for (int c = 0; c < nchannels; ++c) {
        for (int i = THREAD_START_X; i < N_sparse; i += BLOCK_INCR_X) {
            const int idx = c * N_sparse + i;
            const double tau   = (double) i * dt_sparse;
            const double phase = tdi_phase_buf[idx] + phi_ref_buf[i]
                                 - 2.0 * M_PI * f0_grid * tau;
            cmplx s = (cmplx)(tdi_amp_buf[idx]) * gcmplx::exp(I_c * phase);

            // Tukey window with alpha taper at each end; rectangular at
            // alpha=0; full Hann at alpha=1. Taper is alpha/2 of N_sparse
            // samples on each side, cosine half-cycle.
            if (alpha_eff > 0.0 && n_taper > 0.0) {
                double w = 1.0;
                const double di = (double) i;
                const double dlast = (double) (N_sparse - 1);
                if (di < n_taper) {
                    const double xn = di / n_taper;       // 0 -> 1 over taper
                    w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                } else if (di > dlast - n_taper) {
                    const double xn = (dlast - di) / n_taper;
                    w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                }
                s = cmplx(s.real() * w, s.imag() * w);
            }
            slow_buf[idx] = s;
        }
    }
    CUDA_SYNC_THREADS;


    // ---- 4) FFT slow_buf in place, per channel ----------------------------
    for (int c = 0; c < nchannels; ++c) {
        wdm_spline_radix2_fft(&slow_buf[c * N_sparse],
                              N_sparse, log2_N_sparse, /*inverse=*/false);
        CUDA_SYNC_THREADS;
    }


    // ---- 5) Scale and place into chunk_fd_out at [k_f0 + fftfreq] --------
    // fftfreq(N) gives FFT bin indices [0, 1, ..., N/2-1, -N/2, ..., -1].
    // The chunk's dense rfft array has length n_rfft_chunk = N_chunk_td/2+1.
    // Bins outside [0, n_rfft_chunk) are dropped.
    for (int c = 0; c < nchannels; ++c) {
        for (int m_idx = THREAD_START_X; m_idx < N_sparse; m_idx += BLOCK_INCR_X) {
            const int m = (m_idx < half_Nsp) ? m_idx : (m_idx - N_sparse);
            const int kbin = k_f0 + m;
            if (kbin >= 0 && kbin < n_rfft_chunk) {
                const cmplx v = slow_buf[c * N_sparse + m_idx];
                chunk_fd_out[c * n_rfft_chunk + kbin] =
                    cmplx(v.real() * scale_X, v.imag() * scale_X);
            }
        }
    }
    CUDA_SYNC_THREADS;

}


// ============================================================================
// fast_wdm_inner_heterodyne_direct  --  pointwise direct heterodyne path
// ----------------------------------------------------------------------------
//
// Mirrors `fast_wdm_inner_heterodyne` but skips the per-channel (amp,
// phase) extract by calling `get_tdi_Xf_single` per sparse-time-point
// directly into per-thread registers. The complex heterodyne factor
// `exp(-2*pi*i*f0_grid*tau)` is applied via complex multiply, matching
// the established direct pattern in the chunked-het XYZ kernel (where
// `new_extract_amplitude_and_phase` reduces to `slow = conj(M) *
// exp(-2pi i f0 t)` for pjump=0 / typical GB).
//
// Per-source shared workspace drops from
//     t_sparse + tdi_amp + tdi_phase + phi_ref + tdi_channels + slow  (~80 KB)
// to just
//     slow                                                            (~24 KB)
// at `FAST_WDM_N_SPARSE_MAX=512`, `NCHANNELS_MAX=3`. Sky vectors + link
// arrays are computed once and passed in.
//
// Templated on `SourceT` (compile-time dispatch to per-source
// `get_tdi_Xf_single`). Per-source instantiations live in GBGPU
// (`<GBTDIonTheFly>`) and BBHx (`<SOBBHTDIonTheFly>`) once the carve-
// outs of Phase 3L.7 / 3L.8 land; until then lisa-on-gpu's
// `fast_wdm_inner_heterodyne_kernel` instantiates against `GBTDIonTheFly`
// in-place.
// ============================================================================
template <typename SourceT>
CUDA_DEVICE
inline void fast_wdm_inner_heterodyne_direct(
    cmplx *chunk_fd_out,            // (nchannels * n_rfft_chunk); caller zero-inits
    SourceT &src,
    double *params, int bin_i, int carrier_index,
    double chunk_t_start, double T_chunk,
    int N_sparse, int log2_N_sparse,
    int n_rfft_chunk, int nchannels, double tukey_alpha,
    cmplx *slow_buf,                // (nchannels * N_sparse) -- FFT in/out
    Vec k_sky, Vec u_sky, Vec v_sky,
    int *link_sc_rec, int *link_sc_em)
{
    const double dt_sparse = T_chunk / (double) N_sparse;
    const double f0        = params[carrier_index];
    const double df_chunk  = 1.0 / T_chunk;
    const int    k_f0      = (int) round(f0 / df_chunk);
    const double f0_grid   = (double) k_f0 * df_chunk;
    const int    half_Nsp  = N_sparse / 2;
    const double scale_X   = 0.5 * dt_sparse;

    double alpha_eff = tukey_alpha;
    if (alpha_eff == FAST_WDM_TUKEY_ALPHA_AUTO) {
        alpha_eff = (N_sparse >= 512)
            ? FAST_WDM_TUKEY_ALPHA_HET_WIDE
            : FAST_WDM_TUKEY_ALPHA_HET_NARROW;
    }
    const double n_taper = (alpha_eff > 0.0)
        ? 0.5 * alpha_eff * (double) (N_sparse - 1)
        : 0.0;

    // ---- 1) build slow_buf = conj(tdi(t)) * exp(-2pi i f0_grid tau) * tukey
    for (int i = THREAD_START_X; i < N_sparse; i += BLOCK_INCR_X) {
        const double t   = chunk_t_start + (double) i * dt_sparse;
        const double tau = (double) i * dt_sparse;

        cmplx tdi_tmp[FAST_WDM_NCHANNELS_MAX];
        src.get_tdi_Xf_single(&tdi_tmp[0], t, params,
                               k_sky, u_sky, v_sky,
                               link_sc_rec, link_sc_em, bin_i);

        const double het_phase = -2.0 * M_PI * f0_grid * tau;
        const cmplx  het_factor(cos(het_phase), sin(het_phase));

        double w = 1.0;
        if (n_taper > 0.0) {
            const double di    = (double) i;
            const double dlast = (double) (N_sparse - 1);
            if (di < n_taper) {
                const double xn = di / n_taper;
                w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
            } else if (di > dlast - n_taper) {
                const double xn = (dlast - di) / n_taper;
                w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
            }
        }

        for (int c = 0; c < nchannels; ++c) {
            cmplx s = gcmplx::conj(tdi_tmp[c]) * het_factor;
            slow_buf[c * N_sparse + i] = cmplx(s.real() * w, s.imag() * w);
        }
    }
    CUDA_SYNC_THREADS;

    // ---- 2) FFT slow_buf in place, per channel ----------------------------
    for (int c = 0; c < nchannels; ++c) {
        wdm_spline_radix2_fft(&slow_buf[c * N_sparse],
                              N_sparse, log2_N_sparse, /*inverse=*/false);
        CUDA_SYNC_THREADS;
    }

    // ---- 3) Scale and place into chunk_fd_out at [k_f0 + fftfreq] --------
    for (int c = 0; c < nchannels; ++c) {
        for (int m_idx = THREAD_START_X; m_idx < N_sparse; m_idx += BLOCK_INCR_X) {
            const int m = (m_idx < half_Nsp) ? m_idx : (m_idx - N_sparse);
            const int kbin = k_f0 + m;
            if (kbin >= 0 && kbin < n_rfft_chunk) {
                const cmplx v = slow_buf[c * N_sparse + m_idx];
                chunk_fd_out[c * n_rfft_chunk + kbin] =
                    cmplx(v.real() * scale_X, v.imag() * scale_X);
            }
        }
    }
    CUDA_SYNC_THREADS;
}


// ============================================================================
// gb_chunk_fd_to_wdm -- chunk-FD -> chunk-WDM transform
// ----------------------------------------------------------------------------
//
// Despite the historical `gb_` prefix, this function is **source-
// agnostic**: it takes a heterodyned FD chunk array and returns its
// WDM-domain coefficients via per-layer iFFT + parity-sign + Re/Im
// pick. Used by both GB and SOBBH chunked-het kernel families.
//
// Ports `lisatools.domains.FDSignal.wdmtransform` for one chunk's
// dense rfft array (length `n_rfft_chunk = Nf*Nt_sub/2 + 1`, populated
// only in `N_sparse` bins around `k_f0` by
// `fast_wdm_inner_heterodyne`). The output is the
// `(nchannels, Nf, Nt_sub)` real WDM coefficient block for the
// chunk -- caller is responsible for stitching it into the global
// `(nchannels, Nf, Nt)` template buffer (use interior pixels for
// middle chunks, full pixels for first/last; see the Python
// `_stitched_wdm_from_heterodyne` for the convention).
//
// Algorithm per layer m in [0, Nf]:
//
//   1. Build the length-`Nt_sub` windowed FD slice:
//        k_global = m*Nt_sub/2 + (k_idx - Nt_sub/2)
//        Hermitian wrap when k_global < 0 or > N_chunk_td/2.
//        `before_ifft[k_idx] = (Hermitian-folded chunk_fd[k_global]) /
//                              data_dt * wdm_window[k_idx]`
//   2. iFFT length `Nt_sub`  (reuse `wdm_spline_radix2_fft`,
//      `inverse=true`).
//   3. Apply parity factor and pick Re or Im of `conj(C_{m,n}) * out`:
//        sign = (-1)^((m+1)*n)
//        if (m+n) % 2 == 0:  real_part = Re(out[n])     (Cmn=1, conj=1)
//        else:               real_part = Im(out[n])     (Cmn=1j, conj=-1j
//                                                        -> Re(-1j*z)=Im(z))
//        if (m==0 or m==Nf) AND ((m+n) % 2 != 0): tmp_w_mn[m, n] = 0
//        else: tmp_w_mn[m, n] = kappa * sign * real_part
//
// Folding (after collecting all m in [0, Nf]):
//   for n in [0, Nt_sub):
//     if n is even:   w_mn[0, n] = tmp_w_mn[0, n] / sqrt(2)
//     else:           w_mn[0, n] = tmp_w_mn[Nf, n-1] / sqrt(2)
//   for m in [1, Nf): w_mn[m, n] = tmp_w_mn[m, n]
//
// Wavelet window (`wdm_window`) is the `Nt_sub`-length sample of
// `phitilde` at `omega = 2*pi/N_chunk_td * arange(-Nt_sub/2,
// Nt_sub/2)`, precomputed on the host (computing
// `scipy.special.betainc` on-device is impractical). See
// `WDMSettings.setup_window` in Python.
//
// Workspace:
//   * `layer_scratch` -- `Nt_sub`-long cmplx buffer for the per-layer
//                        iFFT.
//   * `tmp_w_mn`      -- `(nchannels, Nf+1, Nt_sub)` real, sized at
//                        compile time with the `FAST_WDM_*` maxima.
//
// Threading: per-block; `THREAD_START_X` / `BLOCK_INCR_X` parallelism
// within the inner `Nt_sub` loops. The outer m-loop is serial within
// a block.
// ============================================================================

CUDA_DEVICE
inline void gb_chunk_fd_to_wdm(
    double *w_mn_out,        // (nchannels, Nf, Nt_sub) -- output, caller-zero'd
    cmplx  *chunk_fd,        // (nchannels, n_rfft_chunk) input
    const double *wdm_window,// (Nt_sub,) precomputed phitilde
    int Nf, int Nt_sub, int log2_Nt_sub,
    int n_rfft_chunk,        // = Nf*Nt_sub/2 + 1
    double data_dt,
    int nchannels,
    cmplx *layer_scratch,    // (Nt_sub,) per-block iFFT scratch
    int m_lo,                // outer loop lower bound (inclusive)
    int m_hi                 // outer loop upper bound (exclusive)
)
{
    const int N_chunk_td = Nf * Nt_sub;
    const int half_Nt_sub = Nt_sub / 2;
    const double kappa = 2.0 * sqrt(M_PI * data_dt) / (double) Nf;
    const double sqrt2 = sqrt(2.0);

    // Output is folded later from tmp_w_mn (Nf+1 rows). For simplicity we
    // process layers one-at-a-time and write directly into w_mn_out for
    // m in [1, Nf-1], and into auxiliary buffers for m=0 / m=Nf so we
    // can fold them at the end.
    //
    // Per-channel m=0 and m=Nf rows go into separate scratch; size Nt_sub each.
    // We allocate this from shared memory via the caller.
    //
    // For now this device function assumes the caller pre-zeros w_mn_out
    // and provides extra m0/mNf scratch buffers. We bake those into the
    // host kernel's shared memory.
    //
    // NOTE: this implementation processes nchannels x (Nf+1) layers in a
    // serial outer loop. Each iteration reuses layer_scratch.
    //
    // m_lo / m_hi (inclusive / exclusive) restrict the outer m-loop to a
    // narrow band -- a ~Nf / band-width speedup when use_layer_groups is
    // active. Pass ``m_lo=0, m_hi=Nf+1`` for the full-Nf (legacy) path.
    // Layers outside [m_lo, m_hi) stay at the caller's pre-zero -- that
    // matches the inner-product / accumulator m-band the layer-groups
    // path already iterates, and matches the mm5/mm2 narrow-band
    // physical model for GBs (see ``gb_chunked_prior_draws.py``).

    for (int c = 0; c < nchannels; ++c) {
        const cmplx *fd_c = &chunk_fd[c * n_rfft_chunk];
        for (int m = m_lo; m < m_hi; ++m) {

            // --- 1) build windowed FD slice (length Nt_sub) -----------------
            for (int k_idx = THREAD_START_X; k_idx < Nt_sub; k_idx += BLOCK_INCR_X) {
                long k_global = (long) m * (long) half_Nt_sub + (long)(k_idx - half_Nt_sub);
                bool herm = false;
                if (k_global < 0)              { k_global = -k_global;       herm = true; }
                if (k_global > N_chunk_td / 2) { k_global = N_chunk_td - k_global; herm = true; }

                cmplx v(0.0, 0.0);
                if (k_global >= 0 && k_global < n_rfft_chunk) {
                    v = fd_c[k_global];
                    if (herm) v = gcmplx::conj(v);
                    v = cmplx(v.real() / data_dt, v.imag() / data_dt);
                    const double w = wdm_window[k_idx];
                    v = cmplx(v.real() * w, v.imag() * w);
                }
                layer_scratch[k_idx] = v;
            }
            CUDA_SYNC_THREADS;

            // --- 2) iFFT length Nt_sub --------------------------------------
            wdm_spline_radix2_fft(layer_scratch, Nt_sub, log2_Nt_sub,
                                  /*inverse=*/true);
            CUDA_SYNC_THREADS;

            // --- 3) parity factor + real/imag pick, write tmp_w_mn ----------
            // For m in [1, Nf-1] -> directly into w_mn_out[c, m, n].
            // For m = 0 -> w_mn_out[c, 0, even n] (folded).
            // For m = Nf -> w_mn_out[c, 0, odd n]  (folded; n_src is even).
            for (int n = THREAD_START_X; n < Nt_sub; n += BLOCK_INCR_X) {
                const bool boundary = (m == 0 || m == Nf);
                const bool mn_parity_even = (((m + n) & 1) == 0);
                if (boundary && !mn_parity_even) {
                    continue;                  // zeroed; caller has output pre-zero'd
                }

                const cmplx z = layer_scratch[n];
                const double real_part = mn_parity_even ? z.real() : z.imag();
                const double sign = ((((m + 1) * n) & 1) == 0) ? 1.0 : -1.0;
                const double val = kappa * sign * real_part;

                if (m >= 1 && m <= Nf - 1) {
                    // direct write -- interior layer
                    w_mn_out[c * Nf * Nt_sub + m * Nt_sub + n] = val;
                } else if (m == 0) {
                    if ((n & 1) == 0) {
                        // even n at m=0 -> w_mn[c, 0, n] (cos@DC); / sqrt(2)
                        w_mn_out[c * Nf * Nt_sub + 0 * Nt_sub + n] = val / sqrt2;
                    }
                } else { // m == Nf
                    // tmp_w_mn[Nf, n_src] with n_src even goes into
                    // w_mn[c, 0, 2*n_src + 1] (odd slots), / sqrt(2).
                    if ((n & 1) == 0) {
                        const int n_out = n + 1;
                        if (n_out < Nt_sub) {
                            w_mn_out[c * Nf * Nt_sub + 0 * Nt_sub + n_out] = val / sqrt2;
                        }
                    }
                }
            }
            CUDA_SYNC_THREADS;
        }
    }
}


#endif  // LAT_CHUNKED_HET_KERNELS_HH
