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
// GBT's Interpolate.hh provides `fit_cubic_spline_thomas` (CPU) and
// `fit_cubic_spline_pcr` (GPU). We deliberately do NOT `#include` it
// here because BBHx ships a same-named, different-purpose
// `Interpolate.hh` for its PhenomHM-mode interp; that local header
// uses the same `__INTERPOLATE_HH__` guard, so whoever is included
// first wins for the whole TU. binding_bbhx.hpp includes the BBHx
// local first (for its own PhenomHM consumers), which would shadow
// GBT's even via `<Interpolate.hh>`. Forward-declare the two GBT
// host launchers directly so this header is self-contained --
// CubicSpline itself is already pulled in via InterpolateDevice.hh
// transitively. Phase 3L.8 (2026-06-04).
class CubicSpline;
CubicSpline fit_cubic_spline_thomas(double *x, double *y,
                                    double *c1, double *c2, double *c3,
                                    double *B,
                                    int length, int spline_type);
#ifdef __CUDACC__
CUDA_DEVICE
CubicSpline fit_cubic_spline_pcr(double *x, double *y,
                                 double *c1, double *c2, double *c3,
                                 double *B, double *pcr_scratch,
                                 int length, int spline_type);
#endif


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
// Block-wide double-sum reductions. `block_reduce` takes a per-thread
// staging slot in shared memory (`array[threadIdx.x]`); `block_reduce_scalar`
// takes a per-thread register value and skips the staging array. Both rely
// on `NUM_THREADS_HERE` (defined just above) so the cub::BlockReduce
// template width matches the launch shape.
//
// Defined here -- not in a host .cu -- because GBGPU's gb_tdi_on_the_fly.cu
// and lisa-on-gpu's TDIonTheFly.cu (and future SOBBH/BBHx kernels)
// all need these. CUDA-only: cub is the underlying primitive and the
// CPU mirror collapses to a single thread (NUM_THREADS_HERE == 1)
// where the reduction is just the scalar identity.
// ----------------------------------------------------------------------------
#ifdef __CUDACC__
#include <cub/cub.cuh>

CUDA_DEVICE
inline double block_reduce(double *array)
{
    using BlockReduce = cub::BlockReduce<double, NUM_THREADS_HERE>;
    int tid = threadIdx.x;
    CUDA_SHARED typename BlockReduce::TempStorage temp_storage;
    CUDA_SYNC_THREADS;
    double thread_data = array[tid];
    return BlockReduce(temp_storage).Sum(thread_data);
}

CUDA_DEVICE
inline double block_reduce_scalar(double thread_data)
{
    using BlockReduce = cub::BlockReduce<double, NUM_THREADS_HERE>;
    CUDA_SHARED typename BlockReduce::TempStorage temp_storage;
    CUDA_SYNC_THREADS;
    return BlockReduce(temp_storage).Sum(thread_data);
}
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




// ============================================================================
// Slice 3 (2026-06-04): templated chunked-het kernel bodies + their
// inline _impl<SourceT> host launchers.
//
// The four kernel templates (wdm_het_{fill_global,get_ll,swap_ll,
// get_fstat_ll}_kernel) are CUDA_KERNEL functions parameterised on
// SourceT (the LISATDIonTheFly subclass providing the per-source
// get_tdi / get_tdi_Xf_single methods). The four `_impl<SourceT>`
// launchers below set up the cuda{Malloc,Memcpy} of wrapper structs
// and dispatch the kernel launch.
//
// `static` was changed to `inline` on the _impl launchers (they used
// to be file-scoped statics in lisa-on-gpu) so multiple downstream
// TUs (lisa-on-gpu now, GBGPU + BBHx after Phase 3L.7 / 3L.8) can
// instantiate against their respective source classes without ODR
// collisions. The kernel templates themselves are already ODR-exempt.
// ============================================================================

// =============================================================================
// NEW shared-memory-only chunked-het kernels.
//
// Design (per user direction 2026-05-29):
//   * One binary per block on the X axis. blockDim.x = NUM_THREADS_HERE (= 64
//     on GPU, 1 on CPU). The block iterates chunks sequentially via a
//     for-loop -- a future commit will move chunks onto blockIdx.y; the
//     sequential loop is marked with a TODO comment.
//   * Per (chunk, m_layer): the per-channel tdi_channel slow-signal samples
//     are computed DIRECTLY from get_tdi (no separate amp/phase cache),
//     heterodyned in time domain via exp(-i 2 pi f0_grid t), FFT'd in shared
//     memory, windowed for this layer's WDM filter, iFFT'd in the same
//     buffer, parity factor applied, then each thread holds one (m, n_loc)
//     WDM coefficient.
//   * Each thread streams data[c, m_act, n_act] and invC[c1, c2, m_act, n_act]
//     from global memory (coalesced -- warp lanes hit adjacent n_act addresses
//     since each thread owns one n_loc), accumulates per-thread partials
//     (registers).
//   * Block-wide partial_dh / partial_hh in shared mem for the final tree
//     reduction; thread 0 atomicAdds into the global outputs.
//
// Shared-memory layout per block (~13 KB at Nt_sub=256, NUM_THREADS=64):
//     tdi_channel_buf[3 * Nt_sub] cmplx  (12 KB)  -- per-channel FFT/iFFT scratch
//     partial_dh[blockDim.x]      double ( 0.5 KB)
//     partial_hh[blockDim.x]      double ( 0.5 KB)
//
// We deliberately do NOT cache:
//   - heterodyne FFT output across layers (recompute per (chunk, m_layer))
//   - tdi_amp/tdi_phase/phi_ref (computed inline)
//   - orbit splines (use raw orbits->get_* per evaluation)
// These caches can be reintroduced as a perf optimization once the kernel
// structure is validated. The point of this rewrite is correctness +
// memory-footprint clarity, not maximum throughput.
//
// Constraints:
//   - Nt_sub and N_sparse must be powers of 2 (radix-2 FFT). The host-side
//     WDMSettings constructor already enforces this.
//   - blockDim.x == Nt_sub is NOT required; the FFT helpers in
//     WDMSplineHelpers.hh thread-stride over the array.
// =============================================================================
template <class SourceT>
CUDA_KERNEL
void wdm_het_get_ll_kernel(
    double *d_h_out, double *h_h_out,        // (num_bin,) outputs (host pre-zero'd)
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all,                      // (num_bin * nparams,)
    int    *data_index_all, int *noise_index_all,
    double *chunk_t_starts,                  // (n_chunks,)
    int    *chunk_keep_lo, int *chunk_keep_hi,
    int    *chunk_n_global_offset,
    double *wdm_window,                      // (Nt_sub,)
    double *data_d, double *invC,            // active-band layout (see contract below)
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int    m_band_half_width,
    int    N_cp_orbit)   // 0 -> raw orbit lookups; >0 -> per-chunk spline cache
{
    // One binary per block (grid.X); chunks iterated sequentially inside the
    // block. See the kernel-section header comment above for the full design.

    // Construct source class (carries orbits / tdi_config pointers).
    SourceT src(orbits, tdi_config, T, t_ref);

    // Hoist scalar WDM grid constants from the device-resident struct into
    // local registers -- the compiler keeps them across the inner loops, so
    // the per-binary inner work pays only one load each, not one per
    // iteration.
    const int Nf         = wdm_settings->Nf;
    const int Nt         = wdm_settings->Nt;
    const int ind_min_f  = wdm_settings->ind_min_f;
    const int ind_min_t  = wdm_settings->ind_min_t;
    const int Nf_active  = wdm_settings->Nf_active;
    const int Nt_active  = wdm_settings->Nt_active;
    (void) Nt;  // unused in get_ll (kept for layout parity with the OLD kernel)

    // Dynamic shared-memory layout (set by ``shared_bytes`` at kernel launch).
    // Two cmplx buffers per (chunk, binary):
    //
    //   fd_chunk_buf   [nchannels * N_sparse] cmplx  -- HOLDS the chunk-FD
    //                                                   (TD build -> heterodyne
    //                                                   -> Tukey -> FFT, done
    //                                                   ONCE per chunk).
    //   layer_buf      [nchannels * Nt_sub]   cmplx  -- per-m_layer scratch
    //                                                   (window+rearrange ->
    //                                                   iFFT -> parity ->
    //                                                   accumulate).
    //   partial_dh     [blockDim.x]           double
    //   partial_hh     [blockDim.x]           double
    //
    // At Nt_sub=N_sparse=256, total ~25 KB (well under 48 KB default on A100).
    // The chunk-FD is computed ONCE per chunk and reused for every m_layer in
    // the binary's band -- previously we did the full TD-build+FFT per m,
    // wasting (m_band_width - 1) x n_chunks worth of FFT work per binary.
    //
    // N_sparse and Nt_sub may differ: fd_chunk_buf is sized by N_sparse
    // (forward-FFT length) and layer_buf is sized by Nt_sub (iFFT length).
    // Step 5 maps from one to the other -- bins of the wider layer that
    // fall outside the narrower FD window get zero-filled by the
    // ``if (fft_bin >= -half_Nsp && fft_bin < half_Nsp)`` guard. Useful
    // when a narrowband source only needs a smaller chunk-FD window.
#ifdef __CUDACC__
    extern CUDA_SHARED char shared_mem[];
    cmplx  *fd_chunk_buf    = (cmplx *) shared_mem;
    cmplx  *layer_buf       = &fd_chunk_buf[(size_t) nchannels * N_sparse];
    double *partial_dh      = (double *) &layer_buf[(size_t) nchannels * Nt_sub];
    double *partial_hh      = &partial_dh[NUM_THREADS_HERE];
    // cufftdx scratch (only used by wdm_fft_dispatch when LISA_USE_CUFFTDX
    // is defined; sized as the max across instantiated FFT lengths).
    char   *fft_scratch     = (char *) &partial_hh[NUM_THREADS_HERE];
#else
    // CPU stubs: stack arrays sized at the compile-time maxima.
    cmplx  fd_chunk_buf_cpu [FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  layer_buf_cpu    [FAST_WDM_NCHANNELS_MAX * FAST_WDM_NT_SUB_MAX];
    double partial_dh_cpu   [1];
    double partial_hh_cpu   [1];
    cmplx  *fd_chunk_buf    = fd_chunk_buf_cpu;
    cmplx  *layer_buf       = layer_buf_cpu;
    double *partial_dh      = partial_dh_cpu;
    double *partial_hh      = partial_hh_cpu;
    char   *fft_scratch     = nullptr;  // unused on CPU
#endif

    const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
    const double df_chunk = 1.0 / T_chunk;

    CUDA_SHARED int link_sc_rec[NLINKS];
    CUDA_SHARED int link_sc_em [NLINKS];
    src.fill_link_arrays(link_sc_rec, link_sc_em);

    // Orbit spline-cache buffers (used only if N_cp_orbit > 0). Mirrors the
    // buffer set consumed by ``populate_orbit_spline_cache``. Sized at
    // FAST_WDM_N_CP_ORBIT_MAX so the kernel JITs once and dispatches against
    // any 0 < N_cp_orbit <= max. At N_cp_orbit=32 this adds ~15.6 KB to
    // shared-mem; capped at 48 -> ~23 KB. Replaces global-mem orbit table
    // reads inside get_tdi_Xf_single with cooperative shared-mem cubic
    // spline evals.
    CUDA_SHARED double orbit_t_cp_buf  [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_y_buf [6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c1_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c2_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c3_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_y_buf [9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c1_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c2_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c3_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_B_buf     [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pcr_buf   [8 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED OrbitsSplineCache orbit_cache_storage;
    const bool use_orbit_cache =
        (N_cp_orbit > 0 && N_cp_orbit <= FAST_WDM_N_CP_ORBIT_MAX);

    CUDA_SYNC_THREADS;

    // One binary per block on grid.X. Grid-stride if num_bin > gridDim.x.
    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
        double *params      = &params_all[(size_t) bin_i * nparams];
        const int data_ind  = data_index_all[bin_i];
        const int noise_ind = noise_index_all[bin_i];
        (void) noise_ind;  // invC already incorporates noise; kept for API parity
        (void) data_ind;   // data_d / invC are indexed directly (no per-binary slab)

        // Per-binary inner-product accumulators (in registers).
        double tmp_dh = 0.0;
        double tmp_hh = 0.0;

        // Carrier bin in chunk-FD coordinates + WDM m-band centred on f0.
        const double f0       = params[src.f0_index];
        const int    k_f0     = (int) round(f0 / df_chunk);
        const double f0_grid  = (double) k_f0 * df_chunk;
        const int    m_floor  = (int) (f0 / layer_df);
        int          m_lo     = m_floor - m_band_half_width;
        int          m_hi     = m_floor + m_band_half_width + 1;   // exclusive
        // Clip to active band -- the accumulator only reads pixels in the
        // active band anyway, so processing layers outside is wasted work.
        if (m_lo < ind_min_f)             m_lo = ind_min_f;
        if (m_hi > ind_min_f + Nf_active) m_hi = ind_min_f + Nf_active;

        Vec k_sky(0.0, 0.0, 0.0);
        Vec u_sky(0.0, 0.0, 0.0);
        Vec v_sky(0.0, 0.0, 0.0);
        src.get_sky_vectors(&k_sky, &u_sky, &v_sky, params);

        // TODO(blockIdx.y on chunks): when chunks move to grid.Y, change
        //   ``for (int j = 0; j < n_chunks; ++j)`` to
        //   ``for (int j = BLOCK_START_Y; j < n_chunks; j += GRID_INCR_Y)``.
        for (int j = 0; j < n_chunks; ++j) {
            const int    keep_lo     = chunk_keep_lo[j];
            const int    keep_hi     = chunk_keep_hi[j];
            const int    n_global_lo = chunk_n_global_offset[j];
            const double chunk_t0    = chunk_t_starts[j];
            const double dt_sparse   = T_chunk / (double) N_sparse;

            // Populate orbit spline cache once over this chunk's time
            // window [chunk_t0, chunk_t0 + T_chunk]. All threads cooperate
            // (PCR solver on GPU, Thomas on CPU). Skipped when N_cp_orbit==0.
            OrbitsSplineCache *orbit_cache_ptr = nullptr;
            if (use_orbit_cache) {
                populate_orbit_spline_cache(
                    &orbit_cache_storage, orbits,
                    chunk_t0, T_chunk, N_cp_orbit,
                    orbit_t_cp_buf,
                    orbit_ltt_y_buf, orbit_ltt_c1_buf,
                    orbit_ltt_c2_buf, orbit_ltt_c3_buf,
                    orbit_pos_y_buf, orbit_pos_c1_buf,
                    orbit_pos_c2_buf, orbit_pos_c3_buf,
                    orbit_B_buf, orbit_pcr_buf);
                CUDA_SYNC_THREADS;
                orbit_cache_ptr = &orbit_cache_storage;
            }

            // ============================================================
            // Steps 1-4 are CHUNK-LEVEL: TD-build -> heterodyne -> Tukey
            // -> FFT into fd_chunk_buf. They do NOT depend on m, so we
            // compute them ONCE per chunk and reuse across all m_layers.
            // ============================================================

            // ---- 1) compute tdi_channel(t) into fd_chunk_buf ----
            // Thread-stride over i; compute t inline as a linear ramp.
            // Writes raw complex TDI values into fd_chunk_buf with layout
            // [c * N_sparse + i]. If orbit_cache_ptr != nullptr, the TDI
            // calls use shared-mem cubic-spline orbit evals instead of
            // global-mem table lookups -- bit-equivalent to the raw path
            // at N_cp_orbit >= 32 over typical chunk lengths (LTT and
            // position residuals well below float64 precision).
            for (int i = THREAD_START_X; i < N_sparse; i += BLOCK_INCR_X) {
                const double t = chunk_t0 + (double) i * dt_sparse;
                cmplx tdi_tmp[3];
                if (orbit_cache_ptr != nullptr) {
                    src.get_tdi_Xf_single_cached(&tdi_tmp[0], t, params,
                                                  k_sky, u_sky, v_sky,
                                                  link_sc_rec, link_sc_em,
                                                  bin_i, orbit_cache_ptr);
                } else {
                    src.get_tdi_Xf_single(&tdi_tmp[0], t, params,
                                          k_sky, u_sky, v_sky,
                                          link_sc_rec, link_sc_em, bin_i);
                }
                for (int c = 0; c < nchannels; ++c)
                    fd_chunk_buf[c * N_sparse + i] = tdi_tmp[c];
            }
            CUDA_SYNC_THREADS;

            // ---- 2) time-domain heterodyne + 3) Tukey window (in place) ----
            // See per-step derivation comments below the loop.
            const double n_taper = (tukey_alpha > 0.0)
                ? 0.5 * tukey_alpha * (double) (N_sparse - 1)
                : 0.0;
            for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                 idx += BLOCK_INCR_X) {
                const int c = idx / N_sparse;
                const int i = idx - c * N_sparse;
                const double tau = (double) i * dt_sparse;
                // Original fast_wdm_inner_heterodyne routes raw cmplx TDI
                // through new_extract_amplitude_and_phase to produce
                //   slow = conj(M) * exp(I*pjump) * exp(-I 2pi f0 t)
                // For typical GB pjump=0; replicate by conjugating the
                // get_tdi_Xf output before the heterodyne multiply.
                const double het_phase = -2.0 * M_PI * f0_grid * tau;
                const cmplx  het_factor(cos(het_phase), sin(het_phase));
                cmplx s = gcmplx::conj(fd_chunk_buf[idx]) * het_factor;
                if (n_taper > 0.0) {
                    double w = 1.0;
                    const double di    = (double) i;
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
                fd_chunk_buf[idx] = s;
            }
            CUDA_SYNC_THREADS;

            // ---- 4) FFT per channel (in place in fd_chunk_buf) ----
            // Per-channel forward FFT. wdm_spline_radix2_fft ends with a
            // CUDA_SYNC_THREADS internally, so no inter-channel sync needed.
            for (int c = 0; c < nchannels; ++c) {
                wdm_fft_dispatch(&fd_chunk_buf[c * N_sparse],
                                  N_sparse, log2_N_sparse,
                                  /*inverse=*/false, fft_scratch);
            }
            // fd_chunk_buf now holds the chunk-FD; reuse for every m below.

            // ============================================================
            // Steps 5-8 are PER M_LAYER: read fd_chunk_buf, window/rearrange
            // into layer_buf, iFFT, parity, accumulate.
            // ============================================================
            const double scale_fd     = 0.5 * dt_sparse / dt;
            for (int m = m_lo; m < m_hi; ++m) {
                const int m_act = m - ind_min_f;

                // ---- 5) window + rearrange for layer m: fd_chunk_buf -> layer_buf ----
                //
                // Since fd_chunk_buf and layer_buf are separate buffers,
                // no per-thread register staging is needed -- each thread
                // reads its source bin from fd_chunk_buf and writes its
                // destination bin in layer_buf with a single sync at the
                // end (before iFFT).
                //
                // Combined heterodyne-place + WDM-read scaling:
                //   chunk_fd  = fft_output * (0.5 * dt_sparse)
                //   layer_fd  = chunk_fd / data_dt * wdm_window[k]
                // -> applied here as one factor: v *= 0.5 * dt_sparse / dt.
                const int    half_Nt_sub  = Nt_sub / 2;
                const int    half_Nsp     = N_sparse / 2;
                const int    fft_offset   = m * half_Nt_sub - half_Nt_sub - k_f0;
                for (int c = 0; c < nchannels; ++c) {
                    for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                         k_idx += BLOCK_INCR_X) {
                        int fft_bin = fft_offset + k_idx;
                        cmplx v(0.0, 0.0);
                        if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                            int read_bin = (fft_bin + N_sparse) % N_sparse;
                            v = fd_chunk_buf[c * N_sparse + read_bin];
                            v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                            const double w = wdm_window[k_idx];
                            v = cmplx(v.real() * w, v.imag() * w);
                        }
                        layer_buf[c * Nt_sub + k_idx] = v;
                    }
                }
                CUDA_SYNC_THREADS;

                // ---- 6) iFFT per channel (in place in layer_buf, length Nt_sub) ----
                // Per-channel iFFT (FFT helper has its own trailing sync).
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                      Nt_sub, log2_Nt_sub,
                                      /*inverse=*/true, fft_scratch);
                }

                // ---- 7+8) FUSED parity + inner-product accumulator ----
                //
                // For layer m and n_loc:
                //   parity_even = ((m + n_loc) & 1) == 0
                //   sign        = ((m + 1) * n_loc) is even ? +1 : -1
                //   w_arr[c]    = kappa * sign * (parity_even ? z.real() : z.imag())
                // We compute w_arr in registers directly from layer_buf and
                // immediately accumulate against global data/invC -- no
                // separate write-back to shared memory. Skip n_locs outside
                // the active time range.
                const double kappa = 2.0 * sqrt(M_PI * dt) / (double) Nf;
                const int ind_max_t_excl = ind_min_t + Nt_active;
                for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                     n_loc += BLOCK_INCR_X) {
                    const int n_glob = n_global_lo + (n_loc - keep_lo);
                    if (n_glob < ind_min_t || n_glob >= ind_max_t_excl) continue;
                    const int n_act = n_glob - ind_min_t;

                    const bool parity_even = (((m + n_loc) & 1) == 0);
                    const double sign      = ((((m + 1) * n_loc) & 1) == 0)
                                              ? 1.0 : -1.0;
                    const double psign     = kappa * sign;

                    double w_arr[FAST_WDM_NCHANNELS_MAX] = {0.};
                    double d_arr[FAST_WDM_NCHANNELS_MAX] = {0.};
                    for (int c = 0; c < nchannels; ++c) {
                        const cmplx z = layer_buf[c * Nt_sub + n_loc];
                        const double rp = parity_even ? z.real() : z.imag();
                        w_arr[c] = psign * rp;
                        const size_t g_d = ((size_t) c * Nf_active + m_act)
                                            * Nt_active + n_act;
                        d_arr[c] = data_d[g_d];
                    }
                    if (tdi_type == TDI_XYZ) {
                        // invC is Hermitian (and real): 3 diag + 3 off-diag
                        // unique reads. Each off-diag pair (c1, c2) with
                        // c1<c2 contributes BOTH (c1,c2) and (c2,c1) terms.
                        //   2D->2F: 3 + 3 = 6 reads instead of 9
                        //   tmp_dh: (c1,c2) + (c2,c1) = d[c1]*w[c2] + d[c2]*w[c1]
                        //   tmp_hh: w[c1]*w[c2] symmetric -> 2 * w[c1]*w[c2]
                        for (int c = 0; c < nchannels; ++c) {
                            const size_t g_inv =
                                (((size_t) c * nchannels + c)
                                   * Nf_active + m_act) * Nt_active + n_act;
                            const double inv = invC[g_inv];
                            tmp_dh += d_arr[c] * w_arr[c] * inv;
                            tmp_hh += w_arr[c] * w_arr[c] * inv;
                        }
                        for (int c1 = 0; c1 < nchannels - 1; ++c1) {
                            for (int c2 = c1 + 1; c2 < nchannels; ++c2) {
                                const size_t g_inv =
                                    (((size_t) c1 * nchannels + c2)
                                       * Nf_active + m_act) * Nt_active + n_act;
                                const double inv = invC[g_inv];
                                tmp_dh += (d_arr[c1] * w_arr[c2]
                                            + d_arr[c2] * w_arr[c1]) * inv;
                                tmp_hh += 2.0 * w_arr[c1] * w_arr[c2] * inv;
                            }
                        }
                    } else {
                        // TDI_AET / TDI_AE: invC is diagonal in channels.
                        for (int c = 0; c < nchannels; ++c) {
                            const size_t g_inv = ((size_t) c * Nf_active + m_act)
                                                  * Nt_active + n_act;
                            const double inv = invC[g_inv];
                            tmp_dh += d_arr[c] * w_arr[c] * inv;
                            tmp_hh += w_arr[c] * w_arr[c] * inv;
                        }
                    }
                }
                CUDA_SYNC_THREADS;
            } // end m_layer
        } // end chunk j

        // ---- per-thread -> shared-mem partials, block-wide tree reduction --
        partial_dh[THREAD_START_X] = tmp_dh;
        partial_hh[THREAD_START_X] = tmp_hh;
        CUDA_SYNC_THREADS;
#ifdef __CUDACC__
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (THREAD_START_X < stride) {
                partial_dh[THREAD_START_X] += partial_dh[THREAD_START_X + stride];
                partial_hh[THREAD_START_X] += partial_hh[THREAD_START_X + stride];
            }
            CUDA_SYNC_THREADS;
        }
        // One binary per block (grid.X) + chunks iterated sequentially
        // INSIDE the block means exactly one block writes to (d_h_out,
        // h_h_out)[bin_i] -- no cross-block race, so a direct store
        // suffices. If we later move chunks onto blockIdx.y (multiple
        // blocks per binary), this must become atomicAdd to combine
        // the per-chunk partials across blocks.
        if (THREAD_START_X == 0) {
            d_h_out[bin_i] = partial_dh[0];
            h_h_out[bin_i] = partial_hh[0];
        }
#else
        // CPU: blockDim.x == 1 (THREAD_START_X / BLOCK_INCR_X stubs collapse
        // to a single virtual thread), so partial_dh[0] already holds the
        // full sum for this binary.
        d_h_out[bin_i] = partial_dh[0];
        h_h_out[bin_i] = partial_hh[0];
#endif
        CUDA_SYNC_THREADS;
    } // end bin_i
}


template <class SourceT>
CUDA_KERNEL
void wdm_het_fill_global_kernel(
    double *template_fill,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, double *factors_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset,
    double *wdm_window,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha,
    int    m_band_half_width)
{
    // Same per-(chunk, m_layer) pipeline as wdm_het_get_ll_kernel:
    //   1. build sparse time grid in shared mem
    //   2. for each m_layer: compute tdi_channel + heterodyne + Tukey
    //   3. FFT length N_sparse per channel in shared mem
    //   4. window + rearrange for layer m in place (per-thread register stage)
    //   5. iFFT length Nt_sub per channel in shared mem
    //   6. parity factor -> real WDM coefficient
    //   7. atomicAdd into template_fill[c, m, n_global] (instead of the
    //      get_ll accumulator + reduction).
    // See get_ll for full design comments; the only difference is the
    // output stage at step 7.
    SourceT src(orbits, tdi_config, T, t_ref);

    const int Nf = wdm_settings->Nf;
    const int Nt = wdm_settings->Nt;

    // Dynamic shared-memory layout (set by ``shared_bytes`` at kernel launch):
    //   fd_chunk_buf [nchannels * N_sparse] cmplx  -- chunk-FD (built ONCE per chunk)
    //   layer_buf    [nchannels * Nt_sub]   cmplx  -- per-m_layer scratch
    // (no per-thread partials; fill_global writes via atomicAdd.)
#ifdef __CUDACC__
    extern CUDA_SHARED char shared_mem[];
    cmplx  *fd_chunk_buf = (cmplx *) shared_mem;
    cmplx  *layer_buf    = &fd_chunk_buf[(size_t) nchannels * N_sparse];
    // cufftdx scratch (unused unless LISA_USE_CUFFTDX is defined).
    char   *fft_scratch  = (char *) &layer_buf[(size_t) nchannels * Nt_sub];
#else
    cmplx  fd_chunk_buf_cpu[FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  layer_buf_cpu   [FAST_WDM_NCHANNELS_MAX * FAST_WDM_NT_SUB_MAX];
    cmplx  *fd_chunk_buf = fd_chunk_buf_cpu;
    cmplx  *layer_buf    = layer_buf_cpu;
    char   *fft_scratch  = nullptr;
#endif

    const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
    const double df_chunk = 1.0 / T_chunk;

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
        double *params       = &params_all[(size_t) bin_i * nparams];
        const double factor  = factors_all[bin_i];

        const double f0      = params[src.f0_index];
        const int    k_f0    = (int) round(f0 / df_chunk);
        const double f0_grid = (double) k_f0 * df_chunk;
        const int    m_floor = (int) (f0 / layer_df);
        int          m_lo    = m_floor - m_band_half_width;
        int          m_hi    = m_floor + m_band_half_width + 1;
        if (m_lo < 0)   m_lo = 0;
        if (m_hi > Nf)  m_hi = Nf;

        for (int j = 0; j < n_chunks; ++j) {
            const int    keep_lo     = chunk_keep_lo[j];
            const int    keep_hi     = chunk_keep_hi[j];
            const int    n_global_lo = chunk_n_global_offset[j];
            const double chunk_t0    = chunk_t_starts[j];
            const double dt_sparse   = T_chunk / (double) N_sparse;

            // ============================================================
            // Steps 1-4 are CHUNK-LEVEL (no m dependence): TD-build,
            // heterodyne, Tukey, forward FFT into fd_chunk_buf. Compute
            // ONCE and reuse across every m_layer below.
            // ============================================================
            {
                CUDA_SHARED int link_sc_rec[NLINKS];
                CUDA_SHARED int link_sc_em [NLINKS];
                src.fill_link_arrays(link_sc_rec, link_sc_em);
                CUDA_SYNC_THREADS;
                Vec k_sky(0.0, 0.0, 0.0);
                Vec u_sky(0.0, 0.0, 0.0);
                Vec v_sky(0.0, 0.0, 0.0);
                src.get_sky_vectors(&k_sky, &u_sky, &v_sky, params);

                // ---- 1) tdi_channel(t) -> fd_chunk_buf ----
                for (int i = THREAD_START_X; i < N_sparse;
                     i += BLOCK_INCR_X) {
                    const double t = chunk_t0 + (double) i * dt_sparse;
                    cmplx tdi_tmp[3];
                    src.get_tdi_Xf_single(&tdi_tmp[0], t, params,
                                          k_sky, u_sky, v_sky,
                                          link_sc_rec, link_sc_em, bin_i);
                    for (int c = 0; c < nchannels; ++c)
                        fd_chunk_buf[c * N_sparse + i] = tdi_tmp[c];
                }
                CUDA_SYNC_THREADS;
            }

            // ---- 2) heterodyne + 3) Tukey (in place in fd_chunk_buf) ----
            const double n_taper = (tukey_alpha > 0.0)
                ? 0.5 * tukey_alpha * (double) (N_sparse - 1) : 0.0;
            for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                 idx += BLOCK_INCR_X) {
                const int i = idx - (idx / N_sparse) * N_sparse;
                const double tau = (double) i * dt_sparse;
                const double het_phase = -2.0 * M_PI * f0_grid * tau;
                const cmplx  het_factor(cos(het_phase), sin(het_phase));
                cmplx s = gcmplx::conj(fd_chunk_buf[idx]) * het_factor;
                if (n_taper > 0.0) {
                    double w = 1.0;
                    const double di    = (double) i;
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
                fd_chunk_buf[idx] = s;
            }
            CUDA_SYNC_THREADS;

            // ---- 4) FFT length N_sparse per channel (in place) ----
            // Per-channel forward FFT. wdm_spline_radix2_fft ends with a
            // CUDA_SYNC_THREADS internally, so no inter-channel sync needed.
            for (int c = 0; c < nchannels; ++c) {
                wdm_fft_dispatch(&fd_chunk_buf[c * N_sparse],
                                  N_sparse, log2_N_sparse,
                                  /*inverse=*/false, fft_scratch);
            }
            // fd_chunk_buf now holds the chunk-FD; reuse below.

            // ============================================================
            // Steps 5-7 per m_layer: window+rearrange -> iFFT -> parity
            //                        -> atomicAdd into template_fill.
            // ============================================================
            const double scale_fd     = 0.5 * dt_sparse / dt;
            for (int m = m_lo; m < m_hi; ++m) {
                // ---- 5) window + rearrange: fd_chunk_buf -> layer_buf ----
                const int    half_Nt_sub  = Nt_sub / 2;
                const int    half_Nsp     = N_sparse / 2;
                const int    fft_offset   = m * half_Nt_sub - half_Nt_sub - k_f0;
                for (int c = 0; c < nchannels; ++c) {
                    for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                         k_idx += BLOCK_INCR_X) {
                        int fft_bin = fft_offset + k_idx;
                        cmplx v(0.0, 0.0);
                        if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                            int read_bin = (fft_bin + N_sparse) % N_sparse;
                            v = fd_chunk_buf[c * N_sparse + read_bin];
                            v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                            const double w = wdm_window[k_idx];
                            v = cmplx(v.real() * w, v.imag() * w);
                        }
                        layer_buf[c * Nt_sub + k_idx] = v;
                    }
                }
                CUDA_SYNC_THREADS;

                // ---- 6) iFFT length Nt_sub per channel (in place in layer_buf) ----
                // Per-channel iFFT (FFT helper has its own trailing sync).
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                      Nt_sub, log2_Nt_sub,
                                      /*inverse=*/true, fft_scratch);
                }

                // ---- 7) parity factor + atomicAdd into template_fill ----
                // atomicAdd required: different binaries with overlapping
                // (m, n_glob) pixels write to the same global cell.
                const double kappa = 2.0 * sqrt(M_PI * dt) / (double) Nf;
                for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                     n_loc += BLOCK_INCR_X) {
                    const int  n_glob       = n_global_lo + (n_loc - keep_lo);
                    const bool parity_even  = (((m + n_loc) & 1) == 0);
                    const double sign       = ((((m + 1) * n_loc) & 1) == 0)
                                                ? 1.0 : -1.0;
                    for (int c = 0; c < nchannels; ++c) {
                        const cmplx z      = layer_buf[c * Nt_sub + n_loc];
                        const double real_part = parity_even ? z.real() : z.imag();
                        const double w     = factor * kappa * sign * real_part;
                        const size_t dst   = ((size_t) c * Nf + m) * Nt + n_glob;
#ifdef __CUDACC__
                        atomicAdd(&template_fill[dst], w);
#else
                        template_fill[dst] += w;
#endif
                    }
                }
                CUDA_SYNC_THREADS;
            } // end m_layer
        } // end chunk j
    } // end bin_i
}


template <class SourceT>
CUDA_KERNEL
void wdm_het_swap_ll_kernel(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset,
    double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int    m_band_half_width)
{
    // Same per-(chunk, m_layer) flow as get_ll, but with TWO template builds
    // (add + rem) and 5 inner-product partials (<d|h_add>, <d|h_rem>,
    // <h_add|h_add>, <h_rem|h_rem>, <h_add|h_rem>).
    //
    // Shared memory budget (single buffer reuse, as in get_ll):
    //   tdi_channel_buf[3 * Nt_sub] cmplx   -- FFT/iFFT scratch (12 KB)
    //   t_arr_buf      [N_sparse]   double  -- sparse time grid  (2 KB)
    //   partial_dh_a/r, partial_aa, partial_rr, partial_ar
    //                  [blockDim.x] double  -- reduction (5 x 0.5 KB)
    // ~17 KB total at Nt_sub=256 / blockDim.x=64.
    //
    // Per-thread register storage for the "w_add held across rem build":
    //   w_add_reg[nchannels * K_PER_THREAD] doubles, where
    //   K_PER_THREAD = ceil(Nt_sub / blockDim.x) = 4 at GPU defaults.
    // 12 doubles = 96 bytes per thread, fits comfortably in registers.
    SourceT src(orbits, tdi_config, T, t_ref);

    const int Nf         = wdm_settings->Nf;
    const int Nt         = wdm_settings->Nt;  (void) Nt;
    const int ind_min_f  = wdm_settings->ind_min_f;
    const int ind_min_t  = wdm_settings->ind_min_t;
    const int Nf_active  = wdm_settings->Nf_active;
    const int Nt_active  = wdm_settings->Nt_active;

    // Dynamic shared-memory layout (must match shared_bytes at launch):
    //   fd_chunk_buf_a [nchannels * N_sparse] cmplx  -- add chunk-FD (built ONCE per chunk)
    //   fd_chunk_buf_r [nchannels * N_sparse] cmplx  -- rem chunk-FD (built ONCE per chunk)
    //   layer_buf      [nchannels * Nt_sub]   cmplx  -- per-m scratch (reused for add + rem)
    //   partial_dh_a / partial_dh_r / partial_aa / partial_rr / partial_ar
    //     each [blockDim.x] double
    // ~38.5 KB at Nt_sub=N_sparse=256, blockDim=64 -- 4 blocks/SM on A100.
    // The add and rem chunk-FDs are computed ONCE per chunk and reused
    // for every m_layer in the (union of) bands.
#ifdef __CUDACC__
    extern CUDA_SHARED char shared_mem[];
    cmplx  *fd_chunk_buf_a  = (cmplx *) shared_mem;
    cmplx  *fd_chunk_buf_r  = &fd_chunk_buf_a[(size_t) nchannels * N_sparse];
    cmplx  *layer_buf       = &fd_chunk_buf_r[(size_t) nchannels * N_sparse];
    double *partial_dh_a    = (double *) &layer_buf[(size_t) nchannels * Nt_sub];
    double *partial_dh_r    = &partial_dh_a[NUM_THREADS_HERE];
    double *partial_aa      = &partial_dh_r[NUM_THREADS_HERE];
    double *partial_rr      = &partial_aa  [NUM_THREADS_HERE];
    double *partial_ar      = &partial_rr  [NUM_THREADS_HERE];
    // cufftdx scratch (unused unless LISA_USE_CUFFTDX is defined).
    char   *fft_scratch     = (char *) &partial_ar[NUM_THREADS_HERE];
#else
    cmplx  fd_chunk_buf_a_cpu[FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  fd_chunk_buf_r_cpu[FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  layer_buf_cpu     [FAST_WDM_NCHANNELS_MAX * FAST_WDM_NT_SUB_MAX];
    double partial_dh_a_cpu[1], partial_dh_r_cpu[1];
    double partial_aa_cpu  [1], partial_rr_cpu  [1], partial_ar_cpu[1];
    cmplx  *fd_chunk_buf_a  = fd_chunk_buf_a_cpu;
    cmplx  *fd_chunk_buf_r  = fd_chunk_buf_r_cpu;
    cmplx  *layer_buf       = layer_buf_cpu;
    double *partial_dh_a    = partial_dh_a_cpu;
    double *partial_dh_r    = partial_dh_r_cpu;
    double *partial_aa      = partial_aa_cpu;
    double *partial_rr      = partial_rr_cpu;
    double *partial_ar      = partial_ar_cpu;
    char   *fft_scratch     = nullptr;
#endif

    const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
    const double df_chunk = 1.0 / T_chunk;

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
        double *p_add = &params_add_all   [(size_t) bin_i * nparams];
        double *p_rem = &params_remove_all[(size_t) bin_i * nparams];
        (void) data_index_all; (void) noise_index_all;

        // Per-thread accumulators.
        double tmp_dh_a = 0.0, tmp_dh_r = 0.0;
        double tmp_aa   = 0.0, tmp_rr   = 0.0, tmp_ar = 0.0;

        // Carrier bins for add and rem (each binary has its own).
        const double f0_a    = p_add[src.f0_index];
        const double f0_r    = p_rem[src.f0_index];
        const int    k_f0_a  = (int) round(f0_a / df_chunk);
        const int    k_f0_r  = (int) round(f0_r / df_chunk);
        const double f0g_a   = (double) k_f0_a * df_chunk;
        const double f0g_r   = (double) k_f0_r * df_chunk;
        // m-band -- take the union of add's and rem's narrow bands so we
        // build both templates over the same m_layer iteration.
        const int    m_floor_a = (int) (f0_a / layer_df);
        const int    m_floor_r = (int) (f0_r / layer_df);
        int          m_lo      = (m_floor_a < m_floor_r ? m_floor_a : m_floor_r) - m_band_half_width;
        int          m_hi      = (m_floor_a > m_floor_r ? m_floor_a : m_floor_r) + m_band_half_width + 1;
        if (m_lo < ind_min_f)             m_lo = ind_min_f;
        if (m_hi > ind_min_f + Nf_active) m_hi = ind_min_f + Nf_active;

        for (int j = 0; j < n_chunks; ++j) {
            const int    keep_lo     = chunk_keep_lo[j];
            const int    keep_hi     = chunk_keep_hi[j];
            const int    n_global_lo = chunk_n_global_offset[j];
            const double chunk_t0    = chunk_t_starts[j];
            const double dt_sparse   = T_chunk / (double) N_sparse;

            // ============================================================
            // CHUNK-LEVEL: build the TWO chunk-FD buffers (add + rem) ONCE
            // per chunk. None of steps 1-4 (TD-build, heterodyne, Tukey,
            // forward FFT) depend on m, so the m loop below only does the
            // per-m work (window+rearrange + iFFT + parity + accumulate).
            // ============================================================
            const double n_taper = (tukey_alpha > 0.0)
                ? 0.5 * tukey_alpha * (double) (N_sparse - 1) : 0.0;

            // ---- Build ADD chunk-FD into fd_chunk_buf_a ----
            {
                CUDA_SHARED int link_sc_rec[NLINKS];
                CUDA_SHARED int link_sc_em [NLINKS];
                src.fill_link_arrays(link_sc_rec, link_sc_em);
                CUDA_SYNC_THREADS;
                Vec k_sky(0.0, 0.0, 0.0);
                Vec u_sky(0.0, 0.0, 0.0);
                Vec v_sky(0.0, 0.0, 0.0);
                src.get_sky_vectors(&k_sky, &u_sky, &v_sky, p_add);
                for (int i = THREAD_START_X; i < N_sparse;
                     i += BLOCK_INCR_X) {
                    const double t = chunk_t0 + (double) i * dt_sparse;
                    cmplx tdi_tmp[3];
                    src.get_tdi_Xf_single(&tdi_tmp[0], t, p_add,
                                          k_sky, u_sky, v_sky,
                                          link_sc_rec, link_sc_em, bin_i);
                    for (int c = 0; c < nchannels; ++c)
                        fd_chunk_buf_a[c * N_sparse + i] = tdi_tmp[c];
                }
                CUDA_SYNC_THREADS;
            }
            for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                 idx += BLOCK_INCR_X) {
                const int i = idx - (idx / N_sparse) * N_sparse;
                const double tau = (double) i * dt_sparse;
                const double het_phase = -2.0 * M_PI * f0g_a * tau;
                const cmplx  het_factor(cos(het_phase), sin(het_phase));
                cmplx s = gcmplx::conj(fd_chunk_buf_a[idx]) * het_factor;
                if (n_taper > 0.0) {
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
                fd_chunk_buf_a[idx] = s;
            }
            CUDA_SYNC_THREADS;
            // Per-channel forward FFT for ADD (helper has trailing sync).
            for (int c = 0; c < nchannels; ++c) {
                wdm_fft_dispatch(&fd_chunk_buf_a[c * N_sparse],
                                  N_sparse, log2_N_sparse,
                                  /*inverse=*/false, fft_scratch);
            }

            // ---- Build REM chunk-FD into fd_chunk_buf_r ----
            {
                CUDA_SHARED int link_sc_rec[NLINKS];
                CUDA_SHARED int link_sc_em [NLINKS];
                src.fill_link_arrays(link_sc_rec, link_sc_em);
                CUDA_SYNC_THREADS;
                Vec k_sky(0.0, 0.0, 0.0);
                Vec u_sky(0.0, 0.0, 0.0);
                Vec v_sky(0.0, 0.0, 0.0);
                src.get_sky_vectors(&k_sky, &u_sky, &v_sky, p_rem);
                for (int i = THREAD_START_X; i < N_sparse;
                     i += BLOCK_INCR_X) {
                    const double t = chunk_t0 + (double) i * dt_sparse;
                    cmplx tdi_tmp[3];
                    src.get_tdi_Xf_single(&tdi_tmp[0], t, p_rem,
                                          k_sky, u_sky, v_sky,
                                          link_sc_rec, link_sc_em, bin_i);
                    for (int c = 0; c < nchannels; ++c)
                        fd_chunk_buf_r[c * N_sparse + i] = tdi_tmp[c];
                }
                CUDA_SYNC_THREADS;
            }
            for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                 idx += BLOCK_INCR_X) {
                const int i = idx - (idx / N_sparse) * N_sparse;
                const double tau = (double) i * dt_sparse;
                const double het_phase = -2.0 * M_PI * f0g_r * tau;
                const cmplx  het_factor(cos(het_phase), sin(het_phase));
                cmplx s = gcmplx::conj(fd_chunk_buf_r[idx]) * het_factor;
                if (n_taper > 0.0) {
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
                fd_chunk_buf_r[idx] = s;
            }
            CUDA_SYNC_THREADS;
            // Per-channel forward FFT for REM (helper has trailing sync).
            for (int c = 0; c < nchannels; ++c) {
                wdm_fft_dispatch(&fd_chunk_buf_r[c * N_sparse],
                                  N_sparse, log2_N_sparse,
                                  /*inverse=*/false, fft_scratch);
            }

            // ============================================================
            // PER-M-LAYER: for each m, read fd_chunk_buf_a -> layer_buf,
            // iFFT, parity into w_add_reg. Then read fd_chunk_buf_r ->
            // layer_buf (overwrites), iFFT, parity -> w_r per thread, then
            // accumulate 5 partials using w_add_reg + w_r.
            // ============================================================
            const double scale_fd     = 0.5 * dt_sparse / dt;
            const int    half_Nt_sub  = Nt_sub / 2;
            const int    half_Nsp     = N_sparse / 2;
            const double kappa        = 2.0 * sqrt(M_PI * dt) / (double) Nf;
            constexpr int K_MAX_REG   = FAST_WDM_K_PER_THREAD_MAX;
            for (int m = m_lo; m < m_hi; ++m) {
                const int m_act        = m - ind_min_f;
                const int fft_offset_a = m * half_Nt_sub - half_Nt_sub - k_f0_a;
                const int fft_offset_r = m * half_Nt_sub - half_Nt_sub - k_f0_r;

                // ---- PHASE 1: ADD layer (fd_chunk_buf_a -> layer_buf -> w_add_reg) ----
                for (int c = 0; c < nchannels; ++c) {
                    for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                         k_idx += BLOCK_INCR_X) {
                        int fft_bin = fft_offset_a + k_idx;
                        cmplx v(0.0, 0.0);
                        if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                            int read_bin = (fft_bin + N_sparse) % N_sparse;
                            v = fd_chunk_buf_a[c * N_sparse + read_bin];
                            v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                            const double w = wdm_window[k_idx];
                            v = cmplx(v.real() * w, v.imag() * w);
                        }
                        layer_buf[c * Nt_sub + k_idx] = v;
                    }
                }
                CUDA_SYNC_THREADS;
                // Per-channel iFFT (FFT helper has its own trailing sync).
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                      Nt_sub, log2_Nt_sub,
                                      /*inverse=*/true, fft_scratch);
                }
                double w_add_reg[FAST_WDM_NCHANNELS_MAX * K_MAX_REG];
                {
                    int k_idx_reg = 0;
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const bool parity_even = (((m + n_loc) & 1) == 0);
                        const double sign      = ((((m + 1) * n_loc) & 1) == 0)
                                                  ? 1.0 : -1.0;
                        for (int c = 0; c < nchannels; ++c) {
                            const cmplx z = layer_buf[c * Nt_sub + n_loc];
                            const double rp = parity_even ? z.real() : z.imag();
                            w_add_reg[c * K_MAX_REG + k_idx_reg] = kappa * sign * rp;
                        }
                        ++k_idx_reg;
                    }
                }
                CUDA_SYNC_THREADS;

                // ---- PHASE 2: REM layer (fd_chunk_buf_r -> layer_buf), accumulate ----
                for (int c = 0; c < nchannels; ++c) {
                    for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                         k_idx += BLOCK_INCR_X) {
                        int fft_bin = fft_offset_r + k_idx;
                        cmplx v(0.0, 0.0);
                        if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                            int read_bin = (fft_bin + N_sparse) % N_sparse;
                            v = fd_chunk_buf_r[c * N_sparse + read_bin];
                            v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                            const double w = wdm_window[k_idx];
                            v = cmplx(v.real() * w, v.imag() * w);
                        }
                        layer_buf[c * Nt_sub + k_idx] = v;
                    }
                }
                CUDA_SYNC_THREADS;
                // Per-channel iFFT (FFT helper has its own trailing sync).
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                      Nt_sub, log2_Nt_sub,
                                      /*inverse=*/true, fft_scratch);
                }

                // ---- Accumulate 5 partials using w_add_reg + freshly-parity'd w_r ----
                const int ind_max_t_excl = ind_min_t + Nt_active;
                {
                    int k_idx_reg = 0;
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const int n_glob = n_global_lo + (n_loc - keep_lo);
                        if (n_glob >= ind_min_t && n_glob < ind_max_t_excl) {
                            const int n_act = n_glob - ind_min_t;
                            const bool parity_even = (((m + n_loc) & 1) == 0);
                            const double sign      = ((((m + 1) * n_loc) & 1) == 0)
                                                      ? 1.0 : -1.0;
                            double w_a_arr[FAST_WDM_NCHANNELS_MAX];
                            double w_r_arr[FAST_WDM_NCHANNELS_MAX];
                            double d_arr  [FAST_WDM_NCHANNELS_MAX];
                            for (int c = 0; c < nchannels; ++c) {
                                w_a_arr[c] = w_add_reg[c * K_MAX_REG + k_idx_reg];
                                const cmplx z = layer_buf[c * Nt_sub + n_loc];
                                const double rp = parity_even ? z.real() : z.imag();
                                w_r_arr[c] = kappa * sign * rp;
                                const size_t g_d = ((size_t) c * Nf_active + m_act)
                                                    * Nt_active + n_act;
                                d_arr[c] = data_d[g_d];
                            }
                            if (tdi_type == TDI_XYZ) {
                                // Symmetric invC: 3 diag + 3 off-diag reads.
                                // tmp_ar is non-symmetric in (a, r) so we
                                // sum both (c1,c2) and (c2,c1) contributions.
                                for (int c = 0; c < nchannels; ++c) {
                                    const size_t g_inv =
                                        (((size_t) c * nchannels + c)
                                          * Nf_active + m_act)
                                          * Nt_active + n_act;
                                    const double inv = invC[g_inv];
                                    tmp_dh_a += d_arr[c]   * w_a_arr[c] * inv;
                                    tmp_dh_r += d_arr[c]   * w_r_arr[c] * inv;
                                    tmp_aa   += w_a_arr[c] * w_a_arr[c] * inv;
                                    tmp_rr   += w_r_arr[c] * w_r_arr[c] * inv;
                                    tmp_ar   += w_a_arr[c] * w_r_arr[c] * inv;
                                }
                                for (int c1 = 0; c1 < nchannels - 1; ++c1) {
                                    for (int c2 = c1 + 1; c2 < nchannels; ++c2) {
                                        const size_t g_inv =
                                            (((size_t) c1 * nchannels + c2)
                                              * Nf_active + m_act)
                                              * Nt_active + n_act;
                                        const double inv = invC[g_inv];
                                        tmp_dh_a += (d_arr[c1]   * w_a_arr[c2]
                                                      + d_arr[c2]   * w_a_arr[c1]) * inv;
                                        tmp_dh_r += (d_arr[c1]   * w_r_arr[c2]
                                                      + d_arr[c2]   * w_r_arr[c1]) * inv;
                                        tmp_aa   += 2.0 * w_a_arr[c1] * w_a_arr[c2] * inv;
                                        tmp_rr   += 2.0 * w_r_arr[c1] * w_r_arr[c2] * inv;
                                        tmp_ar   += (w_a_arr[c1] * w_r_arr[c2]
                                                      + w_a_arr[c2] * w_r_arr[c1]) * inv;
                                    }
                                }
                            } else {
                                for (int c = 0; c < nchannels; ++c) {
                                    const size_t g_inv = ((size_t) c * Nf_active + m_act)
                                                          * Nt_active + n_act;
                                    const double inv = invC[g_inv];
                                    tmp_dh_a += d_arr[c]   * w_a_arr[c] * inv;
                                    tmp_dh_r += d_arr[c]   * w_r_arr[c] * inv;
                                    tmp_aa   += w_a_arr[c] * w_a_arr[c] * inv;
                                    tmp_rr   += w_r_arr[c] * w_r_arr[c] * inv;
                                    tmp_ar   += w_a_arr[c] * w_r_arr[c] * inv;
                                }
                            }
                        }
                        ++k_idx_reg;
                    }
                }
                CUDA_SYNC_THREADS;
            } // end m_layer
        } // end chunk j

        // ---- per-thread -> shared partials -> block tree reduction ----
        partial_dh_a[THREAD_START_X] = tmp_dh_a;
        partial_dh_r[THREAD_START_X] = tmp_dh_r;
        partial_aa  [THREAD_START_X] = tmp_aa;
        partial_rr  [THREAD_START_X] = tmp_rr;
        partial_ar  [THREAD_START_X] = tmp_ar;
        CUDA_SYNC_THREADS;
#ifdef __CUDACC__
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (THREAD_START_X < stride) {
                partial_dh_a[THREAD_START_X] += partial_dh_a[THREAD_START_X + stride];
                partial_dh_r[THREAD_START_X] += partial_dh_r[THREAD_START_X + stride];
                partial_aa  [THREAD_START_X] += partial_aa  [THREAD_START_X + stride];
                partial_rr  [THREAD_START_X] += partial_rr  [THREAD_START_X + stride];
                partial_ar  [THREAD_START_X] += partial_ar  [THREAD_START_X + stride];
            }
            CUDA_SYNC_THREADS;
        }
        // See wdm_het_get_ll_kernel: one binary per block + sequential
        // chunks inside the block -> no cross-block race on these per-binary
        // outputs, so a direct store suffices. Switch to atomicAdd if/when
        // chunks move to blockIdx.y.
        if (THREAD_START_X == 0) {
            d_h_add_out      [bin_i] = partial_dh_a[0];
            d_h_remove_out   [bin_i] = partial_dh_r[0];
            add_add_out      [bin_i] = partial_aa  [0];
            remove_remove_out[bin_i] = partial_rr  [0];
            add_remove_out   [bin_i] = partial_ar  [0];
        }
#else
        d_h_add_out      [bin_i] = partial_dh_a[0];
        d_h_remove_out   [bin_i] = partial_dh_r[0];
        add_add_out      [bin_i] = partial_aa  [0];
        remove_remove_out[bin_i] = partial_rr  [0];
        add_remove_out   [bin_i] = partial_ar  [0];
#endif
        CUDA_SYNC_THREADS;
    } // end bin_i
}


template <class SourceT>
CUDA_KERNEL
void wdm_het_get_fstat_ll_kernel(
    double *N_arr_re_out, double *N_arr_im_out,   // (num_bin, 4) per-binary <d|A_i>
    double *M_mat_re_out, double *M_mat_im_out,   // (num_bin, 10) per-binary <A_i|A_j>
                                                   // (Hermitian; 4 diag + 6 upper)
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset,
    double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int    m_band_half_width)
{
    // F-stat: build 4 basis waveforms per Cornish & Crowder '05 with fixed
    //   (A, iota, psi, phi0) = (2, pi/2, {0, pi/4, 0, pi/4}, {0, pi, 3pi/2, pi/2})
    // For each (chunk, m_layer, filter_i) we build w_i, stage in per-thread
    // registers, then once all 4 are staged we accumulate:
    //   N_arr[bin_i, i]   = sum_pixels    sum_{c1,c2}  d[c1] * w_i[c2] * invC[c1,c2]
    //   M_mat[bin_i, ij]  = sum_pixels    sum_{c1,c2}  w_i[c1] * w_j[c2] * invC[c1,c2]
    // where ij flattens the upper-triangle (i<=j) of the 4x4 Hermitian.
    //
    // WDM coefficients are real -> N and M are real (imag outputs always 0).
    //
    // GB param convention (matches existing chunked-het):
    //   params[0] = A      params[1] = f0   params[2] = fdot  params[3] = fddot
    //   params[4] = phi0   params[5] = iota params[6] = psi
    //   params[7] = lam    params[8] = beta
    SourceT src(orbits, tdi_config, T, t_ref);

    const int Nf         = wdm_settings->Nf;
    const int Nt         = wdm_settings->Nt;  (void) Nt;
    const int ind_min_f  = wdm_settings->ind_min_f;
    const int ind_min_t  = wdm_settings->ind_min_t;
    const int Nf_active  = wdm_settings->Nf_active;
    const int Nt_active  = wdm_settings->Nt_active;

    // F-stat basis filter parameters (Cornish & Crowder '05).
    constexpr int   N_FILTERS  = 4;
    const double A_arr    [N_FILTERS] = {2.0, 2.0, 2.0, 2.0};
    const double iota_arr [N_FILTERS] = {M_PI / 2.0, M_PI / 2.0,
                                          M_PI / 2.0, M_PI / 2.0};
    const double psi_arr  [N_FILTERS] = {0.0, M_PI / 4.0, 0.0, M_PI / 4.0};
    const double phi0_arr [N_FILTERS] = {0.0, M_PI, 3.0 * M_PI / 2.0, M_PI / 2.0};

    // GB param indices (constants for the GB convention; SOBBH would need
    // a trait-based specialization).
    constexpr int IDX_A    = 0;
    constexpr int IDX_PHI0 = 4;
    constexpr int IDX_IOTA = 5;
    constexpr int IDX_PSI  = 6;

    // Dynamic shared-memory layout (must match shared_bytes at launch):
    //   fd_chunk_buf[fi=0..3][nchannels * N_sparse] cmplx
    //                                              -- per-filter chunk-FD,
    //                                                 built ONCE per chunk
    //   layer_buf   [nchannels * Nt_sub]           cmplx -- per-(m, fi) scratch
    //   partial_N   [N_FILTERS  * blockDim.x]      double  ( 4 * NTH)
    //   partial_M   [N_M_PART   * blockDim.x]      double  (10 * NTH)
    // ~67 KB at Nt_sub=N_sparse=256, blockDim=64. Exceeds the 48 KB default
    // limit -- the launcher calls cudaFuncSetAttribute(...,
    // cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes) to raise it.
#ifdef __CUDACC__
    extern CUDA_SHARED char shared_mem[];
    cmplx  *fd_chunk_buf[N_FILTERS];
    fd_chunk_buf[0] = (cmplx *) shared_mem;
    for (int fi_p = 1; fi_p < N_FILTERS; ++fi_p)
        fd_chunk_buf[fi_p] = &fd_chunk_buf[fi_p - 1][(size_t) nchannels * N_sparse];
    cmplx  *layer_buf = &fd_chunk_buf[N_FILTERS - 1][(size_t) nchannels * N_sparse];
    // 4 N + 10 M = 14 partial buffers, each blockDim.x wide.
    double *partial_N = (double *) &layer_buf[(size_t) nchannels * Nt_sub];
    double *partial_M = &partial_N[(size_t) N_FILTERS * NUM_THREADS_HERE];
    // cufftdx scratch (unused unless LISA_USE_CUFFTDX is defined).
    char   *fft_scratch = (char *) &partial_M[(size_t) ((N_FILTERS * (N_FILTERS + 1)) / 2)
                                              * NUM_THREADS_HERE];
#else
    cmplx  fd_chunk_buf_cpu[N_FILTERS][FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  layer_buf_cpu   [FAST_WDM_NCHANNELS_MAX * FAST_WDM_NT_SUB_MAX];
    double partial_N_cpu   [N_FILTERS];
    double partial_M_cpu   [(N_FILTERS * (N_FILTERS + 1)) / 2];
    cmplx  *fd_chunk_buf[N_FILTERS];
    for (int fi_p = 0; fi_p < N_FILTERS; ++fi_p) fd_chunk_buf[fi_p] = fd_chunk_buf_cpu[fi_p];
    cmplx  *layer_buf       = layer_buf_cpu;
    double *partial_N       = partial_N_cpu;
    double *partial_M       = partial_M_cpu;
    char   *fft_scratch     = nullptr;
#endif

    constexpr int N_M_PARTIALS = (N_FILTERS * (N_FILTERS + 1)) / 2;  // = 10

    // (i, j) -> flat upper-triangle index for the 4x4 Hermitian M.
    // ij = i * N_FILTERS - (i*(i+1))/2 + j   for i <= j
    auto m_idx = [] (int i, int j) -> int {
        return i * N_FILTERS - (i * (i + 1)) / 2 + j;
    };

    const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
    const double df_chunk = 1.0 / T_chunk;
    // Narrow band width is configurable via the ``m_band_half_width`` arg
    // (default 1 -> 3 layers, set in the impl wrapper). All 4 het kernels
    // use the same convention.
    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
        double *params      = &params_all[(size_t) bin_i * nparams];
        const int data_ind  = data_index_all [bin_i];  (void) data_ind;
        const int noise_ind = noise_index_all[bin_i];  (void) noise_ind;

        // Per-thread accumulators.
        double tmp_N[N_FILTERS] = {0.0, 0.0, 0.0, 0.0};
        double tmp_M[N_M_PARTIALS];
        for (int k = 0; k < N_M_PARTIALS; ++k) tmp_M[k] = 0.0;

        const double f0      = params[src.f0_index];  // = params[IDX_F0] for GB
        const int    k_f0    = (int) round(f0 / df_chunk);
        const double f0_grid = (double) k_f0 * df_chunk;
        const int    m_floor = (int) (f0 / layer_df);
        int          m_lo    = m_floor - m_band_half_width;
        int          m_hi    = m_floor + m_band_half_width + 1;     // exclusive
        if (m_lo < ind_min_f)             m_lo = ind_min_f;
        if (m_hi > ind_min_f + Nf_active) m_hi = ind_min_f + Nf_active;

        for (int j = 0; j < n_chunks; ++j) {
            const int    keep_lo     = chunk_keep_lo[j];
            const int    keep_hi     = chunk_keep_hi[j];
            const int    n_global_lo = chunk_n_global_offset[j];
            const double chunk_t0    = chunk_t_starts[j];
            const double dt_sparse   = T_chunk / (double) N_sparse;

            // ============================================================
            // CHUNK-LEVEL: build all 4 chunk-FD buffers (one per basis
            // filter). Steps 1-3 (TD-build, heterodyne, Tukey, FFT) do
            // NOT depend on m, so they happen ONCE per chunk per filter
            // -- m loop below only does the per-m work (window+rearrange
            // -> iFFT -> parity -> accumulate).
            // ============================================================
            const double n_taper = (tukey_alpha > 0.0)
                ? 0.5 * tukey_alpha * (double) (N_sparse - 1) : 0.0;
            for (int fi_b = 0; fi_b < N_FILTERS; ++fi_b) {
                double params_basis[16];   // GB has 9; bound generously
                for (int k = 0; k < nparams && k < 16; ++k) {
                    params_basis[k] = params[k];
                }
                params_basis[IDX_A   ] = A_arr   [fi_b];
                params_basis[IDX_IOTA] = iota_arr[fi_b];
                params_basis[IDX_PSI ] = psi_arr [fi_b];
                params_basis[IDX_PHI0] = phi0_arr[fi_b];

                // ---- 1) TD-build into fd_chunk_buf[fi_b] ----
                {
                    CUDA_SHARED int link_sc_rec[NLINKS];
                    CUDA_SHARED int link_sc_em [NLINKS];
                    src.fill_link_arrays(link_sc_rec, link_sc_em);
                    CUDA_SYNC_THREADS;
                    Vec k_sky(0.0, 0.0, 0.0);
                    Vec u_sky(0.0, 0.0, 0.0);
                    Vec v_sky(0.0, 0.0, 0.0);
                    src.get_sky_vectors(&k_sky, &u_sky, &v_sky, params_basis);
                    for (int i = THREAD_START_X; i < N_sparse;
                         i += BLOCK_INCR_X) {
                        const double t = chunk_t0 + (double) i * dt_sparse;
                        cmplx tdi_tmp[3];
                        src.get_tdi_Xf_single(&tdi_tmp[0], t, params_basis,
                                              k_sky, u_sky, v_sky,
                                              link_sc_rec, link_sc_em, bin_i);
                        for (int c = 0; c < nchannels; ++c)
                            fd_chunk_buf[fi_b][c * N_sparse + i] = tdi_tmp[c];
                    }
                    CUDA_SYNC_THREADS;
                }

                // ---- 2) heterodyne + Tukey (in place in fd_chunk_buf[fi_b]) ----
                for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                     idx += BLOCK_INCR_X) {
                    const int i = idx - (idx / N_sparse) * N_sparse;
                    const double tau = (double) i * dt_sparse;
                    const double het_phase = -2.0 * M_PI * f0_grid * tau;
                    const cmplx  het_factor(cos(het_phase), sin(het_phase));
                    cmplx s = gcmplx::conj(fd_chunk_buf[fi_b][idx]) * het_factor;
                    if (n_taper > 0.0) {
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
                    fd_chunk_buf[fi_b][idx] = s;
                }
                CUDA_SYNC_THREADS;

                // ---- 3) FFT length N_sparse per channel (helper has trailing sync) ----
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&fd_chunk_buf[fi_b][c * N_sparse],
                                      N_sparse, log2_N_sparse,
                                      /*inverse=*/false, fft_scratch);
                }
            } // end build-FD per filter
            // All 4 chunk-FDs now resident in shared mem; reuse across m below.

            // ============================================================
            // PER-M-LAYER: for each m, build all 4 basis layer values and
            // accumulate 4 N + 10 M partials.
            // ============================================================
            const double scale_fd    = 0.5 * dt_sparse / dt;
            const int    half_Nt_sub = Nt_sub / 2;
            const int    half_Nsp    = N_sparse / 2;
            const double kappa       = 2.0 * sqrt(M_PI * dt) / (double) Nf;
            constexpr int K_MAX_REG  = FAST_WDM_K_PER_THREAD_MAX;
            for (int m = m_lo; m < m_hi; ++m) {
                const int m_act = m - ind_min_f;
                // Per-thread storage for THIS m's 4 basis WDM coefs.
                double w_basis_reg[N_FILTERS * FAST_WDM_NCHANNELS_MAX * K_MAX_REG];
                const int fft_offset = m * half_Nt_sub - half_Nt_sub - k_f0;

                // Build each of the 4 basis waveforms' layer at this m.
                for (int fi = 0; fi < N_FILTERS; ++fi) {
                    // ---- 4) window + rearrange: fd_chunk_buf[fi] -> layer_buf ----
                    for (int c = 0; c < nchannels; ++c) {
                        for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                             k_idx += BLOCK_INCR_X) {
                            int fft_bin = fft_offset + k_idx;
                            cmplx v(0.0, 0.0);
                            if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                                int read_bin = (fft_bin + N_sparse) % N_sparse;
                                v = fd_chunk_buf[fi][c * N_sparse + read_bin];
                                v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                                const double w = wdm_window[k_idx];
                                v = cmplx(v.real() * w, v.imag() * w);
                            }
                            layer_buf[c * Nt_sub + k_idx] = v;
                        }
                    }
                    CUDA_SYNC_THREADS;

                    // ---- 5) iFFT length Nt_sub per channel (helper has trailing sync) ----
                    for (int c = 0; c < nchannels; ++c) {
                        wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                          Nt_sub, log2_Nt_sub,
                                          /*inverse=*/true, fft_scratch);
                    }

                    // ---- 6) parity factor; stage w_i[c, n_loc] in regs ----
                    int k_idx_reg = 0;
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const bool parity_even = (((m + n_loc) & 1) == 0);
                        const double sign      = ((((m + 1) * n_loc) & 1) == 0)
                                                  ? 1.0 : -1.0;
                        for (int c = 0; c < nchannels; ++c) {
                            const cmplx z  = layer_buf[c * Nt_sub + n_loc];
                            const double rp = parity_even ? z.real() : z.imag();
                            w_basis_reg[(fi * nchannels + c) * K_MAX_REG
                                          + k_idx_reg] = kappa * sign * rp;
                        }
                        ++k_idx_reg;
                    }
                    CUDA_SYNC_THREADS;
                } // end filter fi

                // ---- Accumulate 4 N partials + 10 M partials per pixel ----
                const int ind_max_t_excl = ind_min_t + Nt_active;
                {
                    int k_idx_reg = 0;
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const int n_glob = n_global_lo + (n_loc - keep_lo);
                        if (n_glob >= ind_min_t && n_glob < ind_max_t_excl) {
                            const int n_act = n_glob - ind_min_t;
                            // Read data for all channels once.
                            double d_arr[FAST_WDM_NCHANNELS_MAX];
                            for (int c = 0; c < nchannels; ++c) {
                                const size_t g_d = ((size_t) c * Nf_active + m_act)
                                                    * Nt_active + n_act;
                                d_arr[c] = data_d[g_d];
                            }
                            // 4 N partials: <d | A_i>
                            for (int fi = 0; fi < N_FILTERS; ++fi) {
                                double sum_dh = 0.0;
                                if (tdi_type == TDI_XYZ) {
                                    // Symmetric invC: 3 diag + 3 off-diag reads.
                                    for (int c = 0; c < nchannels; ++c) {
                                        const size_t g_inv =
                                            (((size_t) c * nchannels + c)
                                              * Nf_active + m_act)
                                              * Nt_active + n_act;
                                        const double inv = invC[g_inv];
                                        const double w_i =
                                            w_basis_reg[(fi * nchannels + c)
                                                          * K_MAX_REG
                                                          + k_idx_reg];
                                        sum_dh += d_arr[c] * w_i * inv;
                                    }
                                    for (int c1 = 0; c1 < nchannels - 1; ++c1) {
                                        for (int c2 = c1 + 1; c2 < nchannels; ++c2) {
                                            const size_t g_inv =
                                                (((size_t) c1 * nchannels + c2)
                                                  * Nf_active + m_act)
                                                  * Nt_active + n_act;
                                            const double inv = invC[g_inv];
                                            const double w_c1 =
                                                w_basis_reg[(fi * nchannels + c1)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            const double w_c2 =
                                                w_basis_reg[(fi * nchannels + c2)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            sum_dh += (d_arr[c1] * w_c2
                                                        + d_arr[c2] * w_c1) * inv;
                                        }
                                    }
                                } else {
                                    for (int c = 0; c < nchannels; ++c) {
                                        const size_t g_inv = ((size_t) c * Nf_active
                                                                + m_act)
                                                              * Nt_active + n_act;
                                        const double inv = invC[g_inv];
                                        const double w_i =
                                            w_basis_reg[(fi * nchannels + c)
                                                          * K_MAX_REG
                                                          + k_idx_reg];
                                        sum_dh += d_arr[c] * w_i * inv;
                                    }
                                }
                                tmp_N[fi] += sum_dh;
                            }
                            // 10 M partials: <A_i | A_j> for i <= j
                            for (int fi = 0; fi < N_FILTERS; ++fi) {
                                for (int fj = fi; fj < N_FILTERS; ++fj) {
                                    double sum_hh = 0.0;
                                    if (tdi_type == TDI_XYZ) {
                                        // Symmetric invC. Note: w_i (filter fi)
                                        // and w_j (filter fj) are NOT
                                        // interchangeable when fi != fj, so
                                        // off-diag terms sum both orderings.
                                        for (int c = 0; c < nchannels; ++c) {
                                            const size_t g_inv =
                                                (((size_t) c * nchannels + c)
                                                  * Nf_active + m_act)
                                                  * Nt_active + n_act;
                                            const double inv = invC[g_inv];
                                            const double w_i =
                                                w_basis_reg[(fi * nchannels + c)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            const double w_j =
                                                w_basis_reg[(fj * nchannels + c)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            sum_hh += w_i * w_j * inv;
                                        }
                                        for (int c1 = 0; c1 < nchannels - 1; ++c1) {
                                            for (int c2 = c1 + 1; c2 < nchannels; ++c2) {
                                                const size_t g_inv =
                                                    (((size_t) c1 * nchannels + c2)
                                                      * Nf_active + m_act)
                                                      * Nt_active + n_act;
                                                const double inv = invC[g_inv];
                                                const double w_i_c1 =
                                                    w_basis_reg[(fi * nchannels + c1)
                                                                  * K_MAX_REG
                                                                  + k_idx_reg];
                                                const double w_i_c2 =
                                                    w_basis_reg[(fi * nchannels + c2)
                                                                  * K_MAX_REG
                                                                  + k_idx_reg];
                                                const double w_j_c1 =
                                                    w_basis_reg[(fj * nchannels + c1)
                                                                  * K_MAX_REG
                                                                  + k_idx_reg];
                                                const double w_j_c2 =
                                                    w_basis_reg[(fj * nchannels + c2)
                                                                  * K_MAX_REG
                                                                  + k_idx_reg];
                                                sum_hh += (w_i_c1 * w_j_c2
                                                            + w_i_c2 * w_j_c1) * inv;
                                            }
                                        }
                                    } else {
                                        for (int c = 0; c < nchannels; ++c) {
                                            const size_t g_inv = ((size_t) c
                                                                    * Nf_active
                                                                    + m_act)
                                                                  * Nt_active + n_act;
                                            const double inv = invC[g_inv];
                                            const double w_i =
                                                w_basis_reg[(fi * nchannels + c)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            const double w_j =
                                                w_basis_reg[(fj * nchannels + c)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            sum_hh += w_i * w_j * inv;
                                        }
                                    }
                                    tmp_M[m_idx(fi, fj)] += sum_hh;
                                }
                            }
                        }
                        ++k_idx_reg;
                    }
                }
                CUDA_SYNC_THREADS;
            } // end m_layer
        } // end chunk j

        // ---- per-thread -> shared partials -> block tree reduction ----
        for (int fi = 0; fi < N_FILTERS; ++fi) {
            partial_N[fi * NUM_THREADS_HERE + THREAD_START_X] = tmp_N[fi];
        }
        for (int k = 0; k < N_M_PARTIALS; ++k) {
            partial_M[k * NUM_THREADS_HERE + THREAD_START_X] = tmp_M[k];
        }
        CUDA_SYNC_THREADS;
#ifdef __CUDACC__
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (THREAD_START_X < stride) {
                for (int fi = 0; fi < N_FILTERS; ++fi) {
                    partial_N[fi * NUM_THREADS_HERE + THREAD_START_X] +=
                        partial_N[fi * NUM_THREADS_HERE + THREAD_START_X + stride];
                }
                for (int k = 0; k < N_M_PARTIALS; ++k) {
                    partial_M[k * NUM_THREADS_HERE + THREAD_START_X] +=
                        partial_M[k * NUM_THREADS_HERE + THREAD_START_X + stride];
                }
            }
            CUDA_SYNC_THREADS;
        }
        // See wdm_het_get_ll_kernel: one binary per block + sequential
        // chunks inside the block -> no cross-block race on per-binary N
        // and M outputs, so direct stores suffice. Switch to atomicAdd
        // if/when chunks move to blockIdx.y.
        if (THREAD_START_X == 0) {
            for (int fi = 0; fi < N_FILTERS; ++fi) {
                N_arr_re_out[bin_i * N_FILTERS + fi] =
                    partial_N[fi * NUM_THREADS_HERE];
                // WDM coefficients are real -> imag part is exactly 0.
                N_arr_im_out[bin_i * N_FILTERS + fi] = 0.0;
            }
            for (int k = 0; k < N_M_PARTIALS; ++k) {
                M_mat_re_out[bin_i * N_M_PARTIALS + k] =
                    partial_M[k * NUM_THREADS_HERE];
                M_mat_im_out[bin_i * N_M_PARTIALS + k] = 0.0;
            }
        }
#else
        for (int fi = 0; fi < N_FILTERS; ++fi) {
            N_arr_re_out[bin_i * N_FILTERS + fi] = partial_N[fi];
            N_arr_im_out[bin_i * N_FILTERS + fi] = 0.0;
        }
        for (int k = 0; k < N_M_PARTIALS; ++k) {
            M_mat_re_out[bin_i * N_M_PARTIALS + k] = partial_M[k];
            M_mat_im_out[bin_i * N_M_PARTIALS + k] = 0.0;
        }
#endif
        CUDA_SYNC_THREADS;
    } // end bin_i
}


// =============================================================================
// NEW impl wrappers (paired with the NEW kernels at line ~2520 above).
//
// Signatures kept compatible with the existing public ``gb_wdm_het_*_wrap``
// signatures so the Python ABI is unchanged. The args we no longer use
// (layer-group args, N_cp_sig, N_cp_orbit, grid_dim) are accepted and
// ignored / forwarded as comments below.
// =============================================================================
template <class SourceT>
inline void wdm_het_fill_global_impl(
    double *template_fill,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, double *factors_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset,
    double *wdm_window,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha,
    int grid_dim, int N_cp_sig, int N_cp_orbit,
    int m_band_half_width)
{
    // The new kernel does not use N_cp_sig / N_cp_orbit (no spline / orbit
    // caches in this rewrite). grid_dim selects gridDim.x (binaries per
    // launch); default = num_bin (one block per binary).
    (void) N_cp_sig; (void) N_cp_orbit;
#ifdef __CUDACC__
    // One binary per block on grid.X (always). The grid_dim arg is
    // vestigial -- kept in the signature for ABI stability with the
    // older callers, but the kernel does not grid-stride over binaries.
    (void) grid_dim;
    const int gd_x = num_bin;
    // Shared-mem layout (must match wdm_het_fill_global_kernel):
    //   fd_chunk_buf [nchannels * N_sparse] cmplx  -- chunk-FD (built ONCE per chunk)
    //   layer_buf    [nchannels * Nt_sub]   cmplx  -- per-m_layer scratch
    //   fft_scratch  [wdm_cufftdx_max_scratch()] bytes  -- 0 unless cufftdx is on
    // (no per-thread partials -- fill_global writes directly via atomicAdd.)
    const size_t shared_bytes =
        (size_t) nchannels * (size_t) N_sparse * sizeof(cmplx) +
        (size_t) nchannels * (size_t) Nt_sub   * sizeof(cmplx) +
        wdm_cufftdx_max_scratch();

    // Upload host-side wrapper structs (Orbits / TDIConfig / WDMSettings) to
    // device. Cache the device-side pointers across calls.
    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),       cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings), cudaMemcpyHostToDevice));

    // TODO: when chunks move to gridDim.y, change to dim3(gd_x, n_chunks, 1).
    dim3 grid((unsigned) gd_x, 1u, 1u);
    wdm_het_fill_global_kernel<SourceT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        template_fill, orbits_gpu, tdi_config_gpu, wdm_settings_gpu,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tukey_alpha, m_band_half_width);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    (void) grid_dim;
    wdm_het_fill_global_kernel<SourceT>(
        template_fill, orbits, tdi_config, wdm_settings,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tukey_alpha, m_band_half_width);
#endif
}


template <class SourceT>
inline void wdm_het_get_ll_impl(
    double *d_h_out, double *h_h_out,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset,
    double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int m_band_half_width)
{
    // New kernel does not use group-grouping path: each block handles one
    // binary and determines its own narrow m-band internally. Layer-grouping
    // can be reintroduced later as a perf optimization.
    // N_cp_sig still unused; N_cp_orbit now wired into the kernel for the
    // shared-mem orbit spline cache (raw orbits when 0).
    (void) N_cp_sig;
    (void) binary_perm; (void) group_starts; (void) group_ends;
    (void) group_m_lo; (void) group_m_hi; (void) n_groups;
#ifdef __CUDACC__
    // One binary per block on grid.X (always). The grid_dim arg is
    // vestigial -- kept in the signature for ABI stability with the
    // older callers, but the kernel does not grid-stride over binaries.
    (void) grid_dim;
    const int gd_x = num_bin;
    // Shared-mem layout (must match wdm_het_get_ll_kernel):
    //   fd_chunk_buf [nchannels * N_sparse] cmplx  -- chunk-FD (built ONCE per chunk)
    //   layer_buf    [nchannels * Nt_sub]   cmplx  -- per-m_layer scratch
    //   partial_dh   [blockDim.x]           double
    //   partial_hh   [blockDim.x]           double
    //   fft_scratch  [wdm_cufftdx_max_scratch()] bytes  -- 0 unless cufftdx is on
    const size_t shared_bytes =
        (size_t) nchannels * (size_t) N_sparse * sizeof(cmplx) +
        (size_t) nchannels * (size_t) Nt_sub   * sizeof(cmplx) +
        (size_t) 2 * (size_t) NUM_THREADS_HERE * sizeof(double) +
        wdm_cufftdx_max_scratch();

    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),       cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings), cudaMemcpyHostToDevice));

    // When the kernel's orbit spline cache is enabled, its static shared
    // mem footprint grows by ~26 KB (the orbit_*_buf set). Total can exceed
    // the 48 KB default on sm_70+; opt in to the larger per-block max via
    // cudaFuncSetAttribute before launch.
    if (N_cp_orbit > 0) {
        gpuErrchk(cudaFuncSetAttribute(
            wdm_het_get_ll_kernel<SourceT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    }

    dim3 grid((unsigned) gd_x, 1u, 1u);
    wdm_het_get_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        d_h_out, h_h_out, orbits_gpu, tdi_config_gpu, wdm_settings_gpu,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width,
        N_cp_orbit);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    (void) grid_dim;
    wdm_het_get_ll_kernel<SourceT>(
        d_h_out, h_h_out, orbits, tdi_config, wdm_settings,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width,
        N_cp_orbit);
#endif
}


template <class SourceT>
inline void wdm_het_swap_ll_impl(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset,
    double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int *pair_m_lo_b, int *pair_m_hi_b,
    int m_band_half_width)
{
    (void) N_cp_sig; (void) N_cp_orbit;
    (void) binary_perm; (void) group_starts; (void) group_ends;
    (void) group_m_lo; (void) group_m_hi; (void) n_groups;
    (void) pair_m_lo_b; (void) pair_m_hi_b;
#ifdef __CUDACC__
    // One binary per block on grid.X (always). The grid_dim arg is
    // vestigial -- kept in the signature for ABI stability with the
    // older callers, but the kernel does not grid-stride over binaries.
    (void) grid_dim;
    const int gd_x = num_bin;
    // Shared-mem layout (must match wdm_het_swap_ll_kernel):
    //   fd_chunk_buf_a [nchannels * N_sparse] cmplx  -- add chunk-FD
    //   fd_chunk_buf_r [nchannels * N_sparse] cmplx  -- rem chunk-FD
    //   layer_buf      [nchannels * Nt_sub]   cmplx  -- per-m scratch (reused)
    //   5 * blockDim.x doubles (dh_a, dh_r, aa, rr, ar partial-sum buffers)
    //   fft_scratch    [wdm_cufftdx_max_scratch()] bytes -- 0 unless cufftdx is on
    const size_t shared_bytes =
        (size_t) 2 * (size_t) nchannels * (size_t) N_sparse * sizeof(cmplx) +
        (size_t) nchannels * (size_t) Nt_sub * sizeof(cmplx) +
        (size_t) 5 * (size_t) NUM_THREADS_HERE * sizeof(double) +
        wdm_cufftdx_max_scratch();

    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),       cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) gd_x, 1u, 1u);
    wdm_het_swap_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits_gpu, tdi_config_gpu, wdm_settings_gpu,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    (void) grid_dim;
    wdm_het_swap_ll_kernel<SourceT>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config, wdm_settings,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width);
#endif
}


template <class SourceT>
inline void wdm_het_get_fstat_ll_impl(
    double *N_arr_re_out, double *N_arr_im_out,   // (num_bin, 4)
    double *M_mat_re_out, double *M_mat_im_out,   // (num_bin, 10)
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset,
    double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int grid_dim,
    int m_band_half_width)
{
#ifdef __CUDACC__
    // One binary per block on grid.X (always). The grid_dim arg is
    // vestigial -- kept in the signature for ABI stability with the
    // older callers, but the kernel does not grid-stride over binaries.
    (void) grid_dim;
    const int gd_x = num_bin;
    // Shared-mem layout (must match wdm_het_get_fstat_ll_kernel):
    //   fd_chunk_buf[fi=0..3][nchannels * N_sparse] cmplx -- per-filter chunk-FD
    //   layer_buf            [nchannels * Nt_sub]   cmplx -- per-(m, fi) scratch
    //   partial_N            [ 4 * blockDim.x]      double (4 basis filters)
    //   partial_M            [10 * blockDim.x]      double (upper-tri of 4x4 M)
    //   fft_scratch          [wdm_cufftdx_max_scratch()] bytes -- 0 unless cufftdx is on
    // ~67 KB at Nt_sub=N_sparse=256, blockDim=64 -- exceeds default 48 KB
    // limit, so we raise the cap via cudaFuncSetAttribute below.
    constexpr int N_FILTERS_LAUNCH = 4;
    const size_t shared_bytes =
        (size_t) N_FILTERS_LAUNCH * (size_t) nchannels * (size_t) N_sparse * sizeof(cmplx) +
        (size_t) nchannels * (size_t) Nt_sub * sizeof(cmplx) +
        (size_t) 14        * (size_t) NUM_THREADS_HERE * sizeof(double) +
        wdm_cufftdx_max_scratch();
    if (shared_bytes > 48u * 1024u) {
        gpuErrchk(cudaFuncSetAttribute(
            wdm_het_get_fstat_ll_kernel<SourceT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int) shared_bytes));
    }

    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),       cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) gd_x, 1u, 1u);
    wdm_het_get_fstat_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        N_arr_re_out, N_arr_im_out, M_mat_re_out, M_mat_im_out,
        orbits_gpu, tdi_config_gpu, wdm_settings_gpu,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    (void) grid_dim;
    wdm_het_get_fstat_ll_kernel<SourceT>(
        N_arr_re_out, N_arr_im_out, M_mat_re_out, M_mat_im_out,
        orbits, tdi_config, wdm_settings,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width);
#endif
}


#endif  // LAT_CHUNKED_HET_KERNELS_HH
