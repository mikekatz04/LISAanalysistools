#ifndef LAT_CHUNKED_HET_KERNELS_HH
#define LAT_CHUNKED_HET_KERNELS_HH

// ============================================================================
// Chunked-heterodyne kernel ABI constants + shared-memory POD layout.
//
// This header owns the FAST_WDM_* sizing macros and the WDMHet*Bufs
// shared-memory layout PODs consumed by the chunked-heterodyne
// (`wdm_het_fill_global_kernel` / `wdm_het_get_ll_kernel` /
// `wdm_het_swap_ll_kernel` / `wdm_het_get_fstat_ll_kernel`) templated
// kernel family.
//
// **Status (Phase 3L.7a slice 1, 2026-06-04):** macros + PODs only. The
// templated kernel bodies + `*_impl<SourceT>` host launchers + the
// `fast_wdm_inner_heterodyne_*` and `gb_chunk_fd_to_wdm` helpers still
// live in lisa-on-gpu's `cutils/TDIonTheFly.cu`; they migrate here in
// Slices 2 + 3 of Phase 3L.7a. Once the kernels move here too, the
// GBGPU and BBHx pybind11 modules can instantiate
// `wdm_het_*_impl<GBTDIonTheFly>` / `wdm_het_*_impl<SOBBHTDIonTheFly>`
// against their respective source classes by `#include`-ing this
// header, with no need to link against lisa-on-gpu (which is being
// retired -- see lisa-on-gpu's CLAUDE.md deprecation notice).
//
// **Sprint rule -- no aliasing needed.** `WDMHetDirectBufs` and
// `WDMHetSplineBufs` are file-static shared-memory layouts that never
// escape a translation unit (no pybind11 binding, no exported symbol).
// See LAT CLAUDE.md "CPU/GPU class-name aliasing" rule, point 5.
//
// **Dependencies.** Pulls in `cmplx` via `gbt_global.h` (transitive
// through `global.hpp`). `FAST_WDM_K_PER_THREAD_MAX` is intentionally
// NOT defined here because it references the host-file's
// `NUM_THREADS_HERE` macro (a file-scoped per-kernel block-size knob);
// it will move into this header alongside the templated kernels in
// Slice 2.
// ============================================================================

#include "global.hpp"  // -> gbt_global.h -> cmplx typedef + CUDA macros


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


#endif  // LAT_CHUNKED_HET_KERNELS_HH
