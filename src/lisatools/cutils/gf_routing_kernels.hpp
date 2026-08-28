#ifndef __GF_ROUTING_KERNELS_HPP__
#define __GF_ROUTING_KERNELS_HPP__

// ============================================================================
// gf_routing_kernels -- fused ORCHESTRATION kernels for the global-fit GB
// in-model repeat step (2026-08-27 orchestration audit, candidate 1).
//
// WHAT THIS TU IS FOR. The GB in-model repeat loop
// (``lisatools.globalfit.moves.gbspecialstretch.GBSpecialBase.
// _run_in_model_repeats``) is not compute-bound: at production scale it pays
// ~110-150 SEPARATE array-library launches per repeat step purely on
// bookkeeping -- masks, gates, MH arithmetic, masked state writes and
// scatter-adds -- around ONE real scoring call (``get_add_ll``). At 2e4-7e4
// repeat steps per row that overhead alone is order 1e2 s/row.
//
// The two entry points below fuse that bookkeeping into 3 launches total:
//
//   gb_inmodel_gate_compact  -- the PRE-score chain: f0-window mask,
//       sig-het trust-region gate, cap-drift-gate veto (both the exact
//       partition and the overlapping-cell membership variants), all writing
//       -1e300-flavoured -inf into ``new_logp``; then an ascending, stable
//       device compaction of the surviving rows into ``keep_idx`` plus the
//       count. Python reads that ONE count to size the scoring call -- the
//       single data-dependent host pull the loop is allowed.
//
//   gb_inmodel_accept_apply  -- the POST-score chain: scatter of the scored
//       likelihoods, the SNR prior-boundary clamp, the Metropolis-Hastings
//       ratio and accept decision (with the uniform draws SUPPLIED from
//       python so the RNG stream is untouched), and every masked state write
//       that follows an accept: coords, ll_ref, prior, the per-cell
//       ll-change/proposal/acceptance ledgers, the cap-gate occupancy
//       transition, the sorter <d|h>/<h|h> stashes and the census counters.
//
// WHAT DELIBERATELY STAYS IN PYTHON: the RNG draws, the prior logpdf, the
// parameter ``both_transforms``, eryn's ``periodic.wrap``, the proposal
// itself and the phase-maximization write-back. Those are either
// python-object-shaped or would change the RNG/wrap semantics; the kernels
// take their OUTPUTS as inputs instead.
//
// BACKEND CONTRACT (LISA Analysis Tools-wide rule): the GPU/CUDA build is the
// reference implementation; the CPU build compiles this SAME source (CMake
// copies ``gf_routing_kernels.cu`` to ``gf_routing_kernels.cxx``) with the
// grid-stride macros collapsed to a plain serial loop, and must agree to
// machine precision -- in fact bit-for-bit, since every operation here is
// either integer or a same-order floating-point expression. There is NO
// OpenMP in this file (sprint-wide no-nested-OpenMP rule).
//
// ORDERING / DETERMINISM NOTE. The per-cell scatter-adds use ``atomicAdd`` on
// the GPU. The GB scheduler's serial-within-band guarantee means at most ONE
// picked row per (temp, walker, band) cell is live in a batch, so no two
// lanes ever target the same ledger slot and the atomics are plain adds --
// the float results are order-independent in practice. The python call site
// asserts that uniqueness behind ``GB_INDEX_ASSERTS``.
//
// This TU exposes FREE FUNCTIONS ONLY -- no classes escape it, so the
// sprint's CPU/GPU class-name aliasing rule does not apply here (rule 5:
// only types whose symbols reach the .so's exported interface need aliases).
// ============================================================================

// gbt_global.h brings in the CUDA_KERNEL / CUDA_SHARED / CUDA_SYNC_THREADS /
// THREAD_ZERO macro set and gpuErrchk. There is one sprint-wide copy, owned
// by GPUBackendTools -- do not add a separate cuda_complex.hpp include.
#include "gbt_global.h"
#include <cstdint>

// ---------------------------------------------------------------------------
// Sentinels -- these MUST match the python constants they replace.
// ---------------------------------------------------------------------------
// ``new_logp`` rejections are written as -inf in python
// (``new_logp[mask] = -np.inf``) and detected downstream by
// ``~xp.isinf(new_logp)``; we write the same -inf.
// ``new_ll`` rejections are the -1e300 floor python fills, and the
// out-of-prior "bad" test uses the -1e299 / -1e229 thresholds verbatim.
#define GF_LL_FLOOR (-1e300)
#define GF_BAD_LL_THRESH (-1e299)
#define GF_BAD_LOGP_THRESH (-1e229)

// ---------------------------------------------------------------------------
// gb_inmodel_gate_compact_wrap
// ---------------------------------------------------------------------------
// Fuses the pre-score gate chain and compacts the survivors.
//
// Row addressing. The repeat loop may run over a PARITY HALF of the picked
// pool (eryn red-blue split): ``row_map[i]`` is the index of sub-row ``i``
// inside the full block-width arrays (``curr_coords``, ``anchor_*``,
// ``trust_*``). Pass ``row_map = nullptr`` for the full-batch path, where the
// mapping is the identity.
//
// Outputs. ``new_logp`` is modified IN PLACE (-inf into rejected rows).
// ``keep_flag[i]`` is 1 where the row survives (== ``~isinf(new_logp)``).
// ``keep_idx[0 .. n_keep_out[0])`` holds the surviving row indices in
// ASCENDING order -- identical to ``xp.where(keep)[0]``. ``keep_pos`` is the
// INVERSE map (``keep_pos[i]`` = position of row ``i`` in the scored subset,
// or -1); the accept kernel indexes the scored arrays through it so the whole
// post-score chain stays a single flat loop over sub-rows. ``cur_cells`` and
// ``new_cells`` are 3*n_sub scratch buffers laid out as
// ``[primary (n_sub) | neighbour (n_sub) | has_neighbour (n_sub)]``; the
// accept kernel consumes them so the cap-cell arithmetic runs once.
//
// Optional stages. ``window_on``, ``pc != nullptr`` (trust gate) and
// ``dg_on`` (cap drift gate) each switch a stage off when unset, matching
// the python ``if`` guards exactly. Counter pointers may be null.
void gb_inmodel_gate_compact_wrap(
    // ---- outputs / in-out ----
    double *new_logp,        // (n_sub)     in/out
    uint8_t *keep_flag,      // (n_sub)     out
    int64_t *keep_idx,       // (n_sub)     out
    int64_t *n_keep_out,     // (1)         out
    int32_t *cur_cells,      // (3*n_sub)   out (scratch for accept_apply)
    int32_t *new_cells,      // (3*n_sub)   out
    int64_t *trust_counts,   // (3)  in/out, or null: [dlnA, dphase, either]
    int64_t *dg_count,       // (1)  in/out, or null: cap-gate vetoes
    // ---- f0 window stage ----
    const double *new_coords,   // (n_sub, ndim)   sampling basis
    const double *curr_coords,  // (n_block, ndim) sampling basis
    const int32_t *row_map,     // (n_sub) or null => identity
    const int32_t *n4,          // (n_sub) half-width in FD bins
    const int32_t *lo_bin,      // (n_sub) band-window low edge, FD bins
    const int32_t *hi_bin,      // (n_sub) band-window high edge, FD bins
    int f0_col, int ndim, double df, int window_on,
    // ---- sig-het trust-region stage ----
    const double *pc,           // (n_sub, pc_ncol) both_transforms output, or null
    int pc_ncol,
    const double *anchor_amp,   // (n_block) |A| at the heterodyne anchor
    const double *anchor_f0,    // (n_block)
    const double *anchor_fdot,  // (n_block)
    const double *trust_dlna,   // (n_block) per-source amplitude gate
    const double *trust_dphase, // (n_block) per-source phase gate
    double trust_Tobs,
    // ---- cap drift gate stage ----
    int dg_on, int overlap_on,
    const int32_t *temp_inds,   // (n_sub)
    const int32_t *walker_inds, // (n_sub)
    const int32_t *band_inds,   // (n_sub)
    const double *cap_band_lo,   // (num_bands)
    const double *cap_band_step, // (num_bands)
    const double *cap_edges,     // (num_cap_cells + 1)
    const double *cap_edge_ext,  // (num_cap_cells + 1) or null
    const int32_t *dg_counts,    // (ntemps*nwalkers*num_cap_cells)
    const int32_t *dg_cap,       // (num_cap_cells)
    int cap_divisor, int cap_stagger, int num_cap_cells, int nwalkers,
    // ---- sizes ----
    int n_sub, int n_block,
    int32_t *keep_pos);      // (n_sub) out: inverse of keep_idx, -1 elsewhere

// ---------------------------------------------------------------------------
// gb_inmodel_accept_apply_wrap
// ---------------------------------------------------------------------------
// Fuses the post-score chain: ll scatter, SNR clamp, MH ratio + accept, and
// every masked state write / scatter-add that follows.
//
// ``u`` are the uniform (0, 1] draws, generated in python so the RNG stream
// and its consumption order are untouched: the kernel evaluates
// ``lnpdiff >= log(u)`` exactly as the python did.
//
// ``accept_pre`` is the accept mask BEFORE the out-of-prior "bad" filter, and
// ``accept`` is the final one; python's trace hooks read the pre-filter mask
// (they are called between the two statements), so both are returned.
//
// ``d_h`` / ``h_h`` may be strided views of complex arrays: pass
// ``dh_stride = 2`` (and the base pointer of the complex buffer viewed as
// double) to read the real parts, or 1 for a plain real array. This mirrors
// python's ``cp.asarray(buffer_obj.h_h_out).real`` without materializing a
// contiguous copy.
void gb_inmodel_accept_apply_wrap(
    // ---- in/out state ----
    double *new_ll,          // (n_sub)   in/out (pre-filled with GF_LL_FLOOR)
    double *delta_ll,        // (n_sub)   out
    double *lnpdiff,         // (n_sub)   out
    uint8_t *accept_pre,     // (n_sub)   out (before the bad-mask filter)
    uint8_t *accept,         // (n_sub)   out (final)
    double *curr_coords,     // (n_block, ndim) in/out
    double *ll_ref,          // (n_block)       in/out
    double *curr_prior,      // (n_block)       in/out
    // ---- scored subset ----
    const double *scored_ll, // (n_keep) or null when n_keep == 0
    const int64_t *keep_idx, // (n_keep)
    int n_keep,
    const double *d_h,       // (n_keep * dh_stride) or null
    const double *h_h,       // (n_keep * hh_stride) or null
    int dh_stride, int hh_stride,
    double snr_limit, int snr_detected,
    // ---- MH inputs ----
    const double *new_coords, // (n_sub, ndim)
    const double *new_logp,   // (n_sub)
    const double *factors,    // (n_sub)
    const double *beta_s,     // (n_sub)
    const double *u,          // (n_sub) uniform draws from python
    const int32_t *row_map,   // (n_sub) or null => identity
    int ndim,
    // ---- per-cell ledgers ----
    const int32_t *temp_inds,   // (n_sub)
    const int32_t *walker_inds, // (n_sub)
    const int32_t *band_inds,   // (n_sub)
    const uint8_t *cold_s,      // (n_sub) temp_inds == 0
    double *ll_change_log,      // (ntemps, nwalkers, num_bands)
    int64_t *prop_counts1,      // (ntemps, nwalkers, num_bands)
    int64_t *acc_counts1,       // (ntemps, nwalkers, num_bands)
    int nwalkers, int num_bands,
    int64_t *warn_count,        // (1)  out-of-prior accepts at beta > 0
    int64_t *kind_acc,          // (2) or null: [accepted, cold-accepted]
    // ---- sorter <d|h>/<h|h> stash ----
    double *sorter_dh,          // (n_src) or null
    double *sorter_hh,          // (n_src) or null
    const int32_t *ids_s,       // (n_sub) sorter source ids
    // ---- cap drift gate occupancy ----
    int dg_on, int overlap_on,
    int32_t *dg_counts,         // (ntemps*nwalkers*num_cap_cells)
    const int32_t *cur_cells,   // (3*n_sub) from gb_inmodel_gate_compact
    const int32_t *new_cells,   // (3*n_sub)
    int num_cap_cells,
    // ---- sizes ----
    int n_sub, int n_block,
    const int32_t *keep_pos);   // (n_sub) from gb_inmodel_gate_compact

#endif  // __GF_ROUTING_KERNELS_HPP__
