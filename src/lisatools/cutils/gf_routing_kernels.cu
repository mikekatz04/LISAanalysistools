// gf_routing_kernels.cu -- fused orchestration kernels for the global-fit GB
// in-model repeat step. See gf_routing_kernels.hpp for the full rationale.
//
// This ONE source compiles twice: by nvcc for the GPU backend, and (after
// CMake copies it to gf_routing_kernels.cxx) by the host C++ compiler for the
// CPU backend. Every kernel body is written once; the grid-stride macros
// collapse to a plain serial loop on CPU. No OpenMP (sprint-wide rule).

#include "gf_routing_kernels.hpp"
#include "gbt_global.h"

#include <cmath>
#include <cstdint>

// ============================================================================
// Thread-count knobs. Following the compute_logpdf idiom in PSD.cu: the CPU
// build collapses the block to a single thread so CUDA_SHARED arrays (which
// become plain stack arrays there) size correctly.
// ============================================================================
#ifdef __CUDACC__
#define GF_NUM_THREADS 256
#define GF_SCAN_THREADS 256
#else
#define GF_NUM_THREADS 1
#define GF_SCAN_THREADS 1
#endif

// ============================================================================
// Small device helpers
// ============================================================================

/** Truncation toward zero, matching numpy/cupy ``.astype(int)`` on a float. */
static CUDA_CALLABLE_MEMBER int64_t gf_trunc_i64(double v) {
  return static_cast<int64_t>(v);
}

/** ``atomicAdd`` on a 64-bit signed counter; plain add on the CPU build. */
static CUDA_CALLABLE_MEMBER void gf_atomic_add_i64(int64_t *p, int64_t v) {
#ifdef __CUDA_ARCH__
  // DEVICE compilation pass only: atomicAdd is __device__, and __CUDACC__
  // is ALSO defined during nvcc's HOST pass (repo rule, bbhx ddbe414) --
  // guarding builtins in CUDA_CALLABLE_MEMBER code takes __CUDA_ARCH__.
  atomicAdd(reinterpret_cast<unsigned long long *>(p),
            static_cast<unsigned long long>(v));
#else
  *p += v;
#endif
}

/** ``atomicAdd`` on a 32-bit signed counter; plain add on the CPU build. */
static CUDA_CALLABLE_MEMBER void gf_atomic_add_i32(int32_t *p, int32_t v) {
#ifdef __CUDA_ARCH__  // device pass only (see gf_atomic_add_i64)
  atomicAdd(p, v);
#else
  *p += v;
#endif
}

/** ``atomicAdd`` on a double; plain add on the CPU build. */
static CUDA_CALLABLE_MEMBER void gf_atomic_add_f64(double *p, double v) {
#ifdef __CUDA_ARCH__  // device pass only (see gf_atomic_add_i64)
  atomicAdd(p, v);
#else
  *p += v;
#endif
}

/**
 * @brief Cap-cell index of one source. Mirrors ``_cap_cell_index``.
 *
 * Nested grid: containment makes this pure per-source arithmetic. Staggered
 * grid: same arithmetic with a half-cell offset, no per-band clip, and a
 * global clip into [0, num_cap_cells - 1]. At ``cap_divisor == 1`` (and no
 * stagger) the cell IS the band, which is the python short-circuit.
 */
static CUDA_CALLABLE_MEMBER int32_t gf_cap_cell_index(
    int32_t band, double f_hz, const double *cap_band_lo,
    const double *cap_band_step, int cap_divisor, int cap_stagger,
    int num_cap_cells) {
  // Python short-circuits on ``_cap_is_band_grid`` == (cap_divisor == 1
  // AND NOT stagger) -- match that exactly. K=1 WITH stagger is the
  // midpoint-to-midpoint grid: the cell COUNT still equals the band count
  // but membership is shifted by half a sub-band, so a band index is NOT a
  // cell index and the general formula below must run. If this predicate
  // ever drifts from the python one the disagreement is SILENT -- the two
  // sides simply census a source into different cells.
  if (cap_divisor == 1 && !cap_stagger) {
    return band;
  }
  double sub = floor((f_hz - cap_band_lo[band]) / cap_band_step[band] +
                     (cap_stagger ? 0.5 : 0.0));
  if (cap_stagger) {
    int64_t cell = static_cast<int64_t>(band) * cap_divisor +
                   static_cast<int64_t>(sub);
    if (cell < 0) cell = 0;
    if (cell > num_cap_cells - 1) cell = num_cap_cells - 1;
    return static_cast<int32_t>(cell);
  }
  // Python clips the FLOAT then truncates; ``sub`` is already integral here
  // so the two orders agree exactly.
  if (sub < 0.0) sub = 0.0;
  if (sub > static_cast<double>(cap_divisor - 1)) {
    sub = static_cast<double>(cap_divisor - 1);
  }
  return static_cast<int32_t>(static_cast<int64_t>(band) * cap_divisor +
                              static_cast<int64_t>(sub));
}

/**
 * @brief ``(primary, neighbour, has_neighbour)`` membership. Mirrors
 *        ``_cap_cell_members``.
 *
 * With overlap off, ``has_nb`` is 0 and ``nb`` equals ``primary`` (harmless),
 * which is exactly what the python returns as ``(primary, None, None)`` and
 * what every downstream expression degenerates to.
 */
static CUDA_CALLABLE_MEMBER void gf_cap_cell_members(
    int32_t band, double f_hz, const double *cap_band_lo,
    const double *cap_band_step, const double *cap_edges,
    const double *cap_edge_ext, int cap_divisor, int cap_stagger,
    int num_cap_cells, int overlap_on, int32_t *primary_out, int32_t *nb_out,
    int32_t *has_nb_out) {
  int32_t primary = gf_cap_cell_index(band, f_hz, cap_band_lo, cap_band_step,
                                      cap_divisor, cap_stagger, num_cap_cells);
  *primary_out = primary;
  if (!overlap_on || cap_edges == nullptr || cap_edge_ext == nullptr) {
    *nb_out = primary;
    *has_nb_out = 0;
    return;
  }
  double e_lo = cap_edges[primary];
  double e_hi = cap_edges[primary + 1];
  double x_lo = cap_edge_ext[primary];
  double x_hi = cap_edge_ext[primary + 1];
  bool low = f_hz < (e_lo + x_lo);
  bool high = f_hz > (e_hi - x_hi);
  int32_t nb = low ? (primary - 1) : (high ? (primary + 1) : primary);
  bool has_nb = low || high;
  // Defensive only: the ctor sets the end-edge extensions to 0, so ``low``
  // is unreachable at cell 0 and ``high`` at the last cell -- the python
  // relies on exactly that. Clamping here can never change a result the
  // python would produce; it only turns a would-be out-of-bounds device
  // read into a no-op if that invariant is ever broken.
  if (nb < 0 || nb >= num_cap_cells) {
    nb = primary;
    has_nb = false;
  }
  *has_nb_out = has_nb ? 1 : 0;
  *nb_out = nb;
}

/** Flat ``(temp, walker, cap cell)`` occupancy index. Mirrors ``_cap_flat_index``. */
static CUDA_CALLABLE_MEMBER int64_t gf_cap_flat(int32_t t, int32_t w,
                                                int32_t cell, int nwalkers,
                                                int num_cap_cells) {
  return (static_cast<int64_t>(t) * nwalkers + w) *
             static_cast<int64_t>(num_cap_cells) +
         cell;
}

/** Membership test ``(_c == p) | (has_nb & (_c == nb))`` from the python. */
static CUDA_CALLABLE_MEMBER bool gf_in_set(int32_t cell, int32_t p, int32_t nb,
                                           int32_t has_nb) {
  return (cell == p) || (has_nb && (cell == nb));
}

// ============================================================================
// KERNEL 1a: the pre-score gate chain.
// ============================================================================
CUDA_KERNEL
void gb_inmodel_gate_kernel(
    double *new_logp, uint8_t *keep_flag, int32_t *cur_cells,
    int32_t *new_cells, int64_t *trust_counts, int64_t *dg_count,
    const double *new_coords, const double *curr_coords,
    const int32_t *row_map, const int32_t *n4, const int32_t *lo_bin,
    const int32_t *hi_bin, int f0_col, int ndim, double df, int window_on,
    const double *pc, int pc_ncol, const double *anchor_amp,
    const double *anchor_f0, const double *anchor_fdot,
    const double *trust_dlna, const double *trust_dphase, double trust_Tobs,
    int dg_on, int overlap_on, const int32_t *temp_inds,
    const int32_t *walker_inds, const int32_t *band_inds,
    const double *cap_band_lo, const double *cap_band_step,
    const double *cap_edges, const double *cap_edge_ext,
    const int32_t *dg_counts, const int32_t *dg_cap, int cap_divisor,
    int cap_stagger, int num_cap_cells, int nwalkers, int n_sub) {
#ifdef __CUDACC__
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int incr = gridDim.x * blockDim.x;
#else
  int start = 0;
  int incr = 1;
#endif

  // Constant-folded exactly as python folds them: ``2.0 * np.pi`` and
  // ``np.pi`` are python floats multiplied into the array first, and
  // ``trust_Tobs ** 2`` is a scalar computed once.
  const double two_pi = 2.0 * M_PI;
  const double Tobs2 = trust_Tobs * trust_Tobs;

  for (int i = start; i < n_sub; i += incr) {
    int32_t r = (row_map == nullptr) ? static_cast<int32_t>(i) : row_map[i];
    bool rejected = false;

    // ---- stage 1: the +-N/4 f0 window and the band window --------------
    // Skipped when f0 is a per-leaf fill (not sampled): the proposal
    // cannot move it, which is the python ``self._f0_col is not None`` gate.
    if (window_on) {
      double f_new = new_coords[static_cast<int64_t>(i) * ndim + f0_col];
      double f_cur = curr_coords[static_cast<int64_t>(r) * ndim + f0_col];
      int64_t new_bin = gf_trunc_i64(fabs(f_new / 1e3 / df));
      int64_t step_bins = gf_trunc_i64(fabs(f_new / 1e3 - f_cur / 1e3) / df);
      if (step_bins > static_cast<int64_t>(n4[i])) rejected = true;
      if (new_bin < static_cast<int64_t>(lo_bin[i]) - n4[i]) rejected = true;
      if (new_bin > static_cast<int64_t>(hi_bin[i]) + n4[i]) rejected = true;
    }

    // ---- stage 2: the sig-het TRUST REGION -----------------------------
    // Physical |dlnA| and accumulated carrier-phase gates around the block
    // anchor. The census accumulates unconditionally (the python keeps it
    // knob-free: gated rows are written as -inf priors and would otherwise
    // be indistinguishable from an ordinary prior rejection).
    if (pc != nullptr) {
      double a0 = pc[static_cast<int64_t>(i) * pc_ncol + 0];
      double a1 = pc[static_cast<int64_t>(i) * pc_ncol + 1];
      double a2 = pc[static_cast<int64_t>(i) * pc_ncol + 2];
      double damp = fabs(log(fabs(a0) / anchor_amp[r]));
      double drift = (two_pi * fabs(a1 - anchor_f0[r])) * trust_Tobs +
                     (M_PI * fabs(a2 - anchor_fdot[r])) * Tobs2;
      bool rej_a = damp > trust_dlna[r];
      bool rej_p = drift > trust_dphase[r];
      if (rej_a || rej_p) rejected = true;
      if (trust_counts != nullptr) {
        if (rej_a) gf_atomic_add_i64(&trust_counts[0], 1);
        if (rej_p) gf_atomic_add_i64(&trust_counts[1], 1);
        if (rej_a || rej_p) gf_atomic_add_i64(&trust_counts[2], 1);
      }
    }

    // ---- stage 3: the CAP DRIFT GATE veto ------------------------------
    // Also stashes the covering-cell sets so the accept kernel can apply
    // the occupancy transition without redoing the arithmetic.
    if (dg_on) {
      double f_new_hz =
          new_coords[static_cast<int64_t>(i) * ndim + f0_col] / 1e3;
      double f_cur_hz =
          curr_coords[static_cast<int64_t>(r) * ndim + f0_col] / 1e3;
      int32_t c_p, c_nb, c_hn, n_p, n_nb, n_hn;
      gf_cap_cell_members(band_inds[i], f_cur_hz, cap_band_lo, cap_band_step,
                          cap_edges, cap_edge_ext, cap_divisor, cap_stagger,
                          num_cap_cells, overlap_on, &c_p, &c_nb, &c_hn);
      gf_cap_cell_members(band_inds[i], f_new_hz, cap_band_lo, cap_band_step,
                          cap_edges, cap_edge_ext, cap_divisor, cap_stagger,
                          num_cap_cells, overlap_on, &n_p, &n_nb, &n_hn);
      cur_cells[i] = c_p;
      cur_cells[n_sub + i] = c_nb;
      cur_cells[2 * n_sub + i] = c_hn;
      new_cells[i] = n_p;
      new_cells[n_sub + i] = n_nb;
      new_cells[2 * n_sub + i] = n_hn;

      int32_t t = temp_inds[i];
      int32_t w = walker_inds[i];
      bool veto = false;
      if (overlap_on) {
        // Membership-SET form: a proposal is vetoed when any cell it NEWLY
        // enters (a covering cell of the new f0 that does not cover the
        // current f0) is armed and at cap. Cells already covered never veto.
        for (int s = 0; s < 2; ++s) {
          int32_t cell = (s == 0) ? n_p : n_nb;
          bool memb = (s == 0) ? true : (n_hn != 0);
          bool foreign =
              memb && (cell != c_p) && ((c_hn == 0) || (cell != c_nb));
          if (!foreign) continue;
          int32_t cap = dg_cap[cell];
          if (cap < 0) continue;
          int64_t flat = gf_cap_flat(t, w, cell, nwalkers, num_cap_cells);
          if (dg_counts[flat] >= cap) veto = true;
        }
      } else {
        if (n_p != c_p) {
          int32_t cap = dg_cap[n_p];
          if (cap >= 0) {
            int64_t flat = gf_cap_flat(t, w, n_p, nwalkers, num_cap_cells);
            if (dg_counts[flat] >= cap) veto = true;
          }
        }
      }
      if (veto) {
        rejected = true;
        if (dg_count != nullptr) gf_atomic_add_i64(&dg_count[0], 1);
      }
    }

    if (rejected) new_logp[i] = -INFINITY;
    // ``keep = ~xp.isinf(new_logp)`` -- note this drops +inf too, exactly as
    // the python does (a prior logpdf never produces one).
    keep_flag[i] = isinf(new_logp[i]) ? 0 : 1;
  }
}

// ============================================================================
// KERNEL 1b: stable ascending compaction of the keep mask.
//
// ONE block, tiled Hillis-Steele scan in shared memory. The output order is
// ascending, identical to ``xp.where(keep)[0]``, which matters: every
// downstream gather (the scoring call's row set, the scatter back) must see
// the same rows in the same order for the result to be bit-identical.
//
// On the CPU build GF_SCAN_THREADS is 1, the inner scan loop does not
// execute, and this degenerates to the obvious serial compaction.
// ============================================================================
CUDA_KERNEL
void gb_inmodel_compact_kernel(const uint8_t *keep_flag, int64_t *keep_idx,
                               int32_t *keep_pos, int64_t *n_keep_out,
                               int n_sub) {
  CUDA_SHARED int64_t s_scan[GF_SCAN_THREADS];
  CUDA_SHARED int64_t s_base;
#ifdef __CUDACC__
  int tid = threadIdx.x;
#else
  int tid = 0;
#endif
  if (THREAD_ZERO) s_base = 0;
  CUDA_SYNC_THREADS;

  for (int tile = 0; tile < n_sub; tile += GF_SCAN_THREADS) {
    int i = tile + tid;
    int64_t flag = (i < n_sub && keep_flag[i]) ? 1 : 0;
    s_scan[tid] = flag;
    CUDA_SYNC_THREADS;
    for (int off = 1; off < GF_SCAN_THREADS; off <<= 1) {
      int64_t v = (tid >= off) ? s_scan[tid - off] : 0;
      CUDA_SYNC_THREADS;
      s_scan[tid] += v;
      CUDA_SYNC_THREADS;
    }
    if (flag) {
      int64_t slot = s_base + s_scan[tid] - 1;
      keep_idx[slot] = i;
      keep_pos[i] = static_cast<int32_t>(slot);
    } else if (i < n_sub) {
      keep_pos[i] = -1;
    }
    CUDA_SYNC_THREADS;
    if (THREAD_ZERO) s_base += s_scan[GF_SCAN_THREADS - 1];
    CUDA_SYNC_THREADS;
  }
  if (THREAD_ZERO) n_keep_out[0] = s_base;
}

// ============================================================================
// KERNEL 2: the post-score accept + bookkeeping chain.
// ============================================================================
CUDA_KERNEL
void gb_inmodel_accept_kernel(
    double *new_ll, double *delta_ll, double *lnpdiff, uint8_t *accept_pre,
    uint8_t *accept, double *curr_coords, double *ll_ref, double *curr_prior,
    const double *scored_ll, const int32_t *keep_pos, const double *d_h,
    const double *h_h, int dh_stride, int hh_stride, double snr_limit,
    int snr_detected, const double *new_coords, const double *new_logp,
    const double *factors, const double *beta_s, const double *u,
    const int32_t *row_map, int ndim, const int32_t *temp_inds,
    const int32_t *walker_inds, const int32_t *band_inds,
    const uint8_t *cold_s, double *ll_change_log, int64_t *prop_counts1,
    int64_t *acc_counts1, int nwalkers, int num_bands, int64_t *warn_count,
    int64_t *kind_acc, double *sorter_dh, double *sorter_hh,
    const int32_t *ids_s, int dg_on, int overlap_on, int32_t *dg_counts,
    const int32_t *cur_cells, const int32_t *new_cells, int num_cap_cells,
    int n_sub) {
#ifdef __CUDACC__
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int incr = gridDim.x * blockDim.x;
#else
  int start = 0;
  int incr = 1;
#endif

  for (int i = start; i < n_sub; i += incr) {
    int32_t r = (row_map == nullptr) ? static_cast<int32_t>(i) : row_map[i];
    int32_t k = keep_pos[i];  // position in the scored subset, or -1

    double dh_k = 0.0;
    double hh_k = 0.0;
    bool have_snr = false;

    // ---- scatter of the scored likelihood ------------------------------
    if (k >= 0 && scored_ll != nullptr) {
      new_ll[i] = scored_ll[k];

      // ---- SNR prior-boundary clamp on the NEW point -------------------
      // ONE limit, on the optimal sqrt(h_h) AND (optionally) the detected
      // d_h/sqrt(h_h). A source already below the limit can still move OUT
      // of the violating region; it can never move further in.
      //
      // Both pointers are supplied together or not at all -- the python
      // gates this whole block on ``buffer_obj.d_h_out is not None`` while
      // reading ``h_h_out`` inside it.
      if (d_h != nullptr && h_h != nullptr) {
        hh_k = h_h[static_cast<int64_t>(k) * hh_stride];
        dh_k = d_h[static_cast<int64_t>(k) * dh_stride];
        have_snr = true;
        // NaN-propagating ``xp.maximum`` semantics, not a plain ternary:
        // numpy/cupy maximum(nan, 0.0) is nan, and the downstream
        // comparisons must inherit that (nan < limit is False both ways).
        double hh_cl = (hh_k != hh_k) ? hh_k : (hh_k > 0.0 ? hh_k : 0.0);
        double opt = sqrt(hh_cl);
        bool viol = opt < snr_limit;
        if (snr_detected) {
          double denom = (opt != opt) ? opt : (opt > 1e-300 ? opt : 1e-300);
          if ((dh_k / denom) < snr_limit) viol = true;
        }
        if (viol) new_ll[i] = GF_LL_FLOOR;
      }
    } else {
      // Non-kept lanes take the floor the python fills with
      // ``new_ll = xp.full(n_sub, -1e300)`` at the top of every repeat.
      // Writing it HERE is what lets the caller hand us a block-scope
      // scratch buffer instead of reallocating per repeat -- and it is
      // REQUIRED, not an optimization: without it a lane that was kept in
      // repeat k and rejected in repeat k+1 would still be carrying its old
      // score.
      new_ll[i] = GF_LL_FLOOR;
    }

    double delta = new_ll[i] - ll_ref[r];
    delta_ll[i] = delta;

    // ---- Metropolis-Hastings -------------------------------------------
    // Grouped exactly as the python expression so the floating-point
    // rounding is identical: (beta*delta) + (new_logp - curr_prior) + factors.
    double lp = beta_s[i] * delta + (new_logp[i] - curr_prior[r]) + factors[i];
    lnpdiff[i] = lp;
    bool acc = lp >= log(u[i]);
    accept_pre[i] = acc ? 1 : 0;

    bool bad = (new_ll[i] <= GF_BAD_LL_THRESH) ||
               (new_logp[i] <= GF_BAD_LOGP_THRESH);
    if (acc && bad && beta_s[i] != 0.0 && warn_count != nullptr) {
      gf_atomic_add_i64(&warn_count[0], 1);
    }
    acc = acc && !bad;
    accept[i] = acc ? 1 : 0;

    // ---- per-cell ledgers ----------------------------------------------
    int64_t cell_flat =
        (static_cast<int64_t>(temp_inds[i]) * nwalkers + walker_inds[i]) *
            num_bands +
        band_inds[i];
    gf_atomic_add_i64(&prop_counts1[cell_flat], 1);
    if (kind_acc != nullptr && acc) {
      gf_atomic_add_i64(&kind_acc[0], 1);
      if (cold_s != nullptr && cold_s[i]) gf_atomic_add_i64(&kind_acc[1], 1);
    }

    if (!acc) continue;

    // ---- accepted: the masked state writes ------------------------------
    // ``row_map`` is injective (a parity half is a set of distinct rows) and
    // serial-within-band gives one picked row per cell, so these plain
    // writes cannot race.
    for (int d = 0; d < ndim; ++d) {
      curr_coords[static_cast<int64_t>(r) * ndim + d] =
          new_coords[static_cast<int64_t>(i) * ndim + d];
    }
    ll_ref[r] = new_ll[i];
    curr_prior[r] = new_logp[i];

    // Rejected rows contribute an exact 0.0 in the python
    // (``xp.where(accept, delta_ll, 0.0)``); skipping the add here is the
    // same value -- the ledger starts at +0.0 and only ever receives adds,
    // so no -0.0 can arise for ``+ 0.0`` to normalize.
    gf_atomic_add_f64(&ll_change_log[cell_flat], delta);
    gf_atomic_add_i64(&acc_counts1[cell_flat], 1);

    // ---- sorter <d|h>/<h|h> stash ---------------------------------------
    // ``accept`` implies ``keep`` (a non-keep row has new_logp == -inf, so
    // the bad-mask filter above already cleared it), hence k >= 0 here.
    if (sorter_dh != nullptr && sorter_hh != nullptr && have_snr) {
      int32_t sid = ids_s[i];
      sorter_dh[sid] = dh_k;
      sorter_hh[sid] = hh_k;
    }

    // ---- cap-gate occupancy transition -----------------------------------
    if (dg_on) {
      int32_t t = temp_inds[i];
      int32_t w = walker_inds[i];
      int32_t c_p = cur_cells[i];
      int32_t c_nb = cur_cells[n_sub + i];
      int32_t c_hn = cur_cells[2 * n_sub + i];
      int32_t n_p = new_cells[i];
      int32_t n_nb = new_cells[n_sub + i];
      int32_t n_hn = new_cells[2 * n_sub + i];
      if (overlap_on) {
        // Per-SIDE set difference: +1 into every cell the accepted move
        // NEWLY covers, -1 out of every cell it no longer covers.
        for (int s = 0; s < 4; ++s) {
          int32_t cell;
          bool memb;
          int32_t sign;
          bool covered;
          if (s == 0) {
            cell = n_p; memb = true; sign = 1;
            covered = gf_in_set(cell, c_p, c_nb, c_hn);
          } else if (s == 1) {
            cell = n_nb; memb = (n_hn != 0); sign = 1;
            covered = gf_in_set(cell, c_p, c_nb, c_hn);
          } else if (s == 2) {
            cell = c_p; memb = true; sign = -1;
            covered = gf_in_set(cell, n_p, n_nb, n_hn);
          } else {
            cell = c_nb; memb = (c_hn != 0); sign = -1;
            covered = gf_in_set(cell, n_p, n_nb, n_hn);
          }
          if (!memb || covered) continue;
          int64_t flat = gf_cap_flat(t, w, cell, nwalkers, num_cap_cells);
          gf_atomic_add_i32(&dg_counts[flat], sign);
        }
      } else if (n_p != c_p) {
        gf_atomic_add_i32(
            &dg_counts[gf_cap_flat(t, w, n_p, nwalkers, num_cap_cells)], 1);
        gf_atomic_add_i32(
            &dg_counts[gf_cap_flat(t, w, c_p, nwalkers, num_cap_cells)], -1);
      }
    }
  }
}

// ============================================================================
// Host wrappers
// ============================================================================

void gb_inmodel_gate_compact_wrap(
    double *new_logp, uint8_t *keep_flag, int64_t *keep_idx,
    int64_t *n_keep_out, int32_t *cur_cells, int32_t *new_cells,
    int64_t *trust_counts, int64_t *dg_count, const double *new_coords,
    const double *curr_coords, const int32_t *row_map, const int32_t *n4,
    const int32_t *lo_bin, const int32_t *hi_bin, int f0_col, int ndim,
    double df, int window_on, const double *pc, int pc_ncol,
    const double *anchor_amp, const double *anchor_f0,
    const double *anchor_fdot, const double *trust_dlna,
    const double *trust_dphase, double trust_Tobs, int dg_on, int overlap_on,
    const int32_t *temp_inds, const int32_t *walker_inds,
    const int32_t *band_inds, const double *cap_band_lo,
    const double *cap_band_step, const double *cap_edges,
    const double *cap_edge_ext, const int32_t *dg_counts,
    const int32_t *dg_cap, int cap_divisor, int cap_stagger,
    int num_cap_cells, int nwalkers, int n_sub, int n_block,
    int32_t *keep_pos) {
  if (n_sub <= 0) {
    return;
  }
#ifdef __CUDACC__
  int num_blocks = (n_sub + GF_NUM_THREADS - 1) / GF_NUM_THREADS;
  gb_inmodel_gate_kernel<<<num_blocks, GF_NUM_THREADS>>>(
      new_logp, keep_flag, cur_cells, new_cells, trust_counts, dg_count,
      new_coords, curr_coords, row_map, n4, lo_bin, hi_bin, f0_col, ndim, df,
      window_on, pc, pc_ncol, anchor_amp, anchor_f0, anchor_fdot, trust_dlna,
      trust_dphase, trust_Tobs, dg_on, overlap_on, temp_inds, walker_inds,
      band_inds, cap_band_lo, cap_band_step, cap_edges, cap_edge_ext,
      dg_counts, dg_cap, cap_divisor, cap_stagger, num_cap_cells, nwalkers,
      n_sub);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  gb_inmodel_compact_kernel<<<1, GF_SCAN_THREADS>>>(keep_flag, keep_idx,
                                                    keep_pos, n_keep_out,
                                                    n_sub);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
#else
  gb_inmodel_gate_kernel(new_logp, keep_flag, cur_cells, new_cells,
                         trust_counts, dg_count, new_coords, curr_coords,
                         row_map, n4, lo_bin, hi_bin, f0_col, ndim, df,
                         window_on, pc, pc_ncol, anchor_amp, anchor_f0,
                         anchor_fdot, trust_dlna, trust_dphase, trust_Tobs,
                         dg_on, overlap_on, temp_inds, walker_inds, band_inds,
                         cap_band_lo, cap_band_step, cap_edges, cap_edge_ext,
                         dg_counts, dg_cap, cap_divisor, cap_stagger,
                         num_cap_cells, nwalkers, n_sub);
  gb_inmodel_compact_kernel(keep_flag, keep_idx, keep_pos, n_keep_out, n_sub);
#endif
}

void gb_inmodel_accept_apply_wrap(
    double *new_ll, double *delta_ll, double *lnpdiff, uint8_t *accept_pre,
    uint8_t *accept, double *curr_coords, double *ll_ref, double *curr_prior,
    const double *scored_ll, const int64_t *keep_idx, int n_keep,
    const double *d_h, const double *h_h, int dh_stride, int hh_stride,
    double snr_limit, int snr_detected, const double *new_coords,
    const double *new_logp, const double *factors, const double *beta_s,
    const double *u, const int32_t *row_map, int ndim,
    const int32_t *temp_inds, const int32_t *walker_inds,
    const int32_t *band_inds, const uint8_t *cold_s, double *ll_change_log,
    int64_t *prop_counts1, int64_t *acc_counts1, int nwalkers, int num_bands,
    int64_t *warn_count, int64_t *kind_acc, double *sorter_dh,
    double *sorter_hh, const int32_t *ids_s, int dg_on, int overlap_on,
    int32_t *dg_counts, const int32_t *cur_cells, const int32_t *new_cells,
    int num_cap_cells, int n_sub, int n_block, const int32_t *keep_pos) {
  (void)keep_idx;  // the inverse map (keep_pos) is what the kernel needs
  (void)n_keep;
  (void)n_block;
  if (n_sub <= 0) {
    return;
  }
#ifdef __CUDACC__
  int num_blocks = (n_sub + GF_NUM_THREADS - 1) / GF_NUM_THREADS;
  gb_inmodel_accept_kernel<<<num_blocks, GF_NUM_THREADS>>>(
      new_ll, delta_ll, lnpdiff, accept_pre, accept, curr_coords, ll_ref,
      curr_prior, scored_ll, keep_pos, d_h, h_h, dh_stride, hh_stride,
      snr_limit, snr_detected, new_coords, new_logp, factors, beta_s, u,
      row_map, ndim, temp_inds, walker_inds, band_inds, cold_s, ll_change_log,
      prop_counts1, acc_counts1, nwalkers, num_bands, warn_count, kind_acc,
      sorter_dh, sorter_hh, ids_s, dg_on, overlap_on, dg_counts, cur_cells,
      new_cells, num_cap_cells, n_sub);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
#else
  gb_inmodel_accept_kernel(
      new_ll, delta_ll, lnpdiff, accept_pre, accept, curr_coords, ll_ref,
      curr_prior, scored_ll, keep_pos, d_h, h_h, dh_stride, hh_stride,
      snr_limit, snr_detected, new_coords, new_logp, factors, beta_s, u,
      row_map, ndim, temp_inds, walker_inds, band_inds, cold_s, ll_change_log,
      prop_counts1, acc_counts1, nwalkers, num_bands, warn_count, kind_acc,
      sorter_dh, sorter_hh, ids_s, dg_on, overlap_on, dg_counts, cur_cells,
      new_cells, num_cap_cells, n_sub);
#endif
}
