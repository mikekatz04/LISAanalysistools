/**
 * @file domains.cu
 * @brief Implementation of STFT/FD domain inner-product primitives and CUDA
 *        likelihood kernels for LISA STFT-domain matched filtering.
 *
 * Overview
 * --------
 * This file implements the signal-processing inner products needed to evaluate
 * a Gaussian log-likelihood in the STFT (Short-Time Fourier Transform) or
 * frequency-domain representation of the LISA data stream:
 *
 *   log L ∝  - 1/2 (<d|d> + <h|h> - 2 <>d|h>).real()
 *
 * where the noise-weighted inner product between two complex arrays a and b is:
 *
 *   <a|b> = Σ_{t,f,i,j}  conj(a[t,f,i]) * C^{-1}_{ij}(t,f) * b[t,f,j]
 *
 * with C^{-1} the precomputed inverse noise covariance.  The prefactor 4 (or
 * 2 for auto-products) arises from the one-sided → two-sided PSD convention
 * (the loop only covers positive frequencies).
 *
 * Supported TDI channel configurations
 * --------------------------------------
 * - TDI_AET (diagonal):  A, E, T channels are treated as independent;
 *   C^{-1} is diagonal and only the channel index is needed.
 * - TDI_XYZ (full):      X, Y, Z channels have correlated noise; the full
 *   3×3 matrix C^{-1}_{ij}(t,f) is used.
 *
 * Two-pass GPU reduction
 * ----------------------
 * For GPU execution a two-pass strategy is used to avoid the serialisation
 * bottleneck of an atomic global reduction:
 *
 *   Pass 1 – compute_likelihood_contributions_kernel
 *     Each thread processes one (t,f) pixel.  Threads within a block reduce
 *     their partial sums using cub::BlockReduce and write one complex scalar
 *     per (binary, block) pair to d_h_contrib / h_h_contrib.
 *
 *   Pass 2 – like_sum_from_contrib_cmplx
 *     One block per binary performs a shared-memory tree reduction across the
 *     per-block partial sums and writes the final (d|h) and (h|h) values.
 *
 * On CPU both passes collapse into a single serial loop.
 *
 * Source-swap support
 * -------------------
 * add_ip_swap_contrib() evaluates five inner-product terms in one channel loop,
 * supporting birth/death and swap proposals in a Reversible-Jump MCMC sampler
 * (e.g. Eryn) without redundant noise-matrix lookups.
 *
 * Fresnel evaluator (STFTFresnel)
 * -------------------------------
 * The second half of the file evaluates the STFT pixel value of a locally linear
 * chirp analytically. Three nested layers, each with one shared body that also
 * returns a first moment when the caller asks for one:
 *
 *   lat_fresnel.hh             : the Fresnel series and the auxiliary functions
 *                                f, g (and df, dg), shared with the DFFR kernels
 *   get_phase_kernel_core      : one integration interval, demodulated
 *   get_windowed_fourier_core  : the seven Tukey terms of one STFT window
 *
 * The public entry points sit on top: get_fourier_value, and its factorised
 * pair get_fourier_prefactor * get_fourier_kernel. Only the prefactor depends
 * on amplitude and phase, so the information-matrix kernel in
 * lat_stft_kernels.hh evaluates the kernel once and shares it across templates
 * that agree on (f0, fdot0, window start).
 *
 * @see domains.hpp for class declarations and detailed parameter documentation.
 */

#include <iostream>
#include "domains.hpp"
#include "lat_fresnel.hh"

#ifdef __CUDACC__
#include <cub/cub.cuh>  // CUB block-level primitives for efficient intra-block reductions
/// Number of CUDA threads per block for likelihood kernels.
/// Must be a power of two for CUB/tree reductions.
#define NUM_THREADS 128
#else
/// CPU fallback: treat each "block" as a single thread.
#define NUM_THREADS 1
#endif

#ifdef __CUDACC__
/**
 * @brief Functor for CUB block reduction: element-wise complex addition.
 *
 * CUB's BlockReduce requires a binary operator.  This wraps operator+ for
 * the cmplx (thrust::complex<double>) type.
 */
struct ComplexSum {
  CUDA_DEVICE cmplx operator()(const cmplx& a, const cmplx& b) const {
    return a + b;
  }
};

/**
 * @brief Block-level reduction of a complex shared-memory array.
 *
 * Uses cub::BlockReduce to sum all NUM_THREADS elements of @p array within
 * the current CUDA block.  Must be called by *all* threads in the block
 * (i.e. no early exit before this call).
 *
 * The CUDA_SYNC_THREADS before the reduction ensures that every thread has
 * written its contribution to @p array before the reduction begins, because
 * CUB reuses the caller-provided TempStorage in-place.
 *
 * @param array  Pointer to shared memory of length NUM_THREADS; each thread
 *               contributes array[threadIdx.x].
 * @return       The sum of all elements (valid only on thread 0).
 */
static CUDA_DEVICE cmplx block_reduce_cmplx(cmplx* array) {
  using BlockReduce = cub::BlockReduce<cmplx, NUM_THREADS>;
  CUDA_SHARED typename BlockReduce::TempStorage temp_storage;
  // Synchronise before reading: ensures all threads have written their element.
  CUDA_SYNC_THREADS;
  int tid = threadIdx.x;
  cmplx thread_data = array[tid];
  cmplx output = BlockReduce(temp_storage).Reduce(thread_data, ComplexSum());
  return output;
}

/**
 * @brief Real-valued mirror of block_reduce_cmplx for the WDM kernels
 * (WDM coefficients and inverse-noise weights are real doubles).
 */
static CUDA_DEVICE double block_reduce_double(double* array) {
  using BlockReduce = cub::BlockReduce<double, NUM_THREADS>;
  CUDA_SHARED typename BlockReduce::TempStorage temp_storage;
  // Synchronise before reading: ensures all threads have written their element.
  CUDA_SYNC_THREADS;
  int tid = threadIdx.x;
  double thread_data = array[tid];
  double output = BlockReduce(temp_storage).Sum(thread_data);
  return output;
}
#endif

// ============================================================
// Data indexing
// ============================================================

/**
 * Convert a physical time t [s] to its zero-based grid index.
 *
 * Rounds to the nearest grid step rather than truncating: the template and data
 * grids share the same dt and are bin-aligned by construction, so (t - t0)/dt is
 * an integer up to floating-point noise.  Truncation maps a value that is
 * (integer - 1 ULP) to integer-1; round-to-nearest recovers the intended index.
 * The result may fall outside [0, num_times); the STFT likelihood kernels skip
 * out-of-grid pixels with a bounds guard (the CUDA kernel cannot throw).
 */
CUDA_DEVICE
int STFTDomain::get_time_index(double t) {
  return (int)llround((t - t0) / dt);
}

/**
 * Convert a physical frequency f [Hz] to its zero-based grid index.
 *
 * Rounds to the nearest grid step (see get_time_index).  A template start
 * frequency at the band edge can land ~1 ULP below f_min because it is computed
 * on a different float path than f_min = ind_min*df; truncation plus a strict
 * ``f < f_min`` test would then return an out-of-grid -1 (which the kernel used
 * unchecked -> negative flat offset -> CUDA_ERROR_ILLEGAL_ADDRESS).
 * Round-to-nearest maps it to the intended bin; the kernel bounds guard skips
 * genuinely out-of-grid indices.
 */
CUDA_DEVICE
int STFTDomain::get_freq_index(double f) {
  return (int)llround((f - f_min) / df);
}

/**
 * Compute the flat (row-major) index into the data array.
 * Memory layout: [num_data, num_channels, num_times, num_freqs]
 *   flat = ((data_index * num_channels + channel) * num_times + t_idx) *
 * num_freqs + f_idx
 */
CUDA_DEVICE
int STFTDomain::get_data_index(int t_idx, int f_idx, int channel,
                               int data_index) {
  if (data_index >= num_data) {
#ifdef __CUDACC__
#else
    throw std::invalid_argument(
        "data_index is larger than available data instances.");
#endif
  }
  return ((data_index * num_channels + channel) * num_times + t_idx) *
             num_freqs +
         f_idx;
}

CUDA_DEVICE
cmplx STFTDomain::get_data_value(int t_idx, int f_idx, int channel,
                                 int data_index) {
  return data[get_data_index(t_idx, f_idx, channel, data_index)];
}

// ============================================================
// Noise indexing — diagonal (AET)
// ============================================================

/**
 * Compute the flat index for the diagonal inverse-covariance array.
 * Memory layout: [num_noise, num_channels, num_times, num_freqs]
 *   flat = ((noise_index * num_channels + channel) * num_times + t_idx) *
 * num_freqs + f_idx
 */
CUDA_DEVICE
int STFTDomain::get_noise_index(int t_idx, int f_idx, int channel,
                                int noise_index) {
  if (noise_index >= num_noise) {
#ifdef __CUDACC__
#else
    throw std::invalid_argument(
        "noise_index is larger than available noise instances.");
#endif
  }
  return ((noise_index * num_channels + channel) * num_times + t_idx) *
             num_freqs +
         f_idx;
}

CUDA_DEVICE
cmplx STFTDomain::get_invC_value(int t_idx, int f_idx, int channel,
                                 int noise_index) {
  return invC[get_noise_index(t_idx, f_idx, channel, noise_index)];
}

// ============================================================
// Noise indexing — full matrix (XYZ)
// ============================================================

/**
 * Compute the flat index for the full 3×3 inverse-covariance matrix array.
 * Memory layout: [num_noise, num_channels, num_channels, num_times, num_freqs]
 *   flat = (((noise_index * num_channels + ch_i) * num_channels + ch_j)
 *             * num_times + t_idx) * num_freqs + f_idx
 */
CUDA_DEVICE
int STFTDomain::get_noise_index_cross(int t_idx, int f_idx, int ch_i, int ch_j,
                                      int noise_index) {
  if (noise_index >= num_noise) {
#ifdef __CUDACC__
#else
    throw std::invalid_argument(
        "noise_index is larger than available noise instances.");
#endif
  }
  return (((noise_index * num_channels + ch_i) * num_channels + ch_j) *
              num_times +
          t_idx) *
             num_freqs +
         f_idx;
}

CUDA_DEVICE
cmplx STFTDomain::get_invC_cross_value(int t_idx, int f_idx, int ch_i, int ch_j,
                                       int noise_index) {
  return invC[get_noise_index_cross(t_idx, f_idx, ch_i, ch_j, noise_index)];
}

// ============================================================
// Inner products — per-channel functions
// ============================================================

/**
 * XYZ / full-matrix mode: accumulate one (ch_i, ch_j) cross-term.
 *
 * The inverse covariance element C^{-1}_{ij} is fetched once and reused for
 * both inner products, avoiding a double lookup:
 *
 *   tmp = C^{-1}_{ij}(t,f) * h[ch_j]
 *   *d_h += conj(d[ch_i]) * tmp
 *   *h_h += conj(h[ch_i]) * tmp
 *
 * This function must be called inside a double loop over (ch_i, ch_j) to
 * accumulate the full matrix contraction.
 */
CUDA_DEVICE
void STFTDomain::get_inner_product_cross(cmplx* d_h, cmplx* h_h, cmplx h_val_i,
                                         cmplx h_val_j, int t_idx, int f_idx,
                                         int channel_i, int channel_j,
                                         int data_index, int noise_index) {
  cmplx C_ij =
      get_invC_cross_value(t_idx, f_idx, channel_i, channel_j, noise_index);
  // Pre-multiply template value by the noise weight to share across d_h and
  // h_h.
  cmplx invC_h_j = C_ij * h_val_j;

  cmplx d_i = get_data_value(t_idx, f_idx, channel_i, data_index);
  *d_h += gcmplx::conj(d_i) * invC_h_j;
  *h_h += gcmplx::conj(h_val_i) * invC_h_j;
}

/**
 * AET / diagonal mode: accumulate one channel's contribution to (d|h) and
 * (h|h).
 *
 * The diagonal inverse-covariance weight C^{-1}_{ch}(t,f) is fetched once:
 *
 *   tmp = C^{-1}_{ch}(t,f) * h[ch]
 *   *d_h += conj(d[ch]) * tmp
 *   *h_h += conj(h[ch]) * tmp
 *
 * Call inside a single loop over channels.
 */
CUDA_DEVICE
void STFTDomain::get_inner_product_diag(cmplx* d_h, cmplx* h_h, cmplx h_val,
                                        int t_idx, int f_idx, int channel,
                                        int data_index, int noise_index) {
  cmplx invC_ch = get_invC_value(t_idx, f_idx, channel, noise_index);
  // Pre-multiply template value by the noise weight to share across d_h and
  // h_h.
  cmplx invC_h = invC_ch * h_val;

  cmplx d_ch = get_data_value(t_idx, f_idx, channel, data_index);
  *d_h += gcmplx::conj(d_ch) * invC_h;
  *h_h += gcmplx::conj(h_val) * invC_h;
}

// --- <d|d> inner product — per-channel functions ---

/**
 * XYZ / full-matrix mode: accumulate one (ch_i, ch_j) cross-term of (d|d).
 *
 *   *d_d += conj(d[ch_i]) * C^{-1}_{ij}(t,f) * d[ch_j]
 *
 * Both data values are fetched from the device array; there is no template
 * involved.  Call inside a double loop over (ch_i, ch_j).
 */
CUDA_DEVICE
void STFTDomain::get_d_d_inner_product_cross(cmplx* d_d, int t_idx, int f_idx,
                                             int channel_i, int channel_j,
                                             int data_index, int noise_index) {
  cmplx C_ij =
      get_invC_cross_value(t_idx, f_idx, channel_i, channel_j, noise_index);
  cmplx d_j = get_data_value(t_idx, f_idx, channel_j, data_index);
  cmplx d_i = get_data_value(t_idx, f_idx, channel_i, data_index);
  *d_d += gcmplx::conj(d_i) * (C_ij * d_j);
}

/**
 * AET / diagonal mode: accumulate one channel's contribution to (d|d).
 *
 *   *d_d += conj(d[ch]) * C^{-1}_{ch}(t,f) * d[ch]
 *
 * Call inside a single loop over channels.
 */
CUDA_DEVICE
void STFTDomain::get_d_d_inner_product_diag(cmplx* d_d, int t_idx, int f_idx,
                                            int channel, int data_index,
                                            int noise_index) {
  cmplx invC_ch = get_invC_value(t_idx, f_idx, channel, noise_index);
  cmplx d_ch = get_data_value(t_idx, f_idx, channel, data_index);
  *d_d += gcmplx::conj(d_ch) * (invC_ch * d_ch);
}

// ============================================================
// Unified dispatchers — channel loop lives here
// ============================================================

/**
 * Accumulate (d|h) and (h|h) contributions from all channels at one (t,f)
 * pixel.
 *
 * Dispatches to the cross-channel (XYZ) or diagonal (AET) inner-product
 * primitives based on tdi_type.  Results are added to the per-thread
 * shared arrays at index tid = threadIdx.x (GPU) or 0 (CPU).
 *
 * @param d_h_tmp      Shared accumulator array for (d|h), indexed by tid
 * @param h_h_tmp      Shared accumulator array for (h|h), indexed by tid
 * @param template_vals  Template values for all num_channels at this (t,f)
 * pixel
 */
CUDA_DEVICE
void STFTDomain::add_ip_contrib(cmplx* d_h_tmp, cmplx* h_h_tmp,
                                cmplx* template_vals, int t_idx, int f_idx,
                                int data_index, int noise_index) {
#ifdef __CUDACC__
  int tid = threadIdx.x;
#else
  int tid = 0;
#endif
  cmplx d_h_val = cmplx(0.0, 0.0);
  cmplx h_h_val = cmplx(0.0, 0.0);
  if (tdi_type == TDI_XYZ) {
    for (int ch_i = 0; ch_i < 3; ch_i++) {
      for (int ch_j = 0; ch_j < 3; ch_j++) {
        get_inner_product_cross(&d_h_val, &h_h_val, template_vals[ch_i],
                                template_vals[ch_j], t_idx, f_idx, ch_i, ch_j,
                                data_index, noise_index);
      }
    }
  } else {
    for (int ch = 0; ch < num_channels; ch++) {
      get_inner_product_diag(&d_h_val, &h_h_val, template_vals[ch], t_idx,
                             f_idx, ch, data_index, noise_index);
    }
  }
  d_h_tmp[tid] += d_h_val;
  h_h_tmp[tid] += h_h_val;
}

/**
 * Accumulate (d|d) contributions from all channels at one (t,f) pixel.
 *
 * Dispatches to cross-channel (XYZ) or diagonal (AET) path.  Result is
 * added to d_d_tmp[tid].
 *
 * @param d_d_tmp  Shared accumulator array for (d|d), indexed by tid
 */
CUDA_DEVICE
void STFTDomain::add_d_d_contrib(cmplx* d_d_tmp, int t_idx, int f_idx,
                                 int data_index, int noise_index) {
#ifdef __CUDACC__
  int tid = threadIdx.x;
#else
  int tid = 0;
#endif
  cmplx d_d_val = cmplx(0.0, 0.0);
  if (tdi_type == TDI_XYZ) {
    for (int ch_i = 0; ch_i < 3; ch_i++) {
      for (int ch_j = 0; ch_j < 3; ch_j++) {
        get_d_d_inner_product_cross(&d_d_val, t_idx, f_idx, ch_i, ch_j,
                                    data_index, noise_index);
      }
    }
  } else {
    for (int ch = 0; ch < num_channels; ch++) {
      get_d_d_inner_product_diag(&d_d_val, t_idx, f_idx, ch, data_index,
                                 noise_index);
    }
  }
  d_d_tmp[tid] += d_d_val;
}

/**
 * Accumulate all five inner-product terms needed for a source-swap MCMC step
 * at one (t,f) pixel.
 *
 * This function evaluates (d|h_add), (h_add|h_add), (d|h_remove),
 * (h_remove|h_remove), and (h_add|h_remove) simultaneously within a single
 * channel loop, avoiding duplicate noise-matrix fetches.  The terms correspond
 * to the Metropolis–Hastings acceptance ratio for a birth/death/swap move:
 *
 *   ΔlogL = 2·Re[(d|h_add) - (d|h_remove)]
 *           - [(h_add|h_add) - (h_remove|h_remove)]
 *           - 2·Re[(h_add|h_remove)]
 *           (the last term arises from the cross-correlation between sources)
 *
 * @param d_h_add_tmp        Per-thread accumulator for (d|h_add)
 * @param d_h_remove_tmp     Per-thread accumulator for (d|h_remove)
 * @param add_add_tmp        Per-thread accumulator for (h_add|h_add)
 * @param remove_remove_tmp  Per-thread accumulator for (h_remove|h_remove)
 * @param add_remove_tmp     Per-thread accumulator for (h_add|h_remove)
 * @param template_vals_add     Add-template values, length num_channels
 * @param template_vals_remove  Remove-template values, length num_channels
 */
CUDA_DEVICE
void STFTDomain::add_ip_swap_contrib(
    cmplx* d_h_add_tmp, cmplx* d_h_remove_tmp, cmplx* add_add_tmp,
    cmplx* remove_remove_tmp, cmplx* add_remove_tmp, cmplx* template_vals_add,
    cmplx* template_vals_remove, int t_idx, int f_idx, int data_index,
    int noise_index) {
#ifdef __CUDACC__
  int tid = threadIdx.x;
#else
  int tid = 0;
#endif
  cmplx d_h_add_val = cmplx(0.0, 0.0);
  cmplx d_h_remove_val = cmplx(0.0, 0.0);
  cmplx add_add_val = cmplx(0.0, 0.0);
  cmplx remove_remove_val = cmplx(0.0, 0.0);
  cmplx add_remove_val = cmplx(0.0, 0.0);
  // get_inner_product_{cross,diag} always writes both *d_h and *h_h.
  // When computing <h_add|h_remove> we don't need the
  // (d|h_remove_using_add_row) part, so we route it into this throwaway
  // variable.
  cmplx discard = cmplx(0.0, 0.0);
  if (tdi_type == TDI_XYZ) {
    for (int ch_i = 0; ch_i < 3; ch_i++) {
      for (int ch_j = 0; ch_j < 3; ch_j++) {
        get_inner_product_cross(&d_h_add_val, &add_add_val,
                                template_vals_add[ch_i],
                                template_vals_add[ch_j], t_idx, f_idx, ch_i,
                                ch_j, data_index, noise_index);

        get_inner_product_cross(&d_h_remove_val, &remove_remove_val,
                                template_vals_remove[ch_i],
                                template_vals_remove[ch_j], t_idx, f_idx, ch_i,
                                ch_j, data_index, noise_index);

        get_inner_product_cross(&discard, &add_remove_val,
                                template_vals_add[ch_i],
                                template_vals_remove[ch_j], t_idx, f_idx, ch_i,
                                ch_j, data_index, noise_index);
      }
    }
  } else {
    for (int ch = 0; ch < num_channels; ch++) {
      get_inner_product_diag(&d_h_add_val, &add_add_val, template_vals_add[ch],
                             t_idx, f_idx, ch, data_index, noise_index);

      get_inner_product_diag(&d_h_remove_val, &remove_remove_val,
                             template_vals_remove[ch], t_idx, f_idx, ch,
                             data_index, noise_index);

      get_inner_product_diag(&discard, &add_remove_val, template_vals_add[ch],
                             t_idx, f_idx, ch, data_index, noise_index);
    }
  }
  d_h_add_tmp[tid] += d_h_add_val;
  add_add_tmp[tid] += add_add_val;
  d_h_remove_tmp[tid] += d_h_remove_val;
  remove_remove_tmp[tid] += remove_remove_val;
  add_remove_tmp[tid] += add_remove_val;
}

/**
 * Host wrapper: launch the two-pass GPU likelihood kernels (or CPU loop) for
 * a batch of num_binaries sources.
 *
 * GPU execution strategy
 * ----------------------
 * 1. Allocate temporary device buffers d_h_contrib / h_h_contrib of size
 *    num_binaries × num_blocks_x to hold per-block partial sums.
 * 2. Copy `this` (host STFTDomain object) to device so the kernel can access
 *    grid parameters and device pointers via a uniform interface.
 * 3. Launch Pass 1 (compute_likelihood_contributions_kernel) with a 2-D grid:
 *      gridDim.x = ceil(num_times_template * num_freqs_template / NUM_THREADS)
 *      gridDim.y = num_binaries
 *    Each (blockIdx.y, blockIdx.x) pair handles one (binary, tf-chunk).
 * 4. Launch Pass 2 (like_sum_from_contrib_cmplx) with gridDim = (1,
 * num_binaries) to reduce across the partial sums and write final results to
 * d_h_out / h_h_out.
 * 5. Free temporary buffers and the device domain object.
 *
 * CPU execution strategy
 * ----------------------
 * A single call to compute_likelihood_contributions_kernel with a HOST function
 * pointer writes the complete results directly into d_h_out / h_h_out.
 */
void STFTDomain::compute_likelihood_terms_wrap(
    cmplx* d_h_out, cmplx* h_h_out, cmplx* template_vals,
    double* start_times_all, double* start_freqs_all, int num_binaries,
    int* data_index_all, int* noise_index_all, int num_times_template,
    int num_freqs_template, bool run_async) {
#ifdef __CUDACC__
  cmplx* d_h_contrib;
  cmplx* h_h_contrib;
  // Number of blocks along the (t,f) dimension; each block reduces
  // NUM_THREADS pixels and contributes one partial-sum entry.
  int num_blocks_x =
      (num_times_template * num_freqs_template + NUM_THREADS - 1) / NUM_THREADS;
  int num_blocks_y = num_binaries;  // one row of blocks per binary
  dim3 grid_dim(num_blocks_x, num_blocks_y);

  // Allocate partial-sum buffers: [num_binaries, num_blocks_x]
  if (run_async) {
    gpuErrchk(cudaMallocAsync(&d_h_contrib,
                              num_binaries * num_blocks_x * sizeof(cmplx),
                              cudaStreamDefault));
    gpuErrchk(cudaMallocAsync(&h_h_contrib,
                              num_binaries * num_blocks_x * sizeof(cmplx),
                              cudaStreamDefault));
  } else {
    gpuErrchk(cudaMalloc(&d_h_contrib,
                         num_binaries * num_blocks_x * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&h_h_contrib,
                         num_binaries * num_blocks_x * sizeof(cmplx)));
  }

  // Pass 1: compute per-block partial sums of (d|h) and (h|h). `*this` is passed by value: the
  // struct is copied into kernel-arg space and its data/invC fields are already device pointers.
  compute_likelihood_contributions_kernel<<<grid_dim, NUM_THREADS>>>(
      d_h_contrib, h_h_contrib, *this, template_vals, start_times_all,
      start_freqs_all, num_binaries, data_index_all, noise_index_all,
      num_times_template, num_freqs_template);
  // Pass 2: reduce partial sums across blocks for each binary.
  dim3 reduce_grid_dim(1, num_binaries, 1);  // one block per binary
  like_sum_from_contrib_cmplx<<<reduce_grid_dim, NUM_THREADS>>>(
      d_h_out, h_h_out, d_h_contrib, h_h_contrib, num_blocks_x, num_binaries);

  gpuErrchk(cudaGetLastError());
  if (run_async) {
    gpuErrchk(cudaFreeAsync(d_h_contrib, cudaStreamDefault));
    gpuErrchk(cudaFreeAsync(h_h_contrib, cudaStreamDefault));
  } else {
    gpuErrchk(cudaFree(d_h_contrib));
    gpuErrchk(cudaFree(h_h_contrib));
    cudaDeviceSynchronize();
  }

#else
  // CPU path: the kernel function is a plain C++ function.  Results are
  // written directly to d_h_out / h_h_out (no intermediate buffers needed).
  compute_likelihood_contributions_kernel(
      d_h_out, h_h_out, *this, template_vals, start_times_all, start_freqs_all,
      num_binaries, data_index_all, noise_index_all, num_times_template,
      num_freqs_template);
#endif
};

void FDDomainForStft::compute_likelihood_terms_wrap(
    cmplx* d_h_out, cmplx* h_h_out, cmplx* template_vals,
    double* start_freqs_all, int num_binaries, int* data_index_all,
    int* noise_index_all, int num_freqs_template, bool run_async) {
  // Delegate to the STFT version with num_times_template = 1.
  // start_times_all = nullptr signals the kernel to use start_t_idx = 0.
  STFTDomain::compute_likelihood_terms_wrap(
      d_h_out, h_h_out, template_vals,
      nullptr,  // start_times_all not used in FDDomainForStft
      start_freqs_all, num_binaries, data_index_all, noise_index_all,
      1,  // num_times_template = 1 for FDDomainForStft
      num_freqs_template, run_async);
}

/**
 * Fresnel computation block: compute the frequency domain representation of
 * signals that can be approximated as linear chirps in the time domain, at
 * least locally within each STFT window.
 */
CUDA_DEVICE
void STFTFresnel::get_amp_phase(double* amp, double* phase, cmplx z)
// extract amplitude and phase from complex input
{
  *amp = gcmplx::abs(z);
  *phase = gcmplx::arg(z);
}

CUDA_DEVICE
double STFTFresnel::get_zeta(double f, double f0, double fdot0) {
  double zeta = (f0 - f) / fdot0;
  return zeta;
}

// The Fresnel integrals C(x) and S(x), from the shared functions of lat_fresnel.hh. The phase
// kernel below calls the series directly instead, since it has already tested the seam.
CUDA_DEVICE
void STFTFresnel::get_fresnel_integrals(double* C, double* S, double x) {
  lat_fresnel_integrals(C, S, x);
}

// Auxiliary Fresnel functions f and g for |x| > 1.6 and their derivatives in |x|, from the shared
// functions of lat_fresnel.hh. nullptr for df_out or dg_out skips both derivatives.
CUDA_DEVICE
void STFTFresnel::get_fresnel_aux(double* f_out, double* g_out, double abs_x,
                                  double* df_out, double* dg_out) {
  lat_fresnel_aux(f_out, g_out, abs_x, df_out, dg_out);
}

CUDA_DEVICE
cmplx STFTFresnel::get_phase_kernel_product(double f_eff, double t_ref,
                                            double f0, double fdot0,
                                            double t_start, double t_end,
                                            double t_ft_origin) {
  // The stationary factor exp(-i pi fdot0 zeta^2) and the Fresnel phase pi v^2 / 2 both carry
  // pi (f0 - f_eff)^2 / fdot0 -- 1e12 rad at fdot0 = 1e-19, and 3e15 rad for the taper's
  // f -/+ f_taper terms -- and cancel analytically. Formed apart, their last bits alone cost up to
  // 1e-4 of the value and 1e4 of its f0 derivative. Each endpoint therefore carries the combined
  // phase psi = pi fdot0 tau^2 + 2 pi (f0 - f_eff) tau, which never exceeds a few thousand radians:
  //   C + i s S = sign(v) [ (1 + i s) / 2 - (g + i s f)(|v|) e^{i s pi v^2 / 2} ],  s = sign(fdot0)
  //   e^{i s pi v^2 / 2} e^{-i pi fdot0 zeta^2} = e^{i psi},   since fdot0 zeta = f0 - f_eff.
  // The fdot0 < 0 conjugation, the signed zeta and the magnitude under the root follow the
  // convention in _dev/gbs_derivatives/fresnel_playground_negative_fdot.ipynb.
  cmplx kernel;
  get_phase_kernel_core(f_eff, t_ref, f0, fdot0, t_start, t_end, t_ft_origin,
                        &kernel, nullptr);
  return kernel;
}

// Shared body of the phase-kernel product and its first moment about t_ref. The moment is
//   moment = sqrt(2 |fdot0|) int tau e^{i psi} dtau = (1 / 2 pi i) d(kernel) / d(delta_f),
// so it is the same endpoint decomposition differentiated in delta_f = f0 - f_eff, using
// dv/d(delta_f) = sign(v) sqrt(2|fdot0|) / fdot0 and d(psi)/d(delta_f) = 2 pi tau. Differentiating
// in this form is what keeps the moment conditioned: the textbook expression -zeta * kernel +
// (i / 2 pi) d(kernel)/df subtracts two terms of size |zeta * kernel|, and zeta reaches 1e17 s.
CUDA_DEVICE
void STFTFresnel::get_phase_kernel_core(double f_eff, double t_ref, double f0,
                                        double fdot0, double t_start,
                                        double t_end, double t_ft_origin,
                                        cmplx* kernel_out, cmplx* moment_out) {
  double zeta = get_zeta(f_eff, f0, fdot0);
  double root = std::sqrt(2.0 * std::abs(fdot0));
  double sign_fdot = (fdot0 >= 0.0) ? 1.0 : -1.0;
  double delta_f = f0 - f_eff;  // = fdot0 * zeta, without the round trip through zeta
  bool with_moment = (moment_out != nullptr);

  cmplx oscillating(0.0, 0.0);
  cmplx oscillating_moment(0.0, 0.0);
  cmplx series_part(0.0, 0.0);
  double constant_weight = 0.0;
  bool needs_stationary = false;
  double psi_end = 0.0, psi_start = 0.0;

  for (int endpoint = 0; endpoint < 2; endpoint += 1) {
    double tau = (endpoint == 0) ? (t_end - t_ref) : (t_start - t_ref);
    double weight = (endpoint == 0) ? 1.0 : -1.0;
    double v = root * (tau + zeta);
    double abs_v = std::abs(v);
    double sign_v = (v >= 0.0) ? 1.0 : -1.0;
    double psi = M_PI * fdot0 * tau * tau + 2.0 * M_PI * delta_f * tau;
    if (endpoint == 0) psi_end = psi; else psi_start = psi;
    if (abs_v > LAT_FRESNEL_SEAM) {
      double f_x, g_x, df_x = 0.0, dg_x = 0.0;
      if (with_moment)
        lat_fresnel_aux(&f_x, &g_x, abs_v, &df_x, &dg_x);
      else
        lat_fresnel_aux(&f_x, &g_x, abs_v);
      cmplx rotation = gcmplx::polar(1.0, psi);
      cmplx aux(g_x, sign_fdot * f_x);
      oscillating = oscillating - (weight * sign_v) * rotation * aux;
      if (with_moment) {
        // (1 / 2 pi i) d/d(delta_f) of this endpoint's term.
        cmplx aux_slope(dg_x, sign_fdot * df_x);
        oscillating_moment =
            oscillating_moment +
            weight * rotation *
                (cmplx(0.0, root / (2.0 * M_PI * fdot0)) * aux_slope -
                 (sign_v * tau) * aux);
      }
      constant_weight += weight * sign_v;
    } else {
      // * The endpoint lies within 1.6 / sqrt(2 |fdot0|) of the stationary point, so |zeta| is
      // * bounded by |tau| plus that width and the stationary factor below is safe to form.
      double C_v, S_v;
      lat_fresnel_series(&C_v, &S_v, v);
      series_part = series_part + weight * cmplx(C_v, sign_fdot * S_v);
      needs_stationary = true;
    }
  }

  cmplx bounded(0.0, 0.0);
  bool bounded_zeta = (needs_stationary || constant_weight != 0.0);
  if (bounded_zeta) {
    // ! A surviving constant means v changed sign inside the interval, so -zeta lies in it and
    // ! |zeta| <= the interval length. Where |zeta| is large the two constants cancel exactly and
    // ! this branch is skipped, which is what keeps the large phase out of the evaluation.
    cmplx stationary = gcmplx::polar(1.0, -M_PI * fdot0 * zeta * zeta);
    cmplx flat = series_part + constant_weight * cmplx(0.5, 0.5 * sign_fdot);
    bounded = stationary * flat;
  }
  // Re-references the Fourier transform to the window start when the chirp is anchored elsewhere
  // (t_ref = midpoint); exactly 1 when t_ref == t_ft_origin.
  cmplx origin =
      gcmplx::polar(1.0, -2.0 * M_PI * f_eff * (t_ref - t_ft_origin));
  cmplx kernel = oscillating + bounded;
  *kernel_out = origin * kernel;
  if (!with_moment) return;

  cmplx moment;
  if (bounded_zeta) {
    // Near the stationary point the endpoint derivatives are a difference of two cosines that are
    // both 1 to within the rounding, so that form loses the whole answer. Here |zeta| is bounded, so
    // integrate (tau + zeta) e^{i psi} in closed form instead:
    //   int (tau + zeta) e^{i psi} dtau = [e^{i psi}] / (2 pi i fdot0),
    //   moment = root [e^{i psi}] / (2 pi i fdot0) - zeta * kernel.
    // The endpoint difference is taken as e^{i psi_start} (e^{i dpsi} - 1) = e^{i psi_start} *
    // 2i sin(dpsi/2) e^{i dpsi/2}, which keeps its accuracy when dpsi is small.
    double half = 0.5 * (psi_end - psi_start);
    cmplx difference = (2.0 * std::sin(half)) *
                       (gcmplx::polar(1.0, psi_start + half) * cmplx(0.0, 1.0));
    moment = cmplx(0.0, -root / (2.0 * M_PI * fdot0)) * difference - zeta * kernel;
  } else {
    moment = oscillating_moment;
  }
  *moment_out = origin * moment;
}

// Seven-term Tukey decomposition of the windowed Fourier value, without the amplitude and
// phase prefactor. moment_out may be null; when it is not, every term also returns its first
// moment about t_ref, and the two are combined with the same weights and taper rotations.
CUDA_DEVICE
void STFTFresnel::get_windowed_fourier_core(double f0, double fdot0, double t0,
                                            double f, cmplx* kernel_out,
                                            cmplx* moment_out) {
  double t_end = t0 + dt;
  double t_roll_on = t0 + taper_duration;
  double t_roll_off = t_end - taper_duration;

  // Chirp reference: bin start (default) or bin midpoint (more accurate).  The
  // Fourier-transform origin is always the window start t0, so each sub-term
  // automatically picks up the correct per-effective-frequency compensating
  // phase inside get_phase_kernel_core and the output stays in the standard
  // STFT convention.  Integration bounds remain the physical window times.
  double t_ref = use_midpoint ? (t0 + 0.5 * dt) : t0;
  bool with_moment = (moment_out != nullptr);

  // The seven terms, in order: the rectangular window, then the left ramp's DC and its two
  // taper sidebands, then the same three for the right ramp.
  const double f_shift[7] = {0.0, 0.0, -f_taper, +f_taper, 0.0, -f_taper, +f_taper};
  const double t_lo[7] = {t0, t0, t0, t0, t_roll_off, t_roll_off, t_roll_off};
  const double t_hi[7] = {t_end, t_roll_on, t_roll_on, t_roll_on,
                          t_end, t_end,     t_end};

  cmplx term[7], moment[7];
  for (int i = 0; i < 7; i += 1) {
    get_phase_kernel_core(f + f_shift[i], t_ref, f0, fdot0, t_lo[i], t_hi[i], t0,
                          &term[i], with_moment ? &moment[i] : nullptr);
  }

  // The right-ramp half-cosine is referenced to the segment END,
  // cos(pi*(t_end - t)/taper). Re-expressed as frequency shifts referenced to
  // the Fourier origin t0 it carries constant phases exp(-/+ 2 pi i f_taper*dt)
  // on the (f -/+ f_taper) terms. Those are unity iff f_taper*dt = 1/alpha is an
  // integer (true for all historical alphas: 0.1, 0.5, 1.0); omitting them for
  // an off-grid taper silently degraded the windowed template to mm ~ 1e-2
  // (found 2026-07-02 with taper = 1e4 s at dt = 1 day, alpha = 0.2315).
  double taper_rot = 2.0 * M_PI * f_taper * dt;
  cmplx right_rot_p = gcmplx::polar(1.0, -taper_rot);  // multiplies (f - f_taper)
  cmplx right_rot_m = gcmplx::polar(1.0, +taper_rot);  // multiplies (f + f_taper)

  *kernel_out = term[0] - 0.5 * (term[1] + term[4]) -
                0.25 * (term[2] + term[3] + right_rot_p * term[5] +
                        right_rot_m * term[6]);
  if (!with_moment) return;
  *moment_out = moment[0] - 0.5 * (moment[1] + moment[4]) -
                0.25 * (moment[2] + moment[3] + right_rot_p * moment[5] +
                        right_rot_m * moment[6]);
}

CUDA_DEVICE
cmplx STFTFresnel::get_windowed_fourier_kernel(double f0, double fdot0,
                                               double t0, double f) {
  cmplx kernel;
  get_windowed_fourier_core(f0, fdot0, t0, f, &kernel, nullptr);
  return kernel;
}

CUDA_DEVICE
cmplx STFTFresnel::get_windowed_fourier_value(double amp, double phase0,
                                              double f0, double fdot0,
                                              double t0, double f,
                                              double slope) {
  // account the effect of a tukey window on the fourier value.
  // window_factor is ignored whenever the Tukey evaluator runs (see get_fourier_value), so the
  // 1.0 passed here never reaches the returned prefactor.
  cmplx overall_factor = get_fourier_prefactor(amp, phase0, fdot0, 1.0);

  // Linear-envelope correction: each sub-interval term also gets its own
  // first moment (same 7-term Tukey decomposition, same weights and taper
  // rotations), added as slope * M below. The core returns each term's
  // {value, moment} in one pass -- the values are bit-identical to the
  // moment-free path's, so gating only on (linear_envelope, slope) keeps the
  // value path invariant. slope == 0.0 (astro-fallback columns) skips the
  // moment work entirely: the correction would be exactly zero.
  bool with_moment = linear_envelope && (slope != 0.0);

  // * Without the moment the value is the prefactor times the shared kernel.
  if (!with_moment)
    return overall_factor * get_windowed_fourier_kernel(f0, fdot0, t0, f);

  cmplx kernel, moment_kernel;
  get_windowed_fourier_core(f0, fdot0, t0, f, &kernel, &moment_kernel);
  cmplx out = overall_factor * kernel;
  cmplx moment = overall_factor * moment_kernel;
  return out + slope * moment;
}

CUDA_DEVICE
cmplx STFTFresnel::get_fourier_value(double amp, double phase0, double f0,
                                     double fdot0, double t0, double f,
                                     double window_factor, double slope) {
  // NOTE(window_factor): the scale is applied ONLY on the unwindowed
  // (window_alpha == 0) path below. With window_alpha > 0 the Tukey
  // evaluator models the window exactly and window_factor is intentionally
  // ignored -- callers must not expect it to act on the windowed path.
  if (window_alpha > 0.0)
    return get_windowed_fourier_value(amp, phase0, f0, fdot0, t0, f, slope);

  cmplx pref = get_fourier_prefactor(amp, phase0, fdot0, window_factor);

  // Linear-envelope correction: out += slope * (i/2pi) dF/df, anchored at
  // t_ref, computed by the fused evaluator (value bit-identical to the plain
  // path; Fresnel endpoints, sincos, and polar shared). Fully gated: OFF, or
  // slope == 0.0 (astro-fallback columns, batch API) -> byte-identical
  // const-envelope value.
  if (linear_envelope && slope != 0.0) {
    // Chirp reference: bin start (default) or bin midpoint (more accurate). The
    // Fourier-transform origin stays at the window start t0; the compensating
    // phase added in get_phase_kernel_core keeps the standard STFT convention.
    double t_ref = use_midpoint ? (t0 + 0.5 * dt) : t0;
    cmplx phase_kernel, moment_kernel;
    get_phase_kernel_core(f, t_ref, f0, fdot0, t0, t0 + dt, t0, &phase_kernel,
                          &moment_kernel);
    return pref * phase_kernel + slope * pref * moment_kernel;
  }

  return pref * get_fourier_kernel(f0, fdot0, t0, f);
}

// * The single home of the amplitude and phase prefactor: both get_fourier_value and
// * get_windowed_fourier_value take theirs from here, so the value and the
// * prefactor-times-kernel split the information-matrix kernel uses cannot drift apart.
CUDA_DEVICE
cmplx STFTFresnel::get_fourier_prefactor(double amp, double phase0,
                                         double fdot0, double window_factor) {
  if (window_alpha > 0.0)
    return gcmplx::polar(amp / std::sqrt(2.0 * std::abs(fdot0)), phase0);
  return gcmplx::polar(window_factor * amp / std::sqrt(2.0 * std::abs(fdot0)),
                       phase0);
}

CUDA_DEVICE
cmplx STFTFresnel::get_fourier_kernel(double f0, double fdot0, double t0,
                                      double f) {
  if (window_alpha > 0.0) return get_windowed_fourier_kernel(f0, fdot0, t0, f);
  double t_ref = use_midpoint ? (t0 + 0.5 * dt) : t0;
  return get_phase_kernel_product(f, t_ref, f0, fdot0, t0, t0 + dt, t0);
}

// ============================================================
// Batched Fresnel Fourier-value kernel (map, not reduce)
// ============================================================

CUDA_KERNEL
void compute_phase_kernel_moments_kernel(cmplx* kernel_out, cmplx* moment_out,
                                         STFTFresnel fresnel, double* f_effs,
                                         double* t_refs, double* f0s,
                                         double* fdot0s, double* t_starts,
                                         double* t_ends, double* t_origins,
                                         int num) {
#ifdef __CUDACC__
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num) return;
#else
  for (int i = 0; i < num; i++) {
#endif

  fresnel.get_phase_kernel_core(f_effs[i], t_refs[i], f0s[i], fdot0s[i],
                                t_starts[i], t_ends[i], t_origins[i],
                                &kernel_out[i], &moment_out[i]);

#ifndef __CUDACC__
  }
#endif
}

void STFTFresnel::compute_phase_kernel_moments_wrap(
    cmplx* kernel_out, cmplx* moment_out, double* f_effs, double* t_refs,
    double* f0s, double* fdot0s, double* t_starts, double* t_ends,
    double* t_origins, int num) {
#ifdef __CUDACC__
  int num_blocks = (num + NUM_THREADS - 1) / NUM_THREADS;
  compute_phase_kernel_moments_kernel<<<num_blocks, NUM_THREADS>>>(
      kernel_out, moment_out, *this, f_effs, t_refs, f0s, fdot0s, t_starts,
      t_ends, t_origins, num);
  gpuErrchk(cudaGetLastError());
#else
  compute_phase_kernel_moments_kernel(kernel_out, moment_out, *this, f_effs,
                                      t_refs, f0s, fdot0s, t_starts, t_ends,
                                      t_origins, num);
#endif
}

// * Goes through the two members, so that a test also covers their delegation to lat_fresnel.hh.
CUDA_KERNEL
void compute_fresnel_functions_kernel(double* cosine_out, double* sine_out,
                                      double* aux_f_out, double* aux_g_out,
                                      double* slope_f_out, double* slope_g_out,
                                      STFTFresnel fresnel, double* arguments,
                                      int num) {
#ifdef __CUDACC__
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num) return;
#else
  for (int i = 0; i < num; i++) {
#endif

  fresnel.get_fresnel_integrals(&cosine_out[i], &sine_out[i], arguments[i]);
  fresnel.get_fresnel_aux(&aux_f_out[i], &aux_g_out[i], std::abs(arguments[i]),
                          &slope_f_out[i], &slope_g_out[i]);

#ifndef __CUDACC__
  }
#endif
}

void STFTFresnel::compute_fresnel_functions_wrap(
    double* cosine_out, double* sine_out, double* aux_f_out, double* aux_g_out,
    double* slope_f_out, double* slope_g_out, double* arguments, int num) {
#ifdef __CUDACC__
  int num_blocks = (num + NUM_THREADS - 1) / NUM_THREADS;
  compute_fresnel_functions_kernel<<<num_blocks, NUM_THREADS>>>(
      cosine_out, sine_out, aux_f_out, aux_g_out, slope_f_out, slope_g_out,
      *this, arguments, num);
  gpuErrchk(cudaGetLastError());
#else
  compute_fresnel_functions_kernel(cosine_out, sine_out, aux_f_out, aux_g_out,
                                   slope_f_out, slope_g_out, *this, arguments,
                                   num);
#endif
}

/**
 * @brief Compute Fresnel-based Fourier values for a batch of binaries.
 *
 * This is a map kernel: each thread computes one (binary, freq) output
 * element independently — no shared memory or reduction needed.
 *
 * @param output   Output array, shape [num_binaries * num_freqs]
 * @param fresnel  STFTFresnel object (device copy on GPU)
 * @param amps     Amplitude per binary [num_binaries]
 * @param phase0s  Initial phase per binary [num_binaries]
 * @param f0s      Reference frequency per binary [num_binaries]
 * @param fdot0s   Frequency derivative per binary [num_binaries]
 * @param t0s      Reference time per binary [num_binaries]
 * @param freqs    Evaluation frequencies [num_binaries * num_freqs]
 * @param window_factor  Pre-computed window factor to apply to all outputs.
 * this should be \sum w_i / N_window, where w_i are the window weights for the
 * current STFT window and N_window is the number of time bins in the window.
 * @param num_binaries  Number of sources in the batch
 * @param num_freqs     Number of frequency points per source
 *
 * @note linear_envelope has NO effect on this batch API (by design): the
 * call carries no response stencil, so there is no per-segment amplitude
 * slope to apply -- get_fourier_value is invoked with its default slope=0.
 * The envelope correction acts only on the likelihood/fill paths, where
 * FresnelColumn::setup derives the slope from the +-D TDI samples.
 */
CUDA_KERNEL
void compute_fourier_values_kernel(cmplx* output, STFTFresnel fresnel,
                                   double* amps, double* phase0s, double* f0s,
                                   double* fdot0s, double* t0s, double* freqs,
                                   double window_factor, int num_binaries,
                                   int num_freqs) {
#ifdef __CUDACC__
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_binaries * num_freqs;
  if (tid >= total)
    return;
  int bin = tid / num_freqs;
  int f_idx = tid % num_freqs;
#else
  for (int bin = 0; bin < num_binaries; bin++) {
    for (int f_idx = 0; f_idx < num_freqs; f_idx++) {
#endif

  output[bin * num_freqs + f_idx] = fresnel.get_fourier_value(
      amps[bin], phase0s[bin], f0s[bin], fdot0s[bin], t0s[bin],
      freqs[bin * num_freqs + f_idx], window_factor);

#ifndef __CUDACC__
}
}  // close CPU loops
#endif
}

/**
 * Host wrapper: launch the batched Fresnel kernel for a batch of binaries.
 */
void STFTFresnel::compute_fourier_values_wrap(cmplx* output, double* amps,
                                              double* phase0s, double* f0s,
                                              double* fdot0s, double* t0s,
                                              double* freqs,
                                              double window_factor,
                                              int num_binaries, int num_freqs) {
#ifdef __CUDACC__
  int total = num_binaries * num_freqs;
  int num_blocks = (total + NUM_THREADS - 1) / NUM_THREADS;

  compute_fourier_values_kernel<<<num_blocks, NUM_THREADS>>>(
      output, *this, amps, phase0s, f0s, fdot0s, t0s, freqs, window_factor,
      num_binaries, num_freqs);

  gpuErrchk(cudaGetLastError());
#else
      compute_fourier_values_kernel(output, *this, amps, phase0s, f0s, fdot0s,
                                    t0s, freqs, window_factor, num_binaries,
                                    num_freqs);
#endif
}

/**
 * Pass-1 kernel: accumulate per-block partial sums of (d|h) and (h|h).
 *
 * Grid layout (GPU, 2-D):
 *   blockIdx.y ∈ [0, num_binaries)  — identifies the source.
 *   blockIdx.x ∈ [0, num_blocks_x)  — tile along the flattened (t,f) index.
 *
 * Each thread handles one or more (t_local, f_local) positions within the
 * binary's template sub-grid, accumulating contributions in shared memory.
 * At the end of the inner loop, CUB's block-reduce sums all per-thread values
 * and thread 0 writes the block partial sum to d_h_contrib / h_h_contrib.
 *
 * The factor 4 applied to the partial sums comes from the (one-sided) inner
 * product convention: <a|b>_1-sided = 4 Re ∫ a*(f) C^{-1}(f) b(f) df.
 * (The final likelihood uses 4·Re(d_h_out) - 2·Re(h_h_out).)
 *
 * CPU fallback: the same function body is compiled as a serial loop;
 * d_h_contrib[bin] and h_h_contrib[bin] are written directly.
 *
 * @param d_h_contrib   Output partial sums (d|h): shape [num_binaries,
 * num_blocks_x]
 * @param h_h_contrib   Output partial sums (h|h): shape [num_binaries,
 * num_blocks_x]
 * @param domain        Device pointer to the STFTDomain object
 * @param template_vals Template: [num_binaries, num_channels,
 *                                 num_times_template, num_freqs_template]
 * @param start_times_all  Physical start time for each template sub-grid [s]
 * @param start_freqs_all  Physical start frequency for each template sub-grid
 * [Hz]
 * @param num_binaries   Batch size
 * @param data_index_all  Which data realisation to use per binary
 * @param noise_index_all Which noise realisation to use per binary
 * @param num_times_template  Time bins per template sub-grid
 * @param num_freqs_template  Frequency bins per template sub-grid
 */
CUDA_KERNEL
void compute_likelihood_contributions_kernel(
    cmplx* d_h_contrib,  // [num_binaries * num_blocks_x] partial sums for (d|h)
    cmplx* h_h_contrib,  // [num_binaries * num_blocks_x] partial sums for (h|h)
    STFTDomain domain,
    cmplx* template_vals,  // [num_binaries, num_channels, num_times_template,
                           // num_freqs_template]
    double* start_times_all,  // [num_binaries] physical start time per source
    double*
        start_freqs_all,  // [num_binaries] physical start frequency per source
    int num_binaries,
    int* data_index_all,   // [num_binaries] data instance index per source
    int* noise_index_all,  // [num_binaries] noise instance index per source
    int num_times_template, int num_freqs_template) {
  int tid;                  // thread index within the block (GPU) or 0 (CPU)
  int start_bin, incr_bin;  // binary loop bounds
  int start_idx, incr_idx;  // flat (t,f) loop bounds within this block

#ifdef __CUDACC__
  tid = threadIdx.x;
  // Y dimension maps to binaries; each binary is processed by one row of
  // blocks.
  start_bin = blockIdx.y;
  incr_bin = gridDim.y;
  // X dimension spreads (t,f) pixels across the block grid.
  start_idx = blockIdx.x * blockDim.x + threadIdx.x;
  incr_idx = blockDim.x * gridDim.x;
  // Per-block shared accumulators, one entry per thread.
  CUDA_SHARED cmplx d_h_tmp[NUM_THREADS];
  CUDA_SHARED cmplx h_h_tmp[NUM_THREADS];
#else
                                // CPU: single thread processes everything
                                // serially.
      tid = 0;
      start_bin = 0;
      incr_bin = 1;
      start_idx = 0;
      incr_idx = 1;
      cmplx d_h_tmp[1];
      cmplx h_h_tmp[1];
#endif

  int total_tf = num_times_template * num_freqs_template;

  for (int bin = start_bin; bin < num_binaries; bin += incr_bin) {
    d_h_tmp[tid] = cmplx(0.0, 0.0);
    h_h_tmp[tid] = cmplx(0.0, 0.0);
#ifdef __CUDACC__
    CUDA_SYNC_THREADS;
#endif

    int data_index = data_index_all[bin];
    int noise_index = noise_index_all[bin];
    // For FDDomain (num_times=1), start_times_all is nullptr and dt=0,
    // so we skip get_time_index and default to t_idx=0.
    int start_t_idx = (start_times_all != nullptr)
                          ? domain.get_time_index(start_times_all[bin])
                          : 0;
    int start_f_idx = domain.get_freq_index(start_freqs_all[bin]);

    int num_ch = domain.num_channels;
    // Base offset into template_vals for this binary (row-major).
    int template_base = bin * num_ch * num_times_template * num_freqs_template;

    for (int idx = start_idx; idx < total_tf; idx += incr_idx) {
      int t_local = idx / num_freqs_template;
      int f_local = idx % num_freqs_template;
      int t_idx = start_t_idx + t_local;
      int f_idx = start_f_idx + f_local;

      // Bounds guard: skip template pixels placed outside the data grid.  Two
      // ways this happens: (1) the waveform's STFT window count
      // ceil(N/nperseg) = NT+1 exceeds the data grid num_times = NT (the extra
      // trailing row is safe-masked to zero), and (2) a start freq/time at the
      // band/window edge (get_freq_index / get_time_index round to the nearest
      // bin).  get_data_index / get_noise_index do no bounds checking and the
      // CUDA kernel cannot throw, so an out-of-grid t_idx/f_idx reads past (or
      // before) the data / invC buffers -> CUDA_ERROR_ILLEGAL_ADDRESS.
      // Out-of-grid pixels have no data and contribute zero to the band-limited
      // inner product, so skipping them is correct.
      if (t_idx < 0 || t_idx >= domain.num_times ||
          f_idx < 0 || f_idx >= domain.num_freqs) {
        continue;
      }

      cmplx h_vals[3];
      for (int ch = 0; ch < num_ch; ch++) {
        h_vals[ch] = template_vals[template_base +
                                   (ch * num_times_template + t_local) *
                                       num_freqs_template +
                                   f_local];
      }

      domain.add_ip_contrib(d_h_tmp, h_h_tmp, h_vals, t_idx, f_idx, data_index,
                            noise_index);
    }

#ifdef __CUDACC__
    // Reduce all per-thread contributions within this block using CUB.
    // The factor 4 comes from the one-sided inner-product convention.
    CUDA_SYNC_THREADS;
    cmplx d_h_red = 4.0 * domain.diff_comp * block_reduce_cmplx(d_h_tmp);
    // Must sync again: CUB's TempStorage must not be overwritten until
    // all threads have completed the first reduction.
    CUDA_SYNC_THREADS;
    cmplx h_h_red = 4.0 * domain.diff_comp * block_reduce_cmplx(h_h_tmp);
    if (tid == 0) {
      // Store this block's partial sum; Pass 2 will reduce across blocks.
      d_h_contrib[bin * gridDim.x + blockIdx.x] = d_h_red;
      h_h_contrib[bin * gridDim.x + blockIdx.x] = h_h_red;
    }
    CUDA_SYNC_THREADS;
#else
        // CPU: num_blocks_x == 1, so write directly at index [bin].
        d_h_contrib[bin] = 4.0 * domain.diff_comp * d_h_tmp[0];
        h_h_contrib[bin] = 4.0 * domain.diff_comp * h_h_tmp[0];
#endif
  }
}

/**
 * Pass-2 kernel: reduce per-block partial sums to per-binary scalar results.
 *
 * Grid layout (GPU): gridDim = (1, num_binaries).  Each block (blockIdx.y)
 * is responsible for one binary.  Threads load slices of the partial-sum
 * buffer into shared memory and perform a classic tree-based reduction.
 *
 * CPU fallback: the single partial sum is copied directly to the output.
 *
 * @param d_h_final         Output (d|h) per binary, shape [num_binaries]
 * @param h_h_final         Output (h|h) per binary, shape [num_binaries]
 * @param d_h_contrib       Input partial sums: [num_binaries,
 * num_blocks_per_bin]
 * @param h_h_contrib       Input partial sums: [num_binaries,
 * num_blocks_per_bin]
 * @param num_blocks_per_bin  gridDim.x from the first-pass launch
 * @param num_binaries       Batch size
 */
CUDA_KERNEL
void like_sum_from_contrib_cmplx(
    cmplx* d_h_final,    // [num_binaries] final (d|h)
    cmplx* h_h_final,    // [num_binaries] final (h|h)
    cmplx* d_h_contrib,  // [num_binaries * num_blocks_per_bin] partial sums
    cmplx* h_h_contrib,  // [num_binaries * num_blocks_per_bin] partial sums
    int num_blocks_per_bin, int num_binaries) {
  int tid;
  int bin_i, incr_bin;

#ifdef __CUDACC__
  tid = threadIdx.x;
  bin_i = blockIdx.y;
  incr_bin = gridDim.y;
  CUDA_SHARED cmplx shared_d_h[NUM_THREADS];
  CUDA_SHARED cmplx shared_h_h[NUM_THREADS];
#else
      tid = 0;
      bin_i = 0;
      incr_bin = 1;
      cmplx shared_d_h[1];
      cmplx shared_h_h[1];
#endif

  for (int bin = bin_i; bin < num_binaries; bin += incr_bin) {
    cmplx sum_d_h = cmplx(0.0, 0.0);
    cmplx sum_h_h = cmplx(0.0, 0.0);

#ifdef __CUDACC__
    for (int i = tid; i < num_blocks_per_bin; i += blockDim.x)
#else
        for (int i = 0; i < num_blocks_per_bin; i++)
#endif
    {
      sum_d_h += d_h_contrib[bin * num_blocks_per_bin + i];
      sum_h_h += h_h_contrib[bin * num_blocks_per_bin + i];
    }

    shared_d_h[tid] = sum_d_h;
    shared_h_h[tid] = sum_h_h;

#ifdef __CUDACC__
    CUDA_SYNC_THREADS;
    // Tree-based reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shared_d_h[tid] = shared_d_h[tid] + shared_d_h[tid + s];
        shared_h_h[tid] = shared_h_h[tid] + shared_h_h[tid + s];
      }
      CUDA_SYNC_THREADS;
    }
#endif

    if (tid == 0) {
      d_h_final[bin] = shared_d_h[0];
      h_h_final[bin] = shared_h_h[0];
    }
  }
}

// ============================================================
// WDM batched likelihood kernels — WDM counterparts of the STFT
// two-pass kernels above (2026-06 merge follow-up).
//
// Differences from the STFT pair:
//   - WDM coefficients and inverse-noise weights are real doubles, so all
//     accumulators/outputs are double (no complex conjugation needed).
//   - Template sub-grids are addressed by integer (m, n) start indices on
//     the full WDM grid (the WDMDomain pixel getters offset by
//     ind_min_f / ind_min_t internally) instead of physical start
//     times/frequencies.
//   - The finalization factor is a bare 4.0: WDMDomain's per-pixel
//     primitives already carry the WDM differential component 0.25
//     (see get_inner_product_value), so 4 * sum(d*h*invC*0.25) reproduces
//     the Python convention 4 * sum(...) * differential_component.
//   - tdi_type is a per-call argument (matching the other WDMDomain
//     helpers) rather than a stored member.
// ============================================================

/**
 * Pass-1 kernel: accumulate per-block partial sums of (d|h) and (h|h) on
 * the WDM grid.
 *
 * Grid layout (GPU, 2-D):
 *   blockIdx.y ∈ [0, num_binaries)  — identifies the source.
 *   blockIdx.x ∈ [0, num_blocks_x)  — tile along the flattened (m,n) index.
 *
 * CPU fallback: a single serial loop writes d_h_contrib[bin] directly.
 *
 * @param d_h_contrib   Output partial sums (d|h): [num_binaries, num_blocks_x]
 * @param h_h_contrib   Output partial sums (h|h): [num_binaries, num_blocks_x]
 * @param domain        WDMDomain object (passed by value; pointer fields are
 *                      device pointers in the GPU build)
 * @param template_vals Template: [num_binaries, num_channel, n_m_template,
 *                      n_n_template] (n fastest, matching wdm_data layout)
 * @param start_layer_m_all  Absolute start frequency-layer m per binary
 * @param start_time_n_all   Absolute start time-bin n per binary
 * @param num_binaries  Batch size
 * @param data_index_all  Which data realisation to use per binary
 * @param noise_index_all Which noise realisation to use per binary
 * @param n_m_template  Frequency layers per template sub-grid
 * @param n_n_template  Time bins per template sub-grid
 * @param tdi_type      TDI_XYZ / TDI_AET / TDI_AE
 */
CUDA_KERNEL
void wdm_compute_likelihood_contributions_kernel(
    double* d_h_contrib, double* h_h_contrib, WDMDomain domain,
    double* template_vals, int* start_layer_m_all, int* start_time_n_all,
    int num_binaries, int* data_index_all, int* noise_index_all,
    int n_m_template, int n_n_template, int tdi_type) {
  int tid;                  // thread index within the block (GPU) or 0 (CPU)
  int start_bin, incr_bin;  // binary loop bounds
  int start_idx, incr_idx;  // flat (m,n) loop bounds within this block

#ifdef __CUDACC__
  tid = threadIdx.x;
  start_bin = blockIdx.y;
  incr_bin = gridDim.y;
  start_idx = blockIdx.x * blockDim.x + threadIdx.x;
  incr_idx = blockDim.x * gridDim.x;
  CUDA_SHARED double d_h_tmp[NUM_THREADS];
  CUDA_SHARED double h_h_tmp[NUM_THREADS];
#else
      tid = 0;
      start_bin = 0;
      incr_bin = 1;
      start_idx = 0;
      incr_idx = 1;
      double d_h_tmp[1];
      double h_h_tmp[1];
#endif

  int total_mn = n_m_template * n_n_template;

  for (int bin = start_bin; bin < num_binaries; bin += incr_bin) {
    d_h_tmp[tid] = 0.0;
    h_h_tmp[tid] = 0.0;
#ifdef __CUDACC__
    CUDA_SYNC_THREADS;
#endif

    int data_index = data_index_all[bin];
    int noise_index = noise_index_all[bin];
    int start_m = start_layer_m_all[bin];
    int start_n = start_time_n_all[bin];

    int num_ch = domain.num_channel;
    // Base offset into template_vals for this binary (row-major).
    int template_base = bin * num_ch * n_m_template * n_n_template;

    for (int idx = start_idx; idx < total_mn; idx += incr_idx) {
      int m_local = idx / n_n_template;
      int n_local = idx % n_n_template;
      int m = start_m + m_local;
      int n = start_n + n_local;

      double w_mn[3];
      for (int ch = 0; ch < num_ch; ch++) {
        w_mn[ch] = template_vals[template_base +
                                 (ch * n_m_template + m_local) * n_n_template +
                                 n_local];
      }

      // add_ip_contrib accumulates into d_h_tmp[tid]/h_h_tmp[tid] and
      // dispatches XYZ-cross vs AET/AE-diagonal on tdi_type internally.
      domain.add_ip_contrib(d_h_tmp, h_h_tmp, w_mn, m, n, data_index,
                            noise_index, tdi_type);
    }

#ifdef __CUDACC__
    // The factor 4 pairs with the 0.25 already inside the per-pixel
    // primitives — see the block comment above.
    CUDA_SYNC_THREADS;
    double d_h_red = 4.0 * block_reduce_double(d_h_tmp);
    // Must sync again: CUB's TempStorage must not be overwritten until
    // all threads have completed the first reduction.
    CUDA_SYNC_THREADS;
    double h_h_red = 4.0 * block_reduce_double(h_h_tmp);
    if (tid == 0) {
      d_h_contrib[bin * gridDim.x + blockIdx.x] = d_h_red;
      h_h_contrib[bin * gridDim.x + blockIdx.x] = h_h_red;
    }
    CUDA_SYNC_THREADS;
#else
        // CPU: num_blocks_x == 1, so write directly at index [bin].
        d_h_contrib[bin] = 4.0 * d_h_tmp[0];
        h_h_contrib[bin] = 4.0 * h_h_tmp[0];
#endif
  }
}

/**
 * Pass-2 kernel: reduce per-block partial sums to per-binary scalar results.
 * Real-valued mirror of like_sum_from_contrib_cmplx; identical structure.
 */
CUDA_KERNEL
void like_sum_from_contrib_real(
    double* d_h_final,    // [num_binaries] final (d|h)
    double* h_h_final,    // [num_binaries] final (h|h)
    double* d_h_contrib,  // [num_binaries * num_blocks_per_bin] partial sums
    double* h_h_contrib,  // [num_binaries * num_blocks_per_bin] partial sums
    int num_blocks_per_bin, int num_binaries) {
  int tid;
  int bin_i, incr_bin;

#ifdef __CUDACC__
  tid = threadIdx.x;
  bin_i = blockIdx.y;
  incr_bin = gridDim.y;
  CUDA_SHARED double shared_d_h[NUM_THREADS];
  CUDA_SHARED double shared_h_h[NUM_THREADS];
#else
      tid = 0;
      bin_i = 0;
      incr_bin = 1;
      double shared_d_h[1];
      double shared_h_h[1];
#endif

  for (int bin = bin_i; bin < num_binaries; bin += incr_bin) {
    double sum_d_h = 0.0;
    double sum_h_h = 0.0;

#ifdef __CUDACC__
    for (int i = tid; i < num_blocks_per_bin; i += blockDim.x)
#else
        for (int i = 0; i < num_blocks_per_bin; i++)
#endif
    {
      sum_d_h += d_h_contrib[bin * num_blocks_per_bin + i];
      sum_h_h += h_h_contrib[bin * num_blocks_per_bin + i];
    }

    shared_d_h[tid] = sum_d_h;
    shared_h_h[tid] = sum_h_h;

#ifdef __CUDACC__
    CUDA_SYNC_THREADS;
    // Tree-based reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shared_d_h[tid] = shared_d_h[tid] + shared_d_h[tid + s];
        shared_h_h[tid] = shared_h_h[tid] + shared_h_h[tid + s];
      }
      CUDA_SYNC_THREADS;
    }
#endif

    if (tid == 0) {
      d_h_final[bin] = shared_d_h[0];
      h_h_final[bin] = shared_h_h[0];
    }
  }
}

/**
 * Host wrapper: launch the two-pass GPU WDM likelihood kernels (or the
 * equivalent CPU loop) for a batch of num_binaries sources. Mirrors
 * STFTDomain::compute_likelihood_terms_wrap; see domains.hpp for the
 * parameter documentation.
 */
void WDMDomain::compute_likelihood_terms_wrap(
    double* d_h_out, double* h_h_out, double* template_vals,
    int* start_layer_m_all, int* start_time_n_all, int num_binaries,
    int* data_index_all, int* noise_index_all, int n_m_template,
    int n_n_template, int tdi_type, bool run_async) {
#ifdef __CUDACC__
  double* d_h_contrib;
  double* h_h_contrib;
  // Number of blocks along the (m,n) dimension; each block reduces
  // NUM_THREADS pixels and contributes one partial-sum entry.
  int num_blocks_x =
      (n_m_template * n_n_template + NUM_THREADS - 1) / NUM_THREADS;
  int num_blocks_y = num_binaries;  // one row of blocks per binary
  dim3 grid_dim(num_blocks_x, num_blocks_y);

  // Allocate partial-sum buffers: [num_binaries, num_blocks_x]
  if (run_async) {
    gpuErrchk(cudaMallocAsync(&d_h_contrib,
                              num_binaries * num_blocks_x * sizeof(double),
                              cudaStreamDefault));
    gpuErrchk(cudaMallocAsync(&h_h_contrib,
                              num_binaries * num_blocks_x * sizeof(double),
                              cudaStreamDefault));
  } else {
    gpuErrchk(cudaMalloc(&d_h_contrib,
                         num_binaries * num_blocks_x * sizeof(double)));
    gpuErrchk(cudaMalloc(&h_h_contrib,
                         num_binaries * num_blocks_x * sizeof(double)));
  }

  // Pass 1: compute per-block partial sums of (d|h) and (h|h). `*this` is
  // passed by value (the struct is copied into kernel-arg space, so the
  // host-pointer-dereference pitfall does not apply; its wdm_data/wdm_noise
  // fields are already device pointers).
  wdm_compute_likelihood_contributions_kernel<<<grid_dim, NUM_THREADS>>>(
      d_h_contrib, h_h_contrib, *this, template_vals, start_layer_m_all,
      start_time_n_all, num_binaries, data_index_all, noise_index_all,
      n_m_template, n_n_template, tdi_type);
  // Pass 2: reduce partial sums across blocks for each binary.
  dim3 reduce_grid_dim(1, num_binaries, 1);  // one block per binary
  like_sum_from_contrib_real<<<reduce_grid_dim, NUM_THREADS>>>(
      d_h_out, h_h_out, d_h_contrib, h_h_contrib, num_blocks_x, num_binaries);

  gpuErrchk(cudaGetLastError());
  if (run_async) {
    gpuErrchk(cudaFreeAsync(d_h_contrib, cudaStreamDefault));
    gpuErrchk(cudaFreeAsync(h_h_contrib, cudaStreamDefault));
  } else {
    gpuErrchk(cudaFree(d_h_contrib));
    gpuErrchk(cudaFree(h_h_contrib));
    cudaDeviceSynchronize();
  }

#else
  // CPU path: the kernel function is a plain C++ function.  Results are
  // written directly to d_h_out / h_h_out (no intermediate buffers needed).
  wdm_compute_likelihood_contributions_kernel(
      d_h_out, h_h_out, *this, template_vals, start_layer_m_all,
      start_time_n_all, num_binaries, data_index_all, noise_index_all,
      n_m_template, n_n_template, tdi_type);
#endif
};