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
 * @see domains.hpp for class declarations and detailed parameter documentation.
 */

#include <iostream>
#include "domains.hpp"

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
CUDA_DEVICE
cmplx block_reduce_cmplx(cmplx* array) {
  using BlockReduce = cub::BlockReduce<cmplx, NUM_THREADS>;
  CUDA_SHARED typename BlockReduce::TempStorage temp_storage;
  // Synchronise before reading: ensures all threads have written their element.
  CUDA_SYNC_THREADS;
  int tid = threadIdx.x;
  cmplx thread_data = array[tid];
  cmplx output = BlockReduce(temp_storage).Reduce(thread_data, ComplexSum());
  return output;
}
#endif

// ============================================================
// Data indexing
// ============================================================

/**
 * Convert a physical time t [s] to its zero-based grid index.
 * Returns -1 on GPU if out of bounds; throws on CPU.
 */
CUDA_DEVICE
int STFTDomain::get_time_index(double t) {
  if (t < t0 || t > t0 + num_times * dt) {
#ifdef __CUDACC__
// On GPU we cannot throw; the caller should verify the template
// sub-grid lies within the domain grid before launching.
#else
    throw std::invalid_argument("Time t is out of bounds of the STFT domain.");
#endif
    return -1;
  }
  return (int)((t - t0) / dt);
}

/**
 * Convert a physical frequency f [Hz] to its zero-based grid index.
 * Returns -1 on GPU if out of bounds; throws on CPU.
 */
CUDA_DEVICE
int STFTDomain::get_freq_index(double f) {
  if (f < f_min || f > f_max) {
#ifdef __CUDACC__
#else
    throw std::invalid_argument(
        "Frequency f is out of bounds of the STFT domain.");
#endif
    return -1;
  }
  return (int)((f - f_min) / df);
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
  if (data_index > num_data) {
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
  if (noise_index > num_noise) {
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
  if (noise_index > num_noise) {
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
    int num_freqs_template) {
#ifdef __CUDACC__
  cmplx* d_h_contrib;
  cmplx* h_h_contrib;
  // Number of blocks along the (t,f) dimension; each block reduces
  // NUM_THREADS pixels and contributes one partial-sum entry.
  int num_blocks_x =
      std::ceil((num_times_template * num_freqs_template + NUM_THREADS - 1) /
                NUM_THREADS);
  int num_blocks_y = num_binaries;  // one row of blocks per binary
  dim3 grid_dim(num_blocks_x, num_blocks_y);

  // Allocate partial-sum buffers: [num_binaries, num_blocks_x]
  gpuErrchk(cudaMallocAsync(&d_h_contrib,
                            num_binaries * num_blocks_x * sizeof(cmplx),
                            cudaStreamDefault));
  gpuErrchk(cudaMallocAsync(&h_h_contrib,
                            num_binaries * num_blocks_x * sizeof(cmplx),
                            cudaStreamDefault));

  // Copy the host STFTDomain struct (including its device data/invC pointers)
  // to the device so the kernel can call member functions through the pointer.
  //   STFTDomain* domain_ptr;
  //   gpuErrchk(cudaMalloc(&domain_ptr, sizeof(STFTDomain)));
  //   gpuErrchk(
  //       cudaMemcpy(domain_ptr, this, sizeof(STFTDomain),
  //       cudaMemcpyHostToDevice));

  // Pass 1: compute per-block partial sums of (d|h) and (h|h).
  compute_likelihood_contributions_kernel<<<grid_dim, NUM_THREADS>>>(
      d_h_contrib, h_h_contrib, *this, template_vals, start_times_all,
      start_freqs_all, num_binaries, data_index_all, noise_index_all,
      num_times_template, num_freqs_template);
  // Pass 2: reduce partial sums across blocks for each binary.
  dim3 reduce_grid_dim(1, num_binaries, 1);  // one block per binary
  like_sum_from_contrib_cmplx<<<reduce_grid_dim, NUM_THREADS>>>(
      d_h_out, h_h_out, d_h_contrib, h_h_contrib, num_blocks_x, num_binaries);

  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaFreeAsync(d_h_contrib, cudaStreamDefault));
  gpuErrchk(cudaFreeAsync(h_h_contrib, cudaStreamDefault));
  //   gpuErrchk(cudaFree(domain_ptr));

#else
  // CPU path: the kernel function is a plain C++ function.  Results are
  // written directly to d_h_out / h_h_out (no intermediate buffers needed).
  compute_likelihood_contributions_kernel(
      d_h_out, h_h_out, *this, template_vals, start_times_all, start_freqs_all,
      num_binaries, data_index_all, noise_index_all, num_times_template,
      num_freqs_template);
#endif
};

void FDDomain::compute_likelihood_terms_wrap(
    cmplx* d_h_out, cmplx* h_h_out, cmplx* template_vals,
    double* start_freqs_all, int num_binaries, int* data_index_all,
    int* noise_index_all, int num_freqs_template) {
  // Delegate to the STFT version with num_times_template = 1.
  // start_times_all = nullptr signals the kernel to use start_t_idx = 0.
  STFTDomain::compute_likelihood_terms_wrap(
      d_h_out, h_h_out, template_vals,
      nullptr,  // start_times_all not used in FDDomain
      start_freqs_all, num_binaries, data_index_all, noise_index_all,
      1,  // num_times_template = 1 for FDDomain
      num_freqs_template);
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

CUDA_DEVICE
double STFTFresnel::get_v(double t, double f, double t0, double f0,
                          double fdot0) {
  double zeta = get_zeta(f, f0, fdot0);
  double v = std::sqrt(2.0 * std::abs(fdot0)) *
             (t - t0 + zeta);  // todo: check for negative fdot0
  return v;
}

CUDA_DEVICE
double STFTFresnel::get_auxiliary_f(double x) {
  double f = (1.0 + 0.926 * x) / (2.0 + 1.792 * x + 3.104 * x * x);
  return f;
}

CUDA_DEVICE
double STFTFresnel::get_auxiliary_g(double x) {
  double g = 1.0 / (2.0 + 4.142 * x + 3.492 * x * x + 6.670 * x * x * x);
  return g;
}

CUDA_DEVICE
void STFTFresnel::get_fresnel_integrals(double* C, double* S, double x) {
  double abs_x = std::abs(x);
  double pi_x = M_PI * abs_x;
  double half_pi_x2 = 0.5 * pi_x * abs_x;
  double c_halfpix2 = std::cos(half_pi_x2);
  double s_halfpix2 = std::sin(half_pi_x2);
  double S_val, C_val;

  double threshold = 6.0;

  if (abs_x > threshold) {
    S_val = 0.5 - 1 / pi_x * c_halfpix2;
    C_val = 0.5 + 1 / pi_x * s_halfpix2;
  } else {
    double f_x = get_auxiliary_f(abs_x);
    double g_x = get_auxiliary_g(abs_x);
    S_val = 0.5 - f_x * c_halfpix2 - g_x * s_halfpix2;
    C_val = 0.5 + f_x * s_halfpix2 - g_x * c_halfpix2;
  }

  if (x < 0) {
    *C = -C_val;  // Fresnel C integral
    *S = -S_val;  // Fresnel S integral
  } else {
    *C = C_val;  // Fresnel C integral
    *S = S_val;  // Fresnel S integral
  }
}

CUDA_DEVICE
cmplx STFTFresnel::get_fresnel_kernel(double f, double t0, double f0,
                                      double fdot0) {
  double v0 = get_v(t0, f, t0, f0, fdot0);
  double t1 = t0 + dt;  // End of the current STFT window. we are assuming that
                        // everything is correctly aligned with the stft grid
  double v1 = get_v(t1, f, t0, f0, fdot0);

  double C_0, S_0, C_1, S_1;
  get_fresnel_integrals(&C_0, &S_0, v0);
  get_fresnel_integrals(&C_1, &S_1, v1);

  double delta_C = C_1 - C_0;
  double delta_S = S_1 - S_0;
  cmplx kernel =
      (fdot0 >= 0.0) ? cmplx(delta_C, delta_S) : cmplx(delta_C, -delta_S);

  return kernel;
}

CUDA_DEVICE
cmplx STFTFresnel::get_fourier_value(double amp, double phase0, double f0,
                                     double fdot0, double t0, double f,
                                     double window_factor) {
  cmplx kernel = get_fresnel_kernel(f, t0, f0, fdot0);
  double amplitude =
      window_factor * amp /
      std::sqrt(2.0 * std::abs(fdot0));  // todo: check for negative fdot0?
  double zeta = get_zeta(f, f0, fdot0);
  double phase = phase0 - M_PI * fdot0 * zeta * zeta;

  cmplx out = gcmplx::polar(amplitude, phase) * kernel;
  return out;
}

// ============================================================
// Batched Fresnel Fourier-value kernel (map, not reduce)
// ============================================================

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

  //   STFTFresnel* dev_ptr;
  //   gpuErrchk(cudaMalloc(&dev_ptr, sizeof(STFTFresnel)));
  //   gpuErrchk(
  //       cudaMemcpy(dev_ptr, this, sizeof(STFTFresnel),
  //       cudaMemcpyHostToDevice));

  compute_fourier_values_kernel<<<num_blocks, NUM_THREADS>>>(
      output, *this, amps, phase0s, f0s, fdot0s, t0s, freqs, window_factor,
      num_binaries, num_freqs);

  gpuErrchk(cudaGetLastError());
  // gpuErrchk(cudaFree(dev_ptr));
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