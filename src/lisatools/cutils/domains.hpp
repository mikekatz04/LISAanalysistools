/**
 * @file domains.hpp
 * @brief Declarations for the STFT/FD domain classes and CUDA likelihood
 * kernels.
 *
 * This header defines:
 *  - STFTSettings / FDSettings  – lightweight POD structs holding grid
 * parameters (number of time bins, frequency bins, channels, physical extents).
 *  - STFTDomain / FDDomain      – domain objects that own pointers to the data
 *    array and the precomputed inverse-covariance array, together with all
 *    indexing helpers and per-pixel inner-product primitives.
 *  - The two-pass CUDA kernel pair:
 *      compute_likelihood_contributions_kernel  – first pass: partial (d|h) and
 * (h|h) like_sum_from_contrib_cmplx              – second pass: block-level
 * reduction
 *
 * Memory layout conventions (row-major, zero-based indices)
 * ----------------------------------------------------------
 * data / invC (diagonal / AET mode):
 *   index = ((data_index * num_channels + channel) * num_times + t_idx) *
 * num_freqs + f_idx
 *
 * invC (full matrix / XYZ mode):
 *   index = (((noise_index * num_channels + ch_i) * num_channels + ch_j)
 *             * num_times + t_idx) * num_freqs + f_idx
 *
 * TDI channel types
 * -----------------
 *   TDI_XYZ (0) – uses the full 3×3 cross-channel inverse-covariance matrix.
 *   TDI_AET (1) – uses only the diagonal (per-channel) inverse-covariance.
 */
#ifndef __DOMAINS_HPP__
#define __DOMAINS_HPP__

#include "cuda_complex.hpp"
#include "gbt_global.h"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define STFTSettings STFTSettingsGPU
#define FDSettings FDSettingsGPU
#define STFTDomain STFTDomainGPU
#define FDDomain FDDomainGPU
#define STFTFresnel STFTFresnelGPU
#else
#define STFTSettings STFTSettingsCPU
#define FDSettings FDSettingsCPU
#define STFTDomain STFTDomainCPU
#define FDDomain FDDomainCPU
#define STFTFresnel STFTFresnelCPU
#endif

// TDI channel configuration types
// TDI_XYZ: uses the full 3x3 cross-channel inverse covariance (off-diagonal
// terms are non-zero) TDI_AET: uses the diagonal inverse covariance (A, E, T
// channels are treated as independent)
#define TDI_XYZ 0
#define TDI_AET 1

/**
 * @brief Grid parameters for an STFT (Short-Time Fourier Transform) domain.
 *
 * Stores the integer counts and physical spacings of the time–frequency grid.
 * This is a plain-old-data type so it can be copied to/from the GPU without
 * dynamic allocation.
 */
class STFTSettings {
 public:
  int num_times;     ///< Number of time bins in the STFT grid
  int num_freqs;     ///< Number of frequency bins in the STFT grid
  int num_channels;  ///< Number of TDI channels (typically 2 for AET, 3 for
                     ///< XYZ)
  double t0;         ///< Physical start time of the grid [s]
  double f_min;      ///< Physical minimum frequency of the grid [Hz]
  double f_max;      ///< Physical maximum frequency of the grid [Hz]
  double dt;         ///< Time-bin spacing [s]
  double df;         ///< Frequency-bin spacing [Hz]
  double diff_comp;  ///< Differential component for inner products (= df)

  CUDA_CALLABLE_MEMBER
  STFTSettings(int num_times_, int num_freqs_, int num_channels_, double t0_,
               double f_min_, double f_max_, double dt_, double df_)
      : num_times(num_times_),
        num_freqs(num_freqs_),
        num_channels(num_channels_),
        t0(t0_),
        f_min(f_min_),
        f_max(f_max_),
        dt(dt_),
        df(df_),
        diff_comp(df_) {}
};

/**
 * @brief Grid parameters for a purely frequency-domain (FD) analysis.
 *
 * Specialisation of STFTSettings with num_times = 1 and dt = t0 = 0.
 * Convenient for analyses that work in the frequency domain only (no STFT
 * sliding window).
 */
class FDSettings : public STFTSettings {
 public:
  CUDA_CALLABLE_MEMBER
  FDSettings(int num_freqs_, int num_channels_, double f_min_, double f_max_,
             double df_)
      : STFTSettings(1, num_freqs_, num_channels_, 0.0, f_min_, f_max_, 0.0,
                     df_) {}
};

/**
 * @brief STFT domain with data and inverse-covariance arrays.
 *
 * Extends STFTSettings by attaching device pointers to the complex data array
 * and the precomputed inverse-covariance array, plus convenience methods for
 * flat-array indexing and inner-product accumulation.
 *
 * The two array pointers are *not* owned by this object – their lifetime must
 * be managed by the caller (i.e. the Python wrapper that allocates/frees GPU
 * memory).
 */
class STFTDomain : public STFTSettings {
 public:
  cmplx* data;    ///< Complex data array, shape [num_data, num_channels,
                  ///< num_times, num_freqs]
  cmplx* invC;    ///< Inverse-covariance array; shape depends on TDI mode:
                  ///<   diagonal (AET): [num_noise, num_channels, num_times,
                  ///<   num_freqs] full matrix (XYZ): [num_noise, num_channels,
                  ///<   num_channels, num_times, num_freqs]
  int num_data;   ///< Number of independent data realisations (e.g. MCMC
                  ///< walkers)
  int num_noise;  ///< Number of independent noise/PSD realisations
  int tdi_type;   ///< TDI channel configuration type (TDI_XYZ or TDI_AET)

  CUDA_CALLABLE_MEMBER
  STFTDomain(int num_times_, int num_freqs_, int num_channels_, double t0_,
             double f_min_, double f_max_, double dt_, double df_, cmplx* data_,
             cmplx* invC_, int num_data_, int num_noise_, int tdi_type_)
      : STFTSettings(num_times_, num_freqs_, num_channels_, t0_, f_min_, f_max_,
                     dt_, df_),
        data(data_),
        invC(invC_),
        num_data(num_data_),
        num_noise(num_noise_),
        tdi_type(tdi_type_) {}
  // ----------------------------------------------------------------
  // Data indexing
  // ----------------------------------------------------------------

  /** @brief Convert a physical time @p t [s] to its zero-based grid index.
   *  Throws (CPU) or returns -1 (GPU) if @p t is out of [t0, t0 +
   * num_times*dt]. */
  CUDA_DEVICE
  int get_time_index(double t);

  /** @brief Convert a physical frequency @p f [Hz] to its zero-based grid
   * index. Throws (CPU) or returns -1 (GPU) if @p f is out of [f_min, f_max].
   */
  CUDA_DEVICE
  int get_freq_index(double f);

  /** @brief Compute the flat index into the data array.
   *  Layout: data[(data_index * num_channels + channel) * num_times * num_freqs
   *               + t_idx * num_freqs + f_idx] */
  CUDA_DEVICE
  int get_data_index(int t_idx, int f_idx, int channel, int data_index);

  /** @brief Return the complex data value at position (t_idx, f_idx) for the
   *  given channel and data instance. */
  CUDA_DEVICE
  cmplx get_data_value(int t_idx, int f_idx, int channel, int data_index);

  // ----------------------------------------------------------------
  // Noise indexing — diagonal (AET) mode
  // ----------------------------------------------------------------

  /** @brief Compute the flat index into the diagonal inverse-covariance array.
   *  Layout: invC[(noise_index * num_channels + channel) * num_times *
   * num_freqs
   *               + t_idx * num_freqs + f_idx] */
  CUDA_DEVICE
  int get_noise_index(int t_idx, int f_idx, int channel, int noise_index);

  /** @brief Return the (diagonal) inverse-covariance value C^{-1}_{ch}
   *  at position (t_idx, f_idx) for the given channel and noise instance. */
  CUDA_DEVICE
  cmplx get_invC_value(int t_idx, int f_idx, int channel, int noise_index);

  // ----------------------------------------------------------------
  // Noise indexing — full matrix (XYZ) mode
  // ----------------------------------------------------------------

  /** @brief Compute the flat index into the full (3x3) inverse-covariance
   * matrix array. Layout: invC[((noise_index * num_channels + ch_i) *
   * num_channels + ch_j)
   *               * num_times * num_freqs + t_idx * num_freqs + f_idx] */
  CUDA_DEVICE
  int get_noise_index_cross(int t_idx, int f_idx, int ch_i, int ch_j,
                            int noise_index);

  /** @brief Return the off-diagonal inverse-covariance element C^{-1}_{ch_i,
   * ch_j} at position (t_idx, f_idx) for the given noise instance. */
  CUDA_DEVICE
  cmplx get_invC_cross_value(int t_idx, int f_idx, int ch_i, int ch_j,
                             int noise_index);

  // ----------------------------------------------------------------
  // Inner products — per-channel primitives
  // The caller is responsible for looping over all (channel_i, channel_j)
  // or channel pairs and summing the contributions.
  // ----------------------------------------------------------------

  /** @brief Accumulate one (ch_i, ch_j) cross-term of (d|h) and (h|h).
   *
   *  XYZ / full-matrix mode:
   *    *d_h += conj(d[ch_i]) * C^{-1}_{ch_i,ch_j} * h[ch_j]
   *    *h_h += conj(h[ch_i]) * C^{-1}_{ch_i,ch_j} * h[ch_j]
   *
   * @param d_h       Running accumulator for the data–template inner product
   * @param h_h       Running accumulator for the template–template inner
   * product
   * @param h_val_i   Template value in channel ch_i at this (t,f) pixel
   * @param h_val_j   Template value in channel ch_j at this (t,f) pixel
   */
  CUDA_DEVICE
  void get_inner_product_cross(cmplx* d_h, cmplx* h_h, cmplx h_val_i,
                               cmplx h_val_j, int t_idx, int f_idx,
                               int channel_i, int channel_j, int data_index,
                               int noise_index);

  /** @brief Accumulate one diagonal-channel contribution to (d|h) and (h|h).
   *
   *  AET / diagonal mode:
   *    *d_h += conj(d[ch]) * C^{-1}_{ch} * h[ch]
   *    *h_h += conj(h[ch]) * C^{-1}_{ch} * h[ch]
   *
   * @param d_h     Running accumulator for the data–template inner product
   * @param h_h     Running accumulator for the template–template inner product
   * @param h_val   Template value in this channel at this (t,f) pixel
   */
  CUDA_DEVICE
  void get_inner_product_diag(cmplx* d_h, cmplx* h_h, cmplx h_val, int t_idx,
                              int f_idx, int channel, int data_index,
                              int noise_index);

  // ----------------------------------------------------------------
  // <d|d> inner product — per-channel primitives
  // ----------------------------------------------------------------

  /** @brief Accumulate one (ch_i, ch_j) cross-term of (d|d).
   *
   *  XYZ / full-matrix mode:
   *    *d_d += conj(d[ch_i]) * C^{-1}_{ch_i,ch_j} * d[ch_j]
   */
  CUDA_DEVICE
  void get_d_d_inner_product_cross(cmplx* d_d, int t_idx, int f_idx,
                                   int channel_i, int channel_j, int data_index,
                                   int noise_index);

  /** @brief Accumulate one diagonal-channel contribution to (d|d).
   *
   *  AET / diagonal mode:
   *    *d_d += conj(d[ch]) * C^{-1}_{ch} * d[ch]
   */
  CUDA_DEVICE
  void get_d_d_inner_product_diag(cmplx* d_d, int t_idx, int f_idx, int channel,
                                  int data_index, int noise_index);

  // ----------------------------------------------------------------
  // Unified dispatchers — loop over channels internally
  // ----------------------------------------------------------------

  /** @brief Dispatcher: accumulate (d|h) and (h|h) contributions at one (t,f)
   * pixel.
   *
   *  Selects the cross-channel (XYZ) or diagonal (AET) inner-product path
   *  based on @p tdi_type and loops over all relevant channel combinations.
   *  Results are accumulated in the per-thread shared arrays @p d_h_tmp and
   *  @p h_h_tmp at index `threadIdx.x` (GPU) or index 0 (CPU).
   *
   * @param d_h_tmp       Per-thread accumulator for (d|h), length NUM_THREADS
   * @param h_h_tmp       Per-thread accumulator for (h|h), length NUM_THREADS
   * @param template_vals Template values for all channels at this pixel,
   *                      length num_channels
   */
  CUDA_DEVICE
  void add_ip_contrib(cmplx* d_h_tmp, cmplx* h_h_tmp, cmplx* template_vals,
                      int t_idx, int f_idx, int data_index, int noise_index);

  /** @brief Dispatcher: accumulate (d|d) contribution at one (t,f) pixel.
   *
   *  Selects the cross-channel or diagonal path based on @p tdi_type.
   *  Result is accumulated in @p d_d_tmp[threadIdx.x] (GPU) or [0] (CPU).
   *
   * @param d_d_tmp   Per-thread accumulator for (d|d), length NUM_THREADS
   */
  CUDA_DEVICE
  void add_d_d_contrib(cmplx* d_d_tmp, int t_idx, int f_idx, int data_index,
                       int noise_index);

  /** @brief Dispatcher: accumulate all five inner-product terms needed for a
   *  source-swap (birth/death) MCMC step at one (t,f) pixel.
   *
   *  Computes simultaneously:
   *    (d|h_add), (h_add|h_add), (d|h_remove), (h_remove|h_remove),
   *    and (h_add|h_remove)
   *  avoiding redundant data/covariance lookups by sharing channel loops.
   *
   * @param d_h_add_tmp        Accumulator for (d|h_add)
   * @param d_h_remove_tmp     Accumulator for (d|h_remove)
   * @param add_add_tmp        Accumulator for (h_add|h_add)
   * @param remove_remove_tmp  Accumulator for (h_remove|h_remove)
   * @param add_remove_tmp     Accumulator for (h_add|h_remove)
   * @param template_vals_add     Template for the proposed source, length
   * num_channels
   * @param template_vals_remove  Template for the source being removed, length
   * num_channels
   */
  CUDA_DEVICE
  void add_ip_swap_contrib(cmplx* d_h_add_tmp, cmplx* d_h_remove_tmp,
                           cmplx* add_add_tmp, cmplx* remove_remove_tmp,
                           cmplx* add_remove_tmp, cmplx* template_vals_add,
                           cmplx* template_vals_remove, int t_idx, int f_idx,
                           int data_index, int noise_index);

  /** @brief Host wrapper: launch the two-pass GPU likelihood kernels (or the
   *  equivalent CPU loop) for a batch of @p num_binaries sources.
   *
   *  On GPU the computation is split into two kernel launches:
   *    1. compute_likelihood_contributions_kernel – parallel (t,f) reduction
   *       producing per-block partial sums.
   *    2. like_sum_from_contrib_cmplx – final reduction across blocks per
   * binary. On CPU a single serial loop replaces both kernels.
   *
   * @param d_h_out            Output (d|h) values, shape [num_binaries]
   * @param h_h_out            Output (h|h) values, shape [num_binaries]
   * @param template_vals      Template array,
   *                           shape [num_binaries, num_channels,
   *                                  num_times_template, num_freqs_template]
   * @param start_times_all    Physical start time for each binary's sub-grid
   * [s]
   * @param start_freqs_all    Physical start frequency for each binary's
   * sub-grid [Hz]
   * @param num_binaries       Number of sources to process in this batch
   * @param data_index_all     Data-instance index per binary
   * @param noise_index_all    Noise-instance index per binary
   * @param n_t_template       Number of time bins in each template sub-grid
   * @param n_f_template       Number of frequency bins in each template
   * sub-grid
   */
  void compute_likelihood_terms_wrap(cmplx* d_h_out, cmplx* h_h_out,
                                     cmplx* template_vals,
                                     double* start_times_all,
                                     double* start_freqs_all, int num_binaries,
                                     int* data_index_all, int* noise_index_all,
                                     int n_t_template, int n_f_template);
};

/**
 * @brief Frequency-domain specialisation of STFTDomain (num_times = 1).
 *
 * All STFTDomain methods are available; the time dimension is trivially 1,
 * making all time-related arguments (t_idx, start_times_all, …) effectively
 * no-ops or fixed to zero.
 */
class FDDomain : public STFTDomain {
 public:
  CUDA_CALLABLE_MEMBER
  FDDomain(int num_freqs_, int num_channels_, double f_min_, double f_max_,
           double df_, cmplx* data_, cmplx* invC_, int num_data_,
           int num_noise_, int tdi_type_)
      : STFTDomain(1, num_freqs_, num_channels_, 0.0, f_min_, f_max_, 0.0, df_,
                   data_, invC_, num_data_, num_noise_, tdi_type_){};

  void compute_likelihood_terms_wrap(cmplx* d_h_out, cmplx* h_h_out,
                                     cmplx* template_vals,
                                     double* start_freqs_all, int num_binaries,
                                     int* data_index_all, int* noise_index_all,
                                     int n_f_template);
};

class STFTFresnel : public STFTSettings {
 public:
  double window_alpha;  ///< Window function parameter (e.g. for a Tukey window)
  double taper_duration;  ///< Duration of the window taper (e.g. for a Tukey
                          ///< window) [s] alpha * dt / 2
  double f_taper;  /// 1.0 / (2.0 * taper_duration);  ///< Frequency scale
                   /// associated with the window taper [Hz]
  CUDA_CALLABLE_MEMBER
  STFTFresnel(int num_times_, int num_freqs_, int num_channels_, double t0_,
              double f_min_, double f_max_, double dt_, double df_,
              double window_alpha_)
      : STFTSettings(num_times_, num_freqs_, num_channels_, t0_, f_min_, f_max_,
                     dt_, df_),
        window_alpha(window_alpha_),
        taper_duration(window_alpha_ > 0.0 ? window_alpha_ * dt_ / 2.0 : 0.0),
        f_taper(window_alpha_ > 0.0 ? 1.0 / (2.0 * taper_duration) : 0.0){};

  CUDA_DEVICE
  void get_amp_phase(double* amp, double* phase, cmplx z);
  CUDA_DEVICE
  double get_zeta(double f, double f0, double fdot0);
  CUDA_DEVICE
  double get_v(double t, double f, double t0, double f0, double fdot0);
  CUDA_DEVICE
  double get_auxiliary_f(double x);
  CUDA_DEVICE
  double get_auxiliary_g(double x);
  CUDA_DEVICE
  void get_fresnel_integrals(double* C, double* S,
                             double x);  // Fresnel integrals C(x) and S(x)
                                         // returned in ints[0] and ints[1]
  CUDA_DEVICE
  cmplx get_fresnel_kernel_interval(double f, double t0, double f0,
                                    double fdot0, double t_start, double t_end);
  CUDA_DEVICE
  cmplx get_phase_kernel_product(double f_eff, double t0, double f0,
                                 double fdot0, double t_start, double t_end);
  CUDA_DEVICE
  cmplx get_windowed_fourier_value(double amp, double phase0, double f0,
                                   double fdot0, double t0, double f);
  CUDA_DEVICE
  cmplx get_fresnel_kernel(double f, double t0, double f0, double fdot0);
  CUDA_DEVICE
  cmplx get_fourier_value(double amp, double phase0, double f0, double fdot0,
                          double t0, double f, double window_factor);

  void compute_fourier_values_wrap(cmplx* output, double* amps, double* phase0s,
                                   double* f0s, double* fdot0s, double* t0s,
                                   double* freqs, double window_factor,
                                   int num_binaries, int num_freqs);
};

CUDA_KERNEL
void compute_fourier_values_kernel(cmplx* output, STFTFresnel* fresnel,
                                   double* amps, double* phase0s, double* f0s,
                                   double* fdot0s, double* t0s, double* freqs,
                                   int num_binaries, int num_freqs);

/** @brief First-pass kernel: partial (d|h) and (h|h) sums per CUDA block.
 *
 *  Grid layout (GPU):
 *    - gridDim.y = num_binaries:  each row of blocks handles one source.
 *    - gridDim.x = ceil(num_times_template * num_freqs_template / NUM_THREADS):
 *      blocks tile the (t,f) sub-grid of the template.
 *
 *  Each block performs a CUB block-level reduction over its (t,f) pixels and
 *  stores the partial sums in d_h_contrib[bin * gridDim.x + blockIdx.x] and
 *  h_h_contrib[bin * gridDim.x + blockIdx.x].  A second kernel
 *  (like_sum_from_contrib_cmplx) then reduces across blocks.
 *
 *  CPU fallback: a single serial loop writes d_h_contrib[bin] directly.
 *
 * @param d_h_contrib   Partial sums for (d|h), shape [num_binaries *
 * num_blocks_x]
 * @param h_h_contrib   Partial sums for (h|h), shape [num_binaries *
 * num_blocks_x]
 * @param domain        Pointer to the STFT domain object (device copy on GPU)
 * @param template_vals Template values: [num_binaries, num_channels,
 *                       num_times_template, num_freqs_template]
 * @param start_times_all Physical start time of each template sub-grid [s]
 * @param start_freqs_all Physical start frequency of each template sub-grid
 * [Hz]
 * @param num_binaries  Number of sources in this batch
 * @param data_index_all  Data-instance index per source
 * @param noise_index_all Noise-instance index per source
 * @param n_t_template  Template time bins
 */
CUDA_KERNEL
void compute_likelihood_contributions_kernel(
    cmplx* d_h_contrib, cmplx* h_h_contrib, STFTDomain domain,
    cmplx* template_vals, double* start_times_all, double* start_freqs_all,
    int num_binaries, int* data_index_all, int* noise_index_all,
    int n_t_template, int n_f_template);

/** @brief Second-pass kernel: reduce per-block partial sums to per-binary
 * results.
 *
 *  Grid layout (GPU): gridDim.y = num_binaries, gridDim.x = 1.  Each block
 *  (identified by blockIdx.y) loads the num_blocks_per_bin partial sums for
 *  its binary and performs a shared-memory tree reduction to produce the final
 *  scalar (d|h) and (h|h) values.
 *
 *  CPU fallback: a serial loop over num_blocks_per_bin (= 1) trivially copies
 *  the single partial sum to the output.
 *
 * @param d_h_final        Final (d|h) per binary, shape [num_binaries]
 * @param h_h_final        Final (h|h) per binary, shape [num_binaries]
 * @param d_h_contrib      Input partial sums, shape [num_binaries *
 * num_blocks_per_bin]
 * @param h_h_contrib      Input partial sums, shape [num_binaries *
 * num_blocks_per_bin]
 * @param num_blocks_per_bin   gridDim.x from the first-pass kernel launch
 * @param num_binaries     Number of sources
 */
CUDA_KERNEL
void like_sum_from_contrib_cmplx(cmplx* d_h_final, cmplx* h_h_final,
                                 cmplx* d_h_contrib, cmplx* h_h_contrib,
                                 int num_blocks_per_bin, int num_binaries);

#endif  // __DOMAINS_HPP__
