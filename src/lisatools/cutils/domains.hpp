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

// gbt_global.h transitively brings in cuda_complex.hpp + the CUDA_DEVICE /
// CUDA_KERNEL / cmplx typedef set (one sprint-wide copy, owned by
// GPUBackendTools).
#include "gbt_global.h"
#include <cstddef>    // size_t (FDDomain indexing)
#include <stdexcept>  // std::invalid_argument (WDMDomain CPU-branch checks)

// ----------------------------------------------------------------------------
// Domains consolidation (2026-06, post-stft_tof merge): this header now owns
// ALL of LAT's C++ time-frequency domain descriptors:
//   - STFT family (incoming at the stft_tof merge): STFTSettings, FDSettings,
//     STFTDomain, FDDomainForStft, STFTFresnel. Method bodies in domains.cu.
//   - WDM/FD chunked-het family (Phase 3L.1/3L.2/3L.4, ex lisa-on-gpu):
//     WDMSettings, WDMDomain, FDDomain. Fully header-inline.
// The former standalone headers wdm_settings.hh / wdm_domain.hh / fd_domain.hh
// remain as deprecated #include shims onto this file so lisa-on-gpu-era
// include paths and downstream consumers (GBGPU, BBHx) keep compiling.
//
// NOTE (2026-06 merge): the FD specialisation of STFTDomain is named
// FDDomainForStft because the canonical chunked-het data container (below,
// ex fd_domain.hh) already owns the FDDomain name (and its nanobind
// registration / downstream GBGPU+BBHx consumption).
// TODO(unify): merge FDDomainForStft and FDDomain into one class during a
// follow-up.
//
// Per-backend CPU/GPU class-name alias block (sprint-wide rule): every class
// in this header is compiled into BOTH the CPU and the GPU shared object, so
// each one must alias to a distinct per-backend C++ type name. Both branches
// must carry the same entry set.
// ----------------------------------------------------------------------------
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define STFTSettings STFTSettingsGPU
#define FDSettings FDSettingsGPU
#define STFTDomain STFTDomainGPU
#define FDDomainForStft FDDomainForStftGPU
#define STFTFresnel STFTFresnelGPU
#define WDMSettings WDMSettingsGPU
#define WDMDomain WDMDomainGPU
#define FDDomain FDDomainGPU
#else
#define STFTSettings STFTSettingsCPU
#define FDSettings FDSettingsCPU
#define STFTDomain STFTDomainCPU
#define FDDomainForStft FDDomainForStftCPU
#define STFTFresnel STFTFresnelCPU
#define WDMSettings WDMSettingsCPU
#define WDMDomain WDMDomainCPU
#define FDDomain FDDomainCPU
#endif

// TDI channel configuration types
// TDI_XYZ: uses the full 3x3 cross-channel inverse covariance (off-diagonal
// terms are non-zero) TDI_AET: uses the diagonal inverse covariance (A, E, T
// channels are treated as independent)
//
// VALUES MUST MATCH the canonical flavor ints in wdm_domain.hh
// (TDI_XYZ=1, TDI_AET=2, TDI_AE=3): they cross the Python boundary via
// pycppdetector's TDITypeDict and are shared with the chunked-het kernels.
// (The stft_tof branch used 0/1 here; re-based at the 2026-06 merge.)
#ifndef TDI_XYZ
#define TDI_XYZ 1
#define TDI_AET 2
#define TDI_AE 3
#endif

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
                                     int n_t_template, int n_f_template,
                                     bool run_async = false);
};

/**
 * @brief Frequency-domain specialisation of STFTDomain (num_times = 1).
 *
 * All STFTDomain methods are available; the time dimension is trivially 1,
 * making all time-related arguments (t_idx, start_times_all, …) effectively
 * no-ops or fixed to zero.
 *
 * Named FDDomainForStft (2026-06 merge): the canonical chunked-het data
 * container in fd_domain.hh owns the FDDomain name. TODO(unify): merge the
 * two classes in the domains consolidation follow-up.
 */
class FDDomainForStft : public STFTDomain {
 public:
  CUDA_CALLABLE_MEMBER
  FDDomainForStft(int num_freqs_, int num_channels_, double f_min_, double f_max_,
           double df_, cmplx* data_, cmplx* invC_, int num_data_,
           int num_noise_, int tdi_type_)
      : STFTDomain(1, num_freqs_, num_channels_, 0.0, f_min_, f_max_, 0.0, df_,
                   data_, invC_, num_data_, num_noise_, tdi_type_){};

  void compute_likelihood_terms_wrap(cmplx* d_h_out, cmplx* h_h_out,
                                     cmplx* template_vals,
                                     double* start_freqs_all, int num_binaries,
                                     int* data_index_all, int* noise_index_all,
                                     int n_f_template,
                                     bool run_async = false);
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
  double get_v(double tau, double f, double f0, double fdot0);
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

// ============================================================================
// WDM / FD chunked-het domain descriptors (consolidated from wdm_settings.hh,
// wdm_domain.hh, fd_domain.hh at the 2026-06 domains consolidation; class
// definitions preserved byte-for-byte). All header-inline -- no .cu bodies.
// ============================================================================

// WDMSettings -- POD config describing the WDM (Wilson Daubechies Meyer)
// time-frequency grid and the active (m, n) band of interest.
//
// Phase 3L (2026-06-02): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.hh:459-488
// to LISAanalysistools (ex wdm_settings.hh).

class WDMSettings{
  public:
    int Nt;
    int Nf;
    int num_channel;
    double layer_df;
    double layer_dt;
    int ind_min_t;
    int ind_max_t;
    int ind_min_f;
    int ind_max_f;
    int Nf_active;
    int Nt_active;

    CUDA_CALLABLE_MEMBER
    WDMSettings(double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_){
        Nf = Nf_;
        Nt = Nt_;
        num_channel = num_channel_;
        layer_df = layer_df_;
        layer_dt = layer_dt_;
        ind_min_t = ind_min_t_;
        ind_max_t = ind_max_t_;
        ind_min_f = ind_min_f_;
        ind_max_f = ind_max_f_;
        Nf_active = ind_max_f - ind_min_f + 1; // inclusive
        Nt_active = ind_max_t - ind_min_t + 1; // inclusive
    };
};

// WDMDomain -- WDM (Wilson Daubechies Meyer) time-frequency-domain data
// container + inverse-noise descriptor. Inherits from WDMSettings to share
// grid metadata; adds wdm_data + wdm_noise pointers and the per-pixel
// inner-product / chain-rule helpers used by the chunked-heterodyne (and v2
// signal-heterodyne) kernels.
//
// Phase 3L (2026-06-02): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.hh:466-525
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu:381-846
// to LISAanalysistools (ex wdm_domain.hh). All 12 method bodies are
// header-inline (they are all CUDA_DEVICE-only and small enough to inline).

class WDMDomain : public WDMSettings{
  public:

    double *wdm_data;
    double *wdm_noise;
    int num_data;
    int num_noise;

    CUDA_CALLABLE_MEMBER
    WDMDomain(double *wdm_data_, double *wdm_noise_, double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_, int num_data_, int num_noise_):
    WDMSettings(layer_df_, layer_dt_, Nf_, Nt_, num_channel_, ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_)
    {
        wdm_data = wdm_data_;
        wdm_noise = wdm_noise_;
        num_data = num_data_;
        num_noise = num_noise_;
    };

    CUDA_DEVICE inline
    int get_pixel_index(int m, int n, int channel, int data_index)
    {
        if (data_index >= num_data)
        {
#ifdef __CUDACC__
#else
            throw std::invalid_argument("data_index is larger than available data instances.");
#endif
        }
        return ((data_index * num_channel + channel) * Nf_active + (m - ind_min_f)) * Nt_active + (n - ind_min_t);
    }

    CUDA_DEVICE inline
    int get_pixel_index_noise(int m, int n, int channel, int noise_index)
    {
        if (noise_index >= num_noise)
        {
#ifdef __CUDACC__
#else
            throw std::invalid_argument("noise_index is larger than available noise instances.");
#endif
        }
        return ((noise_index * num_channel + channel) * Nf_active + (m - ind_min_f)) * Nt_active + (n - ind_min_t);
    }

    CUDA_DEVICE inline
    int get_pixel_index_noise_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index)
    {
        return (((noise_index * num_channel + channel_i) * num_channel + channel_j) * Nf_active + (m - ind_min_f)) * Nt_active + (n - ind_min_t);
    }

    CUDA_DEVICE inline
    double get_pixel_data_value(int m, int n, int channel, int data_index)
    {
        return wdm_data[get_pixel_index(m, n, channel, data_index)];
    }

    CUDA_DEVICE inline
    double get_pixel_noise_value(int m, int n, int channel, int noise_index)
    {
        return wdm_noise[get_pixel_index_noise(m, n, channel, noise_index)];
    }

    CUDA_DEVICE inline
    double get_pixel_noise_value_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index)
    {
        return wdm_noise[get_pixel_index_noise_cross_channel(m, n, channel_i, channel_j, noise_index)];
    }

    CUDA_DEVICE inline
    void get_inner_product_value(double *d_h, double *h_h, double wdm_template_nm, int m, int n, int channel, int data_index, int noise_index)
    {
        double wdm_data_nm = get_pixel_data_value(m, n, channel, data_index);
        double wdm_noise_nm = get_pixel_noise_value(m, n, channel, noise_index);
        double val_d_h = wdm_data_nm * wdm_template_nm * wdm_noise_nm * 0.25;
        double val_h_h = wdm_template_nm * wdm_template_nm * wdm_noise_nm * 0.25;

        *d_h = val_d_h;
        *h_h = val_h_h;
    }

    CUDA_DEVICE inline
    void get_inner_product_value_cross_channel(double *d_h, double *h_h, double wdm_template_nm_i, double wdm_template_nm_j, int m, int n, int channel_i, int channel_j, int data_index, int noise_index)
    {
        // assume data is channel_i, template is channel_j
        double wdm_data_nm_i = get_pixel_data_value(m, n, channel_i, data_index);
        double wdm_noise_nm_ij = get_pixel_noise_value_cross_channel(m, n, channel_i, channel_j, noise_index);

        // 0.25 factor is needed. Check python code
        double val_d_h = wdm_data_nm_i * wdm_template_nm_j * wdm_noise_nm_ij * 0.25;
        double val_h_h = wdm_template_nm_i * wdm_template_nm_j * wdm_noise_nm_ij * 0.25;

        *d_h = val_d_h;
        *h_h = val_h_h;
    }

    CUDA_DEVICE inline
    void add_ip_contrib(double *d_h_tmp, double *h_h_tmp, double *w_mn, int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
#ifdef __CUDACC__
        int tid = threadIdx.x;
#else
        int tid = 0;
#endif

        double d_h_val = 0.0;
        double h_h_val = 0.0;
        if (tdi_type == TDI_XYZ)
        {
            for (int channel_i = 0; channel_i < 3; channel_i += 1)
            {
                for (int channel_j = 0; channel_j < 3; channel_j += 1)
                {

                    // TODO: change from 9 to 6 calculations?
                    get_inner_product_value_cross_channel(&d_h_val, &h_h_val, w_mn[channel_i], w_mn[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                    d_h_tmp[tid] += d_h_val;
                    h_h_tmp[tid] += h_h_val;

                }
            }
        }
        else if (tdi_type == TDI_AET)
        {
            // AET: three orthogonal channels, diagonal noise. The caller is
            // responsible for providing AET-projected data/template values and
            // a diagonal-only noise buffer; both the CPU and CUDA builds run
            // the same loop.
            for (int channel_i = 0; channel_i < 3; channel_i += 1)
            {
                get_inner_product_value(&d_h_val, &h_h_val, w_mn[channel_i], layer_m, n, channel_i, data_index, noise_index);
                d_h_tmp[tid] += d_h_val;
                h_h_tmp[tid] += h_h_val;
            }
        }
        else if (tdi_type == TDI_AE)
        {
            // AE: two orthogonal channels (T dropped). Same loop body as AET
            // but truncated to channels {0,1}; the caller must pre-project.
            for (int channel_i = 0; channel_i < 2; channel_i += 1)
            {
                get_inner_product_value(&d_h_val, &h_h_val, w_mn[channel_i], layer_m, n, channel_i, data_index, noise_index);
                d_h_tmp[tid] += d_h_val;
                h_h_tmp[tid] += h_h_val;
            }
        }
    }

    CUDA_DEVICE inline
    void add_ip_swap_contrib(double *d_h_add_acc, double *d_h_remove_acc, double *add_add_acc, double *remove_remove_acc, double *add_remove_acc, double *w_mn_add, double *w_mn_remove, int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
        // Accumulators are per-thread scalars (register-resident in the caller). We
        // sum into local temporaries here and write them back at the end, so the
        // hot channel loop touches no shared/global memory and the previous
        // 5xNUM_THREADS_HERE shared staging buffer is gone.
        double d_h_add_local = 0.0;
        double d_h_remove_local = 0.0;
        double add_add_local = 0.0;
        double remove_remove_local = 0.0;
        double add_remove_local = 0.0;

        double d_h_val = 0.0;
        double hh_val = 0.0;

        int nchannels = 3;
        if (tdi_type == TDI_AE) nchannels = 2;

        if (tdi_type == TDI_XYZ)
        {
            for (int channel_i = 0; channel_i < 3; channel_i += 1)
            {
                for (int channel_j = 0; channel_j < 3; channel_j += 1)
                {
                    get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_add[channel_i], w_mn_add[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                    d_h_add_local += d_h_val;
                    add_add_local += hh_val;

                    get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_remove[channel_i], w_mn_remove[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                    d_h_remove_local += d_h_val;
                    remove_remove_local += hh_val;

                    // <h_add|h_remove>: only hh_val (= add_i * remove_j * noise_ij) is needed.
                    get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_add[channel_i], w_mn_remove[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                    add_remove_local += hh_val;
                }
            }
        }
        else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
        {
            // AET/AE: orthogonal channels, diagonal per-pixel noise. AET keeps
            // all three channels, AE drops T via nchannels=2. Caller must
            // supply data/template/noise in the projected basis. Same loop on
            // CPU and CUDA.
            for (int channel_i = 0; channel_i < nchannels; channel_i += 1)
            {
                get_inner_product_value(&d_h_val, &hh_val, w_mn_add[channel_i], layer_m, n, channel_i, data_index, noise_index);
                d_h_add_local += d_h_val;
                add_add_local += hh_val;

                get_inner_product_value(&d_h_val, &hh_val, w_mn_remove[channel_i], layer_m, n, channel_i, data_index, noise_index);
                d_h_remove_local += d_h_val;
                remove_remove_local += hh_val;

                get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_add[channel_i], w_mn_remove[channel_i], layer_m, n, channel_i, channel_i, data_index, noise_index);
                add_remove_local += hh_val;
            }
        }
        else
        {
#ifdef __CUDACC__
#else
            throw std::invalid_argument("Incorrect TDI type.");
#endif
        }

        *d_h_add_acc += d_h_add_local;
        *d_h_remove_acc += d_h_remove_local;
        *add_add_acc += add_add_local;
        *remove_remove_acc += remove_remove_local;
        *add_remove_acc += add_remove_local;
    }

    // Per-pixel chain-rule contribution:
    //   grad_acc_k += sum_{c,c'} (w_d - w_h)_c * (dw_h/dtheta_k)_{c'} * N^{-1}_{cc'} * 0.25
    // (XYZ cross-channel; the AET / AE branches use the diagonal noise).
    CUDA_DEVICE inline
    void add_grad_contrib(double *grad_acc_k, const double *w_mn, const double *dw_mn_dk,
                          int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
        double local_acc = 0.0;
        if (tdi_type == TDI_XYZ)
        {
            for (int ci = 0; ci < 3; ci += 1)
            {
                double w_d_i = get_pixel_data_value(layer_m, n, ci, data_index);
                double r_i = w_d_i - w_mn[ci];
                for (int cj = 0; cj < 3; cj += 1)
                {
                    double N_ij = get_pixel_noise_value_cross_channel(layer_m, n, ci, cj, noise_index);
                    local_acc += r_i * dw_mn_dk[cj] * N_ij * 0.25;
                }
            }
        }
        else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
        {
            int nchannels = (tdi_type == TDI_AE) ? 2 : 3;
            for (int c = 0; c < nchannels; c += 1)
            {
                double w_d = get_pixel_data_value(layer_m, n, c, data_index);
                double N_c = get_pixel_noise_value(layer_m, n, c, noise_index);
                local_acc += (w_d - w_mn[c]) * dw_mn_dk[c] * N_c * 0.25;
            }
        }
        *grad_acc_k += local_acc;
    }

    // Swap variant: accumulates +/- r_after * dw * N^{-1}, where
    //   r_after = w_d - w_add_center + w_rem_center.
    // `sign` selects between the add side (+1, dw = dw_add) and the remove
    // side (-1, dw = dw_rem); the helper is called once per parameter and
    // once per side.
    CUDA_DEVICE inline
    void add_swap_grad_contrib_one_side(
        double *grad_acc_k, double sign,
        const double *w_mn_add, const double *w_mn_rem, const double *dw_mn_dk,
        int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
        double local_acc = 0.0;
        if (tdi_type == TDI_XYZ)
        {
            for (int ci = 0; ci < 3; ci += 1)
            {
                double w_d_i = get_pixel_data_value(layer_m, n, ci, data_index);
                double r_i = w_d_i - w_mn_add[ci] + w_mn_rem[ci];
                for (int cj = 0; cj < 3; cj += 1)
                {
                    double N_ij = get_pixel_noise_value_cross_channel(layer_m, n, ci, cj, noise_index);
                    local_acc += sign * r_i * dw_mn_dk[cj] * N_ij * 0.25;
                }
            }
        }
        else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
        {
            int nchannels = (tdi_type == TDI_AE) ? 2 : 3;
            for (int c = 0; c < nchannels; c += 1)
            {
                double w_d = get_pixel_data_value(layer_m, n, c, data_index);
                double N_c = get_pixel_noise_value(layer_m, n, c, noise_index);
                double r_c = w_d - w_mn_add[c] + w_mn_rem[c];
                local_acc += sign * r_c * dw_mn_dk[c] * N_c * 0.25;
            }
        }
        *grad_acc_k += local_acc;
    }
};

// FDDomain -- frequency-domain data container + inverse-noise descriptor.
// Used by the chunked-heterodyne and signal-heterodyne kernels to evaluate
// per-bin <d|h> and <h|h>.
//
// Phase 3L (2026-06-02): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.hh:561-610
// to LISAanalysistools (ex fd_domain.hh) as the first installment of the
// C++ TDIonTheFly carve-out. Fully header-inline.
// NOT the same class as FDDomainForStft above -- see TODO(unify).

class FDDomain {
  public:
    cmplx  *fd_data;   // (num_data, num_channel, n_rfft) complex
    double *fd_invC;   // tdi_type=TDI_XYZ: (num_noise, num_channel, num_channel, n_rfft)
                       // tdi_type=TDI_AET/AE: (num_noise, num_channel, n_rfft)
    int    n_rfft;
    int    num_channel;
    int    num_data;
    int    num_noise;
    int    ind_min;    // inclusive
    int    ind_max;    // inclusive
    double df;
    double Tobs;       // = 1/df, kept for convenience

    CUDA_CALLABLE_MEMBER
    FDDomain(cmplx *fd_data_, double *fd_invC_, int n_rfft_,
             int num_channel_, int num_data_, int num_noise_,
             int ind_min_, int ind_max_, double df_)
    {
        fd_data     = fd_data_;
        fd_invC     = fd_invC_;
        n_rfft      = n_rfft_;
        num_channel = num_channel_;
        num_data    = num_data_;
        num_noise   = num_noise_;
        ind_min     = ind_min_;
        ind_max     = ind_max_;
        df          = df_;
        Tobs        = 1.0 / df_;
    };
    CUDA_DEVICE inline cmplx get_data(int k, int channel, int data_index) const
    {
        return fd_data[(size_t) data_index * num_channel * n_rfft
                       + (size_t) channel * n_rfft + k];
    }
    CUDA_DEVICE inline double get_invC_diag(int k, int channel, int noise_index) const
    {
        return fd_invC[(size_t) noise_index * num_channel * n_rfft
                       + (size_t) channel * n_rfft + k];
    }
    CUDA_DEVICE inline double get_invC_cross(int k, int c1, int c2, int noise_index) const
    {
        return fd_invC[(((size_t) noise_index * num_channel + c1)
                        * num_channel + c2) * n_rfft + k];
    }
    CUDA_DEVICE inline bool in_band(int k) const
    {
        return (k >= ind_min) && (k <= ind_max);
    }
};

#endif  // __DOMAINS_HPP__
