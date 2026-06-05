#ifndef LAT_WDM_FFT_HH
#define LAT_WDM_FFT_HH

// ============================================================================
// Cooperative block FFT machinery for WDM-domain kernels.
//
// Owns the two paths shared by every block-cooperative WDM transform in
// the sprint:
//
//   * `wdm_spline_radix2_fft`  -- pure radix-2 in-place FFT (any POT N).
//     Self-contained, uses only `cmplx` + the CUDA thread-stride macros.
//     The hand-rolled fallback that always works.
//   * cufftdx wrappers          -- `wdm_cufftdx_fft_traits<N, inverse>`,
//     `cufftdx_block_fft<N, inverse>(...)`, and
//     `wdm_cufftdx_max_scratch()`. Opt-in via `-DLISA_USE_CUFFTDX`;
//     higher-radix cooperative block FFTs from NVIDIA's MathDx.
//   * `wdm_fft_dispatch`        -- runtime dispatcher: cufftdx if
//     compiled in AND the size is in the dispatch table, else falls
//     back to `wdm_spline_radix2_fft`.
//
// **Status (Phase 3L.7a slice 2a, 2026-06-04):** carved out of
// lisa-on-gpu's `WDMSplineHelpers.hh` so the chunked-heterodyne kernel
// family (currently in lisa-on-gpu's `TDIonTheFly.cu`, slated for LAT
// at slices 2b/3) can consume the FFT helpers from LAT without
// pulling in the spline-path-specific Bluestein/synthesis/extract
// machinery (which stays in `WDMSplineHelpers.hh` -- it's only needed
// by the deprecated `FDSplineTDIWaveform` path).
//
// **Inlining.** All bodies are `inline` (ODR-exempt) so this header
// can be included in multiple TUs without violating ODR. The previous
// `WDM_SPLINE_HELPERS_IMPLEMENTATION` macro-guard pattern that gated
// the bodies in `WDMSplineHelpers.hh` is dropped here as vestigial.
//
// **Dependencies.** Pulls in `cmplx` + the `CUDA_DEVICE` / `CUDA_SHARED`
// / `CUDA_SYNC_THREADS` / `THREAD_START_X` / `BLOCK_INCR_X` macros via
// `gbt_global.h` (transitive through `global.hpp`).
// ============================================================================

#include "global.hpp"  // -> gbt_global.h -> cmplx + CUDA macros

#if defined(__CUDACC__) && defined(LISA_USE_CUFFTDX)
#include <cufftdx.hpp>
#endif


// ---------------------------------------------------------------------------
// Device: in-place radix-2 FFT (forward or inverse). Block-cooperative
// via the `THREAD_START_X` / `BLOCK_INCR_X` stride macros (collapses to
// a single thread on CPU).
// ---------------------------------------------------------------------------
CUDA_DEVICE
inline void wdm_spline_radix2_fft(cmplx *a, int N, int log2N, bool inverse)
{
    // bit-reversal permutation
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X) {
        int r = 0, x = n;
        for (int i = 0; i < log2N; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
        if (r > n) { cmplx t = a[n]; a[n] = a[r]; a[r] = t; }
    }
    CUDA_SYNC_THREADS;
    double sign = inverse ? +1.0 : -1.0;
    for (int s = 1; s <= log2N; ++s) {
        int m  = 1 << s;
        int mh = m >> 1;
        double base = sign * 2.0 * M_PI / (double) m;
        for (int k = THREAD_START_X; k < (N >> 1); k += BLOCK_INCR_X) {
            int g  = k / mh;
            int j  = k - g * mh;
            int i0 = g * m + j;
            int i1 = i0 + mh;
            double th = base * (double) j;
            cmplx w(cos(th), sin(th));
            cmplx u = a[i0];
            cmplx v = w * a[i1];
            a[i0] = u + v;
            a[i1] = u - v;
        }
        CUDA_SYNC_THREADS;
    }
    if (inverse) {
        double inv_N = 1.0 / (double) N;
        for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X) {
            cmplx v = a[n];
            a[n] = cmplx(v.real() * inv_N, v.imag() * inv_N);
        }
        CUDA_SYNC_THREADS;
    }
}


// ---------------------------------------------------------------------------
// cufftdx Block-FFT wrapper (compile-time templated on size + direction).
//
// Drop-in alternative to `wdm_spline_radix2_fft` for sizes the GPU
// dispatcher (`wdm_fft_dispatch` below) hands off to cufftdx.
//
// Design notes:
//
//   * cufftdx is COMPILE-TIME templated on (Size, Direction, Precision,
//     BlockDim, Arch). To handle our runtime-varying Nt_sub / N_sparse,
//     we instantiate a small set of POT sizes (128, 256, 512) and
//     runtime-dispatch via `wdm_fft_dispatch`.
//
//   * Block FFT distributes the FFT cooperatively across all
//     `blockDim.x` threads. Each thread holds `FFT::elements_per_thread`
//     `cmplx` values in registers during the transform.
//
//   * Data flow: load from caller's shared-mem buffer -> per-thread
//     register array (in cufftdx's preferred stride pattern) -> FFT
//     execute -> store back to caller's shared-mem buffer.
//
//   * Each FFT specialisation has a `constexpr shared_memory_size`
//     that gives the scratch needed by cufftdx (twiddle tables,
//     intra-warp comms). The dispatcher takes a caller-provided
//     scratch pointer sized at the max across instantiated
//     specialisations.
//
//   * Falls back to `wdm_spline_radix2_fft` when `LISA_USE_CUFFTDX` is
//     not defined OR when the requested size isn't in the dispatch
//     table.
// ---------------------------------------------------------------------------

#if defined(__CUDACC__) && defined(LISA_USE_CUFFTDX)

// Per-size FFT type alias. We do NOT pin BlockDim or ElementsPerThread
// here -- double-precision FFTs in cufftdx top out at BlockDim<=64 for
// most sizes, so we let cufftdx pick a valid (BlockDim, EPT) combo
// from its database. The block kernel below uses `FFT::block_dim.x`
// to gate which threads of the chunked-het thread block participate.
template <int N, bool inverse>
struct wdm_cufftdx_fft_traits {
    using direction_t = cufftdx::Direction<
        inverse ? cufftdx::fft_direction::inverse
                : cufftdx::fft_direction::forward>;

    using FFT = decltype(
        cufftdx::Size<N>()
      + cufftdx::Precision<double>()
      + cufftdx::Type<cufftdx::fft_type::c2c>()
      + direction_t{}
      + cufftdx::Block()
      + cufftdx::SM<800>());

    using value_type = typename FFT::value_type;  // cuda::std::complex<double>
    static constexpr unsigned int ept    = FFT::elements_per_thread;
    static constexpr unsigned int stride = FFT::stride;
    static constexpr unsigned int fft_threads = FFT::block_dim.x;
    static constexpr size_t scratch_bytes = FFT::shared_memory_size;
};

// Compile-time upper bound on the scratch any of our specialisations
// needs (used by kernel launchers to size the FFT scratch region).
template <int N>
struct wdm_cufftdx_fft_scratch {
    static constexpr size_t value = std::max(
        wdm_cufftdx_fft_traits<N, false>::scratch_bytes,
        wdm_cufftdx_fft_traits<N, true >::scratch_bytes);
};

// Block FFT for one length-N buffer in shared mem. The caller passes:
//   shared_buf : pointer to N cmplx values in shared mem (overwritten)
//   fft_scratch: pointer to `FFT::shared_memory_size` bytes of shared
//                mem (must NOT alias shared_buf)
template <int N, bool inverse>
CUDA_DEVICE inline void cufftdx_block_fft(cmplx *shared_buf, char *fft_scratch)
{
    using Traits     = wdm_cufftdx_fft_traits<N, inverse>;
    using FFT        = typename Traits::FFT;
    using value_type = typename Traits::value_type;
    constexpr unsigned int EPT         = Traits::ept;
    constexpr unsigned int STRIDE      = Traits::stride;
    constexpr unsigned int FFT_THREADS = Traits::fft_threads;

    // cufftdx picked (BlockDim, EPT) for us. Only the first FFT_THREADS
    // threads of the launch participate; the rest sit out and rejoin at
    // the trailing __syncthreads.
    const unsigned int tid = threadIdx.x;
    if (tid < FFT_THREADS) {
        value_type thread_data[EPT];

        // Load shared -> per-thread registers (thread t owns t + i*STRIDE).
        #pragma unroll
        for (unsigned int i = 0; i < EPT; ++i) {
            const unsigned int idx = tid + i * STRIDE;
            thread_data[i] = reinterpret_cast<value_type*>(shared_buf)[idx];
        }

        // Cooperative block FFT (writes thread_data in place).
        FFT().execute(thread_data, fft_scratch);

        // Store regs -> shared. Inverse path applies 1/N to match
        // wdm_spline_radix2_fft's normalisation convention.
        if (inverse) {
            constexpr double inv_N = 1.0 / (double) N;
            #pragma unroll
            for (unsigned int i = 0; i < EPT; ++i) {
                const unsigned int idx = tid + i * STRIDE;
                value_type v = thread_data[i];
                v.real(v.real() * inv_N);
                v.imag(v.imag() * inv_N);
                reinterpret_cast<value_type*>(shared_buf)[idx] = v;
            }
        } else {
            #pragma unroll
            for (unsigned int i = 0; i < EPT; ++i) {
                const unsigned int idx = tid + i * STRIDE;
                reinterpret_cast<value_type*>(shared_buf)[idx] = thread_data[i];
            }
        }
    }
    CUDA_SYNC_THREADS;
}

// Compile-time scratch upper bound across all sizes we dispatch to.
// Used by kernel launchers to size the shared FFT scratch region.
constexpr size_t wdm_cufftdx_max_scratch() {
    size_t s = 0;
    s = std::max(s, wdm_cufftdx_fft_scratch<128>::value);
    s = std::max(s, wdm_cufftdx_fft_scratch<256>::value);
    s = std::max(s, wdm_cufftdx_fft_scratch<512>::value);
    return s;
}

#else  // !LISA_USE_CUFFTDX

// Stub scratch size = 0 when cufftdx is disabled. `wdm_fft_dispatch`
// falls through to `wdm_spline_radix2_fft` and never touches the
// scratch.
inline constexpr size_t wdm_cufftdx_max_scratch() { return 0; }

#endif  // LISA_USE_CUFFTDX


// ---------------------------------------------------------------------------
// Runtime dispatcher: pick cufftdx for supported POT sizes, else fall
// back to the hand-rolled radix-2. `fft_scratch` is only read by the
// cufftdx path -- safe to pass nullptr if `LISA_USE_CUFFTDX` is off.
//
// Supported cufftdx sizes (matches the specialisations above):
//   128, 256, 512
// Sizes outside this set OR when cufftdx is disabled fall back to
// `wdm_spline_radix2_fft`.
// ---------------------------------------------------------------------------
CUDA_DEVICE
inline void wdm_fft_dispatch(cmplx *a, int N, int log2N, bool inverse,
                             char *fft_scratch)
{
#if defined(__CUDACC__) && defined(LISA_USE_CUFFTDX)
    switch (log2N) {
        case 7:  // N = 128
            if (inverse) cufftdx_block_fft<128, true >(a, fft_scratch);
            else         cufftdx_block_fft<128, false>(a, fft_scratch);
            return;
        case 8:  // N = 256
            if (inverse) cufftdx_block_fft<256, true >(a, fft_scratch);
            else         cufftdx_block_fft<256, false>(a, fft_scratch);
            return;
        case 9:  // N = 512
            if (inverse) cufftdx_block_fft<512, true >(a, fft_scratch);
            else         cufftdx_block_fft<512, false>(a, fft_scratch);
            return;
        default:
            break;  // N=1024 and larger fall through to radix-2 fallback
    }
#endif
    (void) fft_scratch;
    wdm_spline_radix2_fft(a, N, log2N, inverse);
}


#endif  // LAT_WDM_FFT_HH
