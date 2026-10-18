#ifndef __LAT_STFT_KERNELS_HH__
#define __LAT_STFT_KERNELS_HH__

// lat_stft_kernels.hh -- source-agnostic STFT/Fresnel galactic-binary-style
// likelihood kernels, templated on the TDI-on-the-fly source class (SourceT).
//
// These are the STFT analog of the chunked-heterodyne `wdm_het_*_impl<SourceT>`
// launchers in lat_chunked_het_kernels.hh: LAT owns the generic kernel bodies;
// per-source packages (GBGPU's GBTDIonTheFly, a future SOBBHTDIonTheFly)
// instantiate `stft_*_impl<TheirSource>` from their own GBComputationGroup wraps.
//
// The kernels build, per STFT time bin, the per-channel TDI value on the fly
// (SourceT::get_tdi_Xf_single), turn it into a windowed Fresnel Fourier value
// per (time, frequency) pixel (STFTFresnel::get_fourier_value), and accumulate
// the noise-weighted inner products via STFTDomain::add_ip_contrib. All of those
// device primitives already live in domains.{hpp,cu} -- this header only adds the
// GB-on-the-fly glue.
//
// Two seams. The per-column source evaluation and per-pixel Fourier value live
// behind a compile-time ColumnT policy (the "Column-producer policy seam" block
// below; FresnelColumn is the production policy). The consumers see a tile
// (the "Tile seam" block): ColumnTile<ColumnT> adapts a column policy, and a
// response that builds its pixels from shared-memory tables plugs in as a
// second tile without touching the consumers. FresnelColumn is the default
// everywhere, so per-source packages instantiate `stft_*_impl<TheirSource>`
// exactly as before.
//
// Conventions (must match domains.cu so the on-the-fly likelihood equals the
// template-based STFTComputationGroup path):
//   amp, phase  <- get_amp_phase(conj(tdi_val[ch]))   (conjugate = Fresnel convention)
//   pixel value =  0.5 * get_fourier_value(amp, phase, f0, fdot0, t, f, window_factor)
//                  (0.5 = real-signal half-amplitude at positive frequencies)
//   accumulate  via add_ip_contrib (no scaling inside)
//   finalize    *= 4.0 * stft->diff_comp  (= 4 df), once, post block-reduction.
//
// Doppler-corrected f0/fdot0 (freq_from_tdi_phase, default true): the chirp model
// uses phase0 = arg(conj(TDI(t))), so the matching instantaneous frequency is
// f0 = (1/2pi) d/dt arg(conj(TDI)). With a exp(-i*Phi) waveform convention this
// reduces to the astrophysical SourceT::get_f in the no-Doppler limit but also
// captures the LISA orbital Doppler (whose rate typically exceeds the
// astrophysical fdot). When freq_from_tdi_phase is false we fall back to
// SourceT::get_f / get_fdot (the legacy astrophysical-only behaviour).

#include "domains.hpp"            // STFTSettings / STFTDomain / STFTFresnel + global.hpp (cmplx, CUDA_* macros, gcmplx)
#include "lat_tdi_on_the_fly.hh"  // LISATDIonTheFly base (SourceT) + Vec / Orbits / TDIConfig
#include <cstring>                // std::memcmp / std::memcpy for the per-device launch structs
#include <stdexcept>              // std::runtime_error
#include <type_traits>            // std::is_same
#include <vector>                 // host arena of a tile launch on the CPU

// blockDim.x for these kernels (GPU) / single-thread CPU mirror. Normally
// already defined by lat_chunked_het_kernels.hh (pulled in ahead of this header
// by the per-source TU); guard so the header is usable on its own too.
#ifndef NUM_THREADS_HERE
#ifdef __CUDA_COMPILATION__
#define NUM_THREADS_HERE 128
#else
#ifdef __CUDACC__
#define NUM_THREADS_HERE 128
#else
#define NUM_THREADS_HERE 1
#endif
#endif
#endif

// Per-binary parameter scratch size. Defined as a .cu-local macro in
// lat_tdi_on_the_fly.cu (not a header), so guard a fallback here to keep this
// header self-contained in any translation unit.
#ifndef N_PARAMS_MAX
#define N_PARAMS_MAX 20
#endif

// * For tile methods called once per pixel. GBGPU's CPU module compiles with -Os, where GCC would
// * keep them out of line and add a call per pixel; nvcc inlines them anyway.
#ifndef STFT_TILE_PIXEL_INLINE
#ifdef __CUDACC__
#define STFT_TILE_PIXEL_INLINE
#else
#define STFT_TILE_PIXEL_INLINE __attribute__((always_inline))
#endif
#endif

// ---------------------------------------------------------------------------
// Device copies of the four host structs every STFT launch reads, one set per device. The objects
// never change after construction, so a launch uploads a struct only when its bytes differ from the
// device's copy: a comp's Orbits and TDIConfig once per device, a group's Fresnel and domain when
// launches switch group. Launches do not synchronise, so the shards of one call overlap.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#ifndef STFT_MAX_DEVICES
#define STFT_MAX_DEVICES 64
#endif

struct STFTDeviceStructs
{
    Orbits*      orbits     = nullptr;
    TDIConfig*   tdi_config = nullptr;
    STFTFresnel* fresnel    = nullptr;
    STFTDomain*  stft       = nullptr;
    unsigned char orbits_bytes[sizeof(Orbits)];
    unsigned char tdi_config_bytes[sizeof(TDIConfig)];
    unsigned char fresnel_bytes[sizeof(STFTFresnel)];
    unsigned char stft_bytes[sizeof(STFTDomain)];
};

template <class StructT>
inline void stft_upload_struct(StructT* device_copy, unsigned char* held_bytes, StructT* host)
{
    gpuErrchk(cudaMemcpy(device_copy, host, sizeof(StructT), cudaMemcpyHostToDevice));
    std::memcpy(held_bytes, host, sizeof(StructT));
}

inline const STFTDeviceStructs& stft_device_structs(
    Orbits* orbits, TDIConfig* tdi_config, STFTFresnel* fresnel, STFTDomain* stft)
{
    static STFTDeviceStructs per_device[STFT_MAX_DEVICES];
    int device = 0;
    gpuErrchk(cudaGetDevice(&device));
    if (device < 0 || device >= STFT_MAX_DEVICES)
        throw std::runtime_error("stft_device_structs: device id is not below STFT_MAX_DEVICES.");
    STFTDeviceStructs& held = per_device[device];

    const bool first = (held.orbits == nullptr);
    if (first)
    {
        gpuErrchk(cudaMalloc(&held.orbits,     sizeof(Orbits)));
        gpuErrchk(cudaMalloc(&held.tdi_config, sizeof(TDIConfig)));
        gpuErrchk(cudaMalloc(&held.fresnel,    sizeof(STFTFresnel)));
        gpuErrchk(cudaMalloc(&held.stft,       sizeof(STFTDomain)));
    }
    const bool new_orbits     = first || std::memcmp(held.orbits_bytes,     orbits,     sizeof(Orbits)) != 0;
    const bool new_tdi_config = first || std::memcmp(held.tdi_config_bytes, tdi_config, sizeof(TDIConfig)) != 0;
    const bool new_fresnel    = first || std::memcmp(held.fresnel_bytes,    fresnel,    sizeof(STFTFresnel)) != 0;
    const bool new_stft       = first || std::memcmp(held.stft_bytes,       stft,       sizeof(STFTDomain)) != 0;
    if (!(new_orbits || new_tdi_config || new_fresnel || new_stft))
        return held;

    cudaDeviceSynchronize();
    if (new_orbits)     stft_upload_struct(held.orbits,     held.orbits_bytes,     orbits);
    if (new_tdi_config) stft_upload_struct(held.tdi_config, held.tdi_config_bytes, tdi_config);
    if (new_fresnel)    stft_upload_struct(held.fresnel,    held.fresnel_bytes,    fresnel);
    if (new_stft)       stft_upload_struct(held.stft,       held.stft_bytes,       stft);
    return held;
}
#endif

// ---------------------------------------------------------------------------
// Self-contained in-block complex sum reduction (GPU only; the CPU mirror reads
// the single accumulator directly). Standalone (no dependency on domains.cu's
// file-static block_reduce_cmplx). Requires blockDim.x a power of two
// (NUM_THREADS_HERE = 128). Overwrites `sdata`; result valid on all threads.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
static CUDA_DEVICE cmplx stft_block_reduce_cmplx(cmplx* sdata)
{
    int tid = threadIdx.x;
    CUDA_SYNC_THREADS;
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = sdata[tid] + sdata[tid + s];
        CUDA_SYNC_THREADS;
    }
    return sdata[0];
}
#endif

// ---------------------------------------------------------------------------
// Instantaneous frequency and frequency rate of the TDI signal at time `t`, per
// channel, in the kernel's chirp convention (phase = arg(conj(TDI))).
//
//     f0    = f_astro + arg( conj(z+) * z- * e^{-i 4 pi f_astro D} ) / (4 pi D)
//     fdot0 = fdot_astro - f0 * (k . a_sc / c)
//
// Rate fdot0 is analytic, with the spacecraft acceleration from Orbits::get_acceleration.
// Carrier demodulation shifts the first difference to baseband (|arg| < 1e-3 rad),
// enabling wide baseline D = 2000 s and suppressing frequency noise as 1/D.
// ---------------------------------------------------------------------------
constexpr double STFT_FREQ_FDOT_DT_MAX = 3600.0;         ///< half-width cap [s]
constexpr double STFT_DT_STENCIL_DEMOD = 2000.0;         ///< demodulated half-width [s]
// Sign-preserving floor on the chirp rate. The Fresnel column carries 1/sqrt(2|fdot0|) and
// zeta = (f0 - f)/fdot0, both singular at fdot0 = 0, which a stencil reaches whenever the fdot step
// exceeds the chirp itself (the production step 1e-19 does so below ~1 mHz). At this floor the phase
// it costs over a segment is pi |dfdot| tau^2 ~ 6e-13 rad, far below every other error here.
constexpr double STFT_FDOT_FLOOR = 1.0e-22;              ///< [Hz/s]

// Astro-model degradation shared by the guards below: fires only where the
// TDI has no differentiable phase (identically-zero response samples at the
// orbit-file boundary, or a fully unusable stencil). The legacy
// astro-heterodyned estimator degraded to exactly these values there.
template <class SourceT>
CUDA_DEVICE inline void stft_freq_fdot_astro_fallback(
    SourceT& src, double t, double* params, int bin_i,
    double* f0_out, double* fdot0_out)
{
    double f_astro = src.get_f(t, params, bin_i);
    double fdot_astro = src.get_fdot(t, params, bin_i);
    // guard for small fdots that explode divisions
    if (std::fabs(fdot_astro) < STFT_FDOT_FLOOR)
        fdot_astro = std::signbit(fdot_astro) ? -STFT_FDOT_FLOOR : STFT_FDOT_FLOOR;
    for (int ch = 0; ch < 3; ch += 1)
    {
        f0_out[ch] = f_astro;
        fdot0_out[ch] = fdot_astro;
    }
}

template <class SourceT>
CUDA_DEVICE void stft_freq_fdot_from_tdi_phase(
    SourceT& src, double t,
    double* params, Vec k, Vec u, Vec v,
    int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
    const cmplx* tdi_center,
    double* f0_out, double* fdot0_out,
    double* amp_p_out, double* amp_m_out, double* D_out)
{
    for (int ch = 0; ch < 3; ch += 1)
    {
        amp_p_out[ch] = 0.0;
        amp_m_out[ch] = 0.0;
    }
    *D_out = STFT_FREQ_FDOT_DT_MAX;

    // f_astro sets the demodulation carrier and nothing else.
    double f_scale = src.get_f(t, params, bin_i);
    if (!(f_scale > 0.0))
    {
        stft_freq_fdot_astro_fallback<SourceT>(src, t, params, bin_i,
                                               f0_out, fdot0_out);
        return;
    }
    double D = STFT_DT_STENCIL_DEMOD;
    if (D > STFT_FREQ_FDOT_DT_MAX)
        D = STFT_FREQ_FDOT_DT_MAX;
    *D_out = D;

    // Active domain is the intersection of position and light travel time tables.
    // Retardation buffer covers the 8 L/c (66.7 s) maximum link delay chain.
    constexpr double DT_RETARD_BUFFER = 70.0; // [s] covers 8 * L/c

    double sc_t_min = src.orbits->sc_t0;
    double sc_t_max = src.orbits->sc_t0
                    + (double)(src.orbits->sc_N - 2) * src.orbits->sc_dt;
    double ltt_t_min = src.orbits->ltt_t0;
    double ltt_t_max = src.orbits->ltt_t0
                     + (double)(src.orbits->ltt_N - 2) * src.orbits->ltt_dt;

    double t_valid_min = (sc_t_min > ltt_t_min ? sc_t_min : ltt_t_min)
                       + DT_RETARD_BUFFER;
    double t_valid_max = (sc_t_max < ltt_t_max ? sc_t_max : ltt_t_max);

    if (t < t_valid_min || t > t_valid_max)
    {
        stft_freq_fdot_astro_fallback<SourceT>(src, t, params, bin_i,
                                               f0_out, fdot0_out);
        return;
    }

    double t_anchor = t;
    if (t_anchor < t_valid_min + D)
        t_anchor = t_valid_min + D;
    else if (t_anchor > t_valid_max - D)
        t_anchor = t_valid_max - D;

    cmplx tdi_p[3];
    cmplx tdi_m[3];
    src.get_tdi_Xf_single(&tdi_p[0], t_anchor + D, params, k, u, v,
                          link_space_craft_rec, link_space_craft_em, bin_i);
    src.get_tdi_Xf_single(&tdi_m[0], t_anchor - D, params, k, u, v,
                          link_space_craft_rec, link_space_craft_em, bin_i);

    double pow_ch[3];
    int ch_ref = 0;
    for (int ch = 0; ch < 3; ch += 1)
    {
        pow_ch[ch] = tdi_center[ch].real() * tdi_center[ch].real()
                   + tdi_center[ch].imag() * tdi_center[ch].imag();
        if (pow_ch[ch] > pow_ch[ch_ref])
            ch_ref = ch;
    }

    double pow_p = tdi_p[ch_ref].real() * tdi_p[ch_ref].real()
                 + tdi_p[ch_ref].imag() * tdi_p[ch_ref].imag();
    double pow_m = tdi_m[ch_ref].real() * tdi_m[ch_ref].real()
                 + tdi_m[ch_ref].imag() * tdi_m[ch_ref].imag();
    if (pow_ch[ch_ref] == 0.0 || pow_p == 0.0 || pow_m == 0.0)
    {
        stft_freq_fdot_astro_fallback<SourceT>(src, t, params, bin_i,
                                               f0_out, fdot0_out);
        return;
    }

    double fdot_astro = src.get_fdot(t, params, bin_i);

    // Counter-rotating carrier demodulation phasor: exp(-i * 4 * pi * f_scale * D) 
    // so arg() evaluates only the small residual Doppler/null perturbation .
    double phi_carrier = 4.0 * M_PI * f_scale * D;
    cmplx demod_phasor(cos(phi_carrier), -sin(phi_carrier));

    for (int ch = 0; ch < 3; ch += 1)
    {
        // First difference with carrier demodulation:
        // arg(conj(z+) * z- * exp(-i * 4 * pi * f_scale * D))
        cmplx z_diff = gcmplx::conj(tdi_p[ch]) * tdi_m[ch] * demod_phasor;
        double dphi1_res = gcmplx::arg(z_diff);

        // Instantaneous frequency = f_astro + delta_f
        f0_out[ch] = f_scale + dphi1_res / (4.0 * M_PI * D);

        // Vertex spacecraft of each channel (X, Y, Z -> SC 1, 2, 3).
        int sc_id = ch + 1;
        Vec a_sc = src.orbits->get_acceleration(t, sc_id);
        double k_dot_a = k.dot(a_sc);

        // fdot_0 = fdot_astro - f_0 * (k . a_sc / c)
        fdot0_out[ch] = fdot_astro - f0_out[ch] * (k_dot_a * C_inv);

        // Sign-preserving zero guard (protects float64 range across side-bins)
        // We only want to guard for exact cancellation of fdot that happen 
        // twice a year for binaries that are dominated by doppler modulation.
        if (std::fabs(fdot0_out[ch]) < STFT_FDOT_FLOOR)
        {
            fdot0_out[ch] = std::signbit(fdot0_out[ch]) ? -STFT_FDOT_FLOOR : STFT_FDOT_FLOOR;
        }
    }
    if (isnan(f0_out[ch_ref]) || isnan(fdot0_out[ch_ref]))
    {
        stft_freq_fdot_astro_fallback<SourceT>(src, t, params, bin_i,
                                               f0_out, fdot0_out);
        return;
    }

    // Near-null channels: their phase is pure roundoff -- copy the loudest
    // channel's estimate (also catches NaN powers via the negated compare).
    for (int ch = 0; ch < 3; ch += 1)
    {
        if (!(pow_ch[ch] >= 1.0e-24 * pow_ch[ch_ref]))
        {
            f0_out[ch] = f0_out[ch_ref];
            fdot0_out[ch] = fdot0_out[ch_ref];
        }
    }

    // Normal path only (all guards passed): export the stencil amplitudes for
    // the linear-envelope slope. Reuses the z+- samples already evaluated above
    // -- no extra response calls.
    for (int ch = 0; ch < 3; ch += 1)
    {
        amp_p_out[ch] = gcmplx::abs(tdi_p[ch]);
        amp_m_out[ch] = gcmplx::abs(tdi_m[ch]);
    }
}

// Resolve (f0, fdot0) per channel for one pixel: TDI-phase derivation
// (small-step central difference, see stft_freq_fdot_from_tdi_phase) or
// astrophysical fallback. amp_p/amp_m/D are the linear-envelope exports
// (zeroed on the astro path -> a_j = 0).
template <class SourceT>
CUDA_DEVICE void stft_pixel_freq_fdot(
    SourceT& src, double t,
    double* params, Vec k, Vec u, Vec v,
    int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
    const cmplx* tdi_center,
    bool freq_from_tdi_phase,
    double* f0_out, double* fdot0_out,
    double* amp_p_out, double* amp_m_out, double* D_out)
{

    if (!freq_from_tdi_phase)
    {
        for (int ch = 0; ch < 3; ch += 1)
        {
            amp_p_out[ch] = 0.0;   // astro path: no stencil amplitudes -> a_j = 0
            amp_m_out[ch] = 0.0;
        }
        *D_out = STFT_FREQ_FDOT_DT_MAX;
        stft_freq_fdot_astro_fallback<SourceT>(src, t, params, bin_i,
                                               f0_out, fdot0_out);
        return;
    }
    stft_freq_fdot_from_tdi_phase<SourceT>(
        src, t, params, k, u, v,
        link_space_craft_rec, link_space_craft_em, bin_i,
        tdi_center, f0_out, fdot0_out, amp_p_out, amp_m_out, D_out);
}

// ===========================================================================
// Column-producer policy seam
// ===========================================================================
// One "column" = one (source, STFT time bin). A ColumnT policy owns the
// per-column SOURCE evaluation and produces the per-(frequency-bin, channel)
// pixel Fourier values; the CONSUMERS (the ll / swap / fill kernels below,
// through ColumnTile; the information-matrix kernel directly)
// keep the frequency loop, the bounds masks, the 0.5 real-signal convention,
// any fill `factor`, and all accumulation/reduction logic. Compile-time
// (static) polymorphism only -- every method CUDA_DEVICE inline, no
// virtuals, no function pointers -- so the compiled inner loops are the
// pre-seam inlined code (outputs byte-identical, validated by
// scripts/validation/stft_column_policy_oracle.py).
//
// Contract:
//   State           POD per-column scratch, policy-defined, register-sized.
//   setup()         everything that does not depend on the pixel frequency:
//                   sample the source at the column anchor, derive the
//                   per-channel chirp (f0, fdot0) and amp/phase, place the
//                   carrier bin. Called once per (source, column).
//   State.carrier_j the column's carrier frequency bin (placement of the
//                   +- n_side_bins stencil by the consumer).
//   value(s, j, freq_j_here, freq_here)
//                   RAW Fourier value for channel j at absolute frequency
//                   bin freq_j_here (frequency freq_here). NOTE: the swap
//                   kernel sums over the UNION of two carriers' stencils, so
//                   value() may be queried up to |freq_j_here - carrier_j|
//                   <= n_side_bins + |carrier_add - carrier_remove|; a
//                   policy with finite tabulated support (e.g. a per-column
//                   FFT) must size or clamp for that.
//
// FresnelColumn is the production policy: the analytic (windowed)
// linear-chirp Fresnel evaluator, byte-identical to the historical inlined
// code (the amp/phase extraction is hoisted from per-pixel to per-column;
// it is a pure function of the anchor TDI sample, so the values are
// unchanged and the atan2/abs work drops by the stencil width).
// ===========================================================================
template <class SourceT>
struct FresnelColumn
{
    using Source = SourceT;

    struct State
    {
        STFTFresnel* fresnel;   // non-const: the evaluator methods are not const-qualified
        double t_seg;           // window start (Fourier origin)
        double window_factor;
        double amp[3];
        double phase[3];
        double f0[3];
        double fdot0[3];
        double a[3];            // per-channel linear-envelope slope a_j (0 if off)
        int carrier_j;          // carrier frequency bin (stencil placement)
    };

    CUDA_DEVICE static void setup(
        State& s, SourceT& src, STFTFresnel* fresnel, STFTDomain* stft,
        double* params, Vec k, Vec u, Vec v,
        int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
        double t_seg, double t_anchor_shift,
        double window_factor, bool freq_from_tdi_phase)
    {
        s.fresnel = fresnel;
        s.t_seg = t_seg;
        s.window_factor = window_factor;
        double t_here = t_seg + t_anchor_shift;  // chirp/sampling anchor
        cmplx tdi_channel_val[3];
        src.get_tdi_Xf_single(&tdi_channel_val[0], t_here, params, k, u, v,
                              link_space_craft_rec, link_space_craft_em, bin_i);
        double amp_p[3], amp_m[3], D_stencil;
        stft_pixel_freq_fdot<SourceT>(
            src, t_here, params, k, u, v,
            link_space_craft_rec, link_space_craft_em, bin_i,
            tdi_channel_val, freq_from_tdi_phase, &s.f0[0], &s.fdot0[0],
            &amp_p[0], &amp_m[0], &D_stencil);
        for (int j = 0; j < 3; j += 1)
            fresnel->get_amp_phase(&s.amp[j], &s.phase[j],
                                   gcmplx::conj(tdi_channel_val[j]));
    
        // Identify loudest channel for relative amplitude thresholding and carrier bin anchoring
        int ch_loudest = 0;
        for (int j = 1; j < 3; j += 1)
        {
            if (s.amp[j] > s.amp[ch_loudest])
                ch_loudest = j;
        }
        
        constexpr double A_J_AMP_GUARD = 1.0e-12;   // relative amplitude floor
        double a_j_max = (s.fresnel->use_midpoint ? 2.0 : 1.0) / s.fresnel->dt;
        double amp_floor = A_J_AMP_GUARD * s.amp[ch_loudest];
        for (int j = 0; j < 3; j += 1)
        {
            if (s.amp[j] > 0.0 && s.amp[j] > amp_floor)
            {
                double slope = (amp_p[j] - amp_m[j])
                             / (2.0 * D_stencil * s.amp[j]);
                if (slope > a_j_max)
                    slope = a_j_max;
                else if (slope < -a_j_max)
                    slope = -a_j_max;
                s.a[j] = slope;
            }
            else
            {
                s.a[j] = 0.0;
            }
        }
        s.carrier_j = stft->get_freq_index(s.f0[ch_loudest]);
    }

    CUDA_DEVICE static cmplx value(const State& s, int j, int freq_j_here,
                                   double freq_here)
    {
        (void) freq_j_here;  // analytic evaluator: any frequency, no tabulation
        // s.a[j] is the linear-envelope slope; get_fourier_value applies it only
        // when fresnel->linear_envelope is on (else slope is ignored -> the
        // const-envelope value is byte-identical).
        return s.fresnel->get_fourier_value(
            s.amp[j], s.phase[j], s.f0[j], s.fdot0[j],
            s.t_seg, freq_here, s.window_factor, s.a[j]);
    }

    // Optional split of value() used by the information-matrix kernel:
    // value == prefactor * kernel, and kernel carries all per-pixel Fresnel work.
    // * A channel with an active envelope moment does not factorise; its kernel is then
    // * the full value and its prefactor is one, so the product still equals value().
    CUDA_DEVICE static bool factorises(const State& s, int j)
    {
        return !(s.fresnel->linear_envelope && s.a[j] != 0.0);
    }

    CUDA_DEVICE static cmplx prefactor(const State& s, int j)
    {
        if (!factorises(s, j))
            return cmplx(1.0, 0.0);
        return s.fresnel->get_fourier_prefactor(s.amp[j], s.phase[j], s.fdot0[j],
                                                s.window_factor);
    }

    CUDA_DEVICE static cmplx kernel(const State& s, int j, int freq_j_here,
                                    double freq_here)
    {
        if (!factorises(s, j))
            return value(s, j, freq_j_here, freq_here);
        return s.fresnel->get_fourier_kernel(s.f0[j], s.fdot0[j], s.t_seg, freq_here);
    }

    // True when two (column, channel) pairs have the same kernel at every pixel: every input
    // the kernel reads compares equal, so sharing it cannot change a finite value.
    CUDA_DEVICE static bool same_kernel(const State& s_a, int j_a, const State& s_b, int j_b)
    {
        return factorises(s_a, j_a) && factorises(s_b, j_b)
            && s_a.f0[j_a] == s_b.f0[j_b] && s_a.fdot0[j_a] == s_b.fdot0[j_b]
            && s_a.t_seg == s_b.t_seg;
    }
};

// ===========================================================================
// Tile seam
// ===========================================================================
// A tile evaluates the pixels of one template. The consumers below (get_ll, fill_global,
// swap_ll, the F-statistic and both gradients) keep the pixel loop, the window mask, the 0.5
// real-signal factor, the fill factor and every accumulation. The contract and its pixel maps
// are described in docs/response/dffr/README.md. A tile type TileT supplies:
//   Inputs                          plain launch inputs; kernels and consumers pass them by value
//   BlockShared                     the block's static shared memory
//   table_bytes()                   dynamic shared memory of one template (host and device)
//   Map                             which thread evaluates which (row, bin) of a pass
//   Context(inputs, stft, shared)   per-thread view of the block; every thread constructs it
//     rows_per_pass()               rows (STFT segments) built per pass
//   Template(context, table)        one template; `table` is its dynamic shared memory
//     load(context, params, bin_i)  once per evaluation, from the shared parameters
//     build(context, inputs, base, rows)   once per pass, called by every thread
//     carrier_bin(row)              the row's band is carrier_bin -+ n_side_bins
//     values(row, bin, f, out)      raw pixel values of the three channels
// ===========================================================================

// Pixel map of a tile. Per pass every thread calls, in this order: first_row, row_stride and
// set_band for the rows whose band it sets; finish_bands; its pixels (first_pixel, end_pixel,
// pixel_stride, locate); finish_pass.
// * One row per thread: thread tid owns row tid of each pass and walks its band in increasing
// * frequency.
// ! Needs rows_per_pass equal to the block size.
struct StftRowPerThreadMap
{
    int row;        // this thread's row in the pass; -1 when it has none
    int band_lo;    // first and last active bin of that row's band
    int band_hi;

    template <class ContextT>
    CUDA_DEVICE StftRowPerThreadMap(const ContextT& context, int rows_here)
        : row(-1), band_lo(0), band_hi(-1)
    {
        (void) context;
        (void) rows_here;
    }

    CUDA_DEVICE int first_row() const { return THREAD_START_X; }
    CUDA_DEVICE int row_stride() const { return BLOCK_INCR_X; }

    CUDA_DEVICE void set_band(int row_in_pass, int lo, int hi)
    {
        row = row_in_pass;
        band_lo = lo;
        band_hi = hi;
    }

    CUDA_DEVICE void finish_bands() const {}

    // A pixel is its active frequency bin.
    CUDA_DEVICE int first_pixel() const { return band_lo; }
    CUDA_DEVICE int end_pixel() const { return band_hi + 1; }
    CUDA_DEVICE int pixel_stride() const { return 1; }

    CUDA_DEVICE bool locate(int pixel, int* row_out, int* freq_j_out) const
    {
        *row_out = row;
        *freq_j_out = pixel;
        return true;
    }

    CUDA_DEVICE void finish_pass() const {}
};

// A column policy (above) as a tile. Thread tid builds and evaluates the columns tid,
// tid + blockDim.x, ... in turn, each over its whole band; the column states stay in registers.
template <class ColumnT>
struct ColumnTile
{
    using SourceT = typename ColumnT::Source;
    using Map = StftRowPerThreadMap;

    struct Inputs
    {
        Orbits* orbits;
        TDIConfig* tdi_config;
        STFTFresnel* fresnel;
        double T;
        double t_ref;
        double window_factor;
        bool freq_from_tdi_phase;
    };

    struct BlockShared
    {
        int link_space_craft_rec[NLINKS];
        int link_space_craft_em[NLINKS];
    };

    CUDA_CALLABLE_MEMBER static size_t table_bytes() { return 0; }

    // * The launch settings reach build() as arguments, not as members: copied into the context,
    // * they cost the gradient kernel its fourth block per SM on sm_80 (129 registers).
    struct Context
    {
        SourceT src;
        STFTDomain* stft;
        int* link_space_craft_rec;
        int* link_space_craft_em;

        // ! Every thread must construct it: fill_link_arrays ends with a barrier.
        CUDA_DEVICE Context(const Inputs& inputs, STFTDomain* stft_in, BlockShared& shared)
            : src(inputs.orbits, inputs.tdi_config, inputs.T, inputs.t_ref),
              stft(stft_in),
              link_space_craft_rec(shared.link_space_craft_rec),
              link_space_craft_em(shared.link_space_craft_em)
        {
            src.fill_link_arrays(link_space_craft_rec, link_space_craft_em);
        }

        CUDA_DEVICE int rows_per_pass() const { return BLOCK_INCR_X; }
    };

    struct Template
    {
        typename ColumnT::State col;   // the column of this thread's row
        Vec k;
        Vec u;
        Vec v;
        double* params;
        int bin_i;

        CUDA_DEVICE Template(Context& context, unsigned char* table)
            : k(0.0, 0.0, 0.0), u(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), params(nullptr), bin_i(0)
        {
            (void) context;
            (void) table;
        }

        // Sky vectors, recomputed per evaluation so that a perturbed parameter reaches them.
        CUDA_DEVICE void load(Context& context, double* params_in, int bin_i_in)
        {
            params = params_in;
            bin_i = bin_i_in;
            context.src.get_sky_vectors(&k, &u, &v, params);
        }

        CUDA_DEVICE void build(Context& context, const Inputs& inputs, int row_base, int rows_here)
        {
            int row = THREAD_START_X;
            if (row >= rows_here)
                return;
            // Midpoint anchoring: sample the source and (f0, fdot0) at the bin midpoint, but pass
            // the WINDOW START to the Fresnel evaluator, which re-anchors internally.
            double t_anchor_shift = inputs.fresnel->use_midpoint ? 0.5 * context.stft->dt : 0.0;
            int time_i = row_base + row;
            double t_seg = context.stft->t0 + time_i * context.stft->dt;   // window start
            ColumnT::setup(col, context.src, inputs.fresnel, context.stft, params, k, u, v,
                           context.link_space_craft_rec, context.link_space_craft_em, bin_i,
                           t_seg, t_anchor_shift, inputs.window_factor,
                           inputs.freq_from_tdi_phase);
        }

        // * The map hands each thread its own row only, so `row` is the row of `col`.
        CUDA_DEVICE int carrier_bin(int row) const
        {
            (void) row;
            return col.carrier_j;
        }

        CUDA_DEVICE STFT_TILE_PIXEL_INLINE void values(int row, int freq_j_here, double freq_here,
                                                       cmplx* out) const
        {
            (void) row;
            for (int j = 0; j < 3; j += 1)
                out[j] = ColumnT::value(col, j, freq_j_here, freq_here);
        }
    };
};

// Dynamic shared memory of a tile launch; on the CPU the launcher passes a host buffer instead.
// TODO: take the budget of tiles with shared tables from the device, not from the 48 kB that a
// launch gets by default. cudaDevAttrMaxSharedMemoryPerBlockOptin is 227 kB on an H100 (228 kB per
// SM) and 163 kB on an A100, so the two DFFR tables of swap (64.8 kB at 32 rows) fit 3 blocks per
// SM on an H100 and 2 on an A100.
#ifdef __CUDACC__
static CUDA_DEVICE unsigned char* stft_tile_arena(unsigned char* host_arena)
{
    extern __shared__ __align__(16) unsigned char stft_tile_dynamic_shared[];
    (void) host_arena;
    return stft_tile_dynamic_shared;
}
#else
inline unsigned char* stft_tile_arena(unsigned char* host_arena)
{
    return host_arena;
}
#endif

// Launch inputs of a column tile; on the GPU they point to the device copies of the structs, and
// *stft_launch receives the domain pointer the kernel reads.
template <class SourceT, class ColumnT>
inline typename ColumnTile<ColumnT>::Inputs stft_column_tile_inputs(
    Orbits* orbits, TDIConfig* tdi_config, STFTFresnel* fresnel, STFTDomain* stft,
    double T, double t_ref, double window_factor, bool freq_from_tdi_phase,
    STFTDomain** stft_launch)
{
    static_assert(std::is_same<typename ColumnT::Source, SourceT>::value,
                  "the column policy must evaluate the source class of the entry point");
#ifdef __CUDACC__
    const STFTDeviceStructs& dev = stft_device_structs(orbits, tdi_config, fresnel, stft);
    *stft_launch = dev.stft;
    return {dev.orbits, dev.tdi_config, dev.fresnel, T, t_ref, window_factor, freq_from_tdi_phase};
#else
    *stft_launch = stft;
    return {orbits, tdi_config, fresnel, T, t_ref, window_factor, freq_from_tdi_phase};
#endif
}

// ===========================================================================
// Per-binary (d|h),(h|h) evaluation for one parameter vector (already loaded
// into the shared `params`). Zeroes the supplied scratch, runs the pass x
// pixel x channel loop of the tile, reduces, and writes the 4*diff_comp-scaled
// (d|h),(h|h) into *d_h_val,*h_h_val (broadcast to all threads on GPU; on CPU
// tid==0 holds the sum). Loads the template from `params` each call so it is
// reusable after a parameter perturbation (the get_ll gradient path).
// Shared by stft_get_ll_kernel and stft_get_ll_grad_kernel so the gradient's
// forward evaluation is byte-identical to get_ll.
// ===========================================================================
template <class TileT>
CUDA_DEVICE void stft_eval_block_ll(
    typename TileT::Context& context, typename TileT::Inputs inputs,
    unsigned char* table, STFTDomain* stft,
    double* params, int bin_i,
    int data_index, int noise_index, int start_j, int n_side_bins,
    cmplx* d_h_tmp, cmplx* h_h_tmp, int tid,
    cmplx* d_h_val, cmplx* h_h_val)
{
    // * Local to one evaluation, so its state is dead between the repeated calls of a kernel.
    typename TileT::Template tpl(context, table);
    cmplx fresnel_val[3];

    double df = stft->df;
    double f_min = stft->f_min;
    int num_times = stft->num_times;
    int num_freqs = stft->num_freqs;
    int rows_per_pass = context.rows_per_pass();

    d_h_tmp[tid] = cmplx(0.0, 0.0);
    h_h_tmp[tid] = cmplx(0.0, 0.0);
    CUDA_SYNC_THREADS;

    tpl.load(context, params, bin_i);
    for (int row_base = 0; row_base < num_times; row_base += rows_per_pass)
    {
        int rows_here = num_times - row_base;
        if (rows_here > rows_per_pass)
            rows_here = rows_per_pass;
        tpl.build(context, inputs, row_base, rows_here);

        typename TileT::Map map(context, rows_here);
        for (int row = map.first_row(); row < rows_here; row += map.row_stride())
        {
            int carrier_j = tpl.carrier_bin(row);
            map.set_band(row, carrier_j - n_side_bins, carrier_j + n_side_bins);
        }
        map.finish_bands();

        for (int pixel = map.first_pixel(); pixel < map.end_pixel(); pixel += map.pixel_stride())
        {
            //  Two indices from here on: freq_j_here is the ACTIVE bin (0 at the domain's
            //  f_min) and sets the frequency; freq_j_local is where this cell STORES it.
            //  They differ by the cell's window start; start_j = 0 is the full grid.
            int row, freq_j_here;
            if (!map.locate(pixel, &row, &freq_j_here))
                continue;
            int time_i = row_base + row;
            int freq_j_local = freq_j_here - start_j;
            if ((freq_j_local >= 0) && (freq_j_local <= num_freqs - 1))
            {
                double freq_here = f_min + freq_j_here * df;
                tpl.values(row, freq_j_here, freq_here, fresnel_val);
                for (int j = 0; j < 3; j += 1)
                    fresnel_val[j] = 0.5 * fresnel_val[j];
                stft->add_ip_contrib(d_h_tmp, h_h_tmp, fresnel_val,
                                     time_i, freq_j_local, data_index, noise_index);
            }
        }
        map.finish_pass();
    }
    CUDA_SYNC_THREADS;
#ifdef __CUDACC__
    cmplx d_h_red = 4.0 * stft->diff_comp * stft_block_reduce_cmplx(d_h_tmp);
    CUDA_SYNC_THREADS;
    cmplx h_h_red = 4.0 * stft->diff_comp * stft_block_reduce_cmplx(h_h_tmp);
    *d_h_val = d_h_red;
    *h_h_val = h_h_red;
    CUDA_SYNC_THREADS;
#else
    *d_h_val = 4.0 * stft->diff_comp * d_h_tmp[0];
    *h_h_val = 4.0 * stft->diff_comp * h_h_tmp[0];
#endif
}

// ===========================================================================
// get_ll : (d|h) and (h|h) per binary.
// ===========================================================================
template <class TileT>
CUDA_KERNEL
void stft_get_ll_kernel(
    cmplx* d_h_out, cmplx* h_h_out,
    typename TileT::Inputs inputs, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, int n_side_bins,
    int* start_freq_inds, unsigned char* host_arena)
{
    CUDA_SHARED cmplx d_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx h_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED double params[N_PARAMS_MAX];
    CUDA_SHARED typename TileT::BlockShared tile_shared;

    typename TileT::Context context(inputs, stft, tile_shared);
    CUDA_SYNC_THREADS;
    unsigned char* arena = stft_tile_arena(host_arena);

#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
        // * Window start of the cell this binary is evaluated against; 0 is the full grid.
        int start_j = (start_freq_inds == nullptr) ? 0 : start_freq_inds[data_index];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;

        cmplx d_h_val, h_h_val;
        stft_eval_block_ll<TileT>(
            context, inputs, arena, stft, params, bin_i,
            data_index, noise_index, start_j, n_side_bins,
            d_h_tmp, h_h_tmp, tid, &d_h_val, &h_h_val);

        if (tid == 0)
        {
            d_h_out[bin_i] = d_h_val;
            h_h_out[bin_i] = h_h_val;
        }
        CUDA_SYNC_THREADS;
    }
}

template <class TileT>
inline void stft_get_ll_launch(
    cmplx* d_h_out, cmplx* h_h_out,
    const typename TileT::Inputs& inputs, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, int n_side_bins, int* start_freq_inds)
{
    const size_t shared_bytes = TileT::table_bytes();
#ifdef __CUDACC__
    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_get_ll_kernel<TileT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        d_h_out, h_h_out, inputs, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds, nullptr);
    gpuErrchk(cudaGetLastError());
#else
    std::vector<unsigned char> host_arena(shared_bytes);
    stft_get_ll_kernel<TileT>(
        d_h_out, h_h_out, inputs, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds, host_arena.data());
#endif
}

template <class SourceT, class ColumnT = FresnelColumn<SourceT>>
inline void stft_get_ll_impl(
    cmplx* d_h_out, cmplx* h_h_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    int* start_freq_inds = nullptr)
{
    STFTDomain* stft_launch = nullptr;
    const auto inputs = stft_column_tile_inputs<SourceT, ColumnT>(
        orbits, tdi_config, fresnel, stft, T, t_ref, window_factor, freq_from_tdi_phase,
        &stft_launch);
    stft_get_ll_launch<ColumnTile<ColumnT>>(
        d_h_out, h_h_out, inputs, stft_launch,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds);
}

// ===========================================================================
// fill_global : scatter 0.5 * factor * fourier_value into a per-template STFT
// grid (active-band layout (num_templates, nchannels, num_times, num_freqs),
// row-major, num_freqs fastest -- the layout STFTComputationGroup consumes).
// Shares get_ll's inner loop exactly, so feeding the produced templates back
// through STFTComputationGroup.compute_signal_likelihood_terms reproduces
// get_ll's (d|h),(h|h) to machine precision.
//
// `active_band` is accepted for parity with the WDM-het fill_global; for STFT
// the domain's num_freqs already IS the active band, so the active-band layout
// is the implemented path.
// ===========================================================================
template <class TileT>
CUDA_KERNEL
void stft_fill_global_kernel(
    cmplx* template_fill,
    typename TileT::Inputs inputs, STFTDomain* stft,
    double* params_all, int* data_index_all, double* factors_all,
    int num_bin, int nparams, int n_side_bins, bool active_band,
    int* start_freq_inds, unsigned char* host_arena)
{
    (void) active_band;  // STFT grid == active band; layout is (.., num_times, num_freqs)
    CUDA_SHARED double params[N_PARAMS_MAX];
    CUDA_SHARED typename TileT::BlockShared tile_shared;

    typename TileT::Context context(inputs, stft, tile_shared);
    CUDA_SYNC_THREADS;
    unsigned char* arena = stft_tile_arena(host_arena);
    cmplx tile_val[3];

    double df = stft->df;
    double f_min = stft->f_min;
    int num_times = stft->num_times;
    int num_freqs = stft->num_freqs;
    int nchannels = stft->num_channels;
    int rows_per_pass = context.rows_per_pass();

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int start_j = (start_freq_inds == nullptr) ? 0 : start_freq_inds[data_index];
        double factor = factors_all[bin_i];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;

        // * One template per source, so its state is dead between the sources of a block.
        typename TileT::Template tpl(context, arena);
        tpl.load(context, params, bin_i);
        for (int row_base = 0; row_base < num_times; row_base += rows_per_pass)
        {
            int rows_here = num_times - row_base;
            if (rows_here > rows_per_pass)
                rows_here = rows_per_pass;
            tpl.build(context, inputs, row_base, rows_here);

            typename TileT::Map map(context, rows_here);
            for (int row = map.first_row(); row < rows_here; row += map.row_stride())
            {
                int carrier_j = tpl.carrier_bin(row);
                map.set_band(row, carrier_j - n_side_bins, carrier_j + n_side_bins);
            }
            map.finish_bands();

            for (int pixel = map.first_pixel(); pixel < map.end_pixel(); pixel += map.pixel_stride())
            {
                int row, freq_j_here;
                if (!map.locate(pixel, &row, &freq_j_here))
                    continue;
                int time_i = row_base + row;
                int freq_j_local = freq_j_here - start_j;
                if ((freq_j_local >= 0) && (freq_j_local <= num_freqs - 1))
                {
                    double freq_here = f_min + freq_j_here * df;
                    tpl.values(row, freq_j_here, freq_here, tile_val);
                    for (int j = 0; j < 3; j += 1)
                    {
                        cmplx val = factor * 0.5 * tile_val[j];
                        // template_fill[(((data_index*nch + j)*num_times + time_i)*num_freqs) + freq_j_local]
                        size_t idx = ((((size_t) data_index * nchannels + j) * num_times
                                       + time_i) * num_freqs) + freq_j_local;
#ifdef __CUDACC__
                        atomicAdd(((double*) &template_fill[idx]) + 0, val.real());
                        atomicAdd(((double*) &template_fill[idx]) + 1, val.imag());
#else
                        template_fill[idx] = template_fill[idx] + val;
#endif
                    }
                }
            }
            map.finish_pass();
        }
        CUDA_SYNC_THREADS;
    }
}

template <class TileT>
inline void stft_fill_global_launch(
    cmplx* template_fill,
    const typename TileT::Inputs& inputs, STFTDomain* stft,
    double* params_all, int* data_index_all, double* factors_all,
    int num_bin, int nparams, int n_side_bins, bool active_band, int* start_freq_inds)
{
    const size_t shared_bytes = TileT::table_bytes();
#ifdef __CUDACC__
    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_fill_global_kernel<TileT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        template_fill, inputs, stft,
        params_all, data_index_all, factors_all,
        num_bin, nparams, n_side_bins, active_band, start_freq_inds, nullptr);
    gpuErrchk(cudaGetLastError());
#else
    std::vector<unsigned char> host_arena(shared_bytes);
    stft_fill_global_kernel<TileT>(
        template_fill, inputs, stft,
        params_all, data_index_all, factors_all,
        num_bin, nparams, n_side_bins, active_band, start_freq_inds, host_arena.data());
#endif
}

template <class SourceT, class ColumnT = FresnelColumn<SourceT>>
inline void stft_fill_global_impl(
    cmplx* template_fill,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, double* factors_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase, bool active_band,
    int* start_freq_inds = nullptr)
{
    STFTDomain* stft_launch = nullptr;
    const auto inputs = stft_column_tile_inputs<SourceT, ColumnT>(
        orbits, tdi_config, fresnel, stft, T, t_ref, window_factor, freq_from_tdi_phase,
        &stft_launch);
    stft_fill_global_launch<ColumnTile<ColumnT>>(
        template_fill, inputs, stft_launch,
        params_all, data_index_all, factors_all,
        num_bin, nparams, n_side_bins, active_band, start_freq_inds);
}

// ===========================================================================
// swap_ll : the five inner-product terms of an RJMCMC source-swap step, per
// binary -- (d|h_add), (d|h_remove), (h_add|h_add), (h_remove|h_remove),
// (h_add|h_remove). Carries an add-track and a remove-track template, each with
// its own params, on-the-fly TDI value and Doppler-corrected (f0, fdot0), and
// sums over the UNION of the two carriers' side-bands so the cross term
// (h_add|h_remove) is captured wherever either track has support. Defers all
// five accumulations to STFTDomain::add_ip_swap_contrib (one channel loop, one
// noise-matrix fetch shared across the terms). Ported from lisa-on-gpu's
// gb_stft_swap_ll_kernel; same per-pixel convention as stft_get_ll_kernel.
//
// With params_add == params_remove this reduces, pixel for pixel, to
// stft_get_ll_kernel: the union band collapses to the single carrier band and
// add_ip_swap_contrib's add/remove/cross terms all evaluate the same template,
// so (d|h_add)=(d|h_remove)=(d|h) and (h_add|h_add)=(h_remove|h_remove)=
// (h_add|h_remove)=(h|h).
// ===========================================================================
// Per-binary 5-term swap evaluation for one (params_add, params_remove) pair
// (both already in shared memory), with one tile template per track. Same
// per-pixel convention as stft_eval_block_ll; sums over the UNION of the
// add/remove carriers' side bands and defers to STFTDomain::add_ip_swap_contrib.
// Writes the five 4*diff_comp-scaled terms (broadcast to all threads on GPU).
// Loads both templates from their params each call so it is reusable after a
// parameter perturbation (the swap gradient path). Shared by
// stft_swap_ll_kernel and stft_swap_ll_grad_kernel.
template <class TileT>
CUDA_DEVICE void stft_eval_block_swap(
    typename TileT::Context& context, typename TileT::Inputs inputs,
    unsigned char* table_add, unsigned char* table_remove,
    STFTDomain* stft,
    double* params_add, double* params_remove, int bin_i,
    int data_index, int noise_index, int start_j, int n_side_bins,
    cmplx* d_h_add_tmp, cmplx* d_h_remove_tmp, cmplx* add_add_tmp,
    cmplx* remove_remove_tmp, cmplx* add_remove_tmp, int tid,
    cmplx* d_h_add_val, cmplx* d_h_remove_val, cmplx* add_add_val,
    cmplx* remove_remove_val, cmplx* add_remove_val)
{
    // * Local to one evaluation, so their state is dead between the repeated calls of a kernel.
    typename TileT::Template tpl_add(context, table_add);
    typename TileT::Template tpl_remove(context, table_remove);
    cmplx fresnel_val_add[3];
    cmplx fresnel_val_remove[3];

    double df = stft->df;
    double f_min = stft->f_min;
    int num_times = stft->num_times;
    int num_freqs = stft->num_freqs;
    int rows_per_pass = context.rows_per_pass();

    d_h_add_tmp[tid] = cmplx(0.0, 0.0);
    d_h_remove_tmp[tid] = cmplx(0.0, 0.0);
    add_add_tmp[tid] = cmplx(0.0, 0.0);
    remove_remove_tmp[tid] = cmplx(0.0, 0.0);
    add_remove_tmp[tid] = cmplx(0.0, 0.0);
    CUDA_SYNC_THREADS;

    tpl_add.load(context, params_add, bin_i);
    tpl_remove.load(context, params_remove, bin_i);
    for (int row_base = 0; row_base < num_times; row_base += rows_per_pass)
    {
        int rows_here = num_times - row_base;
        if (rows_here > rows_per_pass)
            rows_here = rows_per_pass;
        // (f0, fdot0) from each track's own TDI phase (Doppler-corrected
        // when freq_from_tdi_phase; else astrophysical get_f/get_fdot).
        tpl_add.build(context, inputs, row_base, rows_here);
        tpl_remove.build(context, inputs, row_base, rows_here);

        typename TileT::Map map(context, rows_here);
        for (int row = map.first_row(); row < rows_here; row += map.row_stride())
        {
            int freq_j_add = tpl_add.carrier_bin(row);
            int freq_j_remove = tpl_remove.carrier_bin(row);
            int freq_j_min = (freq_j_add < freq_j_remove) ? freq_j_add : freq_j_remove;
            int freq_j_max = (freq_j_add > freq_j_remove) ? freq_j_add : freq_j_remove;
            map.set_band(row, freq_j_min - n_side_bins, freq_j_max + n_side_bins);
        }
        map.finish_bands();

        for (int pixel = map.first_pixel(); pixel < map.end_pixel(); pixel += map.pixel_stride())
        {
            int row, freq_j_here;
            if (!map.locate(pixel, &row, &freq_j_here))
                continue;
            int time_i = row_base + row;
            int freq_j_local = freq_j_here - start_j;
            if ((freq_j_local >= 0) && (freq_j_local <= num_freqs - 1))
            {
                double freq_here = f_min + freq_j_here * df;
                tpl_add.values(row, freq_j_here, freq_here, fresnel_val_add);
                tpl_remove.values(row, freq_j_here, freq_here, fresnel_val_remove);
                for (int j = 0; j < 3; j += 1)
                {
                    fresnel_val_add[j] = 0.5 * fresnel_val_add[j];
                    fresnel_val_remove[j] = 0.5 * fresnel_val_remove[j];
                }
                stft->add_ip_swap_contrib(
                    d_h_add_tmp, d_h_remove_tmp, add_add_tmp, remove_remove_tmp,
                    add_remove_tmp, fresnel_val_add, fresnel_val_remove,
                    time_i, freq_j_local, data_index, noise_index);
            }
        }
        map.finish_pass();
    }
    CUDA_SYNC_THREADS;
#ifdef __CUDACC__
    cmplx d_h_add_red = 4.0 * stft->diff_comp * stft_block_reduce_cmplx(d_h_add_tmp);
    CUDA_SYNC_THREADS;
    cmplx d_h_remove_red = 4.0 * stft->diff_comp * stft_block_reduce_cmplx(d_h_remove_tmp);
    CUDA_SYNC_THREADS;
    cmplx add_add_red = 4.0 * stft->diff_comp * stft_block_reduce_cmplx(add_add_tmp);
    CUDA_SYNC_THREADS;
    cmplx remove_remove_red = 4.0 * stft->diff_comp * stft_block_reduce_cmplx(remove_remove_tmp);
    CUDA_SYNC_THREADS;
    cmplx add_remove_red = 4.0 * stft->diff_comp * stft_block_reduce_cmplx(add_remove_tmp);
    *d_h_add_val = d_h_add_red;
    *d_h_remove_val = d_h_remove_red;
    *add_add_val = add_add_red;
    *remove_remove_val = remove_remove_red;
    *add_remove_val = add_remove_red;
    CUDA_SYNC_THREADS;
#else
    *d_h_add_val = 4.0 * stft->diff_comp * d_h_add_tmp[0];
    *d_h_remove_val = 4.0 * stft->diff_comp * d_h_remove_tmp[0];
    *add_add_val = 4.0 * stft->diff_comp * add_add_tmp[0];
    *remove_remove_val = 4.0 * stft->diff_comp * remove_remove_tmp[0];
    *add_remove_val = 4.0 * stft->diff_comp * add_remove_tmp[0];
#endif
}

template <class TileT>
CUDA_KERNEL
void stft_swap_ll_kernel(
    cmplx* d_h_add_out, cmplx* d_h_remove_out,
    cmplx* add_add_out, cmplx* remove_remove_out, cmplx* add_remove_out,
    typename TileT::Inputs inputs, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, int n_side_bins,
    int* start_freq_inds, unsigned char* host_arena)
{
    CUDA_SHARED cmplx d_h_add_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx d_h_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx add_add_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx remove_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx add_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED double params_add[N_PARAMS_MAX];
    CUDA_SHARED double params_remove[N_PARAMS_MAX];
    CUDA_SHARED typename TileT::BlockShared tile_shared;

    typename TileT::Context context(inputs, stft, tile_shared);
    CUDA_SYNC_THREADS;
    unsigned char* arena = stft_tile_arena(host_arena);

#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
        int start_j = (start_freq_inds == nullptr) ? 0 : start_freq_inds[data_index];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
        {
            params_add[i] = params_add_all[bin_i * nparams + i];
            params_remove[i] = params_remove_all[bin_i * nparams + i];
        }
        CUDA_SYNC_THREADS;

        cmplx d_h_add_val, d_h_remove_val, add_add_val, remove_remove_val, add_remove_val;
        stft_eval_block_swap<TileT>(
            context, inputs, arena, arena + TileT::table_bytes(), stft,
            params_add, params_remove, bin_i,
            data_index, noise_index, start_j, n_side_bins,
            d_h_add_tmp, d_h_remove_tmp, add_add_tmp, remove_remove_tmp, add_remove_tmp, tid,
            &d_h_add_val, &d_h_remove_val, &add_add_val, &remove_remove_val, &add_remove_val);

        if (tid == 0)
        {
            d_h_add_out[bin_i] = d_h_add_val;
            d_h_remove_out[bin_i] = d_h_remove_val;
            add_add_out[bin_i] = add_add_val;
            remove_remove_out[bin_i] = remove_remove_val;
            add_remove_out[bin_i] = add_remove_val;
        }
        CUDA_SYNC_THREADS;
    }
}

template <class TileT>
inline void stft_swap_ll_launch(
    cmplx* d_h_add_out, cmplx* d_h_remove_out,
    cmplx* add_add_out, cmplx* remove_remove_out, cmplx* add_remove_out,
    const typename TileT::Inputs& inputs, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, int n_side_bins, int* start_freq_inds)
{
    const size_t shared_bytes = 2 * TileT::table_bytes();   // add and remove templates
#ifdef __CUDACC__
    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_swap_ll_kernel<TileT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        inputs, stft, params_add_all, params_remove_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds, nullptr);
    gpuErrchk(cudaGetLastError());
#else
    std::vector<unsigned char> host_arena(shared_bytes);
    stft_swap_ll_kernel<TileT>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        inputs, stft, params_add_all, params_remove_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds, host_arena.data());
#endif
}

template <class SourceT, class ColumnT = FresnelColumn<SourceT>>
inline void stft_swap_ll_impl(
    cmplx* d_h_add_out, cmplx* d_h_remove_out,
    cmplx* add_add_out, cmplx* remove_remove_out, cmplx* add_remove_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    int* start_freq_inds = nullptr)
{
    STFTDomain* stft_launch = nullptr;
    const auto inputs = stft_column_tile_inputs<SourceT, ColumnT>(
        orbits, tdi_config, fresnel, stft, T, t_ref, window_factor, freq_from_tdi_phase,
        &stft_launch);
    stft_swap_ll_launch<ColumnTile<ColumnT>>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        inputs, stft_launch, params_add_all, params_remove_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds);
}

// ===========================================================================
// get_fstat_ll : F-statistic per binary. Analytically maximizes the likelihood
// over the 4 extrinsic GB amplitude parameters by building the 4 Cornish &
// Crowder '05 basis filters A_i -- the normal GB waveform at fixed
//   (A, iota, psi, phi0) = (2, pi/2, {0,pi/4,0,pi/4}, {0,pi,3pi/2,pi/2}),
// intrinsic (f0,fdot,fddot,lam,beta) copied from the binary -- and forming
//   N_i  = (d   | A_i)        [4]
//   M_ij = (A_i | A_j)        [4x4 Hermitian; upper triangle = 10]
// from which the caller computes 2F = N^T M^-1 N. Every term is produced by the
// already-validated Stage-1/2 device helpers, so get_fstat is a thin
// orchestration that is byte-identical to {get_ll x4, swap_ll x6}:
//   stft_eval_block_ll(A_i)       -> (d|A_i)=N_i  and  (A_i|A_i)=M_ii (diagonal)
//   stft_eval_block_swap(A_i,A_j) -> add_remove_val=(A_i|A_j)=M_ij    (off-diag)
// This is *more* correct than the WDM common-band approximation: each inner
// product uses its own natural support (per-filter for the diagonal, the
// add/remove union for the off-diagonal). Inner products are complex; the F-stat
// uses the real part (the same convention get_ll's logL uses). Outputs carry
// re+im to mirror the WDM F-stat surface (im is a near-zero diagnostic here, not
// identically 0 as in real-valued WDM). M upper-triangle flatten:
//   m_idx(i,j) = i*4 - i*(i+1)/2 + j   for i <= j.
// ===========================================================================
template <class TileT>
CUDA_KERNEL
void stft_get_fstat_ll_kernel(
    double* N_re_out, double* N_im_out,   // (num_bin, 4)
    double* M_re_out, double* M_im_out,   // (num_bin, 10) upper triangle
    typename TileT::Inputs inputs, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, int n_side_bins,
    int* start_freq_inds, unsigned char* host_arena)
{
    constexpr int N_FILTERS = 4;
    constexpr int N_M = (N_FILTERS * (N_FILTERS + 1)) / 2;   // = 10

    // F-stat basis filter extrinsic params (Cornish & Crowder '05).
    const double A_arr   [N_FILTERS] = {2.0, 2.0, 2.0, 2.0};
    const double iota_arr[N_FILTERS] = {M_PI / 2.0, M_PI / 2.0, M_PI / 2.0, M_PI / 2.0};
    const double psi_arr [N_FILTERS] = {0.0, M_PI / 4.0, 0.0, M_PI / 4.0};
    const double phi0_arr[N_FILTERS] = {0.0, M_PI, 3.0 * M_PI / 2.0, M_PI / 2.0};

    // GB extrinsic param slots (GB convention, matches the WDM F-stat kernel;
    // SOBBH would need a trait-based specialization).
    constexpr int IDX_A = 0, IDX_PHI0 = 4, IDX_IOTA = 5, IDX_PSI = 6;

    // (i,j) -> flat upper-triangle index of the 4x4 Hermitian M (i <= j).
    auto m_idx = [] (int i, int j) -> int {
        return i * N_FILTERS - (i * (i + 1)) / 2 + j;
    };

    // Scratch reused across the eval_block_ll / eval_block_swap calls (each
    // helper re-zeroes its own accumulators on entry).
    CUDA_SHARED cmplx tmp0[NUM_THREADS_HERE];
    CUDA_SHARED cmplx tmp1[NUM_THREADS_HERE];
    CUDA_SHARED cmplx tmp2[NUM_THREADS_HERE];
    CUDA_SHARED cmplx tmp3[NUM_THREADS_HERE];
    CUDA_SHARED cmplx tmp4[NUM_THREADS_HERE];
    CUDA_SHARED double params_i[N_PARAMS_MAX];
    CUDA_SHARED double params_j[N_PARAMS_MAX];
    CUDA_SHARED typename TileT::BlockShared tile_shared;

    typename TileT::Context context(inputs, stft, tile_shared);
    CUDA_SYNC_THREADS;
    unsigned char* arena = stft_tile_arena(host_arena);

#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
        int start_j = (start_freq_inds == nullptr) ? 0 : start_freq_inds[data_index];

        // --- N_i = (d|A_i) and the M diagonal M_ii = (A_i|A_i) ---
        for (int fi = 0; fi < N_FILTERS; ++fi)
        {
            for (int k = THREAD_START_X; k < nparams; k += BLOCK_INCR_X)
                params_i[k] = params_all[bin_i * nparams + k];
            CUDA_SYNC_THREADS;
            if (tid == 0)
            {
                params_i[IDX_A]    = A_arr[fi];
                params_i[IDX_PHI0] = phi0_arr[fi];
                params_i[IDX_IOTA] = iota_arr[fi];
                params_i[IDX_PSI]  = psi_arr[fi];
            }
            CUDA_SYNC_THREADS;

            cmplx d_h_val, h_h_val;
            stft_eval_block_ll<TileT>(
                context, inputs, arena, stft, params_i, bin_i,
                data_index, noise_index, start_j, n_side_bins,
                tmp0, tmp1, tid, &d_h_val, &h_h_val);

            if (tid == 0)
            {
                N_re_out[bin_i * N_FILTERS + fi] = d_h_val.real();
                N_im_out[bin_i * N_FILTERS + fi] = d_h_val.imag();
                int mii = m_idx(fi, fi);
                M_re_out[bin_i * N_M + mii] = h_h_val.real();
                M_im_out[bin_i * N_M + mii] = h_h_val.imag();
            }
            CUDA_SYNC_THREADS;
        }

        // --- off-diagonal M_ij = (A_i|A_j), i < j (the swap add_remove term) ---
        for (int fi = 0; fi < N_FILTERS; ++fi)
        {
            for (int fj = fi + 1; fj < N_FILTERS; ++fj)
            {
                for (int k = THREAD_START_X; k < nparams; k += BLOCK_INCR_X)
                {
                    params_i[k] = params_all[bin_i * nparams + k];
                    params_j[k] = params_all[bin_i * nparams + k];
                }
                CUDA_SYNC_THREADS;
                if (tid == 0)
                {
                    params_i[IDX_A]    = A_arr[fi];    params_i[IDX_PHI0] = phi0_arr[fi];
                    params_i[IDX_IOTA] = iota_arr[fi]; params_i[IDX_PSI]  = psi_arr[fi];
                    params_j[IDX_A]    = A_arr[fj];    params_j[IDX_PHI0] = phi0_arr[fj];
                    params_j[IDX_IOTA] = iota_arr[fj]; params_j[IDX_PSI]  = psi_arr[fj];
                }
                CUDA_SYNC_THREADS;

                cmplx d_h_add_val, d_h_remove_val, add_add_val, remove_remove_val, add_remove_val;
                stft_eval_block_swap<TileT>(
                    context, inputs, arena, arena + TileT::table_bytes(), stft,
                    params_i, params_j, bin_i,
                    data_index, noise_index, start_j, n_side_bins,
                    tmp0, tmp1, tmp2, tmp3, tmp4, tid,
                    &d_h_add_val, &d_h_remove_val, &add_add_val, &remove_remove_val,
                    &add_remove_val);

                if (tid == 0)
                {
                    int mij = m_idx(fi, fj);
                    M_re_out[bin_i * N_M + mij] = add_remove_val.real();
                    M_im_out[bin_i * N_M + mij] = add_remove_val.imag();
                }
                CUDA_SYNC_THREADS;
            }
        }
    }
}

template <class TileT>
inline void stft_get_fstat_ll_launch(
    double* N_re_out, double* N_im_out,
    double* M_re_out, double* M_im_out,
    const typename TileT::Inputs& inputs, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, int n_side_bins, int* start_freq_inds)
{
    const size_t shared_bytes = 2 * TileT::table_bytes();   // two filters at a time
#ifdef __CUDACC__
    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_get_fstat_ll_kernel<TileT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        N_re_out, N_im_out, M_re_out, M_im_out, inputs, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds, nullptr);
    gpuErrchk(cudaGetLastError());
#else
    std::vector<unsigned char> host_arena(shared_bytes);
    stft_get_fstat_ll_kernel<TileT>(
        N_re_out, N_im_out, M_re_out, M_im_out, inputs, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds, host_arena.data());
#endif
}

template <class SourceT, class ColumnT = FresnelColumn<SourceT>>
inline void stft_get_fstat_ll_impl(
    double* N_re_out, double* N_im_out,
    double* M_re_out, double* M_im_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    int* start_freq_inds = nullptr)
{
    STFTDomain* stft_launch = nullptr;
    const auto inputs = stft_column_tile_inputs<SourceT, ColumnT>(
        orbits, tdi_config, fresnel, stft, T, t_ref, window_factor, freq_from_tdi_phase,
        &stft_launch);
    stft_get_fstat_ll_launch<ColumnTile<ColumnT>>(
        N_re_out, N_im_out, M_re_out, M_im_out, inputs, stft_launch,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds);
}

// ===========================================================================
// get_ll_grad : per-binary, per-parameter central finite difference of the
// log-likelihood logL = Re(d|h) - 0.5*(h|h) over the nparams parameters. For
// each param k with eps_k > 0 we perturb the shared params[k] by +-eps_k,
// re-evaluate (d|h),(h|h) via stft_eval_block_ll (so the forward model is
// byte-identical to get_ll), form q_+- = Re(d|h) - 0.5*Re(h|h), and write
// grad[k] = (q_+ - q_-) / (2*eps_k). eps_k <= 0 freezes parameter k (grad 0).
// The constant -0.5*(d|d) term cancels in the difference, so the data
// self-term is never needed. grad_out layout: grad_out[bin*nparams + k].
// (Mirrors the FD/signal-het central-difference gradients.)
// ===========================================================================
template <class TileT>
CUDA_KERNEL
void stft_get_ll_grad_kernel(
    double* grad_out,
    typename TileT::Inputs inputs, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    double* param_eps,
    int num_bin, int nparams, int n_side_bins,
    int* start_freq_inds, unsigned char* host_arena)
{
    CUDA_SHARED cmplx d_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx h_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED double params[N_PARAMS_MAX];
    CUDA_SHARED typename TileT::BlockShared tile_shared;

    typename TileT::Context context(inputs, stft, tile_shared);
    CUDA_SYNC_THREADS;
    unsigned char* arena = stft_tile_arena(host_arena);

#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
        int start_j = (start_freq_inds == nullptr) ? 0 : start_freq_inds[data_index];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;

        for (int kk = 0; kk < nparams; kk += 1)
        {
            double eps_k = param_eps[kk];
            if (eps_k <= 0.0)
            {
                if (tid == 0) grad_out[bin_i * nparams + kk] = 0.0;
                CUDA_SYNC_THREADS;
                continue;
            }
            // Only tid 0 mutates / restores the shared params slot; the eval
            // reads it after the sync below, so no read-before-write race.
            double saved = (tid == 0) ? params[kk] : 0.0;

            if (tid == 0) params[kk] = saved + eps_k;
            CUDA_SYNC_THREADS;
            cmplx d_h_p, h_h_p;
            stft_eval_block_ll<TileT>(
                context, inputs, arena, stft, params, bin_i,
                data_index, noise_index, start_j, n_side_bins,
                d_h_tmp, h_h_tmp, tid, &d_h_p, &h_h_p);
            double q_p = d_h_p.real() - 0.5 * h_h_p.real();

            if (tid == 0) params[kk] = saved - eps_k;
            CUDA_SYNC_THREADS;
            cmplx d_h_m, h_h_m;
            stft_eval_block_ll<TileT>(
                context, inputs, arena, stft, params, bin_i,
                data_index, noise_index, start_j, n_side_bins,
                d_h_tmp, h_h_tmp, tid, &d_h_m, &h_h_m);
            double q_m = d_h_m.real() - 0.5 * h_h_m.real();

            if (tid == 0) params[kk] = saved;
            CUDA_SYNC_THREADS;
            if (tid == 0)
                grad_out[bin_i * nparams + kk] = (q_p - q_m) / (2.0 * eps_k);
            CUDA_SYNC_THREADS;
        }
    }
}

template <class TileT>
inline void stft_get_ll_grad_launch(
    double* grad_out,
    const typename TileT::Inputs& inputs, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    double* param_eps,
    int num_bin, int nparams, int n_side_bins, int* start_freq_inds)
{
    const size_t shared_bytes = TileT::table_bytes();
#ifdef __CUDACC__
    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_get_ll_grad_kernel<TileT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        grad_out, inputs, stft, params_all, data_index_all, noise_index_all, param_eps,
        num_bin, nparams, n_side_bins, start_freq_inds, nullptr);
    gpuErrchk(cudaGetLastError());
#else
    std::vector<unsigned char> host_arena(shared_bytes);
    stft_get_ll_grad_kernel<TileT>(
        grad_out, inputs, stft, params_all, data_index_all, noise_index_all, param_eps,
        num_bin, nparams, n_side_bins, start_freq_inds, host_arena.data());
#endif
}

template <class SourceT, class ColumnT = FresnelColumn<SourceT>>
inline void stft_get_ll_grad_impl(
    double* grad_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    double* param_eps,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    int* start_freq_inds = nullptr)
{
    STFTDomain* stft_launch = nullptr;
    const auto inputs = stft_column_tile_inputs<SourceT, ColumnT>(
        orbits, tdi_config, fresnel, stft, T, t_ref, window_factor, freq_from_tdi_phase,
        &stft_launch);
    stft_get_ll_grad_launch<ColumnTile<ColumnT>>(
        grad_out, inputs, stft_launch, params_all, data_index_all, noise_index_all, param_eps,
        num_bin, nparams, n_side_bins, start_freq_inds);
}

// ===========================================================================
// swap_ll_grad : per-binary central-difference gradients of the swap scalar
//   S = Re(d|h_add) - Re(d|h_remove) - 0.5*Re(h_add|h_add)
//       - 0.5*Re(h_remove|h_remove) + Re(h_add|h_remove)
// (= -0.5*||d - h_add + h_remove||^2 up to the param-independent -0.5*(d|d)),
// matching the FD swap-gradient convention. grad_add[k] perturbs theta_add[k]
// (theta_remove fixed); grad_remove[k] perturbs theta_remove[k] (theta_add
// fixed). Separate eps arrays per track; eps_k <= 0 freezes that component.
// S is re-evaluated each perturbation via stft_eval_block_swap so the forward
// model is byte-identical to swap_ll. Layout: grad[bin*nparams + k].
// ===========================================================================
template <class TileT>
CUDA_KERNEL
void stft_swap_ll_grad_kernel(
    double* grad_add_out, double* grad_remove_out,
    typename TileT::Inputs inputs, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    double* param_eps_add, double* param_eps_remove,
    int num_bin, int nparams, int n_side_bins,
    int* start_freq_inds, unsigned char* host_arena)
{
    CUDA_SHARED cmplx d_h_add_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx d_h_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx add_add_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx remove_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx add_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED double params_add[N_PARAMS_MAX];
    CUDA_SHARED double params_remove[N_PARAMS_MAX];
    CUDA_SHARED typename TileT::BlockShared tile_shared;

    typename TileT::Context context(inputs, stft, tile_shared);
    CUDA_SYNC_THREADS;
    unsigned char* arena = stft_tile_arena(host_arena);
    unsigned char* arena_remove = arena + TileT::table_bytes();

#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
        int start_j = (start_freq_inds == nullptr) ? 0 : start_freq_inds[data_index];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
        {
            params_add[i] = params_add_all[bin_i * nparams + i];
            params_remove[i] = params_remove_all[bin_i * nparams + i];
        }
        CUDA_SYNC_THREADS;

        // ---- add-side gradient: perturb params_add (params_remove fixed) ----
        for (int kk = 0; kk < nparams; kk += 1)
        {
            double eps_k = param_eps_add[kk];
            if (eps_k <= 0.0)
            {
                if (tid == 0) grad_add_out[bin_i * nparams + kk] = 0.0;
                CUDA_SYNC_THREADS;
                continue;
            }
            double saved = (tid == 0) ? params_add[kk] : 0.0;

            if (tid == 0) params_add[kk] = saved + eps_k;
            CUDA_SYNC_THREADS;
            cmplx dha_p, dhr_p, aa_p, rr_p, ar_p;
            stft_eval_block_swap<TileT>(
                context, inputs, arena, arena_remove, stft, params_add, params_remove, bin_i,
                data_index, noise_index, start_j, n_side_bins,
                d_h_add_tmp, d_h_remove_tmp, add_add_tmp, remove_remove_tmp, add_remove_tmp, tid,
                &dha_p, &dhr_p, &aa_p, &rr_p, &ar_p);
            double S_p = dha_p.real() - dhr_p.real()
                         - 0.5 * aa_p.real() - 0.5 * rr_p.real() + ar_p.real();

            if (tid == 0) params_add[kk] = saved - eps_k;
            CUDA_SYNC_THREADS;
            cmplx dha_m, dhr_m, aa_m, rr_m, ar_m;
            stft_eval_block_swap<TileT>(
                context, inputs, arena, arena_remove, stft, params_add, params_remove, bin_i,
                data_index, noise_index, start_j, n_side_bins,
                d_h_add_tmp, d_h_remove_tmp, add_add_tmp, remove_remove_tmp, add_remove_tmp, tid,
                &dha_m, &dhr_m, &aa_m, &rr_m, &ar_m);
            double S_m = dha_m.real() - dhr_m.real()
                         - 0.5 * aa_m.real() - 0.5 * rr_m.real() + ar_m.real();

            if (tid == 0) params_add[kk] = saved;
            CUDA_SYNC_THREADS;
            if (tid == 0)
                grad_add_out[bin_i * nparams + kk] = (S_p - S_m) / (2.0 * eps_k);
            CUDA_SYNC_THREADS;
        }

        // ---- remove-side gradient: perturb params_remove (params_add fixed) ----
        for (int kk = 0; kk < nparams; kk += 1)
        {
            double eps_k = param_eps_remove[kk];
            if (eps_k <= 0.0)
            {
                if (tid == 0) grad_remove_out[bin_i * nparams + kk] = 0.0;
                CUDA_SYNC_THREADS;
                continue;
            }
            double saved = (tid == 0) ? params_remove[kk] : 0.0;

            if (tid == 0) params_remove[kk] = saved + eps_k;
            CUDA_SYNC_THREADS;
            cmplx dha_p, dhr_p, aa_p, rr_p, ar_p;
            stft_eval_block_swap<TileT>(
                context, inputs, arena, arena_remove, stft, params_add, params_remove, bin_i,
                data_index, noise_index, start_j, n_side_bins,
                d_h_add_tmp, d_h_remove_tmp, add_add_tmp, remove_remove_tmp, add_remove_tmp, tid,
                &dha_p, &dhr_p, &aa_p, &rr_p, &ar_p);
            double S_p = dha_p.real() - dhr_p.real()
                         - 0.5 * aa_p.real() - 0.5 * rr_p.real() + ar_p.real();

            if (tid == 0) params_remove[kk] = saved - eps_k;
            CUDA_SYNC_THREADS;
            cmplx dha_m, dhr_m, aa_m, rr_m, ar_m;
            stft_eval_block_swap<TileT>(
                context, inputs, arena, arena_remove, stft, params_add, params_remove, bin_i,
                data_index, noise_index, start_j, n_side_bins,
                d_h_add_tmp, d_h_remove_tmp, add_add_tmp, remove_remove_tmp, add_remove_tmp, tid,
                &dha_m, &dhr_m, &aa_m, &rr_m, &ar_m);
            double S_m = dha_m.real() - dhr_m.real()
                         - 0.5 * aa_m.real() - 0.5 * rr_m.real() + ar_m.real();

            if (tid == 0) params_remove[kk] = saved;
            CUDA_SYNC_THREADS;
            if (tid == 0)
                grad_remove_out[bin_i * nparams + kk] = (S_p - S_m) / (2.0 * eps_k);
            CUDA_SYNC_THREADS;
        }
    }
}

template <class TileT>
inline void stft_swap_ll_grad_launch(
    double* grad_add_out, double* grad_remove_out,
    const typename TileT::Inputs& inputs, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    double* param_eps_add, double* param_eps_remove,
    int num_bin, int nparams, int n_side_bins, int* start_freq_inds)
{
    const size_t shared_bytes = 2 * TileT::table_bytes();   // add and remove templates
#ifdef __CUDACC__
    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_swap_ll_grad_kernel<TileT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        grad_add_out, grad_remove_out, inputs, stft,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        param_eps_add, param_eps_remove,
        num_bin, nparams, n_side_bins, start_freq_inds, nullptr);
    gpuErrchk(cudaGetLastError());
#else
    std::vector<unsigned char> host_arena(shared_bytes);
    stft_swap_ll_grad_kernel<TileT>(
        grad_add_out, grad_remove_out, inputs, stft,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        param_eps_add, param_eps_remove,
        num_bin, nparams, n_side_bins, start_freq_inds, host_arena.data());
#endif
}

template <class SourceT, class ColumnT = FresnelColumn<SourceT>>
inline void stft_swap_ll_grad_impl(
    double* grad_add_out, double* grad_remove_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    double* param_eps_add, double* param_eps_remove,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    int* start_freq_inds = nullptr)
{
    STFTDomain* stft_launch = nullptr;
    const auto inputs = stft_column_tile_inputs<SourceT, ColumnT>(
        orbits, tdi_config, fresnel, stft, T, t_ref, window_factor, freq_from_tdi_phase,
        &stft_launch);
    stft_swap_ll_grad_launch<ColumnTile<ColumnT>>(
        grad_add_out, grad_remove_out, inputs, stft_launch,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        param_eps_add, param_eps_remove,
        num_bin, nparams, n_side_bins, start_freq_inds);
}

// ===========================================================================
// information_matrix : per-source Fisher matrix
//   Gamma_ij = 4 df Re sum_{t,f,c,d} conj(dh_i,c) invC_cd dh_j,d
// where dh_i is the four-point central difference (two-point with easy_central_difference) of the
// fill_global template over parameter inds[i]. Each perturbed template keeps its own
// +-n_side_bins support; the pixel sum runs over the union of the supports.
// Templates on one (f0, fdot0) track share one Fresnel kernel per pixel (ColumnT::same_kernel)
// and differ only in their prefactor. On the astro track that holds for every parameter except
// the frequency ones, so a pixel needs 1 + 4 + 4 kernel sums instead of 32 templates x 3 channels.
// eps <= 0 freezes a parameter (zero row and column). The output is exactly symmetric.
// Layout: info_out[(bin * num_derivs + i) * num_derivs + j].
// ===========================================================================
#ifndef STFT_FISHER_NDERIV_MAX
#define STFT_FISHER_NDERIV_MAX 9 // currently only verified for UCBs, current max cap for memory allocation
#endif
constexpr int STFT_FISHER_NSTENCIL_MAX  = 4;
constexpr int STFT_FISHER_NTEMPLATE_MAX = STFT_FISHER_NDERIV_MAX * STFT_FISHER_NSTENCIL_MAX;
constexpr int STFT_FISHER_NSLOT_MAX     = STFT_FISHER_NTEMPLATE_MAX * 3;  // one kernel per template and channel

#ifdef __CUDACC__
// Real, vector-valued twin of stft_block_reduce_cmplx: sums sdata[tid * width + w] over threads for
// every w, leaving the sums in sdata[0 .. width - 1]. Requires blockDim.x a power of two.
static CUDA_DEVICE void stft_block_reduce_vec(double* sdata, int width)
{
    int tid = threadIdx.x;
    CUDA_SYNC_THREADS;
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            for (int w = 0; w < width; w += 1)
                sdata[tid * width + w] = sdata[tid * width + w] + sdata[(tid + s) * width + w];
        }
        CUDA_SYNC_THREADS;
    }
}
#endif

// Per-source Fisher evaluation (the parameters are already loaded into the shared `params`).
// `inds` and `eps` are per-derivative copies held by the caller. Writes the scaled matrix into
// info_row_out[i * num_derivs + j] from thread 0 only.
template <class SourceT, class ColumnT = FresnelColumn<SourceT>>
CUDA_DEVICE void stft_eval_block_information(
    SourceT& src, STFTFresnel* fresnel, STFTDomain* stft,
    double* params, int nparams,
    int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
    int noise_index, int start_j,
    int* inds, double* eps, int num_derivs,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    bool easy_central_difference,
    double* reduce_tmp, int tid, double* info_row_out)
{
    const int num_stencil = easy_central_difference ? 2 : 4;

    // These arrays are sized at compile time but still too large for registers; if the kernel is
    // slow, their placement in local memory is the first thing to measure.
    typename ColumnT::State cols[STFT_FISHER_NTEMPLATE_MAX];
    cmplx pref[STFT_FISHER_NTEMPLATE_MAX][3];
    int slot_of[STFT_FISHER_NTEMPLATE_MAX][3];
    int slot_template[STFT_FISHER_NSLOT_MAX];
    int slot_channel[STFT_FISHER_NSLOT_MAX];
    int slot_lo[STFT_FISHER_NSLOT_MAX];
    int slot_hi[STFT_FISHER_NSLOT_MAX];
    cmplx slot_val[STFT_FISHER_NSLOT_MAX];
    cmplx dh[STFT_FISHER_NDERIV_MAX][3];
    cmplx invc_dh[STFT_FISHER_NDERIV_MAX][3];   // (invC dh_j)_c
    cmplx invc[3][3];
    double acc[STFT_FISHER_NDERIV_MAX][STFT_FISHER_NDERIV_MAX];
    double params_local[N_PARAMS_MAX];
    bool active[STFT_FISHER_NDERIV_MAX];

    double t0 = stft->t0;
    double dt = stft->dt;
    double df = stft->df;
    double f_min = stft->f_min;
    int num_times = stft->num_times;
    int num_freqs = stft->num_freqs;
    bool cross_channel = (stft->tdi_type == TDI_XYZ);
    int num_contract_channels = cross_channel ? 3 : stft->num_channels;
    double t_anchor_shift = fresnel->use_midpoint ? 0.5 * dt : 0.0;

    for (int i = 0; i < num_derivs; i += 1)
    {
        active[i] = (eps[i] > 0.0);
        for (int j = 0; j < num_derivs; j += 1)
            acc[i][j] = 0.0;
    }

    for (int time_i = THREAD_START_X; time_i < num_times; time_i += BLOCK_INCR_X)
    {
        double t_seg = t0 + time_i * dt;

        // One column per (derivative, stencil point). Stencil point order: +eps, -eps, +2eps, -2eps.
        // The shifts repeat the reference's p[ind] += eps, -= eps, += 2*eps, -= 2*eps exactly.
        int carrier_lo = 0, carrier_hi = -1;
        bool any_active = false;
        // * Lowest and highest carrier among one derivative's stencil templates; their union of
        // * supports is what all four of them are evaluated on (see the pixel loop).
        int deriv_lo[STFT_FISHER_NDERIV_MAX];
        int deriv_hi[STFT_FISHER_NDERIV_MAX];
        for (int d = 0; d < num_derivs; d += 1)
        {
            if (!active[d])
                continue;
            double shift[STFT_FISHER_NSTENCIL_MAX] = {eps[d], -eps[d], 2.0 * eps[d], -(2.0 * eps[d])};
            for (int p = 0; p < num_stencil; p += 1)
            {
                int tpl = d * STFT_FISHER_NSTENCIL_MAX + p;
                for (int k = 0; k < nparams; k += 1)
                    params_local[k] = params[k];
                params_local[inds[d]] = params[inds[d]] + shift[p];

                Vec k_vec(0.0, 0.0, 0.0), u_vec(0.0, 0.0, 0.0), v_vec(0.0, 0.0, 0.0);
                src.get_sky_vectors(&k_vec, &u_vec, &v_vec, params_local);
                ColumnT::setup(cols[tpl], src, fresnel, stft, params_local, k_vec, u_vec, v_vec,
                               link_space_craft_rec, link_space_craft_em, bin_i,
                               t_seg, t_anchor_shift, window_factor, freq_from_tdi_phase);
                for (int c = 0; c < 3; c += 1)
                    pref[tpl][c] = ColumnT::prefactor(cols[tpl], c);

                int carrier = cols[tpl].carrier_j;
                if (p == 0 || carrier < deriv_lo[d]) deriv_lo[d] = carrier;
                if (p == 0 || carrier > deriv_hi[d]) deriv_hi[d] = carrier;
                if (!any_active || carrier < carrier_lo) carrier_lo = carrier;
                if (!any_active || carrier > carrier_hi) carrier_hi = carrier;
                any_active = true;
            }
        }
        if (!any_active)
            continue;

        // Group (template, channel) pairs into kernel slots; a slot is evaluated only on the union
        // of its members' supports.
        int num_slots = 0;
        for (int d = 0; d < num_derivs; d += 1)
        {
            if (!active[d])
                continue;
            for (int p = 0; p < num_stencil; p += 1)
            {
                int tpl = d * STFT_FISHER_NSTENCIL_MAX + p;
                // ! The bounds are the derivative's COMMON support, not this template's own carrier.
                int support_lo = deriv_lo[d] - n_side_bins;
                int support_hi = deriv_hi[d] + n_side_bins;
                for (int c = 0; c < 3; c += 1)
                {
                    int found = -1;
                    for (int s = 0; s < num_slots; s += 1)
                    {
                        if (ColumnT::same_kernel(cols[slot_template[s]], slot_channel[s], cols[tpl], c))
                        {
                            found = s;
                            break;
                        }
                    }
                    if (found < 0)
                    {
                        found = num_slots;
                        slot_template[found] = tpl;
                        slot_channel[found] = c;
                        slot_lo[found] = support_lo;
                        slot_hi[found] = support_hi;
                        num_slots += 1;
                    }
                    else
                    {
                        if (support_lo < slot_lo[found]) slot_lo[found] = support_lo;
                        if (support_hi > slot_hi[found]) slot_hi[found] = support_hi;
                    }
                    slot_of[tpl][c] = found;
                }
            }
        }

        int freq_j_first = carrier_lo - n_side_bins;
        int freq_j_last  = carrier_hi + n_side_bins;
        if (freq_j_first < start_j) freq_j_first = start_j;
        if (freq_j_last > start_j + num_freqs - 1) freq_j_last = start_j + num_freqs - 1;

        for (int freq_j_here = freq_j_first; freq_j_here <= freq_j_last; freq_j_here += 1)
        {
            int freq_j_local = freq_j_here - start_j;
            double freq_here = f_min + freq_j_here * df;

            for (int s = 0; s < num_slots; s += 1)
            {
                if ((freq_j_here >= slot_lo[s]) && (freq_j_here <= slot_hi[s]))
                    slot_val[s] = ColumnT::kernel(cols[slot_template[s]], slot_channel[s],
                                                  freq_j_here, freq_here);
            }

            // Template values as fill_global writes them, combined in the reference's operation order.
            //
            // COMMON SUPPORT. The four templates of one derivative are all evaluated on the union of
            // their carriers' stencils, [min carrier - n_side_bins, max carrier + n_side_bins], not
            // each on its own carrier's. The union is the SAME set for the four, so the cut cancels in
            // the difference. It equals each template's own support whenever the four carriers agree,
            // which is every pixel of every source except a carrier crossing.
            //
            // With each template on its own support instead, a step that moves one carrier across a
            // half bin shifts that template's cut by one bin. The difference then keeps one whole edge
            // pixel that the other three do not have, divided by eps: measured x1e3 to 2e4 in
            // Gamma_f0f0 at n_side_bins = 10, for the ~1.2e-4 of sources where a stencil carrier
            // crosses. The union adds at most the one extra edge bin to all four, which is leakage of
            // order 1e-3 of a segment and cancels to the same order as the templates themselves.
            for (int d = 0; d < num_derivs; d += 1)
            {
                if (!active[d])
                    continue;
                for (int c = 0; c < 3; c += 1)
                {
                    cmplx vals[STFT_FISHER_NSTENCIL_MAX];
                    for (int p = 0; p < num_stencil; p += 1)
                    {
                        int tpl = d * STFT_FISHER_NSTENCIL_MAX + p;
                        if ((freq_j_here >= deriv_lo[d] - n_side_bins)
                            && (freq_j_here <= deriv_hi[d] + n_side_bins))
                            vals[p] = 0.5 * (pref[tpl][c] * slot_val[slot_of[tpl][c]]);
                        else
                            vals[p] = cmplx(0.0, 0.0);
                    }
                    if (easy_central_difference)
                        dh[d][c] = (vals[0] - vals[1]) / (2.0 * eps[d]);
                    else
                        dh[d][c] = (-vals[2] + vals[3] + 8.0 * (vals[0] - vals[1])) / (12.0 * eps[d]);
                }
            }

            if (cross_channel)
            {
                for (int c = 0; c < 3; c += 1)
                    for (int e = 0; e < 3; e += 1)
                        invc[c][e] = stft->get_invC_cross_value(time_i, freq_j_local, c, e, noise_index);
                for (int d = 0; d < num_derivs; d += 1)
                {
                    if (!active[d])
                        continue;
                    for (int c = 0; c < 3; c += 1)
                        invc_dh[d][c] = invc[c][0] * dh[d][0] + invc[c][1] * dh[d][1] + invc[c][2] * dh[d][2];
                }
            }
            else
            {
                for (int c = 0; c < num_contract_channels; c += 1)
                    invc[c][c] = stft->get_invC_value(time_i, freq_j_local, c, noise_index);
                for (int d = 0; d < num_derivs; d += 1)
                {
                    if (!active[d])
                        continue;
                    for (int c = 0; c < num_contract_channels; c += 1)
                        invc_dh[d][c] = invc[c][c] * dh[d][c];
                }
            }

            // Re(conj(x) y) = x.re y.re + x.im y.im, accumulated on the upper triangle only.
            for (int i = 0; i < num_derivs; i += 1)
            {
                if (!active[i])
                    continue;
                for (int j = i; j < num_derivs; j += 1)
                {
                    if (!active[j])
                        continue;
                    double sum = 0.0;
                    for (int c = 0; c < num_contract_channels; c += 1)
                        sum += dh[i][c].real() * invc_dh[j][c].real() + dh[i][c].imag() * invc_dh[j][c].imag();
                    acc[i][j] += sum;
                }
            }
        }
    }
    CUDA_SYNC_THREADS;

    double scale = 4.0 * stft->diff_comp;
    for (int i = 0; i < num_derivs; i += 1)
    {
        int width = num_derivs - i;
#ifdef __CUDACC__
        // * Reduce one row of the upper triangle at a time through the shared staging array.
        for (int w = 0; w < width; w += 1)
            reduce_tmp[tid * width + w] = acc[i][i + w];
        stft_block_reduce_vec(reduce_tmp, width);
        if (tid == 0)
        {
            for (int w = 0; w < width; w += 1)
            {
                double val = scale * reduce_tmp[w];
                info_row_out[i * num_derivs + i + w] = val;
                info_row_out[(i + w) * num_derivs + i] = val;
            }
        }
        CUDA_SYNC_THREADS;
#else
        (void) reduce_tmp;
        (void) tid;
        for (int w = 0; w < width; w += 1)
        {
            double val = scale * acc[i][i + w];
            info_row_out[i * num_derivs + i + w] = val;
            info_row_out[(i + w) * num_derivs + i] = val;
        }
#endif
    }
}

template <class SourceT, class ColumnT = FresnelColumn<SourceT>>
CUDA_KERNEL
void stft_information_matrix_kernel(
    double* info_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* noise_index_all,
    int* inds, double* param_eps,
    int num_bin, int nparams, int num_derivs, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    bool easy_central_difference,
    int* start_freq_inds)
{
    CUDA_SHARED double reduce_tmp[NUM_THREADS_HERE * STFT_FISHER_NDERIV_MAX];
    CUDA_SHARED double params[N_PARAMS_MAX];

    SourceT src(orbits, tdi_config, T, t_ref);

    CUDA_SHARED int link_space_craft_rec[NLINKS];
    CUDA_SHARED int link_space_craft_em[NLINKS];
    src.fill_link_arrays(link_space_craft_rec, link_space_craft_em);
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif

    // * The derivative index and step tables are read once per launch, not per pixel.
    int inds_local[STFT_FISHER_NDERIV_MAX];
    double eps_local[STFT_FISHER_NDERIV_MAX];
    for (int d = 0; d < num_derivs; d += 1)
    {
        inds_local[d] = inds[d];
        eps_local[d] = param_eps[inds[d]];
    }

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int noise_index = noise_index_all[bin_i];
        // The window start belongs to the invC row this source is contracted against.
        int start_j = (start_freq_inds == nullptr) ? 0 : start_freq_inds[noise_index];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;

        stft_eval_block_information<SourceT, ColumnT>(
            src, fresnel, stft, params, nparams,
            link_space_craft_rec, link_space_craft_em, bin_i,
            noise_index, start_j,
            inds_local, eps_local, num_derivs,
            n_side_bins, window_factor, freq_from_tdi_phase, easy_central_difference,
            reduce_tmp, tid, &info_out[(size_t) bin_i * num_derivs * num_derivs]);
        CUDA_SYNC_THREADS;
    }
}

template <class SourceT, class ColumnT = FresnelColumn<SourceT>>
inline void stft_information_matrix_impl(
    double* info_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* noise_index_all,
    int* inds, double* param_eps,
    int num_bin, int nparams, int num_derivs, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    bool easy_central_difference,
    int* start_freq_inds = nullptr)
{
    if (num_derivs < 1 || num_derivs > STFT_FISHER_NDERIV_MAX)
        throw std::invalid_argument("stft_information_matrix: num_derivs must lie in [1, STFT_FISHER_NDERIV_MAX].");
    if (nparams > N_PARAMS_MAX)
        throw std::invalid_argument("stft_information_matrix: nparams exceeds N_PARAMS_MAX.");
    // ! inds is device memory on the GPU path, so its range is checked by the Python caller.
    if (num_bin == 0)
        return;

#ifdef __CUDACC__
    const STFTDeviceStructs& dev = stft_device_structs(orbits, tdi_config, fresnel, stft);

    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_information_matrix_kernel<SourceT, ColumnT><<<grid, NUM_THREADS_HERE>>>(
        info_out, dev.orbits, dev.tdi_config, dev.fresnel, dev.stft,
        params_all, noise_index_all, inds, param_eps,
        num_bin, nparams, num_derivs, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase, easy_central_difference, start_freq_inds);
    gpuErrchk(cudaGetLastError());
#else
    stft_information_matrix_kernel<SourceT, ColumnT>(
        info_out, orbits, tdi_config, fresnel, stft,
        params_all, noise_index_all, inds, param_eps,
        num_bin, nparams, num_derivs, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase, easy_central_difference, start_freq_inds);
#endif
}

#endif // __LAT_STFT_KERNELS_HH__
