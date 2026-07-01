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
#include "lat_tdi_on_the_fly.hh"  // LISATDIonTheFly base (SourceT) + Vec / Orbits / TDIConfig + OrbitsSplineCache
#include "lat_chunked_het_kernels.hh"  // populate_orbit_spline_cache (+ wdm_fit_cubic_spline) for FFTColumn's cached orbit path

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

// Max response sub-samples per STFT segment for the FFTColumn on-stack buffer
// (design 2026-07-01). Bounds register/local pressure; n_sub is clamped to it.
#ifndef STFT_FFT_NSUB_MAX
#define STFT_FFT_NSUB_MAX 64
#endif

// Max orbit spline-cache control points for FFTColumn's cached get_tdi path
// (mirrors FAST_WDM_N_CP_ORBIT_MAX). At 48 the shared buffers add ~23 KB.
#ifndef STFT_ORBIT_NCP_MAX
#define STFT_ORBIT_NCP_MAX 48
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
// Doppler-corrected instantaneous frequency / frequency-rate of the TDI signal
// at time `t`, derived from the TDI phase by a central finite difference around
// `t` of half-width `delta`, in the kernel's chirp convention
// (phase = arg(conj(TDI))). The astrophysical chirp is removed before taking the
// residual argument so there is no phase-wrapping ambiguity even when f*delta >> 1;
// the result reduces exactly to (f_astro, fdot_astro) when there is no Doppler.
//
// `tdi_center[ch_ref]` is the (already-computed) TDI value at `t`. Uses a single
// representative channel `ch_ref` (orbital Doppler is common across X/Y/Z to
// leading order -- matches the source kernels' single-(f0,fdot0)-per-pixel design).
//
// NOTE: this samples the TDI (hence the orbit) at t +- delta. At the first/last
// STFT bins that is +- one bin outside the observation; for accuracy there the
// ORBIT must cover a little beyond the observation window (it normally does --
// orbit files span more than the analysis segment). A one-sided edge stencil was
// tried and is NOT needed: the residual at the first/last bins is the observation
// sitting at the orbit's boundary, not the finite-difference reach.
// ---------------------------------------------------------------------------
template <class SourceT>
CUDA_DEVICE void stft_freq_fdot_from_tdi_phase(
    SourceT& src, double t, double delta,
    double* params, Vec k, Vec u, Vec v,
    int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
    const cmplx* tdi_center, int ch_ref,
    double f_astro, double fdot_astro,
    double* f0_out, double* fdot0_out)
{
    cmplx tdi_p[3];
    cmplx tdi_m[3];
    src.get_tdi_Xf_single(&tdi_p[0], t + delta, params, k, u, v,
                          link_space_craft_rec, link_space_craft_em, bin_i);
    src.get_tdi_Xf_single(&tdi_m[0], t - delta, params, k, u, v,
                          link_space_craft_rec, link_space_craft_em, bin_i);

    // Astrophysical chirp phase advance over +-delta (quadratic model).
    double astro_p = 2.0 * M_PI * f_astro * delta + M_PI * fdot_astro * delta * delta;
    double astro_m = -2.0 * M_PI * f_astro * delta + M_PI * fdot_astro * delta * delta;

    // phi_tilde(t+-delta) - phi_tilde(t) with phi_tilde = arg(conj(TDI)):
    //   = arg( conj(TDI(t+-delta)) * TDI(t) )
    // Remove the astro chirp -> small residual (the Doppler deviation), then add
    // the (unwrapped) astro advance back.
    cmplx c0 = tdi_center[ch_ref];
    double res_p = gcmplx::arg(gcmplx::conj(tdi_p[ch_ref]) * c0 * gcmplx::polar(1.0, -astro_p));
    double res_m = gcmplx::arg(gcmplx::conj(tdi_m[ch_ref]) * c0 * gcmplx::polar(1.0, -astro_m));
    double a_p = res_p + astro_p;
    double a_m = res_m + astro_m;

    // Central differences: a_p = 2pi f0 delta + pi fdot0 delta^2,
    //                      a_m = -2pi f0 delta + pi fdot0 delta^2.
    *f0_out = (a_p - a_m) / (4.0 * M_PI * delta);
    *fdot0_out = (a_p + a_m) / (2.0 * M_PI * delta * delta);
}

// Resolve (f0, fdot0) for one pixel: TDI-phase derivation or astrophysical fallback.
template <class SourceT>
CUDA_DEVICE void stft_pixel_freq_fdot(
    SourceT& src, double t,
    double* params, Vec k, Vec u, Vec v,
    int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
    const cmplx* tdi_center, int ch_ref, double dt,
    bool freq_from_tdi_phase,
    double* f0_out, double* fdot0_out)
{
    double f_astro = src.get_f(t, params, bin_i);
    double fdot_astro = src.get_fdot(t, params, bin_i);
    if (!freq_from_tdi_phase)
    {
        *f0_out = f_astro;
        *fdot0_out = fdot_astro;
        return;
    }
    // Half-width of the central difference: the STFT bin width is far shorter
    // than the (year-scale) Doppler timescale, so it is an accurate, robust
    // sample spacing for the local TDI frequency / rate.
    stft_freq_fdot_from_tdi_phase<SourceT>(
        src, t, dt, params, k, u, v,
        link_space_craft_rec, link_space_craft_em, bin_i,
        tdi_center, ch_ref, f_astro, fdot_astro, f0_out, fdot0_out);
}

// Integer power of a unit-modulus complex number via a short recurrence (|z|==1
// so z^{-k} = conj(z)^k). Used by FFTColumn's twiddle-recurrence DFT; |k| <= n_side.
CUDA_DEVICE inline cmplx stft_unit_cpow(cmplx z, int k)
{
    cmplx r(1.0, 0.0);
    cmplx b = (k >= 0) ? z : gcmplx::conj(z);
    int n = (k >= 0) ? k : -k;
    for (int i = 0; i < n; ++i) r = r * b;
    return r;
}

// ===========================================================================
// Compile-time column-producer policies (design §4). The shared driver
// stft_eval_block_ll<SourceT, ColumnProducer> delegates the per-column inner
// computation to ColumnProducer::produce. FresnelColumn reproduces the merged
// analytic per-pixel path byte-for-byte; FFTColumn (Task 2) is the targeted-DFT
// generator. Zero runtime cost — the policy is resolved at compile time.
// ===========================================================================
struct FresnelColumn {
    template <class SourceT>
    static CUDA_DEVICE void produce(
        SourceT& src, STFTFresnel* fresnel, STFTDomain* stft,
        double* params, Vec k, Vec u, Vec v,
        int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
        int time_i, double t_here, int data_index, int noise_index,
        int n_side_bins, int n_sub, OrbitsSplineCache* orbit_cache,
        double window_factor, bool freq_from_tdi_phase,
        cmplx* d_h_tmp, cmplx* h_h_tmp, int tid)
    {
        (void) n_sub; (void) orbit_cache;  // Fresnel path ignores the FFT-only knobs.
        cmplx tdi_channel_val[3];
        double tdi_channel_amp[3];
        double tdi_channel_phase[3];
        cmplx fresnel_val[3];
        double f0, fdot0;

        double dt = stft->dt;
        double df = stft->df;
        double f_min = stft->f_min;
        int num_freqs = stft->num_freqs;

        src.get_tdi_Xf_single(&tdi_channel_val[0], t_here, params, k, u, v,
                              link_space_craft_rec, link_space_craft_em, bin_i);

        stft_pixel_freq_fdot<SourceT>(
            src, t_here, params, k, u, v,
            link_space_craft_rec, link_space_craft_em, bin_i,
            tdi_channel_val, /*ch_ref=*/0, dt, freq_from_tdi_phase, &f0, &fdot0);

        int freq_j = stft->get_freq_index(f0);
        for (int diff = -n_side_bins; diff <= +n_side_bins; diff += 1)
        {
            int freq_j_here = freq_j + diff;
            if ((freq_j_here >= 0) && (freq_j_here <= num_freqs - 1))
            {
                double freq_here = f_min + freq_j_here * df;
                for (int j = 0; j < 3; j += 1)
                {
                    fresnel->get_amp_phase(&tdi_channel_amp[j], &tdi_channel_phase[j],
                                           gcmplx::conj(tdi_channel_val[j]));
                    fresnel_val[j] = 0.5 * fresnel->get_fourier_value(
                        tdi_channel_amp[j], tdi_channel_phase[j], f0, fdot0,
                        t_here, freq_here, window_factor);
                }
                stft->add_ip_contrib(d_h_tmp, h_h_tmp, fresnel_val,
                                     time_i, freq_j_here, data_index, noise_index);
            }
        }
    }
};

struct FFTColumn {
    template <class SourceT>
    static CUDA_DEVICE void produce(
        SourceT& src, STFTFresnel* fresnel, STFTDomain* stft,
        double* params, Vec k, Vec u, Vec v,
        int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
        int time_i, double t_here, int data_index, int noise_index,
        int n_side_bins, int n_sub, OrbitsSplineCache* orbit_cache,
        double window_factor, bool freq_from_tdi_phase,
        cmplx* d_h_tmp, cmplx* h_h_tmp, int tid)
    {
        // Window = free time-domain multiply (design 2026-07-01). Match the analysis
        // window the data STFT used: Tukey when window_alpha>0 (taper_duration =
        // alpha*dt/2, mirroring get_windowed_fourier_value / scipy.signal.tukey),
        // else a flat window scaled by window_factor (the rectangular Fresnel path).
        double window_alpha = fresnel->window_alpha;
        double taper = fresnel->taper_duration;
        double dt = stft->dt;
        double df = stft->df;
        double f_min = stft->f_min;
        int num_freqs = stft->num_freqs;

        int N = n_sub;
        if (N < 1) N = 1;
        if (N > STFT_FFT_NSUB_MAX) N = STFT_FFT_NSUB_MAX;
        double dts_sub = dt / (double) N;

        // Carrier bin from the (Doppler-corrected) TDI-phase frequency at t_here.
        cmplx tdi_center[3];
        double f0, fdot0;
        src.get_tdi_Xf_single(&tdi_center[0], t_here, params, k, u, v,
                              link_space_craft_rec, link_space_craft_em, bin_i);
        stft_pixel_freq_fdot<SourceT>(
            src, t_here, params, k, u, v,
            link_space_craft_rec, link_space_craft_em, bin_i,
            tdi_center, /*ch_ref=*/0, dt, freq_from_tdi_phase, &f0, &fdot0);
        int freq_j = stft->get_freq_index(f0);

        // Sample conj(response) on the N_sub sub-grid once (reused across bins).
        // Midpoint quadrature (tau_m = t_here + (m+0.5)*dts_sub): 2nd-order accurate
        // Riemann sum of the per-segment integral, vs 1st-order for left endpoints.
        cmplx slow[3 * STFT_FFT_NSUB_MAX];
        for (int m = 0; m < N; ++m)
        {
            double tau = t_here + ((double) m + 0.5) * dts_sub;
            // Analysis window as a plain per-sample multiply (same segment position
            // as the data STFT). Tukey taper over [t_here, t_here+taper] and
            // [t_here+dt-taper, t_here+dt]; flat elsewhere. w_m = window_factor
            // when rectangular (window_alpha == 0), matching the Fresnel path.
            double w_m = window_factor;
            if (window_alpha > 0.0)
            {
                w_m = 1.0;
                if (tau < t_here + taper)
                { double sn = (tau - t_here) / taper; w_m = 0.5 * (1.0 - cos(M_PI * sn)); }
                else if (tau > t_here + dt - taper)
                { double sn = (t_here + dt - tau) / taper; w_m = 0.5 * (1.0 - cos(M_PI * sn)); }
            }
            cmplx tv[3];
            if (orbit_cache != nullptr)
                src.get_tdi_Xf_single_cached(&tv[0], tau, params, k, u, v,
                                             link_space_craft_rec, link_space_craft_em,
                                             bin_i, orbit_cache);
            else
                src.get_tdi_Xf_single(&tv[0], tau, params, k, u, v,
                                      link_space_craft_rec, link_space_craft_em, bin_i);
            for (int c = 0; c < 3; ++c)
            {
                cmplx s = w_m * gcmplx::conj(tv[c]);    // Fresnel convention + window
                if (!isfinite(s.real()) || !isfinite(s.imag()))
                    s = cmplx(0.0, 0.0);                 // NaN scrub (mirror gbfd)
                slow[c * STFT_FFT_NSUB_MAX + m] = s;
            }
        }

        // Heterodyne to the center carrier bin (freq_j) IN PLACE: 1 sincos per
        // sub-sample. slow[] now holds the demodulated (near-DC) column.
        double f_center = f_min + freq_j * df;
        for (int m = 0; m < N; ++m)
        {
            double tau = t_here + ((double) m + 0.5) * dts_sub;
            cmplx het = gcmplx::polar(1.0, -2.0 * M_PI * f_center * tau);
            for (int c = 0; c < 3; ++c)
                slow[c * STFT_FFT_NSUB_MAX + m] = slow[c * STFT_FFT_NSUB_MAX + m] * het;
        }

        // Targeted DFT of the demodulated column via a twiddle recurrence -- only
        // the (2*n_side_bins+1) bins near the carrier, TRANSCENDENTAL-FREE inside:
        //   bin(freq_j+diff) = 0.5*dts_sub * base^diff * sum_m demod[m] * (W^diff)^m,
        //   W = exp(-2pi i/N),  base = exp(-2pi i (df*t_here + 0.5/N)).
        // Identical to the direct DFT (df*tau_m = df*t_here + (m+0.5)/N since
        // df*stft_dt = 1). sincos/column: (2*n_side+1)*N  ->  N + 2.
        cmplx W = gcmplx::polar(1.0, -2.0 * M_PI / (double) N);
        cmplx base = gcmplx::polar(1.0, -2.0 * M_PI * (df * t_here + 0.5 / (double) N));
        cmplx col_val[3];
        for (int diff = -n_side_bins; diff <= +n_side_bins; diff += 1)
        {
            int freq_j_here = freq_j + diff;
            if ((freq_j_here < 0) || (freq_j_here > num_freqs - 1)) continue;
            cmplx Wd = stft_unit_cpow(W, diff);
            cmplx scale = (0.5 * dts_sub) * stft_unit_cpow(base, diff);
            for (int c = 0; c < 3; ++c) col_val[c] = cmplx(0.0, 0.0);
            cmplx tw(1.0, 0.0);
            for (int m = 0; m < N; ++m)
            {
                for (int c = 0; c < 3; ++c)
                    col_val[c] = col_val[c] + slow[c * STFT_FFT_NSUB_MAX + m] * tw;
                tw = tw * Wd;
            }
            for (int c = 0; c < 3; ++c)
                col_val[c] = scale * col_val[c];
            stft->add_ip_contrib(d_h_tmp, h_h_tmp, col_val,
                                 time_i, freq_j_here, data_index, noise_index);
        }
    }
};

// ===========================================================================
// Per-binary (d|h),(h|h) evaluation for one parameter vector (already loaded
// into the shared `params`). Zeroes the supplied scratch, runs the time x
// side-freq x channel Fresnel loop, reduces, and writes the 4*diff_comp-scaled
// (d|h),(h|h) into *d_h_val,*h_h_val (broadcast to all threads on GPU; on CPU
// tid==0 holds the sum). Recomputes the sky vectors from `params` each call so
// it is reusable after a parameter perturbation (the get_ll gradient path).
// Shared by stft_get_ll_kernel and stft_get_ll_grad_kernel so the gradient's
// forward evaluation is byte-identical to get_ll.
// ===========================================================================
template <class SourceT, class ColumnProducer = FresnelColumn>
CUDA_DEVICE void stft_eval_block_ll(
    SourceT& src, STFTFresnel* fresnel, STFTDomain* stft,
    double* params,
    int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
    int data_index, int noise_index,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    cmplx* d_h_tmp, cmplx* h_h_tmp, int tid,
    cmplx* d_h_val, cmplx* h_h_val, int n_sub = 0,
    OrbitsSplineCache* orbit_cache = nullptr)
{
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);

    double t0 = stft->t0;
    double dt = stft->dt;
    int num_times = stft->num_times;

    d_h_tmp[tid] = cmplx(0.0, 0.0);
    h_h_tmp[tid] = cmplx(0.0, 0.0);
    CUDA_SYNC_THREADS;

    src.get_sky_vectors(&k, &u, &v, params);
    for (int time_i = THREAD_START_X; time_i < num_times; time_i += BLOCK_INCR_X)
    {
        double t_here = t0 + time_i * dt;
        ColumnProducer::template produce<SourceT>(
            src, fresnel, stft, params, k, u, v,
            link_space_craft_rec, link_space_craft_em, bin_i,
            time_i, t_here, data_index, noise_index,
            n_side_bins, n_sub, orbit_cache, window_factor, freq_from_tdi_phase,
            d_h_tmp, h_h_tmp, tid);
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
template <class SourceT>
CUDA_KERNEL
void stft_get_ll_kernel(
    cmplx* d_h_out, cmplx* h_h_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
    CUDA_SHARED cmplx d_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx h_h_tmp[NUM_THREADS_HERE];
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

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;

        cmplx d_h_val, h_h_val;
        stft_eval_block_ll<SourceT>(
            src, fresnel, stft, params,
            link_space_craft_rec, link_space_craft_em, bin_i,
            data_index, noise_index,
            n_side_bins, window_factor, freq_from_tdi_phase,
            d_h_tmp, h_h_tmp, tid, &d_h_val, &h_h_val);

        if (tid == 0)
        {
            d_h_out[bin_i] = d_h_val;
            h_h_out[bin_i] = h_h_val;
        }
        CUDA_SYNC_THREADS;
    }
}

template <class SourceT>
inline void stft_get_ll_impl(
    cmplx* d_h_out, cmplx* h_h_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
#ifdef __CUDACC__
    static Orbits*      orbits_gpu     = nullptr;
    static TDIConfig*   tdi_config_gpu = nullptr;
    static STFTFresnel* fresnel_gpu    = nullptr;
    static STFTDomain*  stft_gpu       = nullptr;
    if (orbits_gpu     == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,     sizeof(Orbits)));
    if (tdi_config_gpu == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu, sizeof(TDIConfig)));
    if (fresnel_gpu    == nullptr) gpuErrchk(cudaMalloc(&fresnel_gpu,    sizeof(STFTFresnel)));
    if (stft_gpu       == nullptr) gpuErrchk(cudaMalloc(&stft_gpu,       sizeof(STFTDomain)));
    gpuErrchk(cudaMemcpy(orbits_gpu,     orbits,     sizeof(Orbits),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu, tdi_config, sizeof(TDIConfig),  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fresnel_gpu,    fresnel,    sizeof(STFTFresnel), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(stft_gpu,       stft,       sizeof(STFTDomain), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_get_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        d_h_out, h_h_out, orbits_gpu, tdi_config_gpu, fresnel_gpu, stft_gpu,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    stft_get_ll_kernel<SourceT>(
        d_h_out, h_h_out, orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
#endif
}

// ===========================================================================
// get_ll (FFT-per-column variant): same (d|h),(h|h) surface as stft_get_ll,
// but the per-column template is a targeted DFT of the sub-sampled windowed
// response (FFTColumn policy) instead of the analytic Fresnel per-pixel value.
// ===========================================================================
template <class SourceT>
CUDA_KERNEL
void stft_get_ll_fft_kernel(
    cmplx* d_h_out, cmplx* h_h_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, int n_sub, int n_cp_orbit, double window_factor, bool freq_from_tdi_phase)
{
    CUDA_SHARED cmplx d_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx h_h_tmp[NUM_THREADS_HERE];
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

    // Orbit spline cache: built ONCE over the analysis span [t0, t0+NT*dt] and
    // shared across all binaries (the orbit is source-independent). Replaces the
    // per-sub-sample orbit / light-travel-time lookups inside get_tdi with cheap
    // cached cubic evals; skipped when n_cp_orbit < 4 (FFTColumn falls back to direct).
    CUDA_SHARED double orbit_t_cp_buf  [STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_ltt_y_buf [6 * STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_ltt_c1_buf[6 * STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_ltt_c2_buf[6 * STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_ltt_c3_buf[6 * STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_pos_y_buf [9 * STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_pos_c1_buf[9 * STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_pos_c2_buf[9 * STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_pos_c3_buf[9 * STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_B_buf     [STFT_ORBIT_NCP_MAX];
    CUDA_SHARED double orbit_pcr_buf   [8 * STFT_ORBIT_NCP_MAX];
    CUDA_SHARED OrbitsSplineCache orbit_cache_storage;

    OrbitsSplineCache* orbit_cache_ptr = nullptr;
    {
        int n_cp = (n_cp_orbit > STFT_ORBIT_NCP_MAX) ? STFT_ORBIT_NCP_MAX : n_cp_orbit;
        if (n_cp >= 4)
        {
            populate_orbit_spline_cache(
                &orbit_cache_storage, orbits, stft->t0,
                (double) stft->num_times * stft->dt, n_cp,
                orbit_t_cp_buf, orbit_ltt_y_buf, orbit_ltt_c1_buf,
                orbit_ltt_c2_buf, orbit_ltt_c3_buf, orbit_pos_y_buf,
                orbit_pos_c1_buf, orbit_pos_c2_buf, orbit_pos_c3_buf,
                orbit_B_buf, orbit_pcr_buf);
            CUDA_SYNC_THREADS;
            orbit_cache_ptr = &orbit_cache_storage;
        }
    }

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;

        cmplx d_h_val, h_h_val;
        stft_eval_block_ll<SourceT, FFTColumn>(
            src, fresnel, stft, params,
            link_space_craft_rec, link_space_craft_em, bin_i,
            data_index, noise_index,
            n_side_bins, window_factor, freq_from_tdi_phase,
            d_h_tmp, h_h_tmp, tid, &d_h_val, &h_h_val, n_sub, orbit_cache_ptr);

        if (tid == 0)
        {
            d_h_out[bin_i] = d_h_val;
            h_h_out[bin_i] = h_h_val;
        }
        CUDA_SYNC_THREADS;
    }
}

template <class SourceT>
inline void stft_get_ll_fft_impl(
    cmplx* d_h_out, cmplx* h_h_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, int n_sub, int n_cp_orbit, double window_factor, bool freq_from_tdi_phase)
{
#ifdef __CUDACC__
    static Orbits*      orbits_gpu     = nullptr;
    static TDIConfig*   tdi_config_gpu = nullptr;
    static STFTFresnel* fresnel_gpu    = nullptr;
    static STFTDomain*  stft_gpu       = nullptr;
    if (orbits_gpu     == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,     sizeof(Orbits)));
    if (tdi_config_gpu == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu, sizeof(TDIConfig)));
    if (fresnel_gpu    == nullptr) gpuErrchk(cudaMalloc(&fresnel_gpu,    sizeof(STFTFresnel)));
    if (stft_gpu       == nullptr) gpuErrchk(cudaMalloc(&stft_gpu,       sizeof(STFTDomain)));
    gpuErrchk(cudaMemcpy(orbits_gpu,     orbits,     sizeof(Orbits),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu, tdi_config, sizeof(TDIConfig),  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fresnel_gpu,    fresnel,    sizeof(STFTFresnel), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(stft_gpu,       stft,       sizeof(STFTDomain), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_get_ll_fft_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        d_h_out, h_h_out, orbits_gpu, tdi_config_gpu, fresnel_gpu, stft_gpu,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, n_sub, n_cp_orbit, window_factor, freq_from_tdi_phase);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    stft_get_ll_fft_kernel<SourceT>(
        d_h_out, h_h_out, orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, n_sub, n_cp_orbit, window_factor, freq_from_tdi_phase);
#endif
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
template <class SourceT>
CUDA_KERNEL
void stft_fill_global_kernel(
    cmplx* template_fill,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, double* factors_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase, bool active_band)
{
    (void) active_band;  // STFT grid == active band; layout is (.., num_times, num_freqs)
    CUDA_SHARED double params[N_PARAMS_MAX];

    SourceT src(orbits, tdi_config, T, t_ref);

    cmplx tdi_channel_val[3];
    double tdi_channel_amp[3];
    double tdi_channel_phase[3];
    double f0, fdot0;

    CUDA_SHARED int link_space_craft_rec[NLINKS];
    CUDA_SHARED int link_space_craft_em[NLINKS];
    src.fill_link_arrays(link_space_craft_rec, link_space_craft_em);
    CUDA_SYNC_THREADS;

    double t_here;
    int data_index;
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);

    double t0 = stft->t0;
    double dt = stft->dt;
    double df = stft->df;
    double f_min = stft->f_min;
    int num_times = stft->num_times;
    int num_freqs = stft->num_freqs;
    int nchannels = stft->num_channels;
    int freq_j = 0;

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        data_index = data_index_all[bin_i];
        double factor = factors_all[bin_i];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;

        src.get_sky_vectors(&k, &u, &v, params);
        for (int time_i = THREAD_START_X; time_i < num_times; time_i += BLOCK_INCR_X)
        {
            t_here = t0 + time_i * dt;
            src.get_tdi_Xf_single(&tdi_channel_val[0], t_here, params, k, u, v,
                                  link_space_craft_rec, link_space_craft_em, bin_i);

            stft_pixel_freq_fdot<SourceT>(
                src, t_here, params, k, u, v,
                link_space_craft_rec, link_space_craft_em, bin_i,
                tdi_channel_val, /*ch_ref=*/0, dt, freq_from_tdi_phase, &f0, &fdot0);

            freq_j = stft->get_freq_index(f0);
            for (int diff = -n_side_bins; diff <= +n_side_bins; diff += 1)
            {
                int freq_j_here = freq_j + diff;
                if ((freq_j_here >= 0) && (freq_j_here <= num_freqs - 1))
                {
                    double freq_here = f_min + freq_j_here * df;
                    for (int j = 0; j < 3; j += 1)
                    {
                        fresnel->get_amp_phase(&tdi_channel_amp[j], &tdi_channel_phase[j],
                                               gcmplx::conj(tdi_channel_val[j]));
                        cmplx val = factor * 0.5 * fresnel->get_fourier_value(
                            tdi_channel_amp[j], tdi_channel_phase[j], f0, fdot0,
                            t_here, freq_here, window_factor);
                        // template_fill[(((data_index*nch + j)*num_times + time_i)*num_freqs) + freq_j_here]
                        size_t idx = ((((size_t) data_index * nchannels + j) * num_times
                                       + time_i) * num_freqs) + freq_j_here;
#ifdef __CUDACC__
                        atomicAdd(((double*) &template_fill[idx]) + 0, val.real());
                        atomicAdd(((double*) &template_fill[idx]) + 1, val.imag());
#else
                        template_fill[idx] = template_fill[idx] + val;
#endif
                    }
                }
            }
        }
        CUDA_SYNC_THREADS;
    }
}

template <class SourceT>
inline void stft_fill_global_impl(
    cmplx* template_fill,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, double* factors_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase, bool active_band)
{
#ifdef __CUDACC__
    static Orbits*      orbits_gpu     = nullptr;
    static TDIConfig*   tdi_config_gpu = nullptr;
    static STFTFresnel* fresnel_gpu    = nullptr;
    static STFTDomain*  stft_gpu       = nullptr;
    if (orbits_gpu     == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,     sizeof(Orbits)));
    if (tdi_config_gpu == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu, sizeof(TDIConfig)));
    if (fresnel_gpu    == nullptr) gpuErrchk(cudaMalloc(&fresnel_gpu,    sizeof(STFTFresnel)));
    if (stft_gpu       == nullptr) gpuErrchk(cudaMalloc(&stft_gpu,       sizeof(STFTDomain)));
    gpuErrchk(cudaMemcpy(orbits_gpu,     orbits,     sizeof(Orbits),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu, tdi_config, sizeof(TDIConfig),  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fresnel_gpu,    fresnel,    sizeof(STFTFresnel), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(stft_gpu,       stft,       sizeof(STFTDomain), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_fill_global_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        template_fill, orbits_gpu, tdi_config_gpu, fresnel_gpu, stft_gpu,
        params_all, data_index_all, factors_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase, active_band);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    stft_fill_global_kernel<SourceT>(
        template_fill, orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, factors_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase, active_band);
#endif
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
// (both already in shared memory). Same per-pixel convention as
// stft_eval_block_ll; sums over the UNION of the add/remove carriers' side
// bands and defers to STFTDomain::add_ip_swap_contrib. Writes the five
// 4*diff_comp-scaled terms (broadcast to all threads on GPU). Recomputes both
// tracks' sky vectors from their params each call so it is reusable after a
// parameter perturbation (the swap gradient path). Shared by
// stft_swap_ll_kernel and stft_swap_ll_grad_kernel.
template <class SourceT>
CUDA_DEVICE void stft_eval_block_swap(
    SourceT& src, STFTFresnel* fresnel, STFTDomain* stft,
    double* params_add, double* params_remove,
    int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
    int data_index, int noise_index,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    cmplx* d_h_add_tmp, cmplx* d_h_remove_tmp, cmplx* add_add_tmp,
    cmplx* remove_remove_tmp, cmplx* add_remove_tmp, int tid,
    cmplx* d_h_add_val, cmplx* d_h_remove_val, cmplx* add_add_val,
    cmplx* remove_remove_val, cmplx* add_remove_val)
{
    cmplx tdi_val_add[3];
    double amp_add[3];
    double phase_add[3];
    cmplx fresnel_val_add[3];
    cmplx tdi_val_remove[3];
    double amp_remove[3];
    double phase_remove[3];
    cmplx fresnel_val_remove[3];
    double f0_add, fdot0_add, f0_remove, fdot0_remove;
    Vec k_add(0.0, 0.0, 0.0), u_add(0.0, 0.0, 0.0), v_add(0.0, 0.0, 0.0);
    Vec k_remove(0.0, 0.0, 0.0), u_remove(0.0, 0.0, 0.0), v_remove(0.0, 0.0, 0.0);

    double t0 = stft->t0;
    double dt = stft->dt;
    double df = stft->df;
    double f_min = stft->f_min;
    int num_times = stft->num_times;
    int num_freqs = stft->num_freqs;

    d_h_add_tmp[tid] = cmplx(0.0, 0.0);
    d_h_remove_tmp[tid] = cmplx(0.0, 0.0);
    add_add_tmp[tid] = cmplx(0.0, 0.0);
    remove_remove_tmp[tid] = cmplx(0.0, 0.0);
    add_remove_tmp[tid] = cmplx(0.0, 0.0);
    CUDA_SYNC_THREADS;

    src.get_sky_vectors(&k_add, &u_add, &v_add, params_add);
    src.get_sky_vectors(&k_remove, &u_remove, &v_remove, params_remove);
    for (int time_i = THREAD_START_X; time_i < num_times; time_i += BLOCK_INCR_X)
    {
        double t_here = t0 + time_i * dt;
        src.get_tdi_Xf_single(&tdi_val_add[0], t_here, params_add, k_add, u_add, v_add,
                              link_space_craft_rec, link_space_craft_em, bin_i);
        src.get_tdi_Xf_single(&tdi_val_remove[0], t_here, params_remove, k_remove, u_remove, v_remove,
                              link_space_craft_rec, link_space_craft_em, bin_i);

        // (f0, fdot0) from each track's own TDI phase (Doppler-corrected
        // when freq_from_tdi_phase; else astrophysical get_f/get_fdot).
        stft_pixel_freq_fdot<SourceT>(
            src, t_here, params_add, k_add, u_add, v_add,
            link_space_craft_rec, link_space_craft_em, bin_i,
            tdi_val_add, /*ch_ref=*/0, dt, freq_from_tdi_phase, &f0_add, &fdot0_add);
        stft_pixel_freq_fdot<SourceT>(
            src, t_here, params_remove, k_remove, u_remove, v_remove,
            link_space_craft_rec, link_space_craft_em, bin_i,
            tdi_val_remove, /*ch_ref=*/0, dt, freq_from_tdi_phase, &f0_remove, &fdot0_remove);

        int freq_j_add = stft->get_freq_index(f0_add);
        int freq_j_remove = stft->get_freq_index(f0_remove);
        int freq_j_min = (freq_j_add < freq_j_remove) ? freq_j_add : freq_j_remove;
        int freq_j_max = (freq_j_add > freq_j_remove) ? freq_j_add : freq_j_remove;

        for (int freq_j_here = freq_j_min - n_side_bins;
             freq_j_here <= freq_j_max + n_side_bins; freq_j_here += 1)
        {
            if ((freq_j_here >= 0) && (freq_j_here <= num_freqs - 1))
            {
                double freq_here = f_min + freq_j_here * df;
                for (int j = 0; j < 3; j += 1)
                {
                    fresnel->get_amp_phase(&amp_add[j], &phase_add[j],
                                           gcmplx::conj(tdi_val_add[j]));
                    fresnel->get_amp_phase(&amp_remove[j], &phase_remove[j],
                                           gcmplx::conj(tdi_val_remove[j]));
                    fresnel_val_add[j] = 0.5 * fresnel->get_fourier_value(
                        amp_add[j], phase_add[j], f0_add, fdot0_add,
                        t_here, freq_here, window_factor);
                    fresnel_val_remove[j] = 0.5 * fresnel->get_fourier_value(
                        amp_remove[j], phase_remove[j], f0_remove, fdot0_remove,
                        t_here, freq_here, window_factor);
                }
                stft->add_ip_swap_contrib(
                    d_h_add_tmp, d_h_remove_tmp, add_add_tmp, remove_remove_tmp,
                    add_remove_tmp, fresnel_val_add, fresnel_val_remove,
                    time_i, freq_j_here, data_index, noise_index);
            }
        }
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

template <class SourceT>
CUDA_KERNEL
void stft_swap_ll_kernel(
    cmplx* d_h_add_out, cmplx* d_h_remove_out,
    cmplx* add_add_out, cmplx* remove_remove_out, cmplx* add_remove_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
    CUDA_SHARED cmplx d_h_add_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx d_h_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx add_add_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx remove_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx add_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED double params_add[N_PARAMS_MAX];
    CUDA_SHARED double params_remove[N_PARAMS_MAX];

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

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
        {
            params_add[i] = params_add_all[bin_i * nparams + i];
            params_remove[i] = params_remove_all[bin_i * nparams + i];
        }
        CUDA_SYNC_THREADS;

        cmplx d_h_add_val, d_h_remove_val, add_add_val, remove_remove_val, add_remove_val;
        stft_eval_block_swap<SourceT>(
            src, fresnel, stft, params_add, params_remove,
            link_space_craft_rec, link_space_craft_em, bin_i,
            data_index, noise_index, n_side_bins, window_factor, freq_from_tdi_phase,
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

template <class SourceT>
inline void stft_swap_ll_impl(
    cmplx* d_h_add_out, cmplx* d_h_remove_out,
    cmplx* add_add_out, cmplx* remove_remove_out, cmplx* add_remove_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
#ifdef __CUDACC__
    static Orbits*      orbits_gpu     = nullptr;
    static TDIConfig*   tdi_config_gpu = nullptr;
    static STFTFresnel* fresnel_gpu    = nullptr;
    static STFTDomain*  stft_gpu       = nullptr;
    if (orbits_gpu     == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,     sizeof(Orbits)));
    if (tdi_config_gpu == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu, sizeof(TDIConfig)));
    if (fresnel_gpu    == nullptr) gpuErrchk(cudaMalloc(&fresnel_gpu,    sizeof(STFTFresnel)));
    if (stft_gpu       == nullptr) gpuErrchk(cudaMalloc(&stft_gpu,       sizeof(STFTDomain)));
    gpuErrchk(cudaMemcpy(orbits_gpu,     orbits,     sizeof(Orbits),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu, tdi_config, sizeof(TDIConfig),  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fresnel_gpu,    fresnel,    sizeof(STFTFresnel), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(stft_gpu,       stft,       sizeof(STFTDomain), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_swap_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits_gpu, tdi_config_gpu, fresnel_gpu, stft_gpu,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    stft_swap_ll_kernel<SourceT>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config, fresnel, stft,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
#endif
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
template <class SourceT>
CUDA_KERNEL
void stft_get_fstat_ll_kernel(
    double* N_re_out, double* N_im_out,   // (num_bin, 4)
    double* M_re_out, double* M_im_out,   // (num_bin, 10) upper triangle
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
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

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];

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
            stft_eval_block_ll<SourceT>(
                src, fresnel, stft, params_i,
                link_space_craft_rec, link_space_craft_em, bin_i,
                data_index, noise_index,
                n_side_bins, window_factor, freq_from_tdi_phase,
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
                stft_eval_block_swap<SourceT>(
                    src, fresnel, stft, params_i, params_j,
                    link_space_craft_rec, link_space_craft_em, bin_i,
                    data_index, noise_index, n_side_bins, window_factor, freq_from_tdi_phase,
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

template <class SourceT>
inline void stft_get_fstat_ll_impl(
    double* N_re_out, double* N_im_out,
    double* M_re_out, double* M_im_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
#ifdef __CUDACC__
    static Orbits*      orbits_gpu     = nullptr;
    static TDIConfig*   tdi_config_gpu = nullptr;
    static STFTFresnel* fresnel_gpu    = nullptr;
    static STFTDomain*  stft_gpu       = nullptr;
    if (orbits_gpu     == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,     sizeof(Orbits)));
    if (tdi_config_gpu == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu, sizeof(TDIConfig)));
    if (fresnel_gpu    == nullptr) gpuErrchk(cudaMalloc(&fresnel_gpu,    sizeof(STFTFresnel)));
    if (stft_gpu       == nullptr) gpuErrchk(cudaMalloc(&stft_gpu,       sizeof(STFTDomain)));
    gpuErrchk(cudaMemcpy(orbits_gpu,     orbits,     sizeof(Orbits),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu, tdi_config, sizeof(TDIConfig),  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fresnel_gpu,    fresnel,    sizeof(STFTFresnel), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(stft_gpu,       stft,       sizeof(STFTDomain), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_get_fstat_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        N_re_out, N_im_out, M_re_out, M_im_out,
        orbits_gpu, tdi_config_gpu, fresnel_gpu, stft_gpu,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    stft_get_fstat_ll_kernel<SourceT>(
        N_re_out, N_im_out, M_re_out, M_im_out,
        orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
#endif
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
template <class SourceT>
CUDA_KERNEL
void stft_get_ll_grad_kernel(
    double* grad_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    double* param_eps,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
    CUDA_SHARED cmplx d_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx h_h_tmp[NUM_THREADS_HERE];
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

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
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
            stft_eval_block_ll<SourceT>(
                src, fresnel, stft, params,
                link_space_craft_rec, link_space_craft_em, bin_i,
                data_index, noise_index, n_side_bins, window_factor, freq_from_tdi_phase,
                d_h_tmp, h_h_tmp, tid, &d_h_p, &h_h_p);
            double q_p = d_h_p.real() - 0.5 * h_h_p.real();

            if (tid == 0) params[kk] = saved - eps_k;
            CUDA_SYNC_THREADS;
            cmplx d_h_m, h_h_m;
            stft_eval_block_ll<SourceT>(
                src, fresnel, stft, params,
                link_space_craft_rec, link_space_craft_em, bin_i,
                data_index, noise_index, n_side_bins, window_factor, freq_from_tdi_phase,
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

template <class SourceT>
inline void stft_get_ll_grad_impl(
    double* grad_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_all, int* data_index_all, int* noise_index_all,
    double* param_eps,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
#ifdef __CUDACC__
    static Orbits*      orbits_gpu     = nullptr;
    static TDIConfig*   tdi_config_gpu = nullptr;
    static STFTFresnel* fresnel_gpu    = nullptr;
    static STFTDomain*  stft_gpu       = nullptr;
    if (orbits_gpu     == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,     sizeof(Orbits)));
    if (tdi_config_gpu == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu, sizeof(TDIConfig)));
    if (fresnel_gpu    == nullptr) gpuErrchk(cudaMalloc(&fresnel_gpu,    sizeof(STFTFresnel)));
    if (stft_gpu       == nullptr) gpuErrchk(cudaMalloc(&stft_gpu,       sizeof(STFTDomain)));
    gpuErrchk(cudaMemcpy(orbits_gpu,     orbits,     sizeof(Orbits),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu, tdi_config, sizeof(TDIConfig),  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fresnel_gpu,    fresnel,    sizeof(STFTFresnel), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(stft_gpu,       stft,       sizeof(STFTDomain), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_get_ll_grad_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        grad_out, orbits_gpu, tdi_config_gpu, fresnel_gpu, stft_gpu,
        params_all, data_index_all, noise_index_all, param_eps,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    stft_get_ll_grad_kernel<SourceT>(
        grad_out, orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, noise_index_all, param_eps,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
#endif
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
template <class SourceT>
CUDA_KERNEL
void stft_swap_ll_grad_kernel(
    double* grad_add_out, double* grad_remove_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    double* param_eps_add, double* param_eps_remove,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
    CUDA_SHARED cmplx d_h_add_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx d_h_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx add_add_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx remove_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED cmplx add_remove_tmp[NUM_THREADS_HERE];
    CUDA_SHARED double params_add[N_PARAMS_MAX];
    CUDA_SHARED double params_remove[N_PARAMS_MAX];

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

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
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
            stft_eval_block_swap<SourceT>(
                src, fresnel, stft, params_add, params_remove,
                link_space_craft_rec, link_space_craft_em, bin_i,
                data_index, noise_index, n_side_bins, window_factor, freq_from_tdi_phase,
                d_h_add_tmp, d_h_remove_tmp, add_add_tmp, remove_remove_tmp, add_remove_tmp, tid,
                &dha_p, &dhr_p, &aa_p, &rr_p, &ar_p);
            double S_p = dha_p.real() - dhr_p.real()
                         - 0.5 * aa_p.real() - 0.5 * rr_p.real() + ar_p.real();

            if (tid == 0) params_add[kk] = saved - eps_k;
            CUDA_SYNC_THREADS;
            cmplx dha_m, dhr_m, aa_m, rr_m, ar_m;
            stft_eval_block_swap<SourceT>(
                src, fresnel, stft, params_add, params_remove,
                link_space_craft_rec, link_space_craft_em, bin_i,
                data_index, noise_index, n_side_bins, window_factor, freq_from_tdi_phase,
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
            stft_eval_block_swap<SourceT>(
                src, fresnel, stft, params_add, params_remove,
                link_space_craft_rec, link_space_craft_em, bin_i,
                data_index, noise_index, n_side_bins, window_factor, freq_from_tdi_phase,
                d_h_add_tmp, d_h_remove_tmp, add_add_tmp, remove_remove_tmp, add_remove_tmp, tid,
                &dha_p, &dhr_p, &aa_p, &rr_p, &ar_p);
            double S_p = dha_p.real() - dhr_p.real()
                         - 0.5 * aa_p.real() - 0.5 * rr_p.real() + ar_p.real();

            if (tid == 0) params_remove[kk] = saved - eps_k;
            CUDA_SYNC_THREADS;
            cmplx dha_m, dhr_m, aa_m, rr_m, ar_m;
            stft_eval_block_swap<SourceT>(
                src, fresnel, stft, params_add, params_remove,
                link_space_craft_rec, link_space_craft_em, bin_i,
                data_index, noise_index, n_side_bins, window_factor, freq_from_tdi_phase,
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

template <class SourceT>
inline void stft_swap_ll_grad_impl(
    double* grad_add_out, double* grad_remove_out,
    Orbits* orbits, TDIConfig* tdi_config,
    STFTFresnel* fresnel, STFTDomain* stft,
    double* params_add_all, double* params_remove_all,
    int* data_index_all, int* noise_index_all,
    double* param_eps_add, double* param_eps_remove,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
#ifdef __CUDACC__
    static Orbits*      orbits_gpu     = nullptr;
    static TDIConfig*   tdi_config_gpu = nullptr;
    static STFTFresnel* fresnel_gpu    = nullptr;
    static STFTDomain*  stft_gpu       = nullptr;
    if (orbits_gpu     == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,     sizeof(Orbits)));
    if (tdi_config_gpu == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu, sizeof(TDIConfig)));
    if (fresnel_gpu    == nullptr) gpuErrchk(cudaMalloc(&fresnel_gpu,    sizeof(STFTFresnel)));
    if (stft_gpu       == nullptr) gpuErrchk(cudaMalloc(&stft_gpu,       sizeof(STFTDomain)));
    gpuErrchk(cudaMemcpy(orbits_gpu,     orbits,     sizeof(Orbits),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu, tdi_config, sizeof(TDIConfig),  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fresnel_gpu,    fresnel,    sizeof(STFTFresnel), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(stft_gpu,       stft,       sizeof(STFTDomain), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) num_bin, 1u, 1u);
    stft_swap_ll_grad_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        grad_add_out, grad_remove_out, orbits_gpu, tdi_config_gpu, fresnel_gpu, stft_gpu,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        param_eps_add, param_eps_remove,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    stft_swap_ll_grad_kernel<SourceT>(
        grad_add_out, grad_remove_out, orbits, tdi_config, fresnel, stft,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        param_eps_add, param_eps_remove,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor, freq_from_tdi_phase);
#endif
}

#endif // __LAT_STFT_KERNELS_HH__
