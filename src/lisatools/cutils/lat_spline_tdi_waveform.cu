// === TDSplineTDIWaveform + FDSplineTDIWaveform method bodies + host launchers ===
// Phase 3L.6 (2026-06-03): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu (lines 9659-9774, 9841-9976)
// to LISAanalysistools.

#include "lat_spline_tdi_waveform.hh"
#include "gbt_global.h"
#include "Interpolate.hh"
#include "LISAResponse.hh"
#include "Detector.hpp"
#include "lat_tdi_on_the_fly.hh"

#include <cstdlib>
#include <cstring>
#include <new>

#ifdef __CUDACC__
#define NUM_THREADS_HERE 128
#else
#define NUM_THREADS_HERE 1
#endif

#ifdef __CUDACC__
// ---------------------------------------------------------------
// Shared-memory-vs-global scratch launch helpers (GPU only)
// ---------------------------------------------------------------
// The td/fd spline kernels stage the phase-extract/unwrap scratch
// (get_tdi_buffer_size(N) = 21*N bytes) in dynamic shared memory. For large
// N (dense MBH chirp grids) that overruns the per-block shared budget, so we
// keep it in shared memory where it fits (opting in past the 48 KB default
// when the device allows) and fall back to per-block global memory above the
// device ceiling. See lat_tdi_on_the_fly.cu:get_tdi for the buffer carve.

// Per-block stride into the global-memory scratch fallback. The buffer is
// carved doubles-first (flip/pjump) in LISATDIonTheFly::get_tdi, so every
// block's slice must be 8-byte aligned; buffer_length = 21*N is aligned only
// when N % 8 == 0. Round up to 256 (>= double alignment, matches cudaMalloc
// granularity). MUST be identical host-side (allocation) and device-side
// (per-block offset), so it is __host__ __device__.
__host__ __device__ inline size_t spline_tdi_scratch_stride(int buffer_length)
{
    return ((size_t)buffer_length + 255) & ~(size_t)255;
}

// LISATOOLS_VERBOSE=1 (any non-empty, non-"0" value): print the shared-vs-
// global launch decision per call. Read once via a function-local static so
// there is no wrap/Python signature change and it works from every entry
// point. Matches the project's env-var debug-knob convention (GB_DEBUG=1).
static bool spline_tdi_verbose()
{
    static const bool v = [] {
        const char *e = std::getenv("LISATOOLS_VERBOSE");
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

// Max dynamic shared bytes `kernel_func` may opt into on the CURRENT device:
// the opt-in ceiling minus the kernel's static shared footprint (params_here,
// link arrays, cub::BlockScan TempStorage). This is the shared-vs-global
// decision boundary that also guarantees the cudaFuncSetAttribute below is a
// legal value. Queried per call -- cheap next to the per-call cudaMallocs in
// these wraps, and correct per-device under multi-GPU runs.
static size_t spline_tdi_max_dynamic_shared(const void *kernel_func)
{
    int device = 0;
    gpuErrchk(cudaGetDevice(&device));
    int optin_max = 0;
    gpuErrchk(cudaDeviceGetAttribute(&optin_max,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attrs;
    gpuErrchk(cudaFuncGetAttributes(&attrs, kernel_func));
    return (size_t)optin_max - attrs.sharedSizeBytes;
}

// Shared scratch-placement decision for the td_spline/fd_spline kernels: keep
// dynamic shared memory when it fits (default 48 KB, or the device opt-in
// ceiling after cudaFuncSetAttribute), otherwise allocate a per-block global
// scratch. Mirrors + completes the GB fix (gb_tdi_on_the_fly.cu, commit
// 3cbdcf0), which opts in but has no fallback above the device ceiling.
//
// Returns the dynamic-shared byte count to launch with; 0 means "use the
// global fallback" and *d_scratch_out is set to the freshly cudaMalloc'd
// per-block scratch (caller frees after the launch), else *d_scratch_out is
// nullptr. `kernel_func` is passed only to cudaFunc{Get,Set}Attribute -- the
// actual <<<>>> launch stays on the concrete kernel name in each wrap (the
// codebase's established pattern; see gb_run_wave_tdi_kernel). `tag` names the
// kernel in verbose output ("td_spline" / "fd_spline").
static size_t spline_tdi_prepare_scratch(const char *tag, const void *kernel_func,
    int buffer_length, int N, int num_bin, char **d_scratch_out)
{
    const size_t DEFAULT_DYN_SMEM = 48 * 1024;  // default per-block cap, sm_70+
    size_t shared_bytes = (size_t)buffer_length;
    *d_scratch_out = nullptr;

    if (shared_bytes <= DEFAULT_DYN_SMEM)
    {
        // Legacy fast path: dynamic shared, no opt-in (bit-for-bit unchanged).
        if (spline_tdi_verbose())
            fprintf(stderr, "lisatools %s TDI-on-the-fly: N=%d, scratch=%d B "
                "<= 48 KB default; dynamic shared memory, no opt-in.\n",
                tag, N, buffer_length);
        return shared_bytes;
    }

    const size_t ceiling = spline_tdi_max_dynamic_shared(kernel_func);
    if (shared_bytes <= ceiling)
    {
        // Fast path: opt in to the larger per-block shared cap.
        if (spline_tdi_verbose())
            fprintf(stderr, "lisatools %s TDI-on-the-fly: N=%d, scratch=%d B "
                "> 48 KB default; cudaFuncSetAttribute opt-in "
                "(device ceiling %zu B), still dynamic shared memory.\n",
                tag, N, buffer_length, ceiling);
        gpuErrchk(cudaFuncSetAttribute(kernel_func,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes));
        return shared_bytes;
    }

    // Slow path: per-block global scratch. One-time notice even without
    // VERBOSE; every-call detail with it.
    const size_t stride = spline_tdi_scratch_stride(buffer_length);
    static bool warned = false;
    if (spline_tdi_verbose() || !warned)
    {
        fprintf(stderr, "lisatools %s TDI-on-the-fly: N=%d, scratch=%d B "
            "exceeds device dynamic-shared ceiling (%zu B); global-memory "
            "scratch fallback (%zu B = num_bin %d x %zu B/block; slower).\n",
            tag, N, buffer_length, ceiling,
            (size_t)num_bin * stride, num_bin, stride);
        warned = true;
    }
    // Footprint = num_bin x roundup(21*N, 256) -- one slice per block, smaller
    // than the 3*N*num_bin-double tdi_amp/tdi_phase arrays this call already
    // fills. If it ever bites, cap the grid at min(num_bin, max_blocks) and
    // allocate one slice per launched block: run_wave_tdi already grid-strides
    // bin_i by gridDim.x, so a grid smaller than num_bin is supported as-is.
    gpuErrchk(cudaMalloc(d_scratch_out, (size_t)num_bin * stride));
    return 0;
}
#endif // __CUDACC__

// ---------------------------------------------------------------
// FDSplineTDIWaveform::get_tdi (was TDIonTheFly.cu:9660-9689)
// ---------------------------------------------------------------
CUDA_DEVICE
void FDSplineTDIWaveform::get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels)
{
    LISATDIonTheFly::get_tdi(
        buffer, buffer_length,
        tdi_channels_arr, 
        tdi_amp, tdi_phase,
        phi_ref,
        params, t_arr, N, bin_i, nchannels
    );
    
    CUDA_SYNC_THREADS;
    double amp_f;
    
#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    for (int i = start; i < N; i += incr)
    {
        amp_f = get_amp_f(t_arr[i], params, bin_i);
        for (int chan = 0; chan < tdi_config->num_channels; chan += 1)
        {
            tdi_amp[chan * N + i] *= amp_f;
        }
    }
    CUDA_SYNC_THREADS;
}

// ---------------------------------------------------------------
// TDSplineTDIWaveform accessors (was TDIonTheFly.cu:9692-9705)
// ---------------------------------------------------------------
CUDA_DEVICE
double TDSplineTDIWaveform::get_amp(double t, double *params, int spline_i)
{
    // printf("before amp: %d\n", amp_spline->ninterps);
    return amp_spline->eval_single(t, spline_i);
}

CUDA_DEVICE
double TDSplineTDIWaveform::get_phase(double t, double *params, int spline_i)
{
    // printf("before phase: %d\n", phase_spline->ninterps);
    
    return phase_spline->eval_single(t, spline_i);
}

// ---------------------------------------------------------------
// td_spline kernel + wrap (was TDIonTheFly.cu:9707-9774)
// ---------------------------------------------------------------
#ifdef __CUDACC__
CUDA_KERNEL
void td_spline_run_wave_tdi_kernel(TDSplineTDIWaveform *tdi_on_fly, int buffer_length, char *global_buffer, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    // Scratch selection: dynamic shared when the host launched with it
    // (global_buffer == nullptr), otherwise this block's slice of the global
    // scratch. run_wave_tdi grid-strides bin_i by gridDim.x, so the blockIdx.x
    // slice is private to this block across all its bins.
    void *buffer = (global_buffer != nullptr)
        ? (void*)(global_buffer + (size_t)blockIdx.x * spline_tdi_scratch_stride(buffer_length))
        : (void*)shared_mem;
    tdi_on_fly->run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}

// Construct the polymorphic wave object IN device memory (placement new) so its
// vtable is the DEVICE vtable. A host-`new`'d object cudaMemcpy'd to the device
// carries a HOST vtable pointer; the first virtual get_amp/get_phase call in the
// kernel then dereferences host memory -> illegal access on GPU (silent on CPU,
// which never leaves host). The member pointers are the already-device-mirrored
// d_orbits / d_tdi_config / d_amp_spline / d_phase_spline.
CUDA_KERNEL
void td_spline_construct_kernel(TDSplineTDIWaveform *obj, Orbits *orbits, TDIConfig *tdi_config,
    CubicSpline *amp_spline, CubicSpline *phase_spline)
{
    new (obj) TDSplineTDIWaveform(orbits, tdi_config, amp_spline, phase_spline);
}
#endif

void td_spline_run_wave_tdi_wrap(TDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    CubicSpline *d_amp_spline;
    cudaMalloc(&d_amp_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_amp_spline, tdi_on_fly->amp_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    CubicSpline *d_phase_spline;
    cudaMalloc(&d_phase_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_phase_spline, tdi_on_fly->phase_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    // Build the wave object on the device (device vtable) rather than host-new +
    // cudaMemcpy (which would copy a host vtable pointer).
    TDSplineTDIWaveform *d_wave_here;
    cudaMalloc(&d_wave_here, sizeof(TDSplineTDIWaveform));
    td_spline_construct_kernel<<<1, 1>>>(d_wave_here, d_orbits, d_tdi_config, d_amp_spline, d_phase_spline);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    int buffer_length = tdi_on_fly->get_td_spline_buffer_size(N);

    char *d_scratch = nullptr;
    size_t shared_bytes = spline_tdi_prepare_scratch("td_spline",
        (const void *)td_spline_run_wave_tdi_kernel, buffer_length, N, num_bin, &d_scratch);

    td_spline_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        d_wave_here, buffer_length, d_scratch, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    if (d_scratch != nullptr) gpuErrchk(cudaFree(d_scratch));

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_amp_spline));
    gpuErrchk(cudaFree(d_phase_spline));
    gpuErrchk(cudaFree(d_wave_here));
#else

    // make buffer 
    int buffer_length = tdi_on_fly->get_td_spline_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}

// ---------------------------------------------------------------
// FDSplineTDIWaveform accessors (was TDIonTheFly.cu:9841-9906)
// ---------------------------------------------------------------
CUDA_DEVICE
double FDSplineTDIWaveform::get_amp(double t, double *params, int spline_i)
{
    return 1.0;
}

CUDA_DEVICE
double FDSplineTDIWaveform::get_phase(double t, double *params, int spline_i)
{
    double f = freq_spline->eval_single(t, spline_i);
    return 2. * M_PI * f * t;
}

CUDA_DEVICE
double FDSplineTDIWaveform::get_amp_f(double t, double *params, int spline_i)
{
    // TODO: may want to do this in a fast way
    return amp_spline->eval_single(t, spline_i);
}


// CUDA_DEVICE
// void FDSplineTDIWaveform::run_wave_tdi(cmplx *tdi_channels_arr, 
//     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
//     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
// {
//     for (int bin_i = 0; bin_i < num_sub; bin_i += 1)
//     {
//         // map to Tyson/Neil setup
//         double beta = params[bin_i * n_params + 3];
//         double costh = cos(M_PI / 2.0 - beta);
        
//         double lam = params[bin_i * n_params + 2];
//         double phi = lam;

//         double inc = params[bin_i * n_params + 0];
//         double cosi = cos(inc);

//         double psi = params[bin_i * n_params + 1];
//         double *params_here = &params[bin_i * n_params];
//         double *t_here = &t_arr[bin_i * N];
    
//         // TODO: CHECK THIS!!
//         get_tdi(
//             buffer, buffer_length, &X[bin_i * N], &Y[bin_i * N], &Z[bin_i * N], 
//             &Xamp[bin_i * N], &Xphase[bin_i * N],
//             &Yamp[bin_i * N], &Yphase[bin_i * N],
//             &Zamp[bin_i * N], &Zphase[bin_i * N], &phi_ref[bin_i * N],
//             params_here, t_here, N, costh, phi, cosi, psi, bin_i);
//     }
// }

CUDA_DEVICE
double FDSplineTDIWaveform::get_phase_ref(double t, double *params, int bin_i)
{
    // in FD, has to be fixed to 2 pi f_ssb t_ssb
    // t is t_ssb

    // t_i = t[i];
    // // TODO: should we make it so this is without the spline?
    double f = freq_spline->eval_single(t, bin_i);
    return 2. * M_PI * f * t;
    // phase[i] = 2. * M_PI * f * t_i;
    // phase[index] = phase_ref_store[spline_i * N + index];

}

// ---------------------------------------------------------------
// fd_spline kernel + wrap (was TDIonTheFly.cu:9909-9976)
// ---------------------------------------------------------------
#ifdef __CUDACC__
CUDA_KERNEL
void fd_spline_run_wave_tdi_kernel(FDSplineTDIWaveform *tdi_on_fly, int buffer_length, char *global_buffer, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    // See td_spline_run_wave_tdi_kernel: shared when launched with it,
    // otherwise this block's private slice of the global scratch.
    void *buffer = (global_buffer != nullptr)
        ? (void*)(global_buffer + (size_t)blockIdx.x * spline_tdi_scratch_stride(buffer_length))
        : (void*)shared_mem;
    tdi_on_fly->run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}

// Device-side placement-new construction (device vtable) -- see
// td_spline_construct_kernel for why host-new + cudaMemcpy is wrong for a
// polymorphic object.
CUDA_KERNEL
void fd_spline_construct_kernel(FDSplineTDIWaveform *obj, Orbits *orbits, TDIConfig *tdi_config,
    CubicSpline *amp_spline, CubicSpline *freq_spline)
{
    new (obj) FDSplineTDIWaveform(orbits, tdi_config, amp_spline, freq_spline);
}
#endif

void fd_spline_run_wave_tdi_wrap(FDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    CubicSpline *d_amp_spline;
    cudaMalloc(&d_amp_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_amp_spline, tdi_on_fly->amp_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    CubicSpline *d_freq_spline;
    cudaMalloc(&d_freq_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_freq_spline, tdi_on_fly->freq_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    FDSplineTDIWaveform *d_wave_here;
    cudaMalloc(&d_wave_here, sizeof(FDSplineTDIWaveform));
    fd_spline_construct_kernel<<<1, 1>>>(d_wave_here, d_orbits, d_tdi_config, d_amp_spline, d_freq_spline);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    int buffer_length = tdi_on_fly->get_fd_spline_buffer_size(N);

    char *d_scratch = nullptr;
    size_t shared_bytes = spline_tdi_prepare_scratch("fd_spline",
        (const void *)fd_spline_run_wave_tdi_kernel, buffer_length, N, num_bin, &d_scratch);

    fd_spline_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        d_wave_here, buffer_length, d_scratch, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    if (d_scratch != nullptr) gpuErrchk(cudaFree(d_scratch));

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_amp_spline));
    gpuErrchk(cudaFree(d_freq_spline));
    gpuErrchk(cudaFree(d_wave_here));
#else

    // make buffer 
    int buffer_length = tdi_on_fly->get_fd_spline_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}
