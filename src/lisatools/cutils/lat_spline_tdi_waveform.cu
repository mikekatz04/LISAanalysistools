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

#ifdef __CUDACC__
#define NUM_THREADS_HERE 128
#else
#define NUM_THREADS_HERE 1
#endif

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
void td_spline_run_wave_tdi_kernel(TDSplineTDIWaveform *tdi_on_fly, int buffer_length, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    void *buffer = (void*)shared_mem;
    tdi_on_fly->run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}
#endif

void td_spline_run_wave_tdi_wrap(TDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    TDSplineTDIWaveform *wave_here = new TDSplineTDIWaveform(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->amp_spline, tdi_on_fly->phase_spline);
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

    wave_here->orbits = d_orbits;
    wave_here->tdi_config = d_tdi_config;
    wave_here->amp_spline = d_amp_spline;
    wave_here->phase_spline = d_phase_spline;

    TDSplineTDIWaveform *d_wave_here;
    cudaMalloc(&d_wave_here, sizeof(TDSplineTDIWaveform));
    gpuErrchk(cudaMemcpy(d_wave_here, wave_here, sizeof(TDSplineTDIWaveform), cudaMemcpyHostToDevice));

    int buffer_length = tdi_on_fly->get_td_spline_buffer_size(N); 
    
    td_spline_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, buffer_length>>>(d_wave_here, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_amp_spline));
    gpuErrchk(cudaFree(d_phase_spline));
    gpuErrchk(cudaFree(d_wave_here));
    delete wave_here;
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
void fd_spline_run_wave_tdi_kernel(FDSplineTDIWaveform *tdi_on_fly, int buffer_length, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    void *buffer = (void*)shared_mem;
    tdi_on_fly->run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}
#endif

void fd_spline_run_wave_tdi_wrap(FDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    FDSplineTDIWaveform *wave_here = new FDSplineTDIWaveform(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->amp_spline, tdi_on_fly->freq_spline);
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

    wave_here->orbits = d_orbits;
    wave_here->tdi_config = d_tdi_config;
    wave_here->amp_spline = d_amp_spline;
    wave_here->freq_spline = d_freq_spline;

    FDSplineTDIWaveform *d_wave_here;
    cudaMalloc(&d_wave_here, sizeof(FDSplineTDIWaveform));
    gpuErrchk(cudaMemcpy(d_wave_here, wave_here, sizeof(FDSplineTDIWaveform), cudaMemcpyHostToDevice));

    int buffer_length = tdi_on_fly->get_fd_spline_buffer_size(N); 
    
    fd_spline_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, buffer_length>>>(d_wave_here, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_amp_spline));
    gpuErrchk(cudaFree(d_freq_spline));
    gpuErrchk(cudaFree(d_wave_here));
    delete wave_here;
#else

    // make buffer 
    int buffer_length = tdi_on_fly->get_fd_spline_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}
