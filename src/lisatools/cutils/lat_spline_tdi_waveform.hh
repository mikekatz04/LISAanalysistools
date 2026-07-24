#ifndef __LAT_SPLINE_TDI_WAVEFORM_HH__
#define __LAT_SPLINE_TDI_WAVEFORM_HH__

// LISATDIonTheFly subclasses for spline-fed time-domain (TDSpline) and
// frequency-domain (FDSpline) on-the-fly TDI templates. These exist
// primarily as glue classes that plug a CubicSpline-driven intrinsic
// waveform (amp/phase or amp/freq) into the LAT-owned
// LISATDIonTheFly::run_wave_tdi pipeline.
//
// Phase 3L.6 (2026-06-03): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.hh:377-453
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu:9659-9774
//                                                          :9841-9986
// to LISAanalysistools. lat_spline_tdi_waveform.cu compiles the method
// bodies + the kernel/wrap host launchers. LAT's own CPU (.cxx copy) and
// GPU static libs are the only compilers of this TU -- lisa-on-gpu retired
// its cutils/ copy-compile at Phase 3L.7n (2026-06-04).

#include "gbt_global.h"
#include "Detector.hpp"        // Orbits, Vec
#include "LISAResponse.hh"     // TDIConfig
#include "Interpolate.hh"      // CubicSpline + NLINKS
#include "lat_tdi_on_the_fly.hh"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define TDSplineTDIWaveform TDSplineTDIWaveformGPU
#define FDSplineTDIWaveform FDSplineTDIWaveformGPU
#else
#define TDSplineTDIWaveform TDSplineTDIWaveformCPU
#define FDSplineTDIWaveform FDSplineTDIWaveformCPU
#endif

class TDSplineTDIWaveform : public LISATDIonTheFly{
  public:
    CubicSpline *amp_spline;
    CubicSpline *phase_spline;
    int binary_index_storage;

    CUDA_CALLABLE_MEMBER
    TDSplineTDIWaveform(Orbits* orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *phase_spline_): LISATDIonTheFly(orbits_, tdi_config_, 0, 1, 2, 3){
        amp_spline = amp_spline_;
        phase_spline = phase_spline_;
    };
    CUDA_CALLABLE_MEMBER
    ~TDSplineTDIWaveform(){};
    CUDA_CALLABLE_MEMBER
    int get_td_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_DEVICE
    void check_x();
    CUDA_DEVICE
    double get_amp(double t, double *params, int spline_i);
    CUDA_DEVICE
    double get_phase(double t, double *params, int spline_i);
};

// Host launcher: pulls Orbits/TDIConfig/CubicSpline structs onto the
// device, configures the device-side TDSplineTDIWaveform, runs the
// per-bin kernel, frees the temporary device-side mirrors. CPU branch
// runs the same path on a heap buffer.
void td_spline_run_wave_tdi_wrap(TDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);


class FDSplineTDIWaveform : public LISATDIonTheFly {
    public:
        CubicSpline *amp_spline;
        CubicSpline *freq_spline;

    CUDA_CALLABLE_MEMBER
    FDSplineTDIWaveform(Orbits* orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *freq_spline_): LISATDIonTheFly(orbits_, tdi_config_, 0, 1, 2, 3)
    {
        amp_spline = amp_spline_;
        freq_spline = freq_spline_;
    };
    CUDA_CALLABLE_MEMBER
    ~FDSplineTDIWaveform(){};
    CUDA_CALLABLE_MEMBER
    int get_fd_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_DEVICE
    double get_phase_ref(double t, double *params, int bin_i);
    CUDA_DEVICE
    double get_amp(double t, double *params, int spline_i);
    CUDA_DEVICE
    double get_phase(double t, double *params, int spline_i);
    CUDA_DEVICE
    void get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);
    CUDA_DEVICE
    double get_amp_f(double t, double *params, int spline_i);
};

// Host launcher mirror of td_spline_run_wave_tdi_wrap for the FD path.
void fd_spline_run_wave_tdi_wrap(FDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);

#endif // __LAT_SPLINE_TDI_WAVEFORM_HH__
