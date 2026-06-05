#ifndef __BINDING_LAT_SPLINE_TDI_HPP__
#define __BINDING_LAT_SPLINE_TDI_HPP__

// pybind11 wrappers for the LAT-owned LISATDIonTheFly subclasses:
//   LISATDIonTheFlyWrap     -- base holder for OrbitsWrap + TDIConfigWrap
//                              (used as a common base for all source-class
//                              TDIonTheFly wrappers)
//   TDSplineTDIWaveformWrap -- pybind11 holder for TDSplineTDIWaveform
//   FDSplineTDIWaveformWrap -- pybind11 holder for FDSplineTDIWaveform
//
// Phase 3L.6 (2026-06-03): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.hpp:45-105
// to LISAanalysistools. The pybind11 registrations live in LAT's
// binding_flr.cxx (response_part(m)) so the single-registrant rule
// owns these classes.
//
// GBTDIonTheFlyWrap and SOBBHTDIonTheFlyWrap (still in lisa-on-gpu)
// inherit from LISATDIonTheFlyWrap via the LAT include; that's the
// same pattern the underlying GBTDIonTheFly / SOBBHTDIonTheFly use to
// inherit from the LAT-owned LISATDIonTheFly base.

#include "binding_flr.hpp"     // ReturnPointerBase, TDIConfigWrap,
                               // CubicSplineWrap_responselisa
#include "binding.hpp"         // OrbitsWrap, array_type<T>
#include "lat_spline_tdi_waveform.hh"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define LISATDIonTheFlyWrap     LISATDIonTheFlyWrapGPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapGPU
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapGPU
#else
#define LISATDIonTheFlyWrap     LISATDIonTheFlyWrapCPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapCPU
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapCPU
#endif

// Common base for all LISATDIonTheFly Wrap subclasses. Pure data holder
// for the OrbitsWrap + TDIConfigWrap pair.
class LISATDIonTheFlyWrap : public ReturnPointerBase {
  public:
    OrbitsWrap *orbits;
    TDIConfigWrap *tdi_config;
    LISATDIonTheFlyWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_){
        orbits = orbits_;
        tdi_config = tdi_config_;
    };
};

class FDSplineTDIWaveformWrap : public LISATDIonTheFlyWrap {
  public:
    CubicSplineWrap_responselisa *amp_spline;
    CubicSplineWrap_responselisa *freq_spline;
    FDSplineTDIWaveform *waveform;
    FDSplineTDIWaveformWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_, CubicSplineWrap_responselisa *amp_spline_, CubicSplineWrap_responselisa *freq_spline_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        amp_spline = amp_spline_;
        freq_spline = freq_spline_;
        waveform = new FDSplineTDIWaveform(orbits_->orbits, tdi_config_->tdi_config, amp_spline_->spline, freq_spline_->spline);
    };
    ~FDSplineTDIWaveformWrap(){
        delete waveform;
    };

    inline void run_wave_tdi_wrap(
        array_type<std::complex<double>> tdi_channels_arr,
        array_type<double> tdi_amp, array_type<double> tdi_phase, array_type<double> phi_ref,
        array_type<double> params, array_type<double> t_arr, int N, int num_bin, int n_params, int nchannels)
    {
        fd_spline_run_wave_tdi_wrap(
            waveform,
            (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels),
            return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
            return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
            return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
            return_pointer_and_check_length(params, "params", n_params, num_bin),
            return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
            N, num_bin, n_params, nchannels
        );
    }

    inline int get_buffer_size(int N){return waveform->get_fd_spline_buffer_size(N);}
};


class TDSplineTDIWaveformWrap : public LISATDIonTheFlyWrap {
  public:
    CubicSplineWrap_responselisa *amp_spline;
    CubicSplineWrap_responselisa *phase_spline;
    TDSplineTDIWaveform *waveform;
    TDSplineTDIWaveformWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_, CubicSplineWrap_responselisa *amp_spline_, CubicSplineWrap_responselisa *phase_spline_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        amp_spline = amp_spline_;
        phase_spline = phase_spline_;
        waveform = new TDSplineTDIWaveform(orbits_->orbits, tdi_config_->tdi_config, amp_spline_->spline, phase_spline_->spline);
    };
    ~TDSplineTDIWaveformWrap(){
        delete waveform;
    };

    inline void run_wave_tdi_wrap(
        array_type<std::complex<double>> tdi_channels_arr,
        array_type<double> tdi_amp, array_type<double> tdi_phase, array_type<double> phi_ref,
        array_type<double> params, array_type<double> t_arr, int N, int num_bin, int n_params, int nchannels)
    {
        td_spline_run_wave_tdi_wrap(
            waveform,
            (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels),
            return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
            return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
            return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
            return_pointer_and_check_length(params, "params", n_params, num_bin),
            return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
            N, num_bin, n_params, nchannels
        );
    }

    inline int get_buffer_size(int N){return waveform->get_td_spline_buffer_size(N);}
};

#endif // __BINDING_LAT_SPLINE_TDI_HPP__
