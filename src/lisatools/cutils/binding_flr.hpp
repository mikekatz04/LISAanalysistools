#ifndef __BINDING_FLR_HPP__
#define __BINDING_FLR_HPP__

#include "LISAResponse.hh"
#include "Detector.hpp"
#include <string>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "binding_detector.hpp"
#include "gbt_binding.hpp"
#include "Interpolate.hh"

namespace nb = nanobind;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
template<typename T>
using array_type = nb::ndarray<T, nb::device::cuda>;
#define LISAResponseWrap LISAResponseWrapGPU
#else
template<typename T>
using array_type = nb::ndarray<T, nb::device::cpu>;
#define LISAResponseWrap LISAResponseWrapCPU
#endif

// Phase 3L.7p (2026-06-04): legacy OrbitsWrap_responselisa #define aliases
// dropped; the class was deleted in favor of binding.hpp's canonical OrbitsWrap.
// 2026-06-10: CubicSplineWrap #define alias dropped too -- the class
// is GBT's (gbt_binding.hpp, included above), which carries its own
// CPU/GPU alias block. Same single-registrant pattern as OrbitsWrap:
// GBT's `interp` module is the sole registrant; LAT consumes the class
// through the shared header so the C++ typeid matches GBT's
// registration and nanobind cross-module casting works.
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define TDIConfigWrap TDIConfigWrapGPU
#else
#define TDIConfigWrap TDIConfigWrapCPU
#endif


class ReturnPointerBase {
  public:
    template<typename T>
    static T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = input1.data();
        
#else
        if (input1.size() != static_cast<size_t>(N) * static_cast<size_t>(multiplier))
        {
            std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(static_cast<size_t>(N) * static_cast<size_t>(multiplier)) + ". It's length is " + std::to_string(input1.size()) + ".";
            throw std::invalid_argument(err_out);
        }
        T* ptr1 = input1.data();
#endif
        return ptr1;
    };

    template<typename T>
    static T* return_pointer(array_type<T> input1, std::string name)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = input1.data();
#else
        T* ptr1 = input1.data();
#endif
        return ptr1;
    };

    static cmplx* return_pointer_cmplx(array_type<std::complex<double>> input1, std::string name)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        cmplx *ptr1 = (cmplx*) input1.data();
#else
        cmplx* ptr1 = (cmplx*) input1.data();
#endif
        return ptr1;
    };

};

// 2026-06-10: LAT's local CubicSplineWrap class deleted -- it was a
// byte-for-byte duplicate of GBT's (gbt_binding.hpp), differing only in
// the (unused) ReturnPointerBase inheritance. LAT now consumes GBT's
// class directly, mirroring how GBGPU/BBHx consume LAT's OrbitsWrap.


// Phase 3L.7p (2026-06-04): legacy OrbitsWrap_responselisa deleted -- collapsed
// into the canonical `OrbitsWrap` (binding.hpp). Both classes always
// shipped identical constructor signatures; the only structural difference
// was that this one inherited from ReturnPointerBase for its static
// `return_pointer*` helpers. No downstream code actually accessed those
// through an OrbitsWrap_responselisa pointer (and the GPUs can't dispatch
// virtually anyway), so collapsing was safe. Every downstream `*TDIonTheFlyWrap`
// / LISAResponseWrap constructor now takes `OrbitsWrap *` directly.


class TDIConfigWrap : public ReturnPointerBase{
  public:
    TDIConfig *tdi_config;
    TDIConfigWrap(array_type<int>unit_starts_, array_type<int>unit_lengths_, array_type<int>tdi_base_link_, array_type<int>tdi_link_combinations_, array_type<double>tdi_signs_in_, array_type<int>channels_, int num_units_, int num_channels_)
    {
        // TODO: add check for length of all units
        int *_unit_starts = return_pointer_and_check_length(unit_starts_, "unit_starts", num_units_, 1);
        int *_unit_lengths = return_pointer_and_check_length(unit_lengths_, "unit_lengths", num_units_, 1);

        int *_tdi_base_link = return_pointer(tdi_base_link_, "tdi_base_link");
        int *_tdi_link_combinations = return_pointer(tdi_link_combinations_, "tdi_link_combinations");
        double *_tdi_signs_in = return_pointer(tdi_signs_in_, "tdi_signs_in");
        int *_channels = return_pointer(channels_, "channels");
        tdi_config = new TDIConfig(_unit_starts, _unit_lengths, _tdi_base_link, _tdi_link_combinations, _tdi_signs_in, _channels,  num_units_, num_channels_);
    };
    ~TDIConfigWrap(){
        delete tdi_config;
    };
};

class LISAResponseWrap : public ReturnPointerBase {
  public:
    LISAResponse *response;
    OrbitsWrap *orbits;
    TDIConfigWrap *tdi_config;
    LISAResponseWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_)
    {
        orbits = orbits_;
        tdi_config = tdi_config_;
        response = new LISAResponse(orbits_->orbits, tdi_config_->tdi_config);
    };
    ~LISAResponseWrap(){
        delete response;
    };

    void get_tdi_delays_wrap(array_type<double> delayed_links_, array_type<double> input_links_, int num_inputs, int num_delays, array_type<double> t_arr_,
                    int order, double sampling_frequency, int buffer_integer, array_type<double> A_in_, double deps, int num_A, array_type<double> E_in_, int tdi_start_ind,
                    array_type<double> t0_arr_, int batch_size = 1, bool run_async = false);

    void get_response_wrap(array_type<double> y_gw_, array_type<double> t_data_, array_type<double> k_in_, array_type<double> u_in_, array_type<double> v_in_, double dt,
                  int num_delays,
                  array_type<std::complex<double>> input_in_, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer,
                  array_type<double> A_in_, double deps, int num_A, array_type<double> E_in_, int projections_start_ind,
                  array_type<double> t0_arr_, int batch_size = 1, bool run_async = false);

    void get_response_quintic_wrap(array_type<double> y_gw_, array_type<double> t_data_, array_type<double> k_in_, array_type<double> u_in_, array_type<double> v_in_, double dt,
                  int num_delays,
                  array_type<std::complex<double>> input_in_, int num_inputs, double sampling_frequency,
                  array_type<double> c1r_, array_type<double> c2r_, array_type<double> c3r_, array_type<double> c4r_, array_type<double> c5r_,
                  array_type<double> c1i_, array_type<double> c2i_, array_type<double> c3i_, array_type<double> c4i_, array_type<double> c5i_,
                  int projections_start_ind, int spline_type,
                  array_type<double> t0_arr_, int batch_size = 1, bool run_async = false);

};

#endif // __BINDING_FLR_HPP__