#ifndef __BINDING_HPP__
#define __BINDING_HPP__

#include "L1Detector.hpp"
#include "Detector.hpp"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
template<typename T>
using array_type = cai::cuda_array_t<T>;
#else
template<typename T>
using array_type = py::array_t<T>;
#endif

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define OrbitsWrap OrbitsWrapGPU
#else
#define OrbitsWrap OrbitsWrapCPU
#endif


class OrbitsWrap {
  public:
    Orbits *orbits;
    OrbitsWrap(double dt_, int N_, array_type<double> n_arr_, array_type<double> ltt_arr_, array_type<double> x_arr_, array_type<int> links_, array_type<int> sc_r_, array_type<int> sc_e_, double armlength_)
    {

        double *_n_arr = return_pointer_and_check_length(n_arr_, "n_arr", N_, 6 * 3);
        double *_ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", N_, 6);
        double *_x_arr = return_pointer_and_check_length(x_arr_, "x_arr", N_, 3 * 3);

        int *_sc_r = return_pointer_and_check_length(sc_r_, "sc_r", 6, 1);
        int *_sc_e = return_pointer_and_check_length(sc_e_, "sc_e", 6, 1);
        int *_links = return_pointer_and_check_length(links_, "links", 6, 1);
        
        orbits = new Orbits(dt_, N_, _n_arr, _ltt_arr, _x_arr, _links,  _sc_r, _sc_e, armlength_);
    };
    ~OrbitsWrap(){
        delete orbits;
    };
    void get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num);
    void get_normal_unit_vec_wrap(array_type<double>normal_unit_vec_x, array_type<double>normal_unit_vec_y, array_type<double>normal_unit_vec_z, array_type<double>t, array_type<int>link, int num);
    void get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num);
    template<typename T>
    T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = static_cast<T *>(input1.get_compatible_typed_pointer());
        
#else
        py::buffer_info buf1 = input1.request();

        if (buf1.size != N * multiplier)
        {
            std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(N * multiplier) + ". It's length is " + std::to_string(buf1.size) + ".";
            throw std::invalid_argument(err_out);
        }
        T* ptr1 = static_cast<T *>(buf1.ptr);
#endif
        return ptr1;
    };

};

// now add the L1OrbitsWrap class
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define L1OrbitsWrap L1OrbitsWrapGPU
#else
#define L1OrbitsWrap L1OrbitsWrapCPU
#endif

class L1OrbitsWrap {
  public:
    L1Orbits *orbits;
    L1OrbitsWrap(double sc_t0_, double sc_dt_, int sc_N_, double ltt_t0_, double ltt_dt_, int ltt_N_, array_type<double> n_arr_, array_type<double> ltt_arr_, array_type<double> x_arr_, array_type<int> links_, array_type<int> sc_r_, array_type<int> sc_e_, double armlength_)
    {

        double *_n_arr = return_pointer_and_check_length(n_arr_, "n_arr", sc_N_, 6 * 3);
        double *_ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", ltt_N_, 6);
        double *_x_arr = return_pointer_and_check_length(x_arr_, "x_arr", sc_N_, 3 * 3);

        int *_sc_r = return_pointer_and_check_length(sc_r_, "sc_r", 6, 1);
        int *_sc_e = return_pointer_and_check_length(sc_e_, "sc_e", 6, 1);
        int *_links = return_pointer_and_check_length(links_, "links", 6, 1);
        
        orbits = new L1Orbits(sc_t0_, sc_dt_, sc_N_, ltt_t0_, ltt_dt_, ltt_N_, _n_arr, _ltt_arr, _x_arr, _links,  _sc_r, _sc_e, armlength_);
    };
    ~L1OrbitsWrap(){
        delete orbits;
    };
    void get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num);
    void get_normal_unit_vec_wrap(array_type<double>normal_unit_vec_x, array_type<double>normal_unit_vec_y, array_type<double>normal_unit_vec_z, array_type<double>t, array_type<int>link, int num);
    void get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num);
    template<typename T>
    T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = static_cast<T *>(input1.get_compatible_typed_pointer());
        
#else
        py::buffer_info buf1 = input1.request();

        if (buf1.size != N * multiplier)
        {
            std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(N * multiplier) + ". It's length is " + std::to_string(buf1.size) + ".";
            throw std::invalid_argument(err_out);
        }
        T* ptr1 = static_cast<T *>(buf1.ptr);
#endif
        return ptr1;
    };

};

#endif // __BINDING_HPP__
