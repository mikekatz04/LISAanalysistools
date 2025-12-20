#ifndef __DETECTOR_HPP__
#define __DETECTOR_HPP__

#include "global.hpp"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#ifdef __CUDACC__
#include "pybind11_cuda_array_interface.hpp"
template<typename T>
using array_type = cai::cuda_array_t<T>;
#else
template<typename T>
using array_type = py::array_t<T>;
#endif

class Vec
{
public:
    double x;
    double y;
    double z;

    CUDA_DEVICE
    Vec(double x_, double y_, double z_)
    {
        x = x_;
        y = y_;
        z = z_;
    }
};

class Orbits
{
public:
    double dt;
    int N;
    double *n_arr;
    double *ltt_arr;
    double *x_arr;
    int nlinks;
    int nspacecraft;
    double armlength;
    int *links;
    int *sc_r;
    int *sc_e;

    Orbits(double dt_, int N_, array_type<double> n_arr_, array_type<double> ltt_arr_, array_type<double> x_arr_, array_type<int> links_, array_type<int> sc_r_, array_type<int> sc_e_, double armlength_)
    {
        dt = dt_;
        N = N_;

        n_arr = return_pointer_and_check_length(n_arr_, "n_arr", N, 6 * 3);
        ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", N, 6);
        x_arr = return_pointer_and_check_length(x_arr_, "x_arr", N, 3 * 3);
        nlinks = 6;
        nspacecraft = 3;

        sc_r = return_pointer_and_check_length(sc_r_, "sc_r", nlinks, 1);
        sc_e = return_pointer_and_check_length(sc_e_, "sc_e", nlinks, 1);
        links = return_pointer_and_check_length(links_, "links", nlinks, 1);
        armlength = armlength_;
    };

    template<typename T>
    T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
    {
        #ifdef __CUDACC__
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

    int get_sc_r_from_arr(int i)
    {
        return sc_r[i];
    };

    int get_sc_e_from_arr(int i)
    {
        return sc_e[i];
    };
    int get_link_from_arr(int i)
    {
        return links[i];
    };

    CUDA_DEVICE int get_window(double t);
    CUDA_DEVICE Vec get_normal_unit_vec(double t, int link);
    CUDA_DEVICE double interpolate(double t, double *in_arr, int window, int major_ndim, int major_ind, int ndim, int pos);
    CUDA_DEVICE int get_link_ind(int link);
    CUDA_DEVICE int get_sc_ind(int sc);
    CUDA_DEVICE double get_light_travel_time(double t, int link);
    CUDA_DEVICE Vec get_pos(double t, int sc);
    CUDA_DEVICE void get_normal_unit_vec_ptr(Vec *vec, double t, int link);
    CUDA_DEVICE void get_pos_ptr(Vec *vec, double t, int sc);
    void get_light_travel_time_arr(double *ltt, double *t, int *link, int num);
    void get_pos_arr(double *pos_x, double *pos_y, double *pos_z, double *t, int *sc, int num);
    void get_normal_unit_vec_arr(double *normal_unit_vec_x, double *normal_unit_vec_y, double *normal_unit_vec_z, double *t, int *link, int num);
    void get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num);
    void get_normal_unit_vec_wrap(array_type<double>normal_unit_vec_x, array_type<double>normal_unit_vec_y, array_type<double>normal_unit_vec_z, array_type<double>t, array_type<int>link, int num);
    void get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num);
    void dealloc() {};
};

void detector_part(py::module &m);
// class AddOrbits{
//   public:
//     Orbits *orbits;
    
//     void add_orbit_information(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_)
//     {
//         if (orbits != NULL)
//         {
//             delete orbits;
//         }
//         orbits = new Orbits(dt_, N_, n_arr_, L_arr_, x_arr_, links_, sc_r_, sc_e_, armlength_);
//     };
//     void dealloc(){delete orbits;};
// };


#endif // __DETECTOR_HPP__