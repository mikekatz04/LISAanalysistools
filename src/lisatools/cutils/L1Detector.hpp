#ifndef __L1DETECTOR_HPP__
#define __L1DETECTOR_HPP__

#include "gbt_global.h"
#include "Detector.hpp"
#include "cuda_complex.hpp"
#include <iostream>

#if defined(__CUDACC__) || defined(__CUDA_COMPILATION__)
#define L1Orbits L1OrbitsGPU
#else
#define L1Orbits L1OrbitsCPU
#endif

#define Clight 299792458.

class L1Orbits
{
public:
    double sc_t0;
    double sc_dt;
    int sc_N;
    double ltt_t0;
    double ltt_dt;
    int ltt_N;
    double *n_arr;
    double *ltt_arr;
    double *x_arr;
    int nlinks;
    int nspacecraft;
    double armlength;
    int *links;
    int *sc_r;
    int *sc_e;
    
    L1Orbits(double sc_t0_, double sc_dt_, int sc_N_, double ltt_t0_, double ltt_dt_, int ltt_N_, double *n_arr_, double *ltt_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_)
    {
        sc_t0 = sc_t0_;
        sc_dt = sc_dt_;
        sc_N = sc_N_;

        ltt_t0 = ltt_t0_;
        ltt_dt = ltt_dt_;
        ltt_N = ltt_N_;

        n_arr = n_arr_;
        ltt_arr = ltt_arr_;
        x_arr = x_arr_;
        nlinks = 6;
        nspacecraft = 3;

        sc_r = sc_r_;
        sc_e = sc_e_;
        links = links_;
        armlength = armlength_;

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

    CUDA_DEVICE int get_window(double t, double t0, double dt, int N);
    CUDA_DEVICE Vec get_normal_unit_vec(double t, int link);
    CUDA_DEVICE double interpolate(double t, double *in_arr, double t0, double dt, int window, int major_ndim, int major_ind, int ndim, int pos);
    CUDA_DEVICE int get_link_ind(int link);
    CUDA_DEVICE int get_sc_ind(int sc);
    CUDA_DEVICE double get_light_travel_time(double t, int link);
    CUDA_DEVICE Vec get_pos(double t, int sc);
    CUDA_DEVICE void get_normal_unit_vec_ptr(Vec *vec, double t, int link);
    CUDA_DEVICE void get_pos_ptr(Vec *vec, double t, int sc);
    void get_light_travel_time_arr(double *ltt, double *t, int *link, int num);
    void get_pos_arr(double *pos_x, double *pos_y, double *pos_z, double *t, int *sc, int num);
    void get_normal_unit_vec_arr(double *normal_unit_vec_x, double *normal_unit_vec_y, double *normal_unit_vec_z, double *t, int *link, int num);
    void dealloc() {};
};

#endif // __L1DETECTOR_HPP__
    