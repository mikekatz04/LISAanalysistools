import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from lisatools.utils.pointeradjust import wrapper

cdef extern from "../include/Detector.hpp":
    cdef cppclass VecWrap "Vec":
        VecWrap(double x_, double y_, double z_) except+

    cdef cppclass OrbitsWrap "Orbits":
        OrbitsWrap(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_) except+
        int get_window(double t) except+
        VecWrap get_normal_unit_vec(double t, int link) except+
        double interpolate(double t, double *in_arr, int down_ind, int up_ind, int ndim, int pos) except+
        int get_link_ind(int link) except+
        int get_sc_ind(int sc) except+
        double get_light_travel_time(double t, int link) except+
        VecWrap get_pos(double t, int sc) except+

cdef class pycppDetector:
    cdef OrbitsWrap *g

    def __cinit__(self, 
        *args, 
        **kwargs
    ):
        (
            dt,
            N, 
            n_arr,
            L_arr, 
            x_arr,
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t n_arr_in = n_arr
        cdef size_t L_arr_in = L_arr
        cdef size_t x_arr_in = x_arr
        
        self.g = new OrbitsWrap(
            dt,
            N,
            <double*> n_arr_in,
            <double*> L_arr_in, 
            <double*> x_arr_in, 
        )
    
        