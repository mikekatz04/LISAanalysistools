import cython
import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t


cdef extern from "Detector.hpp":
    cdef cppclass VecWrap "Vec":
        double x
        double y
        double z
        VecWrap(double x_, double y_, double z_) except+

    cdef cppclass OrbitsWrap "Orbits":
        OrbitsWrap(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_) except+
        int get_window(double t) except+
        void get_normal_unit_vec_ptr(VecWrap *vec, double t, int link)
        int get_link_ind(int link) except+
        int get_sc_ind(int sc) except+
        double get_light_travel_time(double t, int link) except+
        VecWrap get_pos_ptr(VecWrap* out, double t, int sc) except+
        void get_light_travel_time_arr(double *ltt, double *t, int *link, int num) except+
        void dealloc();
        void get_pos_arr(double *pos_x, double *pos_y, double *pos_z, double *t, int *sc, int num) except+
        void get_normal_unit_vec_arr(double *normal_unit_vec_x, double *normal_unit_vec_y, double *normal_unit_vec_z, double *t, int *link, int num) except+
        int get_sc_r_from_arr(int i) except+
        int get_sc_e_from_arr(int i) except+
        int get_link_from_arr(int i) except+
        


cdef class pycppDetector:
    cdef OrbitsWrap *g
    cdef double dt
    cdef int N
    cdef size_t n_arr
    cdef size_t L_arr
    cdef size_t x_arr
    cdef size_t links
    cdef size_t sc_r
    cdef size_t sc_e
    cdef double armlength