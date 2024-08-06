import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from lisatools.utils.pointeradjust import wrapper
from libc.stdint cimport uintptr_t


cdef extern from "../include/Detector.hpp":
    cdef cppclass VecWrap "Vec":
        double x
        double y
        double z
        VecWrap(double x_, double y_, double z_) except+

    cdef cppclass OrbitsWrap "Orbits":
        OrbitsWrap(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_) except+
        int get_window(double t) except+
        void get_normal_unit_vec_ptr(VecWrap *vec, double t, int link)
        int get_link_ind(int link) except+
        int get_sc_ind(int sc) except+
        double get_light_travel_time(double t, int link) except+
        VecWrap get_pos_ptr(VecWrap* out, double t, int sc) except+
        void dealloc();
        

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
            links,
            sc_r, 
            sc_e
        ), tkwargs = wrapper(*args, **kwargs)

        self.dt = dt
        self.N = N 
        self.n_arr = n_arr
        self.L_arr = L_arr 
        self.x_arr = x_arr
        self.links = links
        self.sc_r = sc_r
        self.sc_e = sc_e

        cdef size_t n_arr_in = n_arr
        cdef size_t L_arr_in = L_arr
        cdef size_t x_arr_in = x_arr
        cdef size_t links_in = links
        cdef size_t sc_r_in = sc_r
        cdef size_t sc_e_in = sc_e
        
        self.g = new OrbitsWrap(
            dt,
            N,
            <double*> n_arr_in,
            <double*> L_arr_in, 
            <double*> x_arr_in, 
            <int*> links_in, 
            <int*> sc_r_in, 
            <int*> sc_e_in
        )

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __reduce__(self):
        return (rebuild, (self.dt, self.N, self.n_arr, self.L_arr, self.x_arr, self.links, self.sc_r, self.sc_e,))

    def get_window(self, t: float) -> int:
        return self.g.get_window(t)

    def get_normal_unit_vec_single(self, t: float, link: int) -> np.ndarray:
        cdef VecWrap *out = new VecWrap(0.0, 0.0, 0.0)
        self.g.get_normal_unit_vec_ptr(out, t, link)

        return np.array([out.x, out.y, out.z])

    def get_normal_unit_vec_arr(self, t: np.ndarray, link: int) -> np.ndarray:
        output = np.zeros((len(t), 3), dtype=float)
        assert t.ndim == 1
        for i in range(len(t)):
            output[i] = self.get_normal_unit_vec_single(t[i], link)

        return output

    def get_normal_unit_vec(self, t: np.ndarray | float, link: int) -> np.ndarray:

        if isinstance(t, float):
            return self.get_normal_unit_vec_single(t, link)
        elif isinstance(t, np.ndarray):
            return self.get_normal_unit_vec_arr(t, link)

    def get_link_ind(self, link: int) -> int:
        return self.g.get_link_ind(link)

    def get_sc_ind(self, sc: int) -> int:
        return self.g.get_sc_ind(sc)

    def get_light_travel_time_single(self, t: float, link: int) -> float:
        return self.g.get_light_travel_time(t, link)

    def get_light_travel_time_arr(self, t: np.ndarray, link: int) -> np.ndarray:
        output = np.zeros((len(t),), dtype=float)
        assert t.ndim == 1
        for i in range(len(t)):
            output[i] = self.get_light_travel_time_single(t[i], link)

        return output

    def get_light_travel_time(self, t: np.ndarray | float, link: int) -> np.ndarray:

        if isinstance(t, float):
            print("t", t)
            return self.get_light_travel_time_single(t, link)
        elif isinstance(t, np.ndarray):
            return self.get_light_travel_time_arr(t, link)
    
    def get_pos_single(self, t: float, sc: int) -> np.ndarray:
  
        cdef VecWrap *out = new VecWrap(0.0, 0.0, 0.0)
        self.g.get_pos_ptr(out, t, sc)

        return np.array([out.x, out.y, out.z])

    def get_pos_arr(self, t: np.ndarray, sc: int) -> np.ndarray:
        output = np.zeros((len(t), 3), dtype=float)
        assert t.ndim == 1
        for i in range(len(t)):
            output[i] = self.get_pos_single(t[i], sc)

        return output

    def get_pos(self, t: np.ndarray | float, sc: int) -> np.ndarray:

        if isinstance(t, float):
            return self.get_pos_single(t, sc)
        elif isinstance(t, np.ndarray):
            return self.get_pos_arr(t, sc)

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g
    
        
def rebuild(dt,
    N, 
    n_arr,
    L_arr, 
    x_arr,
    links,
    sc_r, 
    sc_e
):
    c = pycppDetector(
        dt,
        N, 
        n_arr,
        L_arr, 
        x_arr,
        links,
        sc_r, 
        sc_e
    )
    return c