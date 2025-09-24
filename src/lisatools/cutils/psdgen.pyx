import numpy as np
cimport numpy as np
from libc.stdint cimport uintptr_t
from gpubackendtools import wrapper

from libcpp cimport bool
assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "PSD.hpp":
    ctypedef void* cmplx 'cmplx'
    
    void psd_likelihood_wrap(double* like_contrib_final, double *f_arr, cmplx* data, int* data_index_all, double* A_Soms_d_in_all, double* A_Sa_a_in_all, double* E_Soms_d_in_all, double* E_Sa_a_in_all, 
                    double* Amp_all, double* alpha_all, double* sl1_all, double* kn_all, double* sl2_all, double df, int data_length, int num_data, int num_psds) except+
        
    void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim) except + 
    void get_psd_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double sl1, double kn, double sl2, int num_f) except +

def psd_likelihood(*args, **kwargs):

    (
        like_contrib_final, f_arr, data, data_index_all,  A_Soms_d_in_all,  A_Sa_a_in_all,  E_Soms_d_in_all,  E_Sa_a_in_all, 
                     Amp_all,  alpha_all,  sl1_all,  kn_all, sl2_all, df, data_length, num_data, num_psds
    ), tkwargs = wrapper(*args, **kwargs)

    cdef size_t like_contrib_final_in = like_contrib_final
    cdef size_t f_arr_in = f_arr
    cdef size_t data_in = data
    cdef size_t data_index_all_in = data_index_all
    cdef size_t A_Soms_d_in_all_in = A_Soms_d_in_all
    cdef size_t A_Sa_a_in_all_in = A_Sa_a_in_all
    cdef size_t E_Soms_d_in_all_in = E_Soms_d_in_all
    cdef size_t E_Sa_a_in_all_in = E_Sa_a_in_all
    cdef size_t Amp_all_in = Amp_all
    cdef size_t alpha_all_in = alpha_all
    cdef size_t sl1_all_in = sl1_all
    cdef size_t kn_all_in = kn_all
    cdef size_t sl2_all_in = sl2_all

    psd_likelihood_wrap(<double*> like_contrib_final_in, <double*> f_arr_in, <cmplx*> data_in, <int*> data_index_all_in, <double*> A_Soms_d_in_all_in, <double*> A_Sa_a_in_all_in, <double*> E_Soms_d_in_all_in, <double*> E_Sa_a_in_all_in, 
                    <double*> Amp_all_in, <double*> alpha_all_in, <double*> sl1_all_in, <double*> kn_all_in, <double*> sl2_all_in, df, data_length, num_data, num_psds)

def compute_logpdf(*args, **kwargs):

    (logpdf_out, component_index, points,
                    weights, mins, maxs, means, invcovs, dets, log_Js, 
                    num_points, start_index, num_components, ndim
    ), tkwargs = wrapper(*args, **kwargs)

    cdef size_t logpdf_out_in = logpdf_out
    cdef size_t component_index_in = component_index
    cdef size_t points_in = points
    cdef size_t weights_in = weights
    cdef size_t mins_in = mins
    cdef size_t maxs_in = maxs
    cdef size_t means_in = means
    cdef size_t invcovs_in = invcovs
    cdef size_t dets_in = dets
    cdef size_t log_Js_in = log_Js
    cdef size_t start_index_in = start_index

    compute_logpdf_wrap(<double*>logpdf_out_in, <int *>component_index_in, <double*>points_in,
                    <double *>weights_in, <double*>mins_in, <double*>maxs_in, <double*>means_in, <double*>invcovs_in, <double*>dets_in, <double *>log_Js_in, 
                    num_points, <int *>start_index_in, num_components, ndim)


def get_psd_val(*args, **kwargs):
    
    (Sn_A_out, Sn_E_out, f_arr, A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in,
                               Amp, alpha, sl1, kn, sl2, num_f
    ), tkwargs = wrapper(*args, **kwargs)
    cdef size_t Sn_A_out_in = Sn_A_out
    cdef size_t Sn_E_out_in = Sn_E_out
    cdef size_t f_arr_in = f_arr

    get_psd_val_wrap(<double *>Sn_A_out_in, <double *>Sn_E_out_in, <double *>f_arr_in, A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in,
                               Amp, alpha, sl1, kn, sl2, num_f)

