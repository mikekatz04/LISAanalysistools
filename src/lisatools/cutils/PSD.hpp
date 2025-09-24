#ifndef __PSD_HPP__
#define __PSD_HPP__

#include "global.hpp"

// void specialty_piece_wise_likelihoods_wrap(
//     double* lnL,
//     cmplx* data,
//     double* noise,
//     int* data_index,
//     int* noise_index,
//     int* start_inds,
//     int* lengths,
//     double df, 
//     int num_parts,
//     int start_freq_ind,
//     int data_length,
//     int tdi_channel_setup,
//     bool do_synchronize,
//     int num_data, 
//     int num_noise
// );


void psd_likelihood_wrap(double* like_contrib_final, double *f_arr, cmplx* data, int* data_index_all, double* A_Soms_d_in_all, double* A_Sa_a_in_all, double* E_Soms_d_in_all, double* E_Sa_a_in_all, 
                    double* Amp_all, double* alpha_all, double* sl1_all, double* kn_all, double* sl2_all, double df, int data_length, int num_data, int num_psds);

void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim);

void get_psd_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double sl1, double kn, double sl2, int num_f);

// void get_lisasens_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
//                                double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f);

#endif // __PSD_HPP__