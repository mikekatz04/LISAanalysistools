#ifndef __PSD_HPP__
#define __PSD_HPP__

#include "gbt_global.h"
#include "cuda_complex.hpp"
#include <iostream>

#if defined(__CUDACC__) || defined(__CUDA_COMPILATION__)
#define XYZSensitivityMatrix XYZSensitivityMatrixGPU
#define NoiseLevels NoiseLevelsGPU
#else
#define XYZSensitivityMatrix XYZSensitivityMatrixCPU
#define NoiseLevels NoiseLevelsCPU
#endif

#define Clight 299792458.

class NoiseLevels {
public:
    bool return_relative_frequency;
    double f_knee_tm;
    double f_break_tm;
    double f_knee_oms;
    
    NoiseLevels(bool return_relative_frequency_, double f_knee_tm_, double f_break_tm_, double f_knee_oms_){
        return_relative_frequency = return_relative_frequency_;
        f_knee_tm = f_knee_tm_;
        f_break_tm = f_break_tm_;
        f_knee_oms = f_knee_oms_;
    };

    CUDA_DEVICE void get_testmass_noise(double *S_tm, double f, double Sa_a_in);
    CUDA_DEVICE void get_isi_oms_noise(double *S_isi_oms, double f, double Soms_d_in);
    CUDA_DEVICE void get_galactic_foreground(double *S_gal, double f, double Amp, double alpha, double slope_1, double f_knee, double slope_2);

    void dealloc() {};
};

class XYZSensitivityMatrix {
    public:
    double *averaged_ltts_arr;
    double *delta_ltts_arr;
    int n_times;
    double armlength;
    int n_links;
    int left_mosas[3];
    int generation;
    NoiseLevels noise_levels;

    XYZSensitivityMatrix(double *averaged_ltts_arr_, double *delta_ltts_arr_, int n_times_, double armlength_, int generation_)
        : noise_levels(true, 0.4e-3, 8e-3, 2.0e-3)
    {
        averaged_ltts_arr = averaged_ltts_arr_; // flattened array of size Ntimes * 6
        delta_ltts_arr = delta_ltts_arr_; // flattened array of size Ntimes * 6
        armlength = armlength_;
        generation = generation_;
        n_links = 6;
        n_times = n_times_;
        // Initialize array manually or via loop
        left_mosas[0] = 12; 
        left_mosas[1] = 23; 
        left_mosas[2] = 31;
    };
    CUDA_DEVICE int get_adjacent_mosa(int mosa);
    CUDA_DEVICE gcmplx::complex<double> oms_xx_unequal_armlength(double f, double avg_d_ij, double avg_d_ik);
    CUDA_DEVICE gcmplx::complex<double> oms_xy_unequal_armlength(double f, double avg_d_ij, double avg_d_ik, double avg_d_jk, double delta_d_ij);
    CUDA_DEVICE gcmplx::complex<double> tm_xx_unequal_armlength(double f, double avg_d_ij, double avg_d_ik);
    CUDA_DEVICE gcmplx::complex<double> tm_xy_unequal_armlength(double f, double avg_d_ij, double avg_d_ik, double avg_d_jk, double delta_d_ij);

    CUDA_DEVICE void get_noise_tfs(double f, 
                                  gcmplx::complex<double> *oms_xx, gcmplx::complex<double> *oms_xy, gcmplx::complex<double> *oms_xz, gcmplx::complex<double> *oms_yy, gcmplx::complex<double> *oms_yz, gcmplx::complex<double> *oms_zz,
                                  gcmplx::complex<double> *tm_xx, gcmplx::complex<double> *tm_xy, gcmplx::complex<double> *tm_xz, gcmplx::complex<double> *tm_yy, gcmplx::complex<double> *tm_yz, gcmplx::complex<double> *tm_zz,
                                  int time_index); 

    CUDA_DEVICE void get_noise_covariance(
        double f, int time_index,
        double Soms_d_in, double Sa_a_in,
        double Amp, double alpha, double slope_1, double f_knee, double slope_2,
        double *c00, gcmplx::complex<double> *c01, gcmplx::complex<double> *c02,
        double *c11, gcmplx::complex<double> *c12, double *c22);
                
    void get_noise_tfs_arr(double *freqs, 
                          gcmplx::complex<double> *oms_xx, gcmplx::complex<double> *oms_xy, gcmplx::complex<double> *oms_xz, gcmplx::complex<double> *oms_yy, gcmplx::complex<double> *oms_yz, gcmplx::complex<double> *oms_zz,
                          gcmplx::complex<double> *tm_xx, gcmplx::complex<double> *tm_xy, gcmplx::complex<double> *tm_xz, gcmplx::complex<double> *tm_yy, gcmplx::complex<double> *tm_yz, gcmplx::complex<double> *tm_zz,
                          int num,
                          int *time_indices);

    void psd_likelihood_wrap(double *like_contrib_final, double *f_arr, gcmplx::complex<double> *data, 
                             int *data_index_all, int *time_index_all,
                             double *Soms_d_in_all, double *Sa_a_in_all, 
                             double *Amp_all, double *alpha_all, double *slope_1_all, double *f_knee_all, double *slope_2_all, 
                             double df, int num_freqs, int num_times, int num_psds);

    // Noise covariance matrix computation
    void get_noise_covariance_arr(
        double *freqs, int *time_indices,
        double Soms_d_in, double Sa_a_in,
        double Amp, double alpha, double slope_1, double f_knee, double slope_2,
        double *c00_arr, gcmplx::complex<double> *c01_arr, gcmplx::complex<double> *c02_arr,
        double *c11_arr, gcmplx::complex<double> *c12_arr, double *c22_arr,
        int num_freqs, int num_times);

    void get_inverse_det_arr(
    double *c00_arr, gcmplx::complex<double> *c01_arr, gcmplx::complex<double> *c02_arr,
    double *c11_arr, gcmplx::complex<double> *c12_arr, double *c22_arr,
    double *i00_arr, gcmplx::complex<double> *i01_arr, gcmplx::complex<double> *i02_arr,
    double *i11_arr, gcmplx::complex<double> *i12_arr, double *i22_arr,
    double *det_arr,
    int num);

    void dealloc() {};

};

// from Sangria setup
void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim);

// LEGACY FUNCTIONS USED FOR SANGRIA, KEPT FOR COMPATIBILITY
void psd_likelihood_wrap(double* like_contrib_final, double *f_arr, cmplx* data, int* data_index_all, double* A_Soms_d_in_all, double* A_Sa_a_in_all, double* E_Soms_d_in_all, double* E_Sa_a_in_all, 
                    double* Amp_all, double* alpha_all, double* sl1_all, double* kn_all, double* sl2_all, double df, int data_length, int num_data, int num_psds);

void get_psd_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double sl1, double kn, double sl2, int num_f);

#endif // __PSD_HPP__