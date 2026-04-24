#ifndef __BINDING_HPP__
#define __BINDING_HPP__

#include "Detector.hpp"
#include "PSD.hpp"
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


void psd_likelihood_legacy_wrap(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                         array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all, 
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> sl1_all, array_type<double> kn_all, array_type<double> sl2_all, double df, int data_length, int num_data, int num_psds);

void get_psd_val_legacy_wrap(array_type<double> Sn_A_out, array_type<double> Sn_E_out, array_type<double> f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double sl1, double kn, double sl2, int num_f);

// Wrapper for psd_likelihood (same as legacy, exposed with different name for consistency)
void psd_likelihood_binding(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                         array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all, 
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> sl1_all, array_type<double> kn_all, array_type<double> sl2_all, double df, int data_length, int num_data, int num_psds);

// Wrapper for compute_logpdf
void compute_logpdf_binding(array_type<double> logpdf_out, array_type<int> component_index, array_type<double> points,
                    array_type<double> weights, array_type<double> mins, array_type<double> maxs, 
                    array_type<double> means, array_type<double> invcovs, array_type<double> dets, array_type<double> log_Js, 
                    int num_points, array_type<int> start_index, int num_components, int ndim);

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


// now add the OrbitsWrap class
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define OrbitsWrap OrbitsWrapGPU
#else
#define OrbitsWrap OrbitsWrapCPU
#endif

class OrbitsWrap {
  public:
    Orbits *orbits;
    OrbitsWrap(double sc_t0_, double sc_dt_, int sc_N_, double ltt_t0_, double ltt_dt_, int ltt_N_, array_type<double> n_arr_, array_type<double> ltt_arr_, array_type<double> x_arr_, array_type<int> links_, array_type<int> sc_r_, array_type<int> sc_e_, double armlength_)
    {

        double *_n_arr = return_pointer_and_check_length(n_arr_, "n_arr", sc_N_, 6 * 3);
        double *_ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", ltt_N_, 6);
        double *_x_arr = return_pointer_and_check_length(x_arr_, "x_arr", sc_N_, 3 * 3);

        int *_sc_r = return_pointer_and_check_length(sc_r_, "sc_r", 6, 1);
        int *_sc_e = return_pointer_and_check_length(sc_e_, "sc_e", 6, 1);
        int *_links = return_pointer_and_check_length(links_, "links", 6, 1);
        
        orbits = new Orbits(sc_t0_, sc_dt_, sc_N_, ltt_t0_, ltt_dt_, ltt_N_, _n_arr, _ltt_arr, _x_arr, _links,  _sc_r, _sc_e, armlength_);
    };
    ~OrbitsWrap(){
        delete orbits;
    };

    OrbitsWrap(const OrbitsWrap& other) {
        orbits = new Orbits(*other.orbits);
    }

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

// XYZ Sensitivity Matrix Wrap
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define XYZSensitivityMatrixWrap XYZSensitivityMatrixWrapGPU
#else
#define XYZSensitivityMatrixWrap XYZSensitivityMatrixWrapCPU
#endif

class XYZSensitivityMatrixWrap {
public:
    XYZSensitivityMatrix *sensitivity_matrix;

    XYZSensitivityMatrixWrap(array_type<double> averaged_ltts_arr_, array_type<double> delta_ltts_arr_, int n_times_, double armlength_, int generation_, bool spline_noise_)
    {
        double *_averaged_ltts_arr = return_pointer_and_check_length(averaged_ltts_arr_, "averaged_ltts_arr", n_times_, 6);
        double *_delta_ltts_arr = return_pointer_and_check_length(delta_ltts_arr_, "delta_ltts_arr", n_times_, 6);
        
        sensitivity_matrix = new XYZSensitivityMatrix(_averaged_ltts_arr, _delta_ltts_arr, n_times_, armlength_, generation_, spline_noise_);
    }

    ~XYZSensitivityMatrixWrap() {
        delete sensitivity_matrix;
    };

    XYZSensitivityMatrixWrap(const XYZSensitivityMatrixWrap& other) {
        sensitivity_matrix = new XYZSensitivityMatrix(*other.sensitivity_matrix);
    }

    void get_noise_tfs_wrap(array_type<double> freqs, 
                          array_type<double> oms_xx, array_type<std::complex<double>> oms_xy, array_type<std::complex<double>> oms_xz, array_type<double> oms_yy, array_type<std::complex<double>> oms_yz, array_type<double> oms_zz,
                          array_type<double> tm_xx, array_type<std::complex<double>> tm_xy, array_type<std::complex<double>> tm_xz, array_type<double> tm_yy, array_type<std::complex<double>> tm_yz, array_type<double> tm_zz,
                          int num_freqs, int num_times,
                          array_type<int> time_indices);
                          
    void psd_likelihood_wrap(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                             array_type<int> data_index_all, array_type<int> time_index_all,
                             array_type<double> Soms_d_in_all, array_type<double> Sa_a_in_all, 
                             array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> slope_1_all, array_type<double> f_knee_all, array_type<double> slope_2_all, 
                             array_type<double> spline_in_isi_oms_all, array_type<double> spline_in_testmass_all,
                             double differential_component, int num_freqs, int num_times, 
                             array_type<bool> dips_mask, int num_psds);

    void get_noise_covariance_wrap(
        array_type<double> freqs, array_type<int> time_indices,
        double Soms_d_in, double Sa_a_in,
        double Amp, double alpha, double slope_1, double f_knee, double slope_2,
        array_type<double> spline_in_isi_oms_arr, array_type<double> spline_in_testmass_arr,
        array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
        array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
        int num_freqs, int num_times);

    void get_inverse_det_wrap(
        array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
        array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
        array_type<double> i00_arr, array_type<std::complex<double>> i01_arr, array_type<std::complex<double>> i02_arr,
        array_type<double> i11_arr, array_type<std::complex<double>> i12_arr, array_type<double> i22_arr,
        array_type<double> det_arr,
        int num);
    
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