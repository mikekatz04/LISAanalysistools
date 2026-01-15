#include "Detector.hpp"
#include "L1Detector.hpp"
#include "PSD.hpp"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding.hpp"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#endif

namespace py = pybind11;


void OrbitsWrap::get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num)
{
    orbits->get_light_travel_time_arr(
        return_pointer_and_check_length(ltt, "ltt", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(link, "sc", num, 1),
        num
    );
}


void OrbitsWrap::get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num)
{
    orbits->get_pos_arr(
        return_pointer_and_check_length(pos_x, "pos_x", num, 1),
        return_pointer_and_check_length(pos_y, "pos_y", num, 1),
        return_pointer_and_check_length(pos_z, "pos_z", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(sc, "sc", num, 1),
        num
    );
}


void OrbitsWrap::get_normal_unit_vec_wrap(array_type<double>normal_unit_vec_x, array_type<double>normal_unit_vec_y, array_type<double>normal_unit_vec_z, array_type<double>t, array_type<int>link, int num)
{
    
// #ifdef __CUDACC__
    orbits->get_normal_unit_vec_arr(
        return_pointer_and_check_length(normal_unit_vec_x, "n_arr_x", num, 1),
        return_pointer_and_check_length(normal_unit_vec_y, "n_arr_y", num, 1),
        return_pointer_and_check_length(normal_unit_vec_z, "n_arr_z", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(link, "link", num, 1),
        num
    );
}

void check_12()
{
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    printf("CHECK 12 GOOD\n");
#else
    printf("CHECK 12 BAD\n");
#endif 
}

void check_orbits(Orbits *orbits)
{
    printf("%e\n", orbits->x_arr[0]);
}

// L1 orbits wrap class

void L1OrbitsWrap::get_light_travel_time_wrap(array_type<double> ltt, array_type<double> t, array_type<int> link, int num)
{
    orbits->get_light_travel_time_arr(
        return_pointer_and_check_length(ltt, "ltt", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(link, "sc", num, 1),
        num
    );
}


void L1OrbitsWrap::get_pos_wrap(array_type<double> pos_x, array_type<double> pos_y, array_type<double> pos_z, array_type<double> t, array_type<int> sc, int num)
{
    orbits->get_pos_arr(
        return_pointer_and_check_length(pos_x, "pos_x", num, 1),
        return_pointer_and_check_length(pos_y, "pos_y", num, 1),
        return_pointer_and_check_length(pos_z, "pos_z", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(sc, "sc", num, 1),
        num
    );
}


void L1OrbitsWrap::get_normal_unit_vec_wrap(array_type<double>normal_unit_vec_x, array_type<double>normal_unit_vec_y, array_type<double>normal_unit_vec_z, array_type<double>t, array_type<int>link, int num)
{
    
// #ifdef __CUDACC__
    orbits->get_normal_unit_vec_arr(
        return_pointer_and_check_length(normal_unit_vec_x, "n_arr_x", num, 1),
        return_pointer_and_check_length(normal_unit_vec_y, "n_arr_y", num, 1),
        return_pointer_and_check_length(normal_unit_vec_z, "n_arr_z", num, 1),
        return_pointer_and_check_length(t, "t", num, 1),
        return_pointer_and_check_length(link, "link", num, 1),
        num
    );
}


void check_L1orbits(L1Orbits *orbits)
{
    printf("%e\n", orbits->x_arr[0]);
}

void XYZSensitivityMatrixWrap::get_noise_tfs_wrap(array_type<double> freqs, 
                          array_type<std::complex<double>> oms_xx, array_type<std::complex<double>> oms_xy, array_type<std::complex<double>> oms_xz, array_type<std::complex<double>> oms_yy, array_type<std::complex<double>> oms_yz, array_type<std::complex<double>> oms_zz,
                          array_type<std::complex<double>> tm_xx, array_type<std::complex<double>> tm_xy, array_type<std::complex<double>> tm_xz, array_type<std::complex<double>> tm_yy, array_type<std::complex<double>> tm_yz, array_type<std::complex<double>> tm_zz,
                          int num,
                          array_type<int> time_indices)
{
    sensitivity_matrix->get_noise_tfs_arr(
        return_pointer_and_check_length(freqs, "freqs", num, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_xx, "oms_xx", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_xy, "oms_xy", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_xz, "oms_xz", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_yy, "oms_yy", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_yz, "oms_yz", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(oms_zz, "oms_zz", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_xx, "tm_xx", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_xy, "tm_xy", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_xz, "tm_xz", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_yy, "tm_yy", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_yz, "tm_yz", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(tm_zz, "tm_zz", num, 1)),
        num,
        return_pointer_and_check_length(time_indices, "time_indices", num, 1)
    );
}

void XYZSensitivityMatrixWrap::psd_likelihood_wrap(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                          array_type<int> data_index_all, array_type<int> time_index_all,
                          array_type<double> Soms_d_in_all, array_type<double> Sa_a_in_all, 
                          array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> slope_1_all, array_type<double> f_knee_all, array_type<double> slope_2_all, 
                          double df, int num_freqs, int num_times, int num_psds)
{
    int total_tf_pairs = num_times * num_freqs;
    sensitivity_matrix->psd_likelihood_wrap(
        return_pointer_and_check_length(like_contrib_final, "like_contrib_final", num_psds, 1),
        return_pointer_and_check_length(f_arr, "f_arr", num_freqs, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(data, "data", num_psds * 3 * total_tf_pairs, 1)),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_psds, 1),
        return_pointer_and_check_length(time_index_all, "time_index_all", num_times, 1),
        return_pointer_and_check_length(Soms_d_in_all, "Soms_d_in_all", num_psds, 1),
        return_pointer_and_check_length(Sa_a_in_all, "Sa_a_in_all", num_psds, 1),
        return_pointer_and_check_length(Amp_all, "Amp_all", num_psds, 1),
        return_pointer_and_check_length(alpha_all, "alpha_all", num_psds, 1),
        return_pointer_and_check_length(slope_1_all, "slope_1_all", num_psds, 1),
        return_pointer_and_check_length(f_knee_all, "f_knee_all", num_psds, 1),
        return_pointer_and_check_length(slope_2_all, "slope_2_all", num_psds, 1),
        df, num_freqs, num_times, num_psds
    );
}

void XYZSensitivityMatrixWrap::get_noise_covariance_wrap(
    array_type<double> freqs, array_type<int> time_indices,
    double Soms_d_in, double Sa_a_in,
    double Amp, double alpha, double slope_1, double f_knee, double slope_2,
    array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
    array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
    int num_freqs, int num_times)
{
    int total_size = num_freqs * num_times;
    sensitivity_matrix->get_noise_covariance_arr(
        return_pointer_and_check_length(freqs, "freqs", num_freqs, 1),
        return_pointer_and_check_length(time_indices, "time_indices", num_times, 1),
        Soms_d_in, Sa_a_in,
        Amp, alpha, slope_1, f_knee, slope_2,
        return_pointer_and_check_length(c00_arr, "c00_arr", total_size, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c01_arr, "c01_arr", total_size, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c02_arr, "c02_arr", total_size, 1)),
        return_pointer_and_check_length(c11_arr, "c11_arr", total_size, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c12_arr, "c12_arr", total_size, 1)),
        return_pointer_and_check_length(c22_arr, "c22_arr", total_size, 1),
        num_freqs, num_times
    );
}

void XYZSensitivityMatrixWrap::get_inverse_logdet_wrap(
    array_type<double> c00_arr, array_type<std::complex<double>> c01_arr, array_type<std::complex<double>> c02_arr,
    array_type<double> c11_arr, array_type<std::complex<double>> c12_arr, array_type<double> c22_arr,
    array_type<double> i00_arr, array_type<std::complex<double>> i01_arr, array_type<std::complex<double>> i02_arr,
    array_type<double> i11_arr, array_type<std::complex<double>> i12_arr, array_type<double> i22_arr,
    array_type<double> log_det_arr,
    int num)
{
    sensitivity_matrix->get_inverse_logdet_arr(
        return_pointer_and_check_length(c00_arr, "c00_arr", num, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c01_arr, "c01_arr", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c02_arr, "c02_arr", num, 1)),
        return_pointer_and_check_length(c11_arr, "c11_arr", num, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(c12_arr, "c12_arr", num, 1)),
        return_pointer_and_check_length(c22_arr, "c22_arr", num, 1),
        return_pointer_and_check_length(i00_arr, "i00_arr", num, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(i01_arr, "i01_arr", num, 1)),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(i02_arr, "i02_arr", num, 1)),
        return_pointer_and_check_length(i11_arr, "i11_arr", num, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_pointer_and_check_length(i12_arr, "i12_arr", num, 1)),
        return_pointer_and_check_length(i22_arr, "i22_arr", num, 1),
        return_pointer_and_check_length(log_det_arr, "log_det_arr", num, 1),
        num
    );
}


std::string get_module_path() {
    // Acquire the GIL if it's not already held (safe to call multiple times)
    py::gil_scoped_acquire acquire;

    // Import the module by its name
    // Note: The module name here ("pycppdetector") must match the name used in PYBIND11_MODULE
    py::object module = py::module::import("pycppdetector");

    // Access the __file__ attribute and cast it to a C++ string
    try {
        std::string path = module.attr("__file__").cast<std::string>();
        return path;
    } catch (const py::error_already_set& e) {
        // Handle the error if __file__ attribute is missing (e.g., if module is a namespace package)
        std::cerr << "Error getting __file__ attribute: " << e.what() << std::endl;
        return "";
    }
}

// PYBIND11_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt
void detector_part(py::module &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<OrbitsWrap>(m, "OrbitsWrapGPU")
#else
    py::class_<OrbitsWrap>(m, "OrbitsWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<double, int, array_type<double>, array_type<double>, array_type<double>, array_type<int>, array_type<int>, array_type<int>, double>(), 
         py::arg("dt"), py::arg("N"), py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"), py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))
    // Bind member functions
    .def("get_light_travel_time_wrap", &OrbitsWrap::get_light_travel_time_wrap, "Get the light travel time.")
    .def("get_pos_wrap", &OrbitsWrap::get_pos_wrap, "Get spacecraft position.")
    .def("get_normal_unit_vec_wrap", &OrbitsWrap::get_normal_unit_vec_wrap, "Get link normal vector.")
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("orbits", &OrbitsWrap::orbits)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<Orbits>(m, "OrbitsGPU")
#else
    py::class_<Orbits>(m, "OrbitsCPU")
#endif

    // Bind the constructor
    .def(py::init<double, int, double *, double *, double *, int *, int *, int *, double>(),
         py::arg("dt"), py::arg("N"), py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"), py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))

    ;
}

template<typename T>
T* return_ptr(array_type<T> input1, std::string name, int N, int multiplier)
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

void psd_likelihood_legacy_wrap(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                         array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all, 
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> sl1_all, array_type<double> kn_all, array_type<double> sl2_all, double df, int data_length, int num_data, int num_psds)
{
    psd_likelihood_wrap(
        return_ptr(like_contrib_final, "like_contrib_final", num_psds, 1),
        return_ptr(f_arr, "f_arr", data_length, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_ptr(data, "data", data_length * num_data, 1)),
        return_ptr(data_index_all, "data_index_all", num_psds, 1),
        return_ptr(Soms_d_in_all, "Soms_d_in_all", num_psds, 1),
        return_ptr(Sa_a_in_all, "Sa_a_in_all", num_psds, 1),
        return_ptr(E_Soms_d_in_all, "E_Soms_d_in_all", num_psds, 1),
        return_ptr(E_Sa_a_in_all, "E_Sa_a_in_all", num_psds, 1),
        return_ptr(Amp_all, "Amp_all", num_psds, 1),
        return_ptr(alpha_all, "alpha_all", num_psds, 1),
        return_ptr(sl1_all, "sl1_all", num_psds, 1),
        return_ptr(kn_all, "kn_all", num_psds, 1),
        return_ptr(sl2_all, "sl2_all", num_psds, 1),
        df, data_length, num_data, num_psds
    );

}

void get_psd_val_legacy_wrap(array_type<double> Sn_A_out, array_type<double> Sn_E_out, array_type<double> f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double sl1, double kn, double sl2, int num_f)
{
    get_psd_val_wrap(
        return_ptr(Sn_A_out, "Sn_A_out", num_f, 1),
        return_ptr(Sn_E_out, "Sn_E_out", num_f, 1),
        return_ptr(f_arr, "f_arr", num_f, 1),
        A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in,
        Amp, alpha, sl1, kn, sl2, num_f
    );
}

void psd_likelihood_binding(array_type<double> like_contrib_final, array_type<double> f_arr, array_type<std::complex<double>> data, 
                         array_type<int> data_index_all, array_type<double>Soms_d_in_all, array_type<double>Sa_a_in_all, array_type<double>E_Soms_d_in_all, array_type<double>E_Sa_a_in_all, 
                         array_type<double> Amp_all, array_type<double> alpha_all, array_type<double> sl1_all, array_type<double> kn_all, array_type<double> sl2_all, double df, int data_length, int num_data, int num_psds)
{
    psd_likelihood_wrap(
        return_ptr(like_contrib_final, "like_contrib_final", num_psds, 1),
        return_ptr(f_arr, "f_arr", data_length, 1),
        reinterpret_cast<gcmplx::complex<double>*>(return_ptr(data, "data", data_length * num_data, 1)),
        return_ptr(data_index_all, "data_index_all", num_psds, 1),
        return_ptr(Soms_d_in_all, "Soms_d_in_all", num_psds, 1),
        return_ptr(Sa_a_in_all, "Sa_a_in_all", num_psds, 1),
        return_ptr(E_Soms_d_in_all, "E_Soms_d_in_all", num_psds, 1),
        return_ptr(E_Sa_a_in_all, "E_Sa_a_in_all", num_psds, 1),
        return_ptr(Amp_all, "Amp_all", num_psds, 1),
        return_ptr(alpha_all, "alpha_all", num_psds, 1),
        return_ptr(sl1_all, "sl1_all", num_psds, 1),
        return_ptr(kn_all, "kn_all", num_psds, 1),
        return_ptr(sl2_all, "sl2_all", num_psds, 1),
        df, data_length, num_data, num_psds
    );
}

void compute_logpdf_binding(array_type<double> logpdf_out, array_type<int> component_index, array_type<double> points,
                    array_type<double> weights, array_type<double> mins, array_type<double> maxs, 
                    array_type<double> means, array_type<double> invcovs, array_type<double> dets, array_type<double> log_Js, 
                    int num_points, array_type<int> start_index, int num_components, int ndim)
{
    compute_logpdf_wrap(
        return_ptr(logpdf_out, "logpdf_out", num_points, 1),
        return_ptr(component_index, "component_index", num_points, 1),
        return_ptr(points, "points", num_points * ndim, 1),
        return_ptr(weights, "weights", num_components, 1),
        return_ptr(mins, "mins", num_components * ndim, 1),
        return_ptr(maxs, "maxs", num_components * ndim, 1),
        return_ptr(means, "means", num_components * ndim, 1),
        return_ptr(invcovs, "invcovs", num_components * ndim * ndim, 1),
        return_ptr(dets, "dets", num_components, 1),
        return_ptr(log_Js, "log_Js", num_components, 1),
        num_points,
        return_ptr(start_index, "start_index", num_components + 1, 1),
        num_components,
        ndim
    );
}

void L1detector_part(py::module &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<L1OrbitsWrap>(m, "L1OrbitsWrapGPU")
#else
    py::class_<L1OrbitsWrap>(m, "L1OrbitsWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<double, double, int, double, double, int, array_type<double>, array_type<double>, array_type<double>, array_type<int>, array_type<int>, array_type<int>, double>(), 
         py::arg("sc_t0"), py::arg("sc_dt"), py::arg("sc_N"), py::arg("ltt_t0"), py::arg("ltt_dt"), py::arg("ltt_N"), py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"), py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))
    // Bind member functions
    .def("get_light_travel_time_wrap", &L1OrbitsWrap::get_light_travel_time_wrap, "Get the light travel time.")
    .def("get_pos_wrap", &L1OrbitsWrap::get_pos_wrap, "Get spacecraft position.")
    .def("get_normal_unit_vec_wrap", &L1OrbitsWrap::get_normal_unit_vec_wrap, "Get link normal vector.")
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("orbits", &L1OrbitsWrap::orbits)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<L1Orbits>(m, "L1OrbitsGPU")
#else
    py::class_<L1Orbits>(m, "L1OrbitsCPU")
#endif

    // Bind the constructor
    .def(py::init<double, double, int, double, double, int, double *, double *, double *, int *, int *, int *, double>(),
         py::arg("sc_t0"), py::arg("sc_dt"), py::arg("sc_N"), py::arg("ltt_t0"), py::arg("ltt_dt"), py::arg("ltt_N"), py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"), py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))

    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<XYZSensitivityMatrixWrap>(m, "XYZSensitivityMatrixWrapGPU")
#else
    py::class_<XYZSensitivityMatrixWrap>(m, "XYZSensitivityMatrixWrapCPU")
#endif
    .def(py::init<array_type<double>, array_type<double>, int, double, int>(),
            py::arg("averaged_ltts_arr"), py::arg("delta_ltts_arr"), py::arg("n_times"), py::arg("armlength"), py::arg("generation"))
    .def("get_noise_tfs_wrap", &XYZSensitivityMatrixWrap::get_noise_tfs_wrap, "Get noise transfer functions.")
    .def("psd_likelihood_wrap", &XYZSensitivityMatrixWrap::psd_likelihood_wrap, "Compute PSD likelihood.")
    .def("get_noise_covariance_wrap", &XYZSensitivityMatrixWrap::get_noise_covariance_wrap, "Compute noise covariance matrix.")
    .def("get_inverse_logdet_wrap", &XYZSensitivityMatrixWrap::get_inverse_logdet_wrap, "Batch invert 3x3 Hermitian matrices and compute log determinants.")
    .def_readwrite("sensitivity_matrix", &XYZSensitivityMatrixWrap::sensitivity_matrix)
    ;
    
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<XYZSensitivityMatrix>(m, "XYZSensitivityMatrixGPU")
#else
    py::class_<XYZSensitivityMatrix>(m, "XYZSensitivityMatrixCPU")
#endif
    .def(py::init<double *, double *, int, double, int>(),
            py::arg("averaged_ltts_arr"), py::arg("delta_ltts_arr"), py::arg("n_times"), py::arg("armlength"), py::arg("generation"))
    ;

    m.def("psd_likelihood_legacy_wrap", &psd_likelihood_legacy_wrap, "Legacy PSD likelihood wrapping");
    m.def("get_psd_val_legacy_wrap", &get_psd_val_legacy_wrap, "Legacy PSD val wrapping");
    m.def("psd_likelihood", &psd_likelihood_binding, "PSD likelihood computation");
    m.def("compute_logpdf", &compute_logpdf_binding, "Compute log PDF from GMM");
}


PYBIND11_MODULE(pycppdetector, m) {
    m.doc() = "Orbits/Detector C++ plug-in"; // Optional module docstring

    // Call initialization functions from other files
    detector_part(m);
    m.def("check_orbits", &check_orbits, "Make sure that we can insert orbits properly.");

    m.def("get_module_path_cpp", &get_module_path, "Returns the file path of the module");
    m.def("check_12", &check_12, "Check12");

    L1detector_part(m);
    m.def("check_L1orbits", &check_L1orbits, "Make sure that we can insert L1orbits properly.");
    // Optionally, get the path during module initialization and store it
    // This can cause an AttributeError if not handled carefully, as m.attr("__file__")
    // might not be fully set during the initial call if the module is loaded in
    // a specific way (e.g., via pythonw or as a namespace package).
    try {
        std::string path_at_init = m.attr("__file__").cast<std::string>();
        // std::cout << "Module loaded from: " << path_at_init << std::endl;
        m.attr("module_dir") = py::cast(path_at_init.substr(0, path_at_init.find_last_of("/\\")));
    } catch (py::error_already_set &e) {
         // Handle potential error here, e.g., by logging or setting a default value
        std::cerr << "Could not capture __file__ at init time." << std::endl;
        e.restore(); // Restore exception state for proper Python handling
        PyErr_Clear();
    }
}

