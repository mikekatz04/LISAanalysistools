#include "LISAResponse.hh"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding_flr.hpp"
#include "binding.hpp"
#include "gbt_binding.hpp"
// Phase 3L (2026-06-02): generic classes absorbed from lisa-on-gpu's
// TDIonTheFly carve-out.
#include "binding_fd_domain.hpp"
#include "binding_wdm_settings.hpp"
#include "binding_wdm_domain.hpp"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#endif

namespace py = pybind11;


void LISAResponseWrap::get_tdi_delays_wrap(array_type<double> delayed_links_, array_type<double> input_links_, int num_inputs, int num_delays, array_type<double> t_arr_,
                    int order, double sampling_frequency, int buffer_integer, array_type<double> A_in_, double deps, int num_A, array_type<double> E_in_, int tdi_start_ind)
{
    response->get_tdi_delays(
        return_pointer_and_check_length(delayed_links_, "delayed_links", num_delays, 3),
        return_pointer_and_check_length(input_links_, "input_links", num_inputs, 6),
        num_inputs, num_delays,
        return_pointer_and_check_length(t_arr_, "t_arr", num_delays, 1),
        order, sampling_frequency, buffer_integer, 
        return_pointer_and_check_length(A_in_, "A_in", num_A, 1),
        deps, num_A,
        return_pointer(E_in_, "E_in"), 
        tdi_start_ind
    );
}

void LISAResponseWrap::get_response_wrap(array_type<double> y_gw_, array_type<double> t_data_, array_type<double> k_in_, array_type<double> u_in_, array_type<double> v_in_, double dt,
    int num_delays,
    array_type<std::complex<double>> input_in_, int num_inputs, int order,
    double sampling_frequency, int buffer_integer,
    array_type<double> A_in_, double deps, int num_A, array_type<double> E_in_, int projections_start_ind, double t0)
{
    response->get_response(
        return_pointer_and_check_length(y_gw_, "y_gw", num_delays, 6),
        return_pointer_and_check_length(t_data_, "t_data", num_delays, 1),
        return_pointer_and_check_length(k_in_, "k_in", 3, 1),
        return_pointer_and_check_length(u_in_, "u_in", 3, 1),
        return_pointer_and_check_length(v_in_, "v_in", 3, 1),
        dt, num_delays,
        return_pointer_cmplx(input_in_, "input_in"),
        num_inputs, order, sampling_frequency, buffer_integer,  
        return_pointer_and_check_length(A_in_, "A_in", num_A, 1),
        deps, num_A,
        return_pointer(E_in_, "E_in"), 
        projections_start_ind, t0
    );
}
    

void check_response(LISAResponse *response)
{
    printf("%e\n", response->orbits->x_arr[0]);
}



std::string get_module_path_responselisa() {
    // Acquire the GIL if it's not already held (safe to call multiple times)
    py::gil_scoped_acquire acquire;

    // Import the module by its name
    // Note: The module name here ("responselisa") must match the name used in PYBIND11_MODULE
    py::object module = py::module::import("responselisa");

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
void response_part(py::module &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<LISAResponseWrap>(m, "LISAResponseWrapGPU")
#else
    py::class_<LISAResponseWrap>(m, "LISAResponseWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<OrbitsWrap_responselisa *, TDIConfigWrap *>(), 
         py::arg("orbits"), py::arg("tdi_config"))
    // Bind member functions
    .def("get_tdi_delays_wrap", &LISAResponseWrap::get_tdi_delays_wrap, "Preform TDI combinations.")
    .def("get_response_wrap", &LISAResponseWrap::get_response_wrap, "Get detector projections.")
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("orbits", &LISAResponseWrap::orbits)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<LISAResponse>(m, "LISAResponseGPU")
#else
    py::class_<LISAResponse>(m, "LISAResponseCPU")
#endif

    // Bind the constructor
    .def(py::init<Orbits *, TDIConfig *>(), 
         py::arg("orbits"), py::arg("tdi_config"))
    ;


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<TDIConfigWrap>(m, "TDIConfigWrapGPU")
#else
    py::class_<TDIConfigWrap>(m, "TDIConfigWrapCPU")
#endif

    // Bind the constructor
    .def(py::init<array_type<int>, array_type<int>, array_type<int>, array_type<int>, array_type<double>, array_type<int>, int, int>(), 
         py::arg("unit_starts"), py::arg("unit_lengths"), py::arg("tdi_base_link"), py::arg("tdi_link_combinations"), py::arg("tdi_signs_in"), py::arg("channels"), py::arg("num_units"), py::arg("num_channels"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<TDIConfig>(m, "TDIConfigGPU")
#else
    py::class_<TDIConfig>(m, "TDIConfigCPU")
#endif

    // Bind the constructor
    .def(py::init<int*, int*, int*, int*, double*, int*, int, int>(), 
         py::arg("unit_starts"), py::arg("unit_lengths"), py::arg("tdi_base_link"), py::arg("tdi_link_combinations"), py::arg("tdi_signs_in"), py::arg("channels"), py::arg("num_units"), py::arg("num_channels"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<OrbitsWrap_responselisa>(m, "OrbitsWrapGPU_responselisa")
#else
    py::class_<OrbitsWrap_responselisa>(m, "OrbitsWrapCPU_responselisa")
#endif

    // Bind the constructor
    .def(py::init<double, double, int, double, double, int, array_type<double>, array_type<double>, array_type<double>, array_type<int>, array_type<int>, array_type<int>, double>(),
         py::arg("sc_t0"), py::arg("sc_dt"), py::arg("sc_N"), py::arg("ltt_t0"), py::arg("ltt_dt"), py::arg("ltt_N"), py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"), py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))
    // .def(py::init<double, double, int, double, double, int, array_type<double>, array_type<double>, array_type<double>, array_type<int>, array_type<int>, array_type<int>, double>(), 
    //      py::arg("sc_t0"), py::arg("sc_dt"), py::arg("sc_N"), py::arg("ltt_t0"), py::arg("ltt_dt"), py::arg("ltt_N"), py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"), py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))
    // Bind member functions
    // .def("get_light_travel_time_wrap", &OrbitsWrap::get_light_travel_time_wrap, "Get the light travel time.")
    // .def("get_pos_wrap", &OrbitsWrap::get_pos_wrap, "Get spacecraft position.")
    // .def("get_normal_unit_vec_wrap", &OrbitsWrap::get_normal_unit_vec_wrap, "Get link normal vector.")
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("orbits", &OrbitsWrap_responselisa::orbits)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<CubicSplineWrap_responselisa>(m, "CubicSplineWrapGPU_responselisa")
#else
    py::class_<CubicSplineWrap_responselisa>(m, "CubicSplineWrapCPU_responselisa")
#endif

    // Bind the constructor
    .def(py::init<array_type<double>, array_type<double>, array_type<double>, array_type<double>, array_type<double>, int, int, int>(),
         py::arg("x0"), py::arg("y0"), py::arg("c1"), py::arg("c2"), py::arg("c3"), py::arg("ninterps"), py::arg("length"), py::arg("spline_type"))
    // Bind member functions

    // You can also expose public data members directly using def_readwrite
    .def_readwrite("spline", &CubicSplineWrap_responselisa::spline)
    // .def("get_link_ind", &CubicSplineWrap::get_link_ind, "Get link index.")
    ;

    // Phase 3L: FDDomain + FDDomainWrap absorbed from lisa-on-gpu's
    // TDIonTheFly.hh / binding_tof.{cxx,hpp}. The C++ class definitions
    // live in fd_domain.hh / binding_fd_domain.hpp (included above);
    // this is the SOLE pybind11 registration. lisa-on-gpu's
    // binding_tof.cxx no longer registers them -- the LISATOOLS_IS_WRAPPER_OWNER
    // static_assert in its TU guards against any future regression.

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<FDDomainWrap>(m, "FDDomainWrapGPU")
#else
    py::class_<FDDomainWrap>(m, "FDDomainWrapCPU")
#endif
    .def(py::init<array_type<std::complex<double>>, array_type<double>,
                  int, int, int, int, int, int, double>(),
         py::arg("fd_data"), py::arg("fd_invC"),
         py::arg("n_rfft"), py::arg("num_channel"),
         py::arg("num_data"), py::arg("num_noise"),
         py::arg("ind_min"), py::arg("ind_max"), py::arg("df"))
    .def_readwrite("fd", &FDDomainWrap::fd)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<FDDomain>(m, "FDDomainGPU")
#else
    py::class_<FDDomain>(m, "FDDomainCPU")
#endif
    .def(py::init<cmplx*, double*, int, int, int, int, int, int, double>())
    ;

    // Phase 3L: WDMSettingsWrap absorbed from lisa-on-gpu's
    // binding_tof.{cxx,hpp}. The class definition lives in
    // binding_wdm_settings.hpp (included above).
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<WDMSettingsWrap>(m, "WDMSettingsWrapGPU")
#else
    py::class_<WDMSettingsWrap>(m, "WDMSettingsWrapCPU")
#endif
    .def(py::init<double, double, int, int, int, int, int, int, int>(),
         py::arg("layer_df"), py::arg("layer_dt"), py::arg("Nf"), py::arg("Nt"),
         py::arg("num_channel"), py::arg("ind_min_t"), py::arg("ind_max_t"),
         py::arg("ind_min_f"), py::arg("ind_max_f"))
    .def_readwrite("wdm_settings", &WDMSettingsWrap::wdm_settings)
    ;

    // Phase 3L: WDMDomainWrap + WDMDomain absorbed from lisa-on-gpu's
    // binding_tof.{cxx,hpp}. The class definitions live in
    // binding_wdm_domain.hpp / wdm_domain.hh (included above).
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<WDMDomainWrap, WDMSettingsWrap>(m, "WDMDomainWrapGPU")
#else
    py::class_<WDMDomainWrap, WDMSettingsWrap>(m, "WDMDomainWrapCPU")
#endif
    .def(py::init<array_type<double>, array_type<double>, double, double, int, int, int, int, int, int, int, int, int>(),
         py::arg("wdm_data"), py::arg("wdm_noise"), py::arg("layer_df"), py::arg("layer_dt"),
         py::arg("Nf"), py::arg("Nt"), py::arg("num_channel"),
         py::arg("ind_min_t"), py::arg("ind_max_t"),
         py::arg("ind_min_f"), py::arg("ind_max_f"),
         py::arg("num_data"), py::arg("num_noise"))
    .def_readwrite("wdm", &WDMDomainWrap::wdm)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<WDMDomain>(m, "WDMDomainGPU")
#else
    py::class_<WDMDomain>(m, "WDMDomainCPU")
#endif
    .def(py::init<double*, double*, double, double, int, int, int, int, int, int, int, int, int>(),
         py::arg("wdm_data"), py::arg("wdm_noise"), py::arg("layer_df"), py::arg("layer_dt"),
         py::arg("Nf"), py::arg("Nt"), py::arg("num_channel"),
         py::arg("ind_min_t"), py::arg("ind_max_t"),
         py::arg("ind_min_f"), py::arg("ind_max_f"),
         py::arg("num_data"), py::arg("num_noise"))
    ;

}



// PYBIND11_MODULE(responselisa, ...) removed during Phase 3E (2026-06-02):
// the response classes (LISAResponseWrap, TDIConfigWrap, OrbitsWrap_responselisa,
// CubicSplineWrap_responselisa) are now registered into LAT's `pycppdetector`
// pybind11 module via response_part(m) called from binding.cxx's
// PYBIND11_MODULE(pycppdetector, m) body.
//
// Helpers `check_response` and `get_module_path_responselisa` were also
// only used by the deleted PYBIND11_MODULE body; the latter referenced a
// module name that no longer exists.

