#include "Detector.hpp"
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



PYBIND11_MODULE(pycppdetector, m) {
    m.doc() = "Orbits/Detector C++ plug-in"; // Optional module docstring

    // Call initialization functions from other files
    detector_part(m);
    m.def("check_orbits", &check_orbits, "Make sure that we can insert orbits properly.");

    m.def("get_module_path_cpp", &get_module_path, "Returns the file path of the module");
    m.def("check_12", &check_12, "Check12");
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

