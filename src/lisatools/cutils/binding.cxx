#include "Detector.hpp"
#include <pybind11/pybind11.h>
#include <string>
#include <iostream>


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


PYBIND11_MODULE(pycppdetector, m) {
    m.doc() = "Orbits/Detector C++ plug-in"; // Optional module docstring

    // Call initialization functions from other files
    detector_part(m);

    m.def("get_module_path_cpp", &get_module_path, "Returns the file path of the module");

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

