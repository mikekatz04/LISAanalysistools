#include "LISAResponse.hh"
#include <string>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "binding_flr.hpp"
#include "binding_detector.hpp"
#include "gbt_binding.hpp"
// Phase 3L (2026-06-02): generic classes absorbed from lisa-on-gpu's
// TDIonTheFly carve-out.
#include "binding_fd_domain.hpp"
#include "binding_wdm_settings.hpp"
#include "binding_wdm_domain.hpp"
#include "binding_lat_spline_tdi.hpp"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#endif

namespace nb = nanobind;


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



// NB_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt
void response_part(nb::module_ &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<LISAResponseWrap>(m, "LISAResponseWrapGPU")
#else
    nb::class_<LISAResponseWrap>(m, "LISAResponseWrapCPU")
#endif 

    // Bind the constructor
    .def(nb::init<OrbitsWrap *, TDIConfigWrap *>(),
         nb::arg("orbits"), nb::arg("tdi_config"))
    // Bind member functions
    .def("get_tdi_delays_wrap", &LISAResponseWrap::get_tdi_delays_wrap, "Preform TDI combinations.")
    .def("get_response_wrap", &LISAResponseWrap::get_response_wrap, "Get detector projections.")
    // You can also expose public data members directly using def_rw
    .def_rw("orbits", &LISAResponseWrap::orbits)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<LISAResponse>(m, "LISAResponseGPU")
#else
    nb::class_<LISAResponse>(m, "LISAResponseCPU")
#endif

    // Bind the constructor
    .def(nb::init<Orbits *, TDIConfig *>(), 
         nb::arg("orbits"), nb::arg("tdi_config"))
    ;


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<TDIConfigWrap>(m, "TDIConfigWrapGPU")
#else
    nb::class_<TDIConfigWrap>(m, "TDIConfigWrapCPU")
#endif

    // Bind the constructor
    .def(nb::init<array_type<int>, array_type<int>, array_type<int>, array_type<int>, array_type<double>, array_type<int>, int, int>(), 
         nb::arg("unit_starts"), nb::arg("unit_lengths"), nb::arg("tdi_base_link"), nb::arg("tdi_link_combinations"), nb::arg("tdi_signs_in"), nb::arg("channels"), nb::arg("num_units"), nb::arg("num_channels"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<TDIConfig>(m, "TDIConfigGPU")
#else
    nb::class_<TDIConfig>(m, "TDIConfigCPU")
#endif

    // Bind the constructor
    .def(nb::init<int*, int*, int*, int*, double*, int*, int, int>(), 
         nb::arg("unit_starts"), nb::arg("unit_lengths"), nb::arg("tdi_base_link"), nb::arg("tdi_link_combinations"), nb::arg("tdi_signs_in"), nb::arg("channels"), nb::arg("num_units"), nb::arg("num_channels"))
    ;

    // Phase 3L.7p (2026-06-04): legacy OrbitsWrap_responselisa class + its
    // pybind registration deleted. Use OrbitsWrap (binding.hpp) directly -- the two
    // shipped identical constructor signatures + identical Orbits* fields;
    // the only structural difference was the (unused) ReturnPointerBase
    // inheritance. All downstream *TDIonTheFlyWrap / LISAResponseWrap
    // constructors now take OrbitsWrap *.

    // 2026-06-10: CubicSplineWrap registration removed -- the class is
    // GBT's and GBT's `interp` module is its single registrant (same
    // pattern as downstream packages consuming LAT's OrbitsWrap). LAT's
    // FDSpline/TDSplineTDIWaveformWrap constructors take
    // `CubicSplineWrap *`; nanobind resolves the shared typeid against
    // GBT's registration at call time. Python code reaches the class
    // via `gbt_backend_<flavor>.interp.CubicSplineWrap{CPU,GPU}`
    // (re-exported on the LAT/GBGPU/BBHx backend objects).

    // Phase 3L: FDDomain + FDDomainWrap absorbed from lisa-on-gpu's
    // TDIonTheFly.hh / binding_tof.{cxx,hpp}. The C++ class definitions
    // live in fd_domain.hh / binding_fd_domain.hpp (included above);
    // this is the SOLE pybind11 registration. lisa-on-gpu's
    // binding_tof.cxx no longer registers them -- the LISATOOLS_IS_WRAPPER_OWNER
    // static_assert in its TU guards against any future regression.

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<FDDomainWrap>(m, "FDDomainWrapGPU")
#else
    nb::class_<FDDomainWrap>(m, "FDDomainWrapCPU")
#endif
    .def(nb::init<array_type<std::complex<double>>, array_type<double>,
                  int, int, int, int, int, int, double>(),
         nb::arg("fd_data"), nb::arg("fd_invC"),
         nb::arg("n_rfft"), nb::arg("num_channel"),
         nb::arg("num_data"), nb::arg("num_noise"),
         nb::arg("ind_min"), nb::arg("ind_max"), nb::arg("df"))
    .def_rw("fd", &FDDomainWrap::fd)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<FDDomain>(m, "FDDomainGPU")
#else
    nb::class_<FDDomain>(m, "FDDomainCPU")
#endif
    .def(nb::init<cmplx*, double*, int, int, int, int, int, int, double>())
    ;

    // Phase 3L: WDMSettingsWrap absorbed from lisa-on-gpu's
    // binding_tof.{cxx,hpp}. The class definition lives in
    // binding_wdm_settings.hpp (included above).
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<WDMSettingsWrap>(m, "WDMSettingsWrapGPU")
#else
    nb::class_<WDMSettingsWrap>(m, "WDMSettingsWrapCPU")
#endif
    .def(nb::init<double, double, int, int, int, int, int, int, int>(),
         nb::arg("layer_df"), nb::arg("layer_dt"), nb::arg("Nf"), nb::arg("Nt"),
         nb::arg("num_channel"), nb::arg("ind_min_t"), nb::arg("ind_max_t"),
         nb::arg("ind_min_f"), nb::arg("ind_max_f"))
    .def_rw("wdm_settings", &WDMSettingsWrap::wdm_settings)
    ;

    // Phase 3L: WDMDomainWrap + WDMDomain absorbed from lisa-on-gpu's
    // binding_tof.{cxx,hpp}. The class definitions live in
    // binding_wdm_domain.hpp / wdm_domain.hh (included above).
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<WDMDomainWrap, WDMSettingsWrap>(m, "WDMDomainWrapGPU")
#else
    nb::class_<WDMDomainWrap, WDMSettingsWrap>(m, "WDMDomainWrapCPU")
#endif
    .def(nb::init<array_type<double>, array_type<double>, double, double, int, int, int, int, int, int, int, int, int>(),
         nb::arg("wdm_data"), nb::arg("wdm_noise"), nb::arg("layer_df"), nb::arg("layer_dt"),
         nb::arg("Nf"), nb::arg("Nt"), nb::arg("num_channel"),
         nb::arg("ind_min_t"), nb::arg("ind_max_t"),
         nb::arg("ind_min_f"), nb::arg("ind_max_f"),
         nb::arg("num_data"), nb::arg("num_noise"))
    .def_rw("wdm", &WDMDomainWrap::wdm)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<WDMDomain>(m, "WDMDomainGPU")
#else
    nb::class_<WDMDomain>(m, "WDMDomainCPU")
#endif
    .def(nb::init<double*, double*, double, double, int, int, int, int, int, int, int, int, int>(),
         nb::arg("wdm_data"), nb::arg("wdm_noise"), nb::arg("layer_df"), nb::arg("layer_dt"),
         nb::arg("Nf"), nb::arg("Nt"), nb::arg("num_channel"),
         nb::arg("ind_min_t"), nb::arg("ind_max_t"),
         nb::arg("ind_min_f"), nb::arg("ind_max_f"),
         nb::arg("num_data"), nb::arg("num_noise"))
    ;

    // Phase 3L.6: LISATDIonTheFlyWrap + FDSpline/TDSpline wrap+underlying
    // absorbed from lisa-on-gpu's binding_tof.{cxx,hpp}. Class defs live in
    // binding_lat_spline_tdi.hpp / lat_spline_tdi_waveform.hh (included above).

    // FDSplineTDIWaveformWrap
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<FDSplineTDIWaveformWrap>(m, "FDSplineTDIWaveformWrapGPU")
#else
    nb::class_<FDSplineTDIWaveformWrap>(m, "FDSplineTDIWaveformWrapCPU")
#endif
    .def(nb::init<OrbitsWrap *, TDIConfigWrap *, CubicSplineWrap *, CubicSplineWrap *>(),
         nb::arg("orbits"), nb::arg("tdi_config"), nb::arg("amp_spline"), nb::arg("freq_spline"))
    .def("run_wave_tdi_wrap", &FDSplineTDIWaveformWrap::run_wave_tdi_wrap, "Preform TDI combinations.")
    .def("get_buffer_size", &FDSplineTDIWaveformWrap::get_buffer_size, "Get needed buffer size.")
    .def_rw("orbits", &FDSplineTDIWaveformWrap::orbits)
    .def_rw("tdi_config", &FDSplineTDIWaveformWrap::tdi_config)
    .def_rw("amp_spline", &FDSplineTDIWaveformWrap::amp_spline)
    .def_rw("freq_spline", &FDSplineTDIWaveformWrap::freq_spline)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<FDSplineTDIWaveform>(m, "FDSplineTDIWaveformGPU")
#else
    nb::class_<FDSplineTDIWaveform>(m, "FDSplineTDIWaveformCPU")
#endif
    .def(nb::init<Orbits *, TDIConfig*, CubicSpline*, CubicSpline*>(),
         nb::arg("orbits"), nb::arg("tdi_config"), nb::arg("amp_spline"), nb::arg("freqs_spline"))
    ;

    // TDSplineTDIWaveformWrap
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<TDSplineTDIWaveformWrap>(m, "TDSplineTDIWaveformWrapGPU")
#else
    nb::class_<TDSplineTDIWaveformWrap>(m, "TDSplineTDIWaveformWrapCPU")
#endif
    .def(nb::init<OrbitsWrap *, TDIConfigWrap *, CubicSplineWrap *, CubicSplineWrap *>(),
         nb::arg("orbits"), nb::arg("tdi_config"), nb::arg("amp_spline"), nb::arg("phase_spline"))
    .def("run_wave_tdi_wrap", &TDSplineTDIWaveformWrap::run_wave_tdi_wrap, "Preform TDI combinations.")
    .def("get_buffer_size", &TDSplineTDIWaveformWrap::get_buffer_size, "Get needed buffer size.")
    .def_rw("orbits", &TDSplineTDIWaveformWrap::orbits)
    .def_rw("tdi_config", &TDSplineTDIWaveformWrap::tdi_config)
    .def_rw("amp_spline", &TDSplineTDIWaveformWrap::amp_spline)
    .def_rw("phase_spline", &TDSplineTDIWaveformWrap::phase_spline)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<TDSplineTDIWaveform>(m, "TDSplineTDIWaveformGPU")
#else
    nb::class_<TDSplineTDIWaveform>(m, "TDSplineTDIWaveformCPU")
#endif
    .def(nb::init<Orbits *, TDIConfig*, CubicSpline*, CubicSpline*>(),
         nb::arg("orbits"), nb::arg("tdi_config"), nb::arg("amp_spline"), nb::arg("phase_spline"))
    ;

}



// NB_MODULE(responselisa, ...) removed during Phase 3E (2026-06-02):
// the response classes (LISAResponseWrap, TDIConfigWrap) are now
// registered into LAT's `pycppdetector` module via response_part(m)
// called from binding.cxx's NB_MODULE(pycppdetector, m) body.
// (CubicSplineWrap moved to GBT ownership 2026-06-10 -- registered
// solely by GBT's `interp` module.)
//
// Helpers `check_response` and `get_module_path` were also
// only used by the deleted NB_MODULE body; the latter referenced a
// module name that no longer exists.

