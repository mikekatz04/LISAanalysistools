# =============================
# ==== BACKEND COMPILATION ====
# =============================

# TODO: how can we avoid moving source and header files in this directory ?

# In the project root CMakeLists.txt, we defined a "lisaanalysistools" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(LISATOOLS_WITH_CPU lisaanalysistools WITH_CPU)
get_target_property(LISATOOLS_WITH_GPU lisaanalysistools WITH_GPU)

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
# TODO: is this needed ?
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${lisaanalysistools_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools.__file__.split('__init__.py')[0] + 'cutils')"
  OUTPUT_VARIABLE GBT_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(GBT_CMAKE_TOOLS "${GBT_CUTILS}/cmake_functions.cmake")

# Check if the file exists
if(EXISTS "${GBT_CMAKE_TOOLS}")
    message("INFO: ${GBT_CMAKE_TOOLS} exists.")
else()
    message(FATAL_ERROR "${GBT_CMAKE_TOOLS} not found.")
    # Handle the case where the file is missing
endif()

include(${GBT_CMAKE_TOOLS})
set(HERE_CUDA_ARCH ${LISATOOLS_CUDA_ARCH})

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# ----------------
# --- utils ---
# ----------------

# I. Process pycppdetector.pyx into a C++ file

if(EXISTS "${GBT_CUTILS}/gbt_global.h")
    message("INFO: ${GBT_CUTILS}/gbt_global.h exists.")
else()
    message(FATAL_ERROR "${GBT_CUTILS}/gbt_global.h not found.")
    # Handle the case where the file is missing
endif()

# II. Declare the CPU backend
if(LISATOOLS_WITH_CPU)
  add_custom_command(
    OUTPUT "Detector.cxx"
    COMMENT "Copy Detector.cu to Detector.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "Detector.cu"
    VERBATIM
    )

  # Copy PSD.cu to PSD.cxx for CPU build
  add_custom_command(
    OUTPUT "PSD.cxx"
    COMMENT "Copy PSD.cu to PSD.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/PSD.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/PSD.cxx"
    DEPENDS "PSD.cu"
    VERBATIM
    )

  # Phase 3E: copy LISAResponse.cu to LISAResponse.cxx for CPU build
  add_custom_command(
    OUTPUT "LISAResponse.cxx"
    COMMENT "Copy LISAResponse.cu to LISAResponse.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/LISAResponse.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/LISAResponse.cxx"
    DEPENDS "LISAResponse.cu"
    VERBATIM
    )

  # Phase 3L.5: copy lat_tdi_on_the_fly.cu to .cxx for CPU build.
  add_custom_command(
    OUTPUT "lat_tdi_on_the_fly.cxx"
    COMMENT "Copy lat_tdi_on_the_fly.cu to lat_tdi_on_the_fly.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/lat_tdi_on_the_fly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/lat_tdi_on_the_fly.cxx"
    DEPENDS "lat_tdi_on_the_fly.cu"
    VERBATIM
    )

  # Phase 3L.6: copy lat_spline_tdi_waveform.cu to .cxx for CPU build.
  add_custom_command(
    OUTPUT "lat_spline_tdi_waveform.cxx"
    COMMENT "Copy lat_spline_tdi_waveform.cu to lat_spline_tdi_waveform.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/lat_spline_tdi_waveform.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/lat_spline_tdi_waveform.cxx"
    DEPENDS "lat_spline_tdi_waveform.cu"
    VERBATIM
    )

  # Phase 3E adds LISAResponse.cxx to the detector static library so all
  # LISA-side C++ symbols live in one .so per backend wheel.
  # Phase 3L.5+3L.6 add lat_tdi_on_the_fly.cxx + lat_spline_tdi_waveform.cxx
  # so the virtual-method vtables/typeinfo for LISATDIonTheFly +
  # FDSpline/TDSpline classes are emitted into LAT's own .so (LAT is the
  # pybind11 registrant; lisa-on-gpu's wheel recompiles the same .cu
  # files into its own .so under the per-backend alias).
  add_library(lisaanalysistools_cpu_detector_static STATIC Detector.cxx PSD.cxx LISAResponse.cxx lat_tdi_on_the_fly.cxx lat_spline_tdi_waveform.cxx)
  apply_cpu_backend_common_options(detector lisatools lisaanalysistools true)
  target_include_directories(lisaanalysistools_cpu_detector_static PUBLIC ${GBT_CUTILS})

  target_sources(lisaanalysistools_cpu_detector_static PUBLIC FILE_SET HEADERS FILES
                                        Detector.hpp  PSD.hpp  LISAResponse.hh)

  # Phase 3E adds binding_flr.cxx to the pybind11 module so response_part(m)
  # is linked in alongside detector_part(m).
  nanobind_add_module(lisaanalysistools_cpu_pycppdetector binding.cxx binding_flr.cxx)
  apply_cpu_backend_common_options(pycppdetector lisatools lisaanalysistools false)
  target_link_libraries(lisaanalysistools_cpu_pycppdetector PRIVATE lisaanalysistools_cpu_detector_static)
  target_include_directories(lisaanalysistools_cpu_pycppdetector PUBLIC ${GBT_CUTILS})
  target_sources(lisaanalysistools_cpu_pycppdetector PUBLIC FILE_SET HEADERS FILES
                                      binding.hpp  binding_flr.hpp)
endif()

# III. Declare the GPU backend
if(LISATOOLS_WITH_GPU)
  # Phase 3E adds LISAResponse.cu to the GPU sources.
  # Phase 3L.5+3L.6 add lat_tdi_on_the_fly.cu + lat_spline_tdi_waveform.cu
  # so LAT's GPU .so emits the typeinfo / vtable for LISATDIonTheFly +
  # FDSpline/TDSpline (LAT is the pybind11 registrant).
  add_library(lisaanalysistools_gpu_detector_gpu_static STATIC Detector.cu PSD.cu LISAResponse.cu lat_tdi_on_the_fly.cu lat_spline_tdi_waveform.cu)
  apply_gpu_backend_common_options(detector_gpu lisatools lisaanalysistools true)
  target_include_directories(lisaanalysistools_gpu_detector_gpu_static PUBLIC ${GBT_CUTILS})

  target_sources(lisaanalysistools_gpu_detector_gpu_static PUBLIC FILE_SET HEADERS FILES
                                        Detector.hpp PSD.hpp LISAResponse.hh)

  nanobind_add_module(lisaanalysistools_gpu_pycppdetector binding.cxx binding_flr.cxx)
  apply_gpu_backend_common_options(pycppdetector lisatools lisaanalysistools false)
  target_link_libraries(lisaanalysistools_gpu_pycppdetector PRIVATE lisaanalysistools_gpu_detector_gpu_static)

  target_include_directories(lisaanalysistools_gpu_pycppdetector PUBLIC ${GBT_CUTILS})
  target_sources(lisaanalysistools_gpu_pycppdetector PUBLIC FILE_SET HEADERS FILES
                                      binding.hpp  binding_flr.hpp)

endif()
