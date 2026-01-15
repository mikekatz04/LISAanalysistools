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

# ----------------------
# --- pycppdetector ----
# ----------------------

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
  
  # NEW: Copy L1Detector.cu to L1Detector.cxx for CPU build
  add_custom_command(
    OUTPUT "L1Detector.cxx"
    COMMENT "Copy L1Detector.cu to L1Detector.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/L1Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/L1Detector.cxx"
    DEPENDS "L1Detector.cu"
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

  # Add L1Detector.cxx to the sources
  add_library(lisaanalysistools_cpu_detector_static STATIC Detector.cxx L1Detector.cxx PSD.cxx)
  apply_cpu_backend_common_options(detector lisatools lisaanalysistools true)
  target_include_directories(lisaanalysistools_cpu_detector_static PUBLIC ${GBT_CUTILS})
  
  # Add L1Detector.hpp to header files
  target_sources(lisaanalysistools_cpu_detector_static PUBLIC FILE_SET HEADERS FILES
                                        Detector.hpp L1Detector.hpp PSD.hpp)

  pybind11_add_module(lisaanalysistools_cpu_pycppdetector binding.cxx)
  apply_cpu_backend_common_options(pycppdetector lisatools lisaanalysistools false)
  target_link_libraries(lisaanalysistools_cpu_pycppdetector PRIVATE lisaanalysistools_cpu_detector_static)
  target_include_directories(lisaanalysistools_cpu_pycppdetector PUBLIC ${GBT_CUTILS})
  target_sources(lisaanalysistools_cpu_pycppdetector PUBLIC FILE_SET HEADERS FILES
                                      binding.hpp)
endif()

# III. Declare the GPU backend
if(LISATOOLS_WITH_GPU)
  # Add L1Detector.cu to the GPU sources
  add_library(lisaanalysistools_gpu_detector_gpu_static STATIC Detector.cu L1Detector.cu PSD.cu)
  apply_gpu_backend_common_options(detector_gpu lisatools lisaanalysistools true)
  target_include_directories(lisaanalysistools_gpu_detector_gpu_static PUBLIC ${GBT_CUTILS})
  
  # Add L1Detector.hpp to header files
  target_sources(lisaanalysistools_gpu_detector_gpu_static PUBLIC FILE_SET HEADERS FILES
                                        Detector.hpp L1Detector.hpp PSD.hpp)

  pybind11_add_module(lisaanalysistools_gpu_pycppdetector binding.cxx)
  apply_gpu_backend_common_options(pycppdetector lisatools lisaanalysistools false)
  target_link_libraries(lisaanalysistools_gpu_pycppdetector PRIVATE lisaanalysistools_gpu_detector_gpu_static)
  
  target_include_directories(lisaanalysistools_gpu_pycppdetector PUBLIC ${GBT_CUTILS})
  target_sources(lisaanalysistools_gpu_pycppdetector PUBLIC FILE_SET HEADERS FILES
                                      binding.hpp)

endif()



# ----------------------
# --- psd ----
# ----------------------

# I. Process pycppdetector.pyx into a C++ file
add_custom_command(
  OUTPUT "psdgen.cxx"
  COMMENT "Cythonize psdgen.pyx into psdgen.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/psdgen.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/psdgen.cxx" -3 -+ --module-name
    "psd" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "psdgen.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(LISATOOLS_WITH_CPU)
  add_custom_command(
    OUTPUT "PSD.cxx"
    COMMENT "Copy PSD.cu to PSD.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/PSD.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/PSD.cxx"
    DEPENDS "PSD.cu"
    VERBATIM)

  python_add_library(lisaanalysistools_cpu_psd MODULE WITH_SOABI psdgen.cxx PSD.cxx)
  apply_cpu_backend_common_options(psd lisatools lisaanalysistools false)

  target_sources(lisaanalysistools_cpu_psd PUBLIC FILE_SET HEADERS FILES
                                         global.hpp PSD.hpp)
endif()

# III. Declare the GPU backend
if(LISATOOLS_WITH_GPU)
  python_add_library(lisaanalysistools_gpu_psd MODULE WITH_SOABI psdgen.cxx PSD.cu)
  apply_gpu_backend_common_options(psd lisatools lisaanalysistools false)
  target_sources(lisaanalysistools_gpu_psd PUBLIC FILE_SET HEADERS FILES
                                         global.hpp PSD.hpp)
endif()

