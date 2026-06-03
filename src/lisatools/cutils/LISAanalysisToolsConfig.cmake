# LISAanalysisToolsConfig.cmake
#
# Forward-compatibility export so downstream sprint packages
# (GBGPU, BBHx, FastEMRIWaveforms) can pick up LAT's public C++/CUDA
# headers via find_package(...) instead of the Python shell-out
# get_include() path. Both mechanisms point at the same directory and
# are equivalent.
#
# Usage (downstream CMakeLists.txt):
#
#   execute_process(
#     COMMAND ${Python_EXECUTABLE} -c
#     "import lisatools; print(lisatools.get_cmake_module_path())"
#     OUTPUT_VARIABLE LAT_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
#   find_package(LISAanalysisTools CONFIG REQUIRED PATHS ${LAT_CMAKE_DIR})
#   target_link_libraries(my_kernel PRIVATE LISAanalysisTools::headers)
#
# This config file does NOT export compiled targets -- per the sprint
# rule, each downstream wheel re-compiles its required C++/CUDA against
# these headers. POD `*View` structs (e.g. OrbitsView) keep the
# duplication minimal: downstreams typically only need the headers, not
# the C++ class implementations.
#
# Pair with gpubackendtools' Config: downstreams should call
# find_package(GPUBackendTools CONFIG REQUIRED ...) BEFORE find_package
# (LISAanalysisTools CONFIG REQUIRED ...).

get_filename_component(_LAT_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

set(LISAanalysisTools_INCLUDE_DIR  "${_LAT_INCLUDE_DIR}")
set(LISAanalysisTools_INCLUDE_DIRS "${_LAT_INCLUDE_DIR}")

if(NOT TARGET LISAanalysisTools::headers)
  add_library(LISAanalysisTools::headers INTERFACE IMPORTED)
  set_target_properties(LISAanalysisTools::headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_LAT_INCLUDE_DIR}")
endif()

set(LISAanalysisTools_FOUND TRUE)

unset(_LAT_INCLUDE_DIR)
