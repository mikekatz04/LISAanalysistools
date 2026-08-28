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

  # stft_tof merge: copy galactic_response.cu to .cxx for CPU build.
  add_custom_command(
    OUTPUT "galactic_response.cxx"
    COMMENT "Copy galactic_response.cu to galactic_response.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/galactic_response.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/galactic_response.cxx"
    DEPENDS "galactic_response.cu"
    VERBATIM
    )

  # stft_tof merge: copy domains.cu to domains.cxx for CPU build.
  add_custom_command(
    OUTPUT "domains.cxx"
    COMMENT "Copy domains.cu to domains.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/domains.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/domains.cxx"
    DEPENDS "domains.cu"
    VERBATIM
    )

  # Orchestration audit (2026-08-27): copy gf_routing_kernels.cu to .cxx for
  # the CPU build. Same single-source rule as every TU above -- the CPU build
  # compiles the identical file with the grid-stride macros collapsed.
  add_custom_command(
    OUTPUT "gf_routing_kernels.cxx"
    COMMENT "Copy gf_routing_kernels.cu to gf_routing_kernels.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gf_routing_kernels.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/gf_routing_kernels.cxx"
    DEPENDS "gf_routing_kernels.cu"
    VERBATIM
    )

  # Phase 3E adds LISAResponse.cxx to the detector static library so all
  # LISA-side C++ symbols live in one .so per backend wheel.
  # Phase 3L.5+3L.6 add lat_tdi_on_the_fly.cxx + lat_spline_tdi_waveform.cxx
  # so the virtual-method vtables/typeinfo for LISATDIonTheFly +
  # FDSpline/TDSpline classes are emitted into LAT's own .so (LAT is the
  # nanobind registrant; downstream wheels recompile the same .cu
  # files into their own .so under the per-backend alias).
  # stft_tof merge adds galactic_response.cxx + domains.cxx (XYZ sensitivity
  # matrix backend + STFT/FD domain likelihood kernels).
  # The 2026-08-27 orchestration audit adds gf_routing_kernels.cxx (fused GB
  # in-model gate/compaction + accept/bookkeeping kernels).
  add_library(lisaanalysistools_cpu_detector_static STATIC Detector.cxx PSD.cxx LISAResponse.cxx lat_tdi_on_the_fly.cxx lat_spline_tdi_waveform.cxx galactic_response.cxx domains.cxx gf_routing_kernels.cxx)
  apply_cpu_backend_common_options(detector lisatools lisaanalysistools true)
  target_include_directories(lisaanalysistools_cpu_detector_static PUBLIC ${GBT_CUTILS})

  target_sources(lisaanalysistools_cpu_detector_static PUBLIC FILE_SET HEADERS FILES
                                        Detector.hpp  PSD.hpp  LISAResponse.hh domains.hpp galactic_response.hpp gf_routing_kernels.hpp)

  # Phase 3E adds binding_flr.cxx to the nanobind module so response_part(m)
  # is linked in alongside detector_part(m). (binding.cxx renamed to
  # binding_detector.cxx at the 2026-06 stft_tof merge.)
  nanobind_add_module(lisaanalysistools_cpu_pycppdetector binding_detector.cxx binding_flr.cxx)
  apply_cpu_backend_common_options(pycppdetector lisatools lisaanalysistools false)
  target_link_libraries(lisaanalysistools_cpu_pycppdetector PRIVATE lisaanalysistools_cpu_detector_static)
  target_include_directories(lisaanalysistools_cpu_pycppdetector PUBLIC ${GBT_CUTILS})
  target_sources(lisaanalysistools_cpu_pycppdetector PUBLIC FILE_SET HEADERS FILES
                                      binding_detector.hpp  binding_flr.hpp  binding_domains.hpp)
endif()

# III. Declare the GPU backend
if(LISATOOLS_WITH_GPU)
  # Phase 3E adds LISAResponse.cu to the GPU sources.
  # Phase 3L.5+3L.6 add lat_tdi_on_the_fly.cu + lat_spline_tdi_waveform.cu
  # so LAT's GPU .so emits the typeinfo / vtable for LISATDIonTheFly +
  # FDSpline/TDSpline (LAT is the nanobind registrant).
  # stft_tof merge adds galactic_response.cu + domains.cu (XYZ sensitivity
  # matrix backend + STFT/FD domain likelihood kernels).
  # The 2026-08-27 orchestration audit adds gf_routing_kernels.cu (fused GB
  # in-model gate/compaction + accept/bookkeeping kernels); no copy step on
  # this side -- the same .cu goes straight to nvcc.
  add_library(lisaanalysistools_gpu_detector_gpu_static STATIC Detector.cu PSD.cu LISAResponse.cu lat_tdi_on_the_fly.cu lat_spline_tdi_waveform.cu galactic_response.cu domains.cu gf_routing_kernels.cu)
  apply_gpu_backend_common_options(detector_gpu lisatools lisaanalysistools true)
  target_include_directories(lisaanalysistools_gpu_detector_gpu_static PUBLIC ${GBT_CUTILS})
  target_include_directories(lisaanalysistools_gpu_detector_gpu_static SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

  target_sources(lisaanalysistools_gpu_detector_gpu_static PUBLIC FILE_SET HEADERS FILES
                                        Detector.hpp PSD.hpp LISAResponse.hh domains.hpp galactic_response.hpp gf_routing_kernels.hpp)

  nanobind_add_module(lisaanalysistools_gpu_pycppdetector binding_detector.cxx binding_flr.cxx)
  apply_gpu_backend_common_options(pycppdetector lisatools lisaanalysistools false)
  target_link_libraries(lisaanalysistools_gpu_pycppdetector PRIVATE lisaanalysistools_gpu_detector_gpu_static)

  target_include_directories(lisaanalysistools_gpu_pycppdetector PUBLIC ${GBT_CUTILS})
  target_sources(lisaanalysistools_gpu_pycppdetector PUBLIC FILE_SET HEADERS FILES
                                      binding_detector.hpp  binding_flr.hpp  binding_domains.hpp)

endif()

# Generate lisatools_sources.cmake for downstream packages.
# 2026-06 domains consolidation: wdm_settings.hh / wdm_domain.hh /
# fd_domain.hh are now deprecated #include shims onto domains.hpp (listed so
# downstream installs keep the lisa-on-gpu-era include paths working);
# binding_domains.hpp carries the STFT/FD domain wrap classes.
set(LISATOOLS_CU_SOURCE_NAMES "Detector.cu" "PSD.cu" "domains.cu" "galactic_response.cu")
set(LISATOOLS_HEADER_NAMES "Detector.hpp" "PSD.hpp" "domains.hpp" "binding_detector.hpp" "binding_domains.hpp" "galactic_response.hpp" "wdm_settings.hh" "wdm_domain.hh" "fd_domain.hh")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lisatools_sources.cmake.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/lisatools_sources.cmake"
    @ONLY
)
