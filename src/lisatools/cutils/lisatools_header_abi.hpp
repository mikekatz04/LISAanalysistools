#ifndef __LISATOOLS_HEADER_ABI_HPP__
#define __LISATOOLS_HEADER_ABI_HPP__

// LISAanalysistools cross-wheel ABI marker.
//
// This header ships in LAT's installed cutils/ and is consumed by every
// downstream sprint wheel (GBGPU, BBHx, FastEMRIWaveforms) that includes
// LAT public headers via gpubackendtools.get_include() / lisatools.get_include().
//
// =====================================================================
// LISATOOLS_HEADER_ABI_VERSION
// =====================================================================
//
// Monotonically incremented when the layout / contract of any of these
// changes:
//   - OrbitsView                       (cutils/orbits_view.hpp)
//   - WDMSettingsView                  (cutils/wdm_settings_view.hpp, future)
//   - TDIConfigView                    (cutils/tdi_config_view.hpp, future)
//   - Orbits class POD field layout    (cutils/Detector.hpp)
//   - WDMSettings class POD field layout
//   - TDIConfig class POD field layout
//
// Downstream binding TUs read this macro and static_assert the value
// they were compiled against. Mismatch = downstream wheel must rebuild.
// This converts the current silent-corruption-at-runtime failure mode
// (downstream wheel built against stale header layout) into a loud
// compile-time error before the user ever loads the broken process.
//
// Bump ONE value, in ONE PR, when any of the above layouts changes.

#define LISATOOLS_HEADER_ABI_VERSION 1

// =====================================================================
// LISATOOLS_IS_WRAPPER_OWNER
// =====================================================================
//
// Single-registrant rule. Only LISAanalysistools may register the shared
// pybind11 wrapper classes (OrbitsWrap, WDMSettingsWrap, WDMDomainWrap,
// FDDomainWrap, TDIConfigWrap, TDIOnTheFlyBaseWrap). Downstream wheels
// (GBGPU, BBHx, FastEMRIWaveforms) #include this header (default 0)
// and add to the top of every binding TU:
//
//     static_assert(!LISATOOLS_IS_WRAPPER_OWNER,
//         "Single-registrant rule: only LISAanalysistools may register "
//         "OrbitsWrap / WDM*Wrap / TDIConfig* with pybind11. "
//         "See plan section OrbitsWrap-symbol-unification.");
//
// LAT's own binding_detector.cxx #defines LISATOOLS_IS_WRAPPER_OWNER 1 before
// including this header, so the assertion passes there. Any downstream
// TU that accidentally calls py::class_<OrbitsWrap>(...) without
// flipping the toggle gets a compile-time error pointing at this rule
// instead of a silent runtime cast failure later.
//
// See plan section "OrbitsWrap (and friends) symbol unification across
// packages and CI" for the full rationale.

#ifndef LISATOOLS_IS_WRAPPER_OWNER
#define LISATOOLS_IS_WRAPPER_OWNER 0
#endif

#endif // __LISATOOLS_HEADER_ABI_HPP__
