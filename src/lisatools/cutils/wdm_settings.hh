#ifndef __WDM_SETTINGS_HH__
#define __WDM_SETTINGS_HH__

// DEPRECATED include shim (2026-06 domains consolidation).
//
// WDMSettings now lives in domains.hpp, together with the rest of LAT's
// C++ time-frequency domain descriptors (WDMDomain, FDDomain, and the STFT
// family). This header remains only so lisa-on-gpu-era include paths and
// downstream consumers (GBGPU, BBHx, lat_tdi_on_the_fly, ...) keep
// compiling. Do not add declarations here; the per-backend CPU/GPU alias
// (#define WDMSettings WDMSettings{GPU,CPU}) lives in domains.hpp and must
// not be duplicated in this shim.

#include "domains.hpp"

#endif // __WDM_SETTINGS_HH__
