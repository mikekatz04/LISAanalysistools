#ifndef __WDM_DOMAIN_HH__
#define __WDM_DOMAIN_HH__

// DEPRECATED include shim (2026-06 domains consolidation).
//
// WDMDomain now lives in domains.hpp, together with the rest of LAT's
// C++ time-frequency domain descriptors (WDMSettings, FDDomain, and the
// STFT family). The canonical TDI flavor ints (TDI_XYZ=1, TDI_AET=2,
// TDI_AE=3) also come from domains.hpp now. This header remains only so
// lisa-on-gpu-era include paths and downstream consumers (GBGPU, BBHx,
// lat_chunked_het_kernels, ...) keep compiling. Do not add declarations
// here; the per-backend CPU/GPU alias (#define WDMDomain
// WDMDomain{GPU,CPU}) lives in domains.hpp and must not be duplicated in
// this shim.

#include "domains.hpp"

#endif // __WDM_DOMAIN_HH__
