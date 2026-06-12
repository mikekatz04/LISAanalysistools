#ifndef __FD_DOMAIN_HH__
#define __FD_DOMAIN_HH__

// DEPRECATED include shim (2026-06 domains consolidation).
//
// FDDomain (the chunked-het / signal-het frequency-domain data container)
// now lives in domains.hpp, together with the rest of LAT's C++
// time-frequency domain descriptors (WDMSettings, WDMDomain, and the STFT
// family -- note the STFT-side FD specialisation is the SEPARATE class
// FDDomainForStft). This header remains only so lisa-on-gpu-era include
// paths and downstream consumers (GBGPU, BBHx, lat_tdi_on_the_fly, ...)
// keep compiling. Do not add declarations here; the per-backend CPU/GPU
// alias (#define FDDomain FDDomain{GPU,CPU}) lives in domains.hpp and must
// not be duplicated in this shim.

#include "domains.hpp"

#endif // __FD_DOMAIN_HH__
