#ifndef __LAT_DFFR_DIAGNOSTICS_HH__
#define __LAT_DFFR_DIAGNOSTICS_HH__

// Host entry points of the source-independent DFFR diagnostics, which tests/test_dffr.py checks
// against reference values. They are defined in lat_dffr_diagnostics.cu, which only LAT's own
// module compiles; every array lives on the device of the backend that calls them. The bindings
// are methods of STFTFresnelWrap and KinematicsTableWrap.

#include "gbt_global.h"  // cmplx
#include "Detector.hpp"  // KinematicsTable
#include "domains.hpp"   // STFTFresnel

// The bounded sub-interval integral int_a^b exp(i [2 pi detuning tau + pi alpha tau^2]) dtau and
// its first moment about tau = 0. Chirp rates closer to zero than the floor are moved to it, sign
// kept.
void dffr_sub_intervals_wrap(cmplx *kernel_out, cmplx *moment_out, const double *lower_bounds,
                             const double *upper_bounds, const double *detunings,
                             const double *chirp_rates, double chirp_rate_floor, int num_points);

// The windowed transform of exp(i [2 pi f_eff tau + pi alpha tau^2]) at the frequency f, with the
// window and segment of fresnel and the Fourier origin at the segment start, and its first moment
// about the segment midpoint.
void dffr_windowed_kernels_wrap(cmplx *kernel_out, cmplx *moment_out, const STFTFresnel *fresnel,
                                const double *effective_frequencies, const double *chirp_rates,
                                const double *frequencies, double chirp_rate_floor,
                                int num_points);

// The TDI coefficient of every link in every channel at zeroth order in the delays, for each
// segment of the table, coefficients_out[(segment * 3 + channel) * 6 + link]. The delays are the
// table's light travel times at the carrier frequency of the segment.
void dffr_link_coefficients_wrap(cmplx *coefficients_out, const KinematicsTable *table,
                                 const double *carrier_frequencies, int tdi_generation);

#endif  // __LAT_DFFR_DIAGNOSTICS_HH__
