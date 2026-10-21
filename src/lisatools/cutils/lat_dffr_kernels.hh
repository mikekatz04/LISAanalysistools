#ifndef __LAT_DFFR_KERNELS_HH__
#define __LAT_DFFR_KERNELS_HH__

// lat_dffr_kernels.hh -- the Direct Fast Fresnel Response (DFFR) of the STFT likelihood: settings,
// source contract, windowed chirp transform, TDI weights, segment table, the tile that hands the
// pixels to the consumers of lat_stft_kernels.hh (get_ll, fill_global, swap_ll, the F-statistic and
// both gradients), and two diagnostics of a batch of sources (compact template, segment table).
// Ported from the standalone package direct-fast-fresnel-response at commit 01530a5
// (src/dffr/cutils/dffr.cu). A source package (GBGPU for galactic binaries) instantiates the tile
// and the diagnostics with its source model; the source-independent diagnostics of LAT are in
// lat_dffr_diagnostics.cu. docs/response/dffr/README.md describes the response and its integration.
//
// * Response. Within one segment the signal that reaches spacecraft s is a linear chirp under a
// * slowly varying envelope, expanded about the segment midpoint, so its windowed transform is a
// * sum of bounded Fresnel integrals at any frequency. A pixel evaluates three such reception
// * states, each a kernel K_s and its first moment M_s, and sums
// *     value_c = sum_s (w_{c,s} K_s + v_{c,s} M_s),   c = X, Y, Z,
// * with eighteen weights per segment. The weights fold the six links and the TDI combination:
// * each emission state is the reception state of its sender, rotated by the delay, so it needs no
// * transform of its own. The moment carries the antenna and amplitude slopes (linear_envelope) and
// * the first-order change of every delay across the segment (first_order_delays).
// *
// * Conventions, those of GBTDIonTheFly and STFTFresnel: a value is dt * rfft(w * x) of the real
// * TDI x at f = f_min + j df, with the Fourier origin at the segment start and w the periodic
// * Tukey window of taper fraction window_alpha; it is the positive-frequency half, so it compares
// * with the transform of a real time series directly. Strain of LAT, -2 times the strain of the
// * derivation (DFFR_LAT_STRAIN_FACTOR). Links in the order 12, 23, 31, 13, 32, 21 of Orbits, link
// * ij received by spacecraft i and sent by spacecraft j.
// *
// * Accuracy of the standalone package against a time-domain TDI 2.0 truth (lisagwresponse,
// * lisainstrument without noise, pytdi; 30 one-day segments on the ESA orbits, Tukey 0.116,
// * sources from 3 to 40 mHz): median error per segment 3.2e-5 to 4.7e-5 of the segment's peak,
// * largest 4.8e-4, template mismatch 6.5e-10 to 1.2e-9. The Fresnel functions of lat_fresnel.hh
// * contribute at most 1.6e-10.
// ! Assumptions, checked on the host by dffr_make_settings and dffr_check_table:
// !   df * dt = 1 to within 1e-12 (contiguous segments); otherwise the band-bin phases of the
// !   window endpoints are wrong without any sign of it.
// !   The kinematics table holds the midpoints of the grid's segments, in the time frame of the
// !   orbits, the frame of t_ref.
// !   XYZ channels, with the combinations of TDIConfig("1st generation") or ("2nd generation").
// ! The chirp-rate floor (default 1e-22 Hz/s) changes the phase by at most pi floor (dt / 2)^2,
// ! 6e-13 rad on a one-day segment.
// * Speed: everything that does not depend on the bin is in the segment table, built once per
// * pass by all threads of a block. A pixel then costs three windowed transforms, of six bounded
// * integrals each (one on a boxcar), and 18 complex multiply-adds that apply the weights.

#include "gbt_global.h"        // cmplx, gcmplx, CUDA_* macros, THREAD_START_X, BLOCK_INCR_X
#include "Detector.hpp"        // KinematicsTable
#include "domains.hpp"         // STFTFresnel, STFTDomain
#include "lat_fresnel.hh"      // lat_fresnel_series, lat_fresnel_aux, LAT_FRESNEL_SEAM
#include "lat_stft_kernels.hh" // StftBandBinMap, stft_tile_arena, the launch helpers

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#define DFFR_NUM_SPACECRAFT 3
#define DFFR_NUM_LINKS 6
#define DFFR_NUM_CHANNELS 3
#define DFFR_TDI_WEIGHTS 18
#define DFFR_LIGHT_SPEED 299792458.0
#define DFFR_TWO_PI 6.283185307179586476925286766559
#define DFFR_LAT_STRAIN_FACTOR (-2.0)
// * Default floor on the local chirp rate [Hz/s], the STFT_FDOT_FLOOR of the existing response.
#define DFFR_CHIRP_RATE_FLOOR 1.0e-22

// ===========================================================================
// Links
// ===========================================================================
// ! Receivers, senders and channels are packed two bits per link, because a local array indexed
// ! at run time lands in local memory whenever a loop fails to unroll.
#define DFFR_LINK_RECEIVERS 0x624
#define DFFR_LINK_SENDERS 0x189
// The two TDI channels a link feeds: arm 12 (links 0 and 5) feeds X and Y, arm 23 (links 1 and 4)
// Y and Z, arm 31 (links 2 and 3) X and Z.
#define DFFR_LINK_CHANNELS_FIRST 0x104
#define DFFR_LINK_CHANNELS_SECOND 0x6A9

// * The sender of a link is the receiver of the reverse link, which checks the two tables.
static_assert(((DFFR_LINK_SENDERS >> 0) & 3) == ((DFFR_LINK_RECEIVERS >> 10) & 3) &&
                  ((DFFR_LINK_SENDERS >> 2) & 3) == ((DFFR_LINK_RECEIVERS >> 8) & 3) &&
                  ((DFFR_LINK_SENDERS >> 4) & 3) == ((DFFR_LINK_RECEIVERS >> 6) & 3),
              "The sender table disagrees with the receiver table.");

// Zero-based receiving spacecraft of a link.
CUDA_CALLABLE_MEMBER
inline int dffr_link_receiver(int link_index) {
  return (DFFR_LINK_RECEIVERS >> (2 * link_index)) & 3;
}

// Zero-based sending spacecraft of a link.
CUDA_CALLABLE_MEMBER
inline int dffr_link_sender(int link_index) {
  return (DFFR_LINK_SENDERS >> (2 * link_index)) & 3;
}

CUDA_CALLABLE_MEMBER
inline int dffr_link_channel_first(int link_index) {
  return (DFFR_LINK_CHANNELS_FIRST >> (2 * link_index)) & 3;
}

CUDA_CALLABLE_MEMBER
inline int dffr_link_channel_second(int link_index) {
  return (DFFR_LINK_CHANNELS_SECOND >> (2 * link_index)) & 3;
}

// Unit phasor exp(i theta), with one sincos on the device.
CUDA_CALLABLE_MEMBER
inline cmplx dffr_polar_unit(double theta) {
#if defined(__CUDA_ARCH__)
  double sine_value, cosine_value;
  sincos(theta, &sine_value, &cosine_value);
  return cmplx(cosine_value, sine_value);
#else
  return cmplx(std::cos(theta), std::sin(theta));
#endif
}

// ! Complex product without the infinity recovery of the library operator: every operand here is
// ! finite, and the recovery code costs instruction cache.
CUDA_CALLABLE_MEMBER
inline cmplx dffr_mul(const cmplx &left, const cmplx &right) {
  return cmplx(left.real() * right.real() - left.imag() * right.imag(),
               left.real() * right.imag() + left.imag() * right.real());
}

// ===========================================================================
// Settings
// ===========================================================================

struct DFFRSettings {
  int num_time_segments;
  double segment_duration;  // Delta T, the dt of the STFT grid, 1 / df [s]
  double half_duration;

  double f_min;  // frequency of active bin zero [Hz]
  double df;     // bin spacing [Hz]
  int num_bins;  // active bins of the grid

  double window_taper;        // Tukey taper fraction rho, STFTFresnel::window_alpha; 0 is a boxcar
  double plateau_half_width;  // tau_0 = (Delta T / 2) (1 - rho)
  double taper_frequency;     // 1 / (rho Delta T), the shift of the taper sidebands
  double taper_phase_offset;  // theta_rho = pi (1 / rho - 1)
  cmplx taper_phasor;         // exp(i theta_rho)
  cmplx origin_phasor_base;   // exp(-i pi f_min Delta T), the Fourier origin at the segment start

  bool linear_envelope;     // carry the antenna and amplitude slopes through the moment
  bool first_order_delays;  // carry every delay to first order across the segment
  int tdi_generation;       // 1 or 2
  int num_side_bins;        // half-width of a row's band around its carrier bin
  double chirp_rate_floor;  // sign-preserving floor on the local chirp rate [Hz/s]
};

// Settings of a launch: the grid and window of an STFTFresnel and the DFFR knobs. Host only.
// ! Raises std::invalid_argument for a setup DFFR cannot represent.
inline DFFRSettings dffr_make_settings(const STFTFresnel &fresnel, bool linear_envelope,
                                       bool first_order_delays, int tdi_generation,
                                       int num_side_bins, double chirp_rate_floor) {
  const double segment_duration = fresnel.dt;
  const double window_taper = fresnel.window_alpha;
  if (!(segment_duration > 0.0))
    throw std::invalid_argument("dffr: the segment duration dt must be positive.");
  if (!(window_taper >= 0.0 && window_taper <= 1.0))
    throw std::invalid_argument("dffr: window_alpha must lie in [0, 1], got " +
                                std::to_string(window_taper) + ".");
  // ! The band-bin phases of the window endpoints reduce to signs only when df * dt = 1.
  if (!(std::fabs(fresnel.df * segment_duration - 1.0) <= 1.0e-12))
    throw std::invalid_argument("dffr: df * dt must equal one to within 1e-12, got " +
                                std::to_string(fresnel.df * segment_duration) + ".");
  if (fresnel.num_times < 1 || fresnel.num_freqs < 1)
    throw std::invalid_argument("dffr: the grid needs at least one segment and one bin.");
  if (tdi_generation != 1 && tdi_generation != 2)
    throw std::invalid_argument("dffr: tdi_generation must be 1 or 2, got " +
                                std::to_string(tdi_generation) + ".");
  if (num_side_bins < 0)
    throw std::invalid_argument("dffr: n_side_bins must not be negative.");
  if (!(chirp_rate_floor > 0.0))
    throw std::invalid_argument("dffr: chirp_rate_floor must be positive.");

  DFFRSettings settings;
  settings.num_time_segments = fresnel.num_times;
  settings.segment_duration = segment_duration;
  settings.half_duration = 0.5 * segment_duration;
  settings.f_min = fresnel.f_min;
  settings.df = fresnel.df;
  settings.num_bins = fresnel.num_freqs;
  settings.window_taper = window_taper;
  settings.plateau_half_width = 0.5 * segment_duration * (1.0 - window_taper);
  if (window_taper > 0.0) {
    settings.taper_frequency = 1.0 / (window_taper * segment_duration);
    settings.taper_phase_offset = M_PI * ((1.0 / window_taper) - 1.0);
  } else {
    settings.taper_frequency = 0.0;
    settings.taper_phase_offset = 0.0;
  }
  // * Built from the phase offset, never from 2 pi f_taper tau_0: only this form makes the taper
  // * sidebands collapse exactly to +-1 at the window edges.
  settings.taper_phasor = dffr_polar_unit(settings.taper_phase_offset);
  settings.origin_phasor_base = dffr_polar_unit(-M_PI * settings.f_min * segment_duration);
  settings.linear_envelope = linear_envelope;
  settings.first_order_delays = first_order_delays;
  settings.tdi_generation = tdi_generation;
  settings.num_side_bins = num_side_bins;
  settings.chirp_rate_floor = chirp_rate_floor;
  return settings;
}

// Check that a kinematics table holds the midpoints t0 + (i + 1/2) dt of the grid. Host only.
// ! Raises std::invalid_argument otherwise.
inline void dffr_check_table(const DFFRSettings &settings, const KinematicsTable &table,
                             double grid_t0) {
  if (table.num_time_segments != settings.num_time_segments)
    throw std::invalid_argument("dffr: the kinematics table has " +
                                std::to_string(table.num_time_segments) +
                                " segments, the grid " +
                                std::to_string(settings.num_time_segments) + ".");
  // * One microsecond: the table is built from the same t0 and dt, so the two agree to rounding.
  const double expected_midpoint = grid_t0 + 0.5 * settings.segment_duration;
  if (!(std::fabs(table.first_segment_midpoint - expected_midpoint) <= 1.0e-6))
    throw std::invalid_argument("dffr: the kinematics table starts at the midpoint " +
                                std::to_string(table.first_segment_midpoint) +
                                " s, the grid at " + std::to_string(expected_midpoint) + " s.");
  if (!(std::fabs(table.segment_time - settings.segment_duration) <=
        1.0e-12 * settings.segment_duration))
    throw std::invalid_argument("dffr: the kinematics table has a segment spacing of " +
                                std::to_string(table.segment_time) + " s, the grid dt " +
                                std::to_string(settings.segment_duration) + " s.");
}

// ===========================================================================
// Quadratic phase
// ===========================================================================

// Quadratic phase expansion of one state at a retarded time.
struct DFFRQuadPhaseExpansion {
  double reference_phase;
  double instantaneous_frequency;  // in retarded time [Hz]
  double frequency_rate;           // d f / d T [Hz/s]
  double amplitude_ratio;          // A(T) over the amplitude that enters the link amplitude
};

// Sign-preserving floor on the local chirp rate. The transform carries 1 / sqrt(2 |alpha|) and a
// stationary offset (f_eff - f) / alpha, and the Doppler-dominated alpha crosses zero twice a year.
CUDA_DEVICE
inline double dffr_apply_chirp_rate_floor(double chirp_rate, double floor_value) {
  if (std::fabs(chirp_rate) < floor_value) return (chirp_rate < 0.0) ? -floor_value : floor_value;
  return chirp_rate;
}

// What a sub-interval needs from the chirp rate, formed once per reception state and segment.
struct DFFRQuadPhaseScalars {
  double chirp_rate;
  double scale_factor;      // sqrt(2 |alpha|)
  double inverse_scale;     // 1 / scale_factor
  double reciprocal_chirp;  // 1 / alpha, signed
  double moment_prefactor;  // scale_factor / (2 pi alpha)
  double chirp_sign;
};

CUDA_DEVICE
inline DFFRQuadPhaseScalars dffr_make_quad_phase_scalars(double chirp_rate) {
  DFFRQuadPhaseScalars scalars;
  scalars.chirp_rate = chirp_rate;
  scalars.scale_factor = std::sqrt(2.0 * std::fabs(chirp_rate));
  scalars.inverse_scale = 1.0 / scalars.scale_factor;
  scalars.reciprocal_chirp = 1.0 / chirp_rate;
  scalars.moment_prefactor = scalars.scale_factor * scalars.reciprocal_chirp / DFFR_TWO_PI;
  scalars.chirp_sign = (chirp_rate > 0.0) ? 1.0 : -1.0;
  return scalars;
}

// ===========================================================================
// The bounded sub-interval integral and its first moment
// ===========================================================================
//   I(a, b) = int_a^b exp(i [2 pi detuning tau + pi alpha tau^2]) dtau,
// and the same integrand weighted by tau for the moment.
//
// The stationary point zeta = detuning / alpha and the phase exp(-i pi alpha zeta^2) diverge as
// alpha tends to zero; zeta reaches 1e17 s. The same phase appears with the opposite sign inside
// the auxiliary form and cancels analytically, which leaves the bounded endpoint phase
// psi(tau) = pi alpha tau^2 + 2 pi detuning tau. The constant term that still carries the large
// phase vanishes whenever both endpoints lie on one side of the stationary point, which is
// exactly where that phase is large.
//
// The endpoint phasors exp(i psi) arrive precomputed and already carry the window weight and the
// taper sideband phase; outer_weight is the window weight alone, which the bounded branch needs
// because it does not attach to an endpoint.

CUDA_DEVICE
inline void dffr_accumulate_sub_interval(const DFFRQuadPhaseScalars &scalars, double lower_bound,
                                         double upper_bound, double detuning,
                                         double stationary_offset, const cmplx &rotation_lower,
                                         const cmplx &rotation_upper, const cmplx &outer_weight,
                                         bool with_moment, cmplx *total, cmplx *total_moment) {
  if (!(upper_bound > lower_bound)) return;

  cmplx oscillating_part(0.0, 0.0);
  cmplx oscillating_moment(0.0, 0.0);
  cmplx series_part(0.0, 0.0);
  double constant_weight = 0.0;
  bool needs_stationary = false;

  // * Index 0 is the upper endpoint with weight +1, index 1 the lower with weight -1.
  const double endpoint_time[2] = {upper_bound, lower_bound};
  const double endpoint_weight[2] = {1.0, -1.0};

#pragma unroll
  for (int which = 0; which < 2; which += 1) {
    const double tau = endpoint_time[which];
    const double weight = endpoint_weight[which];
    const cmplx rotation = (which == 0) ? rotation_upper : rotation_lower;

    const double fresnel_argument = scalars.scale_factor * (tau + stationary_offset);
    const double absolute_argument = std::fabs(fresnel_argument);
    const double argument_sign = (fresnel_argument >= 0.0) ? 1.0 : -1.0;

    if (absolute_argument > LAT_FRESNEL_SEAM) {
      double auxiliary_f, auxiliary_g, slope_f, slope_g;
      lat_fresnel_aux(&auxiliary_f, &auxiliary_g, absolute_argument,
                      with_moment ? &slope_f : nullptr, with_moment ? &slope_g : nullptr);
      const cmplx auxiliary_complex(auxiliary_g, scalars.chirp_sign * auxiliary_f);
      const double signed_weight = weight * argument_sign;
      oscillating_part = oscillating_part - signed_weight * dffr_mul(rotation, auxiliary_complex);
      constant_weight += signed_weight;
      if (with_moment) {
        // * (1 / 2 pi i) d/d(detuning) of this endpoint term, with the bounded phase.
        const cmplx auxiliary_slope(slope_g, scalars.chirp_sign * slope_f);
        const cmplx bracket = dffr_mul(cmplx(0.0, scalars.moment_prefactor), auxiliary_slope) -
                              (argument_sign * tau) * auxiliary_complex;
        oscillating_moment = oscillating_moment + weight * dffr_mul(rotation, bracket);
      }
    } else {
      // * Inside the seam the series is used directly; the stationary factor applied below is
      // * then bounded, because |zeta| <= |tau| + 1.6 / sqrt(2 |alpha|).
      double cosine_value, sine_value;
      lat_fresnel_series(&cosine_value, &sine_value, fresnel_argument);
      series_part = series_part + weight * cmplx(cosine_value, scalars.chirp_sign * sine_value);
      needs_stationary = true;
    }
  }

  // ! The stationary factor is formed only near the stationary point or where the two constant
  // ! terms fail to cancel; everywhere else the large phase never appears.
  const bool bounded = needs_stationary || (constant_weight != 0.0);

  cmplx kernel = oscillating_part;
  if (bounded) {
    const cmplx stationary_factor =
        dffr_polar_unit(-M_PI * scalars.chirp_rate * stationary_offset * stationary_offset);
    const cmplx flat_part = series_part + constant_weight * cmplx(0.5, 0.5 * scalars.chirp_sign);
    kernel = kernel + dffr_mul(outer_weight, dffr_mul(stationary_factor, flat_part));
  }
  *total = *total + scalars.inverse_scale * kernel;

  if (!with_moment) return;

  cmplx moment = oscillating_moment;
  if (bounded) {
    // ! Near the stationary point the endpoint difference is written as
    // ! 2 i sin(dpsi / 2) exp(i (psi_a + dpsi / 2)), because both cosines round to one.
    const double phase_upper = M_PI * scalars.chirp_rate * upper_bound * upper_bound +
                               DFFR_TWO_PI * detuning * upper_bound;
    const double phase_lower = M_PI * scalars.chirp_rate * lower_bound * lower_bound +
                               DFFR_TWO_PI * detuning * lower_bound;
    const double half_difference = 0.5 * (phase_upper - phase_lower);
    const cmplx phase_difference = dffr_mul(cmplx(0.0, 2.0 * std::sin(half_difference)),
                                            dffr_polar_unit(phase_lower + half_difference));
    moment = dffr_mul(outer_weight,
                      dffr_mul(cmplx(0.0, -scalars.moment_prefactor), phase_difference)) -
             stationary_offset * kernel;
  }
  *total_moment = *total_moment + scalars.inverse_scale * moment;
}

// ===========================================================================
// The windowed chirp transform
// ===========================================================================
// A centred Tukey window splits into six bounded integrals: the full span and the plateau at the
// unshifted detuning, plus two sidebands on each taper with the phases exp(+-i theta_rho).
// Anchoring at the segment midpoint makes those phases constant.
//
// The twelve endpoint phases take four values of tau and three detunings, and each factorises into
// a per-segment piece, a per-band-bin piece and a constant. The per-segment and per-band-bin
// products arrive as outer_plus, outer_minus, inner_plus and inner_minus.
// * With f_taper = 1 / (rho Delta T) and tau_0 = (Delta T / 2)(1 - rho), each sideband's weight
// * times its own shift phasor is exactly +1 at the plateau edge and -1 at the outer edge, so both
// * sidebands of one taper share endpoint phasors.

// * The six integrals before the overall phasor: the sums that the stencil variants of the Fisher
// * matrix share when their reception states agree.
CUDA_DEVICE
inline void dffr_windowed_sums(const DFFRSettings &settings, const DFFRQuadPhaseScalars &scalars,
                               double effective_frequency, double frequency,
                               const cmplx &outer_plus, const cmplx &outer_minus,
                               const cmplx &inner_plus, const cmplx &inner_minus, bool with_moment,
                               cmplx *total_out, cmplx *moment_out) {
  const double detuning_base = effective_frequency - frequency;
  const double offset_base = detuning_base * scalars.reciprocal_chirp;
  const double half_duration = settings.half_duration;
  const double plateau = settings.plateau_half_width;
  const cmplx unit_weight(1.0, 0.0);

  cmplx total(0.0, 0.0);
  cmplx total_moment(0.0, 0.0);

  if (settings.window_taper <= 0.0) {
    // * A boxcar keeps only the full span.
    dffr_accumulate_sub_interval(scalars, -half_duration, half_duration, detuning_base,
                                 offset_base, outer_minus, outer_plus, unit_weight, with_moment,
                                 &total, &total_moment);
  } else {
    const cmplx half_weight(0.5, 0.0);
    dffr_accumulate_sub_interval(scalars, -half_duration, half_duration, detuning_base,
                                 offset_base, 0.5 * outer_minus, 0.5 * outer_plus, half_weight,
                                 with_moment, &total, &total_moment);
    dffr_accumulate_sub_interval(scalars, -plateau, plateau, detuning_base, offset_base,
                                 0.5 * inner_minus, 0.5 * inner_plus, half_weight, with_moment,
                                 &total, &total_moment);

    const double detuning_plus = detuning_base + settings.taper_frequency;
    const double detuning_minus = detuning_base - settings.taper_frequency;
    // * One stationary offset per detuning, not per sub-interval: half the divisions.
    const double offset_plus = detuning_plus * scalars.reciprocal_chirp;
    const double offset_minus = detuning_minus * scalars.reciprocal_chirp;
    const cmplx weight_positive = 0.25 * settings.taper_phasor;
    const cmplx weight_negative = 0.25 * gcmplx::conj(settings.taper_phasor);

    const cmplx left_lower = -0.25 * outer_minus;
    const cmplx left_upper = 0.25 * inner_minus;
    dffr_accumulate_sub_interval(scalars, -half_duration, -plateau, detuning_plus, offset_plus,
                                 left_lower, left_upper, weight_positive, with_moment, &total,
                                 &total_moment);
    dffr_accumulate_sub_interval(scalars, -half_duration, -plateau, detuning_minus, offset_minus,
                                 left_lower, left_upper, weight_negative, with_moment, &total,
                                 &total_moment);

    const cmplx right_lower = 0.25 * inner_plus;
    const cmplx right_upper = -0.25 * outer_plus;
    dffr_accumulate_sub_interval(scalars, plateau, half_duration, detuning_plus, offset_plus,
                                 right_lower, right_upper, weight_negative, with_moment, &total,
                                 &total_moment);
    dffr_accumulate_sub_interval(scalars, plateau, half_duration, detuning_minus, offset_minus,
                                 right_lower, right_upper, weight_positive, with_moment, &total,
                                 &total_moment);
  }

  *total_out = total;
  *moment_out = total_moment;
}

// The windowed transform of one reception state: its windowed sums times the overall phasor.
CUDA_DEVICE
inline void dffr_evaluate_windowed_kernel(const DFFRSettings &settings,
                                          const DFFRQuadPhaseScalars &scalars,
                                          double effective_frequency, double frequency,
                                          const cmplx &outer_plus, const cmplx &outer_minus,
                                          const cmplx &inner_plus, const cmplx &inner_minus,
                                          const cmplx &overall, bool with_moment,
                                          cmplx *kernel_out, cmplx *moment_out) {
  cmplx total, total_moment;
  dffr_windowed_sums(settings, scalars, effective_frequency, frequency, outer_plus, outer_minus,
                     inner_plus, inner_minus, with_moment, &total, &total_moment);
  *kernel_out = dffr_mul(overall, total);
  *moment_out = with_moment ? dffr_mul(overall, total_moment) : cmplx(0.0, 0.0);
}

// ===========================================================================
// Source-model contract
// ===========================================================================
// A source type SourceT supplies, all as CUDA_DEVICE static members:
//   Constants                       POD, one per template in shared memory, with a
//                                   DFFRSourceGeometry member named geometry
//   num_components                  compile-time number of components (phase expansions)
//   load(Constants*, params)        fill the constants from the parameters; one thread calls it
//   time_reference(c)               epoch of the phase model, in the time frame of the orbits [s]
//   amplitude(c)                    amplitude shared by the components
//   plus_weight(c, k)               weights of the plus and cross projections of component k in
//   cross_weight(c, k)                the link amplitude, LAT's strain factor -2 included
//   expand(c, k, T)                 DFFRQuadPhaseExpansion of component k at retarded time T,
//                                   counted from time_reference
//   amplitude_log_slope(c, k, T)    d ln A_k / dt at the segment midpoint T, also from the epoch
//   reception_invariant_params      bit mask, bit k set when parameter k changes no reception
//                                   state (effective frequency, chirp rate, band start) but only
//                                   the reference phasors and the TDI weights; the Fisher matrix
//                                   evaluates the reception sums of such variants once
// ! The segment table and the pixel evaluate component 0 only, and a row's band follows its
// ! carrier: components belong in one source only if they share that band. Harmonics, hundreds of
// ! bins apart, belong in separate templates.

// Sky basis and polarisation of one source, shared by its components; the basis of
// LISATDIonTheFly::get_sky_vectors.
struct DFFRSourceGeometry {
  double propagation_vector[3];  // from the source towards the barycentre
  double u_vector[3];
  double v_vector[3];
  double cos_two_polarization;
  double sin_two_polarization;
};

CUDA_DEVICE
inline void dffr_fill_source_geometry(DFFRSourceGeometry *geometry, double polarization,
                                      double longitude, double latitude) {
  const double cos_latitude = std::cos(latitude);
  const double sin_latitude = std::sin(latitude);
  const double cos_longitude = std::cos(longitude);
  const double sin_longitude = std::sin(longitude);

  // * The wave travels from the source towards the barycentre; the minus sign makes the
  // * propagation delay enter as t + d_A(t).
  geometry->propagation_vector[0] = -cos_latitude * cos_longitude;
  geometry->propagation_vector[1] = -cos_latitude * sin_longitude;
  geometry->propagation_vector[2] = -sin_latitude;

  geometry->u_vector[0] = sin_longitude;
  geometry->u_vector[1] = -cos_longitude;
  geometry->u_vector[2] = 0.0;

  geometry->v_vector[0] = -sin_latitude * cos_longitude;
  geometry->v_vector[1] = -sin_latitude * sin_longitude;
  geometry->v_vector[2] = cos_latitude;

  geometry->cos_two_polarization = std::cos(2.0 * polarization);
  geometry->sin_two_polarization = std::sin(2.0 * polarization);
}

// ===========================================================================
// The segment table
// ===========================================================================
// Everything in a segment that does not depend on the frequency bin, for Rows segments. Reception
// states 0..2 are those of spacecraft 1..3, at slot row * 3 + spacecraft; per link only what the
// sender delay changes, at slot row * 6 + link. Complex arrays first, so that a 16-byte aligned
// table keeps all of them aligned. Fixed sizes keep every offset a compile-time constant.

template <int Rows>
struct DFFRSegmentTable {
  static_assert(Rows >= 1, "a segment table holds at least one row");
  cmplx reference_phasor[Rows * DFFR_NUM_SPACECRAFT];    // amplitude ratio times exp(i phase)
  cmplx anchor_outer_plus[Rows * DFFR_NUM_SPACECRAFT];   // endpoint phasor at +Delta T / 2
  cmplx anchor_outer_minus[Rows * DFFR_NUM_SPACECRAFT];  // at -Delta T / 2
  cmplx anchor_inner_plus[Rows * DFFR_NUM_SPACECRAFT];   // at +tau_0
  cmplx anchor_inner_minus[Rows * DFFR_NUM_SPACECRAFT];  // at -tau_0
  cmplx link_amplitude[Rows * DFFR_NUM_LINKS];
  cmplx envelope_slope[Rows * DFFR_NUM_LINKS];
  cmplx emission_phasor[Rows * DFFR_NUM_LINKS];
  cmplx tdi_weights[Rows * DFFR_TDI_WEIGHTS];  // per channel and spacecraft: kernel, then moment
  double effective_frequency[Rows * DFFR_NUM_SPACECRAFT];
  double chirp_rate[Rows * DFFR_NUM_SPACECRAFT];
  double scale_factor[Rows * DFFR_NUM_SPACECRAFT];
  double inverse_scale[Rows * DFFR_NUM_SPACECRAFT];
  double reciprocal_chirp[Rows * DFFR_NUM_SPACECRAFT];
  double moment_prefactor[Rows * DFFR_NUM_SPACECRAFT];
  double emission_frequency[Rows * DFFR_NUM_LINKS];  // carrier of the emission state
  int start_freq_ind[Rows];                          // carrier bin - num_side_bins
};

// * Per row: 3 reception states of 5 complex and 6 double, 6 links of 3 complex and one double,
// * 18 complex weights and one int.
#define DFFR_SEGMENT_TABLE_BYTES_PER_ROW 1012
static_assert(sizeof(DFFRSegmentTable<32>) == 32 * DFFR_SEGMENT_TABLE_BYTES_PER_ROW,
              "the segment table no longer holds 1012 bytes per row");

// Store the bin-independent quantities of one reception state.
template <int Rows>
CUDA_DEVICE inline void dffr_store_reception_state(DFFRSegmentTable<Rows> &table,
                                                   const DFFRSettings &settings, int slot,
                                                   const DFFRQuadPhaseExpansion &expansion,
                                                   double first_derivative,
                                                   double second_derivative) {
  const double chirp_rate = dffr_apply_chirp_rate_floor(
      expansion.frequency_rate * first_derivative * first_derivative +
          expansion.instantaneous_frequency * second_derivative,
      settings.chirp_rate_floor);
  const DFFRQuadPhaseScalars scalars = dffr_make_quad_phase_scalars(chirp_rate);

  table.effective_frequency[slot] = expansion.instantaneous_frequency * first_derivative;
  table.chirp_rate[slot] = scalars.chirp_rate;
  table.scale_factor[slot] = scalars.scale_factor;
  table.inverse_scale[slot] = scalars.inverse_scale;
  table.reciprocal_chirp[slot] = scalars.reciprocal_chirp;
  table.moment_prefactor[slot] = scalars.moment_prefactor;
  // * The amplitude ratio multiplies the kernel and its moment alike, so it folds into the
  // * reference phasor together with the phase.
  table.reference_phasor[slot] =
      expansion.amplitude_ratio * dffr_polar_unit(expansion.reference_phase);
}

// The emission phasor of a link over the reception phasor of its sender: their exact amplitude
// ratio and phase difference at the segment anchor.
template <int Rows>
CUDA_DEVICE inline cmplx dffr_emission_rotation(const DFFRSegmentTable<Rows> &table, int link_slot,
                                                int sender_slot) {
  const cmplx sender_reference = table.reference_phasor[sender_slot];
  const double sender_norm = sender_reference.real() * sender_reference.real() +
                             sender_reference.imag() * sender_reference.imag();
  // ! A sender without signal has no ratio; zero keeps NaN out of the weights and of the global
  // ! grid that the atomics share between sources.
  if (!(sender_norm > 0.0)) return cmplx(0.0, 0.0);
  return (1.0 / sender_norm) *
         dffr_mul(table.emission_phasor[link_slot], gcmplx::conj(sender_reference));
}

// Add one link's two channel coefficients to the eighteen weights of its row. Each link difference
// becomes a combination of the three reception kernels and moments.
template <int Rows>
CUDA_DEVICE inline void dffr_fold_link_weights(const DFFRSegmentTable<Rows> &table, int row,
                                               int link_index, const cmplx &first_coefficient,
                                               const cmplx &second_coefficient,
                                               bool sender_delay_moment, cmplx *weights) {
  const int link_slot = row * DFFR_NUM_LINKS + link_index;
  const int sender_index = dffr_link_sender(link_index);
  const int receiver_index = dffr_link_receiver(link_index);
  const int sender_slot = row * DFFR_NUM_SPACECRAFT + sender_index;
  const cmplx antenna = 0.5 * gcmplx::conj(table.link_amplitude[link_slot]);
  const cmplx slope = table.envelope_slope[link_slot];
  const cmplx rotation = dffr_emission_rotation(table, link_slot, sender_slot);
  // * The carrier offset shifts the rebuilt state by 2 pi i offset times its first moment;
  // * switched off, the sender keeps only its exact phase at the anchor.
  const cmplx moment_factor =
      sender_delay_moment ? cmplx(0.0, DFFR_TWO_PI * (table.emission_frequency[link_slot] -
                                                     table.effective_frequency[sender_slot])) +
                                slope
                          : slope;

  const int channel_of[2] = {dffr_link_channel_first(link_index),
                             dffr_link_channel_second(link_index)};
  const cmplx coefficient_of[2] = {first_coefficient, second_coefficient};
#pragma unroll
  for (int which = 0; which < 2; which += 1) {
    const cmplx link_weight = dffr_mul(coefficient_of[which], antenna);
    const cmplx sender_weight = dffr_mul(link_weight, rotation);
    cmplx *channel_weights = weights + 6 * channel_of[which];
    channel_weights[2 * sender_index] = channel_weights[2 * sender_index] + sender_weight;
    channel_weights[2 * sender_index + 1] =
        channel_weights[2 * sender_index + 1] + dffr_mul(sender_weight, moment_factor);
    channel_weights[2 * receiver_index] = channel_weights[2 * receiver_index] - link_weight;
    channel_weights[2 * receiver_index + 1] =
        channel_weights[2 * receiver_index + 1] - dffr_mul(link_weight, slope);
  }
}

// The two TDI coefficients of every link at zeroth order in the delays: link l enters channel
// dffr_link_channel_first(l) with first[l] and dffr_link_channel_second(l) with second[l].
// Channel X of the 2nd generation is
//   (1 - R_13 R_12) [(1 - R_12)(eta_13 + z_13 eta_31) - (1 - R_13)(eta_12 + z_12 eta_21)],
// with round trips R_ij = z_ij z_ji; the 1st generation drops the factor (1 - R_13 R_12), and Y
// and Z follow by cyclic permutation. Both are the combinations of LAT's TDIConfig.
CUDA_CALLABLE_MEMBER
inline void dffr_link_coefficients(const cmplx *delay_phasor, int generation, cmplx *first,
                                   cmplx *second) {
  const cmplx one(1.0, 0.0);
  const cmplx round_trip_121 = dffr_mul(delay_phasor[0], delay_phasor[5]);
  const cmplx round_trip_232 = dffr_mul(delay_phasor[1], delay_phasor[4]);
  const cmplx round_trip_313 = dffr_mul(delay_phasor[2], delay_phasor[3]);

  const cmplx complement_121 = one - round_trip_121;
  const cmplx complement_232 = one - round_trip_232;
  const cmplx complement_313 = one - round_trip_313;

  // * The factor that turns the 1st generation into the 2nd.
  const bool second_generation = (generation == 2);
  const cmplx second_x = second_generation ? one - dffr_mul(round_trip_313, round_trip_121) : one;
  const cmplx second_y = second_generation ? one - dffr_mul(round_trip_121, round_trip_232) : one;
  const cmplx second_z = second_generation ? one - dffr_mul(round_trip_232, round_trip_313) : one;

  const cmplx x_on_131 = dffr_mul(second_x, complement_121);
  const cmplx x_on_121 = dffr_mul(second_x, complement_313);
  const cmplx y_on_212 = dffr_mul(second_y, complement_232);
  const cmplx y_on_232 = dffr_mul(second_y, complement_121);
  const cmplx z_on_323 = dffr_mul(second_z, complement_313);
  const cmplx z_on_313 = dffr_mul(second_z, complement_232);

  first[0] = -x_on_121;
  second[0] = dffr_mul(y_on_212, delay_phasor[5]);
  first[1] = -y_on_232;
  second[1] = dffr_mul(z_on_323, delay_phasor[4]);
  first[2] = dffr_mul(x_on_131, delay_phasor[3]);
  second[2] = -z_on_313;
  first[3] = x_on_131;
  second[3] = -dffr_mul(z_on_313, delay_phasor[2]);
  first[4] = -dffr_mul(y_on_232, delay_phasor[1]);
  second[4] = z_on_323;
  first[5] = -dffr_mul(x_on_121, delay_phasor[0]);
  second[5] = y_on_212;
}

// The eighteen weights of one row at zeroth order in the delays.
template <int Rows>
CUDA_DEVICE inline void dffr_store_channel_weights(DFFRSegmentTable<Rows> &table, int row,
                                                   const cmplx *delay_phasor,
                                                   bool sender_delay_moment, int generation) {
  cmplx first[DFFR_NUM_LINKS];
  cmplx second[DFFR_NUM_LINKS];
  dffr_link_coefficients(delay_phasor, generation, first, second);

  // ? One thread owns a row here, so the weights accumulate without atomics.
  cmplx *weights = table.tdi_weights + row * DFFR_TDI_WEIGHTS;
  for (int entry = 0; entry < DFFR_TDI_WEIGHTS; entry += 1) weights[entry] = cmplx(0.0, 0.0);
#pragma unroll
  for (int link_index = 0; link_index < DFFR_NUM_LINKS; link_index += 1)
    dffr_fold_link_weights(table, row, link_index, first[link_index], second[link_index],
                           sender_delay_moment, weights);
}

// ===========================================================================
// TDI delays to first order across the segment
// ===========================================================================
// A delayed copy carries the carrier offset df = -(f dl/dt + alpha l) and the antenna factor of an
// earlier epoch, F(t - D) = F (1 - kappa D). Through the product rule each TDI coefficient W gains
// a rate W', on the link moment, and a delay-weighted part E, which the antenna slope turns into
// -kappa E.

struct DFFRTDIChainExpansion {
  cmplx value;     // the phasor product
  cmplx rate;      // its derivative across the segment
  cmplx weighted;  // each phasor-product term times its total delay
};

CUDA_DEVICE
inline DFFRTDIChainExpansion dffr_chain_mul(const DFFRTDIChainExpansion &first,
                                            const DFFRTDIChainExpansion &second) {
  DFFRTDIChainExpansion product;
  product.value = dffr_mul(first.value, second.value);
  product.rate = dffr_mul(first.rate, second.value) + dffr_mul(first.value, second.rate);
  product.weighted =
      dffr_mul(first.weighted, second.value) + dffr_mul(first.value, second.weighted);
  return product;
}

CUDA_DEVICE
inline DFFRTDIChainExpansion dffr_chain_complement(const DFFRTDIChainExpansion &jet) {
  DFFRTDIChainExpansion complement;
  complement.value = cmplx(1.0, 0.0) - jet.value;
  complement.rate = -jet.rate;
  complement.weighted = -jet.weighted;
  return complement;
}

CUDA_DEVICE
inline DFFRTDIChainExpansion dffr_chain_negate(const DFFRTDIChainExpansion &jet) {
  DFFRTDIChainExpansion negated;
  negated.value = -jet.value;
  negated.rate = -jet.rate;
  negated.weighted = -jet.weighted;
  return negated;
}

// One delay at the segment midpoint as a jet.
CUDA_DEVICE
inline DFFRTDIChainExpansion dffr_make_chain_expansion(const KinematicsTable &kinematics,
                                                       int link_index, int segment_index,
                                                       double carrier_frequency,
                                                       double delay_chirp) {
  const int table_slot = link_index * kinematics.num_time_segments + segment_index;
  const double light_travel_time = kinematics.light_travel_time[table_slot];
  const cmplx phasor = dffr_polar_unit(-DFFR_TWO_PI * carrier_frequency * light_travel_time);
  // * Carrier offset of the delayed copy: the breathing arm and the chirp across the delay.
  const double delay_offset = -(carrier_frequency * kinematics.light_travel_time_rate[table_slot] +
                                delay_chirp * light_travel_time);
  DFFRTDIChainExpansion jet;
  jet.value = phasor;
  jet.rate = dffr_mul(cmplx(0.0, DFFR_TWO_PI * delay_offset), phasor);
  jet.weighted = light_travel_time * phasor;
  return jet;
}

// Add one link's first-order terms to a channel's six increments, held by place relative to the
// channel's vertex (i, j, k at places 0, 1, 2).
template <int Rows>
CUDA_DEVICE inline void dffr_fold_first_order_link(const DFFRSegmentTable<Rows> &table, int row,
                                                   int link_index, int sender_place,
                                                   int receiver_place,
                                                   const DFFRTDIChainExpansion &coefficient,
                                                   cmplx *increment) {
  const int link_slot = row * DFFR_NUM_LINKS + link_index;
  const int sender_slot = row * DFFR_NUM_SPACECRAFT + dffr_link_sender(link_index);
  const cmplx antenna = 0.5 * gcmplx::conj(table.link_amplitude[link_slot]);
  const cmplx slope = table.envelope_slope[link_slot];
  const cmplx rotation = dffr_emission_rotation(table, link_slot, sender_slot);
  const cmplx moment_factor =
      cmplx(0.0, DFFR_TWO_PI * (table.emission_frequency[link_slot] -
                                table.effective_frequency[sender_slot])) +
      slope;

  const cmplx link_change = -dffr_mul(antenna, dffr_mul(slope, coefficient.weighted));
  const cmplx link_rate = dffr_mul(antenna, coefficient.rate);
  increment[2 * sender_place] = increment[2 * sender_place] + dffr_mul(link_change, rotation);
  increment[2 * sender_place + 1] =
      increment[2 * sender_place + 1] +
      dffr_mul(rotation, dffr_mul(link_change, moment_factor) + link_rate);
  increment[2 * receiver_place] = increment[2 * receiver_place] - link_change;
  increment[2 * receiver_place + 1] =
      increment[2 * receiver_place + 1] - (dffr_mul(link_change, slope) + link_rate);
}

// The first-order increments of one channel of one row. Channel c is the combination at vertex
// i = c + 1 with arms to j = i + 1 and k = i - 1:
//   X_i = (1 - a)(1 - b a) eta_iki - (1 - b)(1 - a b) eta_iji,  a = z_ij z_ji,  b = z_ik z_ki,
// and the 1st generation drops the factors (1 - b a) and (1 - a b).
template <int Rows>
CUDA_DEVICE inline void dffr_first_order_delay_increments(const DFFRSegmentTable<Rows> &table,
                                                          const KinematicsTable &kinematics,
                                                          int row, int segment_index, int channel,
                                                          int generation, cmplx *increment) {
  // ? Link order 12, 23, 31, 13, 32, 21: the forward link of vertex i is i, its reverse 5 - i,
  // ? and the arm to k = i - 1 is ki = (i + 2) mod 3 with ik = 5 - ki.
  const int link_ij = channel;
  const int link_ji = 5 - channel;
  const int link_ki = (channel + 2) % 3;
  const int link_ik = 5 - link_ki;

  // * The carrier of the zeroth-order phasors, recomputed, and the mean reception chirp.
  double carrier_sum = 0.0;
  for (int link_index = 0; link_index < DFFR_NUM_LINKS; link_index += 1) {
    const int receiver_index = dffr_link_receiver(link_index);
    carrier_sum += 0.5 * (table.effective_frequency[row * DFFR_NUM_SPACECRAFT + receiver_index] +
                          table.emission_frequency[row * DFFR_NUM_LINKS + link_index]);
  }
  const double carrier_frequency = carrier_sum / (double)DFFR_NUM_LINKS;
  double chirp_sum = 0.0;
  for (int spacecraft_index = 0; spacecraft_index < DFFR_NUM_SPACECRAFT; spacecraft_index += 1)
    chirp_sum += table.chirp_rate[row * DFFR_NUM_SPACECRAFT + spacecraft_index];
  const double delay_chirp = chirp_sum / (double)DFFR_NUM_SPACECRAFT;

  const DFFRTDIChainExpansion delay_ij =
      dffr_make_chain_expansion(kinematics, link_ij, segment_index, carrier_frequency, delay_chirp);
  const DFFRTDIChainExpansion delay_ji =
      dffr_make_chain_expansion(kinematics, link_ji, segment_index, carrier_frequency, delay_chirp);
  const DFFRTDIChainExpansion delay_ik =
      dffr_make_chain_expansion(kinematics, link_ik, segment_index, carrier_frequency, delay_chirp);
  const DFFRTDIChainExpansion delay_ki =
      dffr_make_chain_expansion(kinematics, link_ki, segment_index, carrier_frequency, delay_chirp);

  const DFFRTDIChainExpansion round_trip_ij = dffr_chain_mul(delay_ij, delay_ji);
  const DFFRTDIChainExpansion round_trip_ik = dffr_chain_mul(delay_ik, delay_ki);
  DFFRTDIChainExpansion on_iki = dffr_chain_complement(round_trip_ij);
  DFFRTDIChainExpansion on_iji = dffr_chain_negate(dffr_chain_complement(round_trip_ik));
  if (generation == 2) {
    const DFFRTDIChainExpansion second =
        dffr_chain_complement(dffr_chain_mul(round_trip_ik, round_trip_ij));
    on_iki = dffr_chain_mul(second, dffr_chain_complement(round_trip_ij));
    on_iji = dffr_chain_negate(dffr_chain_mul(second, dffr_chain_complement(round_trip_ik)));
  }

#pragma unroll
  for (int entry = 0; entry < 6; entry += 1) increment[entry] = cmplx(0.0, 0.0);
  // * Link ij is received by i and sent by j: places (receiver 0, sender 1). Likewise ji sits at
  // * (1, 0), ik at (0, 2) and ki at (2, 0).
  dffr_fold_first_order_link(table, row, link_ik, 2, 0, on_iki, increment);
  dffr_fold_first_order_link(table, row, link_ki, 0, 2, dffr_chain_mul(on_iki, delay_ik),
                             increment);
  dffr_fold_first_order_link(table, row, link_ij, 1, 0, on_iji, increment);
  dffr_fold_first_order_link(table, row, link_ji, 0, 1, dffr_chain_mul(on_iji, delay_ij),
                             increment);
}

// Add a channel's increments to its six weights, mapping places back to spacecraft.
template <int Rows>
CUDA_DEVICE inline void dffr_add_first_order_increments(DFFRSegmentTable<Rows> &table, int row,
                                                        int channel, const cmplx *increment) {
  cmplx *channel_weights = table.tdi_weights + row * DFFR_TDI_WEIGHTS + 6 * channel;
#pragma unroll
  for (int place = 0; place < DFFR_NUM_SPACECRAFT; place += 1) {
    const int spacecraft_index = (channel + place) % DFFR_NUM_SPACECRAFT;
    channel_weights[2 * spacecraft_index] =
        channel_weights[2 * spacecraft_index] + increment[2 * place];
    channel_weights[2 * spacecraft_index + 1] =
        channel_weights[2 * spacecraft_index + 1] + increment[2 * place + 1];
  }
}

// ===========================================================================
// Building the segment table
// ===========================================================================
// Fill the table for rows_here segments from row_base on. Every thread of the block must enter:
// the body has barriers between stages that read what the previous stage wrote. The caller
// separates it from earlier readers of the table and from later readers by barriers of its own.
template <class SourceT, int Rows>
CUDA_DEVICE inline void dffr_build_segment_table(DFFRSegmentTable<Rows> &table,
                                                 const DFFRSettings &settings,
                                                 const KinematicsTable &kinematics,
                                                 const typename SourceT::Constants &source,
                                                 int row_base, int rows_here) {
  const double time_reference = SourceT::time_reference(source);
  const DFFRSourceGeometry &geometry = source.geometry;

  // --- reception events, three per segment -------------------------------------------------------
  for (int item = THREAD_START_X; item < rows_here * DFFR_NUM_SPACECRAFT; item += BLOCK_INCR_X) {
    const int spacecraft_index = item / rows_here;
    const int row = item - spacecraft_index * rows_here;
    const int segment_index = row_base + row;
    const double segment_midpoint = kinematics.segment_midpoint(segment_index);

    double projection_delay = 0.0;
    double projection_delay_rate = 0.0;
    double projection_delay_acceleration = 0.0;
    for (int xyz = 0; xyz < 3; xyz += 1) {
      const int slot = (spacecraft_index * 3 + xyz) * kinematics.num_time_segments + segment_index;
      projection_delay -= geometry.propagation_vector[xyz] * kinematics.position_receiver[slot];
      projection_delay_rate -=
          geometry.propagation_vector[xyz] * kinematics.velocity_receiver[slot];
      projection_delay_acceleration -=
          geometry.propagation_vector[xyz] * kinematics.acceleration_receiver[slot];
    }
    projection_delay /= DFFR_LIGHT_SPEED;
    projection_delay_rate /= DFFR_LIGHT_SPEED;
    projection_delay_acceleration /= DFFR_LIGHT_SPEED;

    const double retarded_time = (segment_midpoint - time_reference) + projection_delay;
    const DFFRQuadPhaseExpansion expansion = SourceT::expand(source, 0, retarded_time);
    dffr_store_reception_state(table, settings, row * DFFR_NUM_SPACECRAFT + spacecraft_index,
                               expansion, 1.0 + projection_delay_rate,
                               projection_delay_acceleration);
  }

  // --- emission events, one per link ------------------------------------------------------------
  // * Only what the delay changes is kept: the emission state's phasor and carrier.
  for (int item = THREAD_START_X; item < rows_here * DFFR_NUM_LINKS; item += BLOCK_INCR_X) {
    const int link_index = item / rows_here;
    const int row = item - link_index * rows_here;
    const int segment_index = row_base + row;
    const double segment_midpoint = kinematics.segment_midpoint(segment_index);
    const int link_slot = link_index * kinematics.num_time_segments + segment_index;

    const double light_travel_time = kinematics.light_travel_time[link_slot];
    const double light_travel_time_rate = kinematics.light_travel_time_rate[link_slot];
    const double emission_time = segment_midpoint - light_travel_time;

    double projection_delay = 0.0;
    double projection_delay_rate = 0.0;
    for (int xyz = 0; xyz < 3; xyz += 1) {
      const int slot = (link_index * 3 + xyz) * kinematics.num_time_segments + segment_index;
      projection_delay -= geometry.propagation_vector[xyz] * kinematics.position_sender[slot];
      projection_delay_rate -= geometry.propagation_vector[xyz] * kinematics.velocity_sender[slot];
    }
    projection_delay /= DFFR_LIGHT_SPEED;
    projection_delay_rate /= DFFR_LIGHT_SPEED;

    const double retarded_time = (emission_time - time_reference) + projection_delay;
    const DFFRQuadPhaseExpansion expansion = SourceT::expand(source, 0, retarded_time);

    // * The emission epoch advances at (1 - dL/dt), which rescales the sender's Doppler factor.
    const double first_derivative = (1.0 - light_travel_time_rate) * (1.0 + projection_delay_rate);
    const int slot = row * DFFR_NUM_LINKS + link_index;
    table.emission_phasor[slot] =
        expansion.amplitude_ratio * dffr_polar_unit(expansion.reference_phase);
    table.emission_frequency[slot] = expansion.instantaneous_frequency * first_derivative;
  }

  // --- link amplitude and its logarithmic slope -------------------------------------------------
  for (int item = THREAD_START_X; item < rows_here * DFFR_NUM_LINKS; item += BLOCK_INCR_X) {
    const int link_index = item / rows_here;
    const int row = item - link_index * rows_here;
    const int segment_index = row_base + row;

    double unit_direction[3], unit_direction_rate[3];
    for (int xyz = 0; xyz < 3; xyz += 1) {
      const int slot = (link_index * 3 + xyz) * kinematics.num_time_segments + segment_index;
      unit_direction[xyz] = kinematics.unit_direction[slot];
      unit_direction_rate[xyz] = kinematics.unit_direction_rate[slot];
    }

    double propagation_dot_beam = 0.0, u_dot_beam = 0.0, v_dot_beam = 0.0;
    double propagation_dot_rate = 0.0, u_dot_rate = 0.0, v_dot_rate = 0.0;
    for (int xyz = 0; xyz < 3; xyz += 1) {
      propagation_dot_beam += unit_direction[xyz] * geometry.propagation_vector[xyz];
      u_dot_beam += unit_direction[xyz] * geometry.u_vector[xyz];
      v_dot_beam += unit_direction[xyz] * geometry.v_vector[xyz];
      propagation_dot_rate += unit_direction_rate[xyz] * geometry.propagation_vector[xyz];
      u_dot_rate += unit_direction_rate[xyz] * geometry.u_vector[xyz];
      v_dot_rate += unit_direction_rate[xyz] * geometry.v_vector[xyz];
    }

    // * The obliquity denominator comes from integrating the null geodesic along the arm.
    const double denominator = 2.0 * (1.0 - propagation_dot_beam);
    const double inverse_denominator = 1.0 / denominator;

    const double numerator_plus = u_dot_beam * u_dot_beam - v_dot_beam * v_dot_beam;
    const double numerator_cross = 2.0 * u_dot_beam * v_dot_beam;
    const double antenna_plus = numerator_plus * inverse_denominator;
    const double antenna_cross = numerator_cross * inverse_denominator;

    const double amplitude_plus = antenna_plus * geometry.cos_two_polarization +
                                  antenna_cross * geometry.sin_two_polarization;
    const double amplitude_cross = -antenna_plus * geometry.sin_two_polarization +
                                   antenna_cross * geometry.cos_two_polarization;

    const double amplitude = SourceT::amplitude(source);
    const double plus_weight = SourceT::plus_weight(source, 0);
    const double cross_weight = SourceT::cross_weight(source, 0);
    const int slot = row * DFFR_NUM_LINKS + link_index;
    const cmplx factor(amplitude * plus_weight * amplitude_plus,
                       amplitude * cross_weight * amplitude_cross);
    table.link_amplitude[slot] = factor;

    if (settings.linear_envelope) {
      const double numerator_plus_rate = 2.0 * (u_dot_beam * u_dot_rate - v_dot_beam * v_dot_rate);
      const double numerator_cross_rate = 2.0 * (u_dot_rate * v_dot_beam + u_dot_beam * v_dot_rate);
      // * d(1 / denominator)/dt = 2 (k . ndot) / denominator^2.
      const double antenna_plus_rate =
          numerator_plus_rate * inverse_denominator +
          2.0 * numerator_plus * propagation_dot_rate * inverse_denominator * inverse_denominator;
      const double antenna_cross_rate =
          numerator_cross_rate * inverse_denominator +
          2.0 * numerator_cross * propagation_dot_rate * inverse_denominator * inverse_denominator;

      const double amplitude_plus_rate = antenna_plus_rate * geometry.cos_two_polarization +
                                         antenna_cross_rate * geometry.sin_two_polarization;
      const double amplitude_cross_rate = -antenna_plus_rate * geometry.sin_two_polarization +
                                          antenna_cross_rate * geometry.cos_two_polarization;

      const cmplx factor_rate(amplitude * plus_weight * amplitude_plus_rate,
                              amplitude * cross_weight * amplitude_cross_rate);
      // ! Where the link amplitude vanishes the slope multiplies a zero term; dividing would give
      // ! NaN, which the reductions and atomics would spread.
      const double factor_magnitude_squared =
          factor.real() * factor.real() + factor.imag() * factor.imag();
      // ! The response pairs the kernel with conj(F), so the slope is the conjugate of the
      // ! logarithmic derivative. The source amplitude slope adds to it.
      const double amplitude_slope = SourceT::amplitude_log_slope(
          source, 0, kinematics.segment_midpoint(segment_index) - time_reference);
      table.envelope_slope[slot] = ((factor_magnitude_squared > 0.0)
                                        ? gcmplx::conj(factor_rate / factor)
                                        : cmplx(0.0, 0.0)) +
                                   cmplx(amplitude_slope, 0.0);
    } else {
      table.envelope_slope[slot] = cmplx(0.0, 0.0);
    }
  }

  CUDA_SYNC_THREADS;

  // --- segment carrier, band start, delay phasors -----------------------------------------------
  for (int row = THREAD_START_X; row < rows_here; row += BLOCK_INCR_X) {
    const int segment_index = row_base + row;
    double carrier_sum = 0.0;
    for (int link_index = 0; link_index < DFFR_NUM_LINKS; link_index += 1) {
      const int receiver_index = dffr_link_receiver(link_index);
      // * The two states of a link differ only by inter-spacecraft Doppler.
      carrier_sum += 0.5 * (table.effective_frequency[row * DFFR_NUM_SPACECRAFT + receiver_index] +
                            table.emission_frequency[row * DFFR_NUM_LINKS + link_index]);
    }
    const double carrier_frequency = carrier_sum / (double)DFFR_NUM_LINKS;

    const int carrier_bin =
        (int)std::floor((carrier_frequency - settings.f_min) / settings.df + 0.5);
    table.start_freq_ind[row] = carrier_bin - settings.num_side_bins;

    cmplx delay_phasor[DFFR_NUM_LINKS];
    for (int link_index = 0; link_index < DFFR_NUM_LINKS; link_index += 1) {
      const double light_travel_time =
          kinematics.light_travel_time[link_index * kinematics.num_time_segments + segment_index];
      delay_phasor[link_index] =
          dffr_polar_unit(-DFFR_TWO_PI * carrier_frequency * light_travel_time);
    }
    dffr_store_channel_weights(table, row, delay_phasor, settings.first_order_delays,
                               settings.tdi_generation);
  }

  CUDA_SYNC_THREADS;

  // --- first-order delay terms, one channel of one row per thread -------------------------------
  if (settings.first_order_delays) {
    for (int item = THREAD_START_X; item < rows_here * DFFR_NUM_CHANNELS; item += BLOCK_INCR_X) {
      const int channel = item / rows_here;
      const int row = item - channel * rows_here;
      cmplx increment[6];
      dffr_first_order_delay_increments(table, kinematics, row, row_base + row, channel,
                                        settings.tdi_generation, increment);
      dffr_add_first_order_increments(table, row, channel, increment);
    }
  }

  // --- endpoint phasor anchors, three per segment -----------------------------------------------
  // * Anchored on the row's band start, so that both phases of the later split stay near 1e2 rad.
  for (int item = THREAD_START_X; item < rows_here * DFFR_NUM_SPACECRAFT; item += BLOCK_INCR_X) {
    const int spacecraft_index = item / rows_here;
    const int row = item - spacecraft_index * rows_here;
    const int slot = row * DFFR_NUM_SPACECRAFT + spacecraft_index;

    const double band_anchor = settings.f_min + (double)table.start_freq_ind[row] * settings.df;
    const double delta_anchor = table.effective_frequency[slot] - band_anchor;
    const double chirp_rate = table.chirp_rate[slot];

    const double outer = settings.half_duration;
    const double quadratic_outer = M_PI * chirp_rate * outer * outer;
    const double linear_outer = DFFR_TWO_PI * delta_anchor * outer;
    const cmplx outer_plus = dffr_polar_unit(quadratic_outer + linear_outer);
    const cmplx outer_minus = dffr_polar_unit(quadratic_outer - linear_outer);
    table.anchor_outer_plus[slot] = outer_plus;
    table.anchor_outer_minus[slot] = outer_minus;

    if (settings.window_taper > 0.0) {
      const double inner = settings.plateau_half_width;
      const double quadratic_inner = M_PI * chirp_rate * inner * inner;
      const double linear_inner = DFFR_TWO_PI * delta_anchor * inner;
      table.anchor_inner_plus[slot] = dffr_polar_unit(quadratic_inner + linear_inner);
      table.anchor_inner_minus[slot] = dffr_polar_unit(quadratic_inner - linear_inner);
    } else {
      // ? The boxcar never reads these; they are set so that no NaN can reach them.
      table.anchor_inner_plus[slot] = outer_plus;
      table.anchor_inner_minus[slot] = outer_minus;
    }
  }
}

// ===========================================================================
// One pixel: three windowed chirp transforms and the eighteen weights
// ===========================================================================

// Per-band-bin phase factors that every reception state of a pixel shares.
struct DFFRBandBinPhasors {
  double outer_sign;  // (-1)^band_bin for the outer endpoints
  cmplx inner;        // exp(-i pi (1 - rho) band_bin) for the plateau endpoints
  cmplx inner_conj;
};

// * The outer endpoints sit at +-Delta T / 2 and df Delta T = 1, so their band-bin phase is
// * pi band_bin: a sign. The plateau phase pi (1 - rho) band_bin is split into that sign and
// * -pi rho band_bin, folded to its nearest even multiple so the argument stays bounded. Negative
// * band bins, below the band start, come out right as well.
CUDA_DEVICE
inline DFFRBandBinPhasors dffr_make_band_bin_phasors(const DFFRSettings &settings, int band_bin) {
  DFFRBandBinPhasors band_bin_phasors;
  band_bin_phasors.outer_sign = (band_bin & 1) ? -1.0 : 1.0;
  if (settings.window_taper > 0.0) {
    const double product = (double)band_bin * settings.window_taper;
    const double folded = product - 2.0 * std::floor(0.5 * product + 0.5);
    const double residual = std::fma((double)band_bin, settings.window_taper, -product);
    band_bin_phasors.inner =
        band_bin_phasors.outer_sign * dffr_polar_unit(M_PI * (folded + residual));
  } else {
    band_bin_phasors.inner = cmplx(band_bin_phasors.outer_sign, 0.0);
  }
  band_bin_phasors.inner_conj = gcmplx::conj(band_bin_phasors.inner);
  return band_bin_phasors;
}

// The windowed sums of one reception state at one pixel, before its reference phasor and the
// origin phasor. StatesT holds the bin-independent quantities of the states at slot
// row * 3 + spacecraft: a DFFRSegmentTable, or the states of a Fisher variant
// (DFFRFisherReception).
template <class StatesT>
CUDA_DEVICE inline void dffr_reception_sums(const StatesT &states, const DFFRSettings &settings,
                                            int slot, double frequency,
                                            const DFFRBandBinPhasors &band_bin_phasors,
                                            cmplx *total_out, cmplx *moment_out) {
  DFFRQuadPhaseScalars scalars;
  scalars.chirp_rate = states.chirp_rate[slot];
  scalars.scale_factor = states.scale_factor[slot];
  scalars.inverse_scale = states.inverse_scale[slot];
  scalars.reciprocal_chirp = states.reciprocal_chirp[slot];
  scalars.moment_prefactor = states.moment_prefactor[slot];
  scalars.chirp_sign = (scalars.chirp_rate > 0.0) ? 1.0 : -1.0;

  const cmplx outer_plus = band_bin_phasors.outer_sign * states.anchor_outer_plus[slot];
  const cmplx outer_minus = band_bin_phasors.outer_sign * states.anchor_outer_minus[slot];
  const cmplx inner_plus = dffr_mul(states.anchor_inner_plus[slot], band_bin_phasors.inner);
  const cmplx inner_minus = dffr_mul(states.anchor_inner_minus[slot], band_bin_phasors.inner_conj);

  dffr_windowed_sums(settings, scalars, states.effective_frequency[slot], frequency, outer_plus,
                     outer_minus, inner_plus, inner_minus, true, total_out, moment_out);
}

template <int Rows>
CUDA_DEVICE inline void dffr_evaluate_reception_state(const DFFRSegmentTable<Rows> &table,
                                                      const DFFRSettings &settings, int row,
                                                      int spacecraft_index, double frequency,
                                                      const DFFRBandBinPhasors &band_bin_phasors,
                                                      const cmplx &origin_phasor, cmplx *kernel_out,
                                                      cmplx *moment_out) {
  const int slot = row * DFFR_NUM_SPACECRAFT + spacecraft_index;
  cmplx total, total_moment;
  dffr_reception_sums(table, settings, slot, frequency, band_bin_phasors, &total, &total_moment);
  const cmplx overall = dffr_mul(table.reference_phasor[slot], origin_phasor);
  *kernel_out = dffr_mul(overall, total);
  *moment_out = dffr_mul(overall, total_moment);
}

// Add the kernel and moment of one reception state to the three channels through the row's
// eighteen weights.
CUDA_DEVICE inline void dffr_add_reception_state(const cmplx *weights, int spacecraft_index,
                                                 const cmplx &kernel, const cmplx &moment,
                                                 cmplx *channels) {
#pragma unroll
  for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1) {
    const int entry = 6 * channel + 2 * spacecraft_index;
    channels[channel] = channels[channel] + dffr_mul(weights[entry], kernel) +
                        dffr_mul(weights[entry + 1], moment);
  }
}

// The three channels of one pixel: band_bin counts from the row's band start, bin_index from the
// grid's f_min, and frequency = f_min + bin_index df. The row's eighteen weights carry the six
// links and the whole TDI combination, so each reception state feeds the channels directly.
template <int Rows>
CUDA_DEVICE inline void dffr_evaluate_pixel(const DFFRSegmentTable<Rows> &table,
                                            const DFFRSettings &settings, int row, int band_bin,
                                            int bin_index, double frequency, cmplx *channels) {
  const DFFRBandBinPhasors band_bin_phasors = dffr_make_band_bin_phasors(settings, band_bin);

  // ? exp(-i pi f Delta T) moves the origin to the segment start; with df Delta T = 1 it is a
  // ? launch constant times (-1)^j.
  const double bin_sign = (bin_index & 1) ? -1.0 : 1.0;
  const cmplx origin_phasor = bin_sign * settings.origin_phasor_base;

  const cmplx *weights = table.tdi_weights + row * DFFR_TDI_WEIGHTS;
#pragma unroll
  for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1)
    channels[channel] = cmplx(0.0, 0.0);

  // ! Not unrolled: each iteration inlines a whole reception state, and copies of it cost more.
  for (int spacecraft_index = 0; spacecraft_index < DFFR_NUM_SPACECRAFT; spacecraft_index += 1) {
    cmplx expansion_kernel, expansion_moment;
    dffr_evaluate_reception_state(table, settings, row, spacecraft_index, frequency,
                                  band_bin_phasors, origin_phasor, &expansion_kernel,
                                  &expansion_moment);
    dffr_add_reception_state(weights, spacecraft_index, expansion_kernel, expansion_moment,
                             channels);
  }
}

// ===========================================================================
// The DFFR tile of the STFT consumers
// ===========================================================================
// * The tile builds the segment table of Rows segments per pass in its template's dynamic shared
// * memory, with all threads of the block, and maps the pixels band bin first (StftBandBinMap).
// * On the GPU Rows is the warp size, 32, so that a warp takes one band bin of 32 segments. One
// * template, 32 rows of 1012 bytes plus the source constants and the settings, takes 32 672
// * bytes, within the 49 152 a launch gets. The consumers that hold two templates (swap_ll, its
// * gradient and the F-statistic) take 65 344 bytes through the opt-in launch path, or 32 960 at
// * Rows = 16 (DFFR_ROWS_PER_PASS_HALF); both are instantiated, and the A100 benchmarks of the
// * integration choose. On the CPU a block is one thread and Rows is 1, which evaluates the pixels
// * in the order of the standalone package's CPU backend.
// * values() doubles DFFR's pixel: the consumers halve every value, the positive-frequency half of
// * a real signal, and a DFFR pixel is that half already. Doubling and halving are exact.

#ifdef __CUDACC__
#define DFFR_ROWS_PER_PASS 32
#define DFFR_ROWS_PER_PASS_HALF 16
#else
#define DFFR_ROWS_PER_PASS 1
#define DFFR_ROWS_PER_PASS_HALF 1
#endif

// Launch inputs of the DFFR tile, passed by value to the kernels: the settings, the kinematics
// table of the grid (its arrays on the device of the launch) and the source model's own inputs.
template <class SourceT>
struct DFFRTileInputs {
  DFFRSettings settings;
  KinematicsTable kinematics;
  typename SourceT::Model model;
};

template <class SourceT, int Rows>
struct DFFRTile {
  static_assert(SourceT::num_components == 1,
                "the DFFR tile evaluates component 0 only; see the source-model contract");
  using Map = StftBandBinMap<Rows>;
  using Inputs = DFFRTileInputs<SourceT>;

  struct BlockShared {
    typename Map::Shared bands;
    typename SourceT::Model model;
  };

  // * One template's dynamic shared memory: its source constants, the settings its pixels read and
  // * its segment table, so that a template keeps one pointer in registers.
  struct Arena {
    typename SourceT::Constants source;
    DFFRSettings settings;
    DFFRSegmentTable<Rows> table;
  };

  CUDA_CALLABLE_MEMBER static size_t table_bytes() { return sizeof(Arena); }

  struct Context {
    BlockShared *shared;

    // * Thread 0 publishes the source model; every consumer synchronises after the context.
    CUDA_DEVICE Context(const Inputs &inputs, STFTDomain *stft, BlockShared &shared_in)
        : shared(&shared_in) {
      (void)stft;
      if (THREAD_START_X == 0) shared->model = inputs.model;
    }

    CUDA_DEVICE int rows_per_pass() const { return Rows; }
    CUDA_DEVICE typename Map::Shared *band_shared() const { return &shared->bands; }
  };

  struct Template {
    Arena *arena;

    CUDA_DEVICE Template(Context &context, unsigned char *table)
        : arena(reinterpret_cast<Arena *>(table)) {
      (void)context;
    }

    // * One thread fills the constants; build() starts with the barrier that publishes them.
    CUDA_DEVICE void load(Context &context, double *params, int bin_i) {
      (void)bin_i;
      if (THREAD_START_X == 0) SourceT::load(&arena->source, context.shared->model, params);
    }

    CUDA_DEVICE void build(Context &context, const Inputs &inputs, int row_base, int rows_here) {
      (void)context;
      CUDA_SYNC_THREADS;
      if (THREAD_START_X == 0) arena->settings = inputs.settings;
      dffr_build_segment_table<SourceT, Rows>(arena->table, inputs.settings, inputs.kinematics,
                                              arena->source, row_base, rows_here);
      CUDA_SYNC_THREADS;
    }

    CUDA_DEVICE int carrier_bin(int row) const {
      return arena->table.start_freq_ind[row] + arena->settings.num_side_bins;
    }

    CUDA_DEVICE STFT_TILE_PIXEL_INLINE void values(int row, int freq_j_here, double freq_here,
                                                   cmplx *out) const {
      dffr_evaluate_pixel(arena->table, arena->settings, row,
                          freq_j_here - arena->table.start_freq_ind[row], freq_j_here, freq_here,
                          out);
      for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1)
        out[channel] = 2.0 * out[channel];
    }
  };
};

// Launch inputs of the tile from the bound grid, the kinematics table and the knobs; *stft_launch
// receives the domain pointer the consumers read, the device copy on the GPU. Host only.
// ! Raises std::invalid_argument for a setup DFFR cannot represent.
template <class SourceT>
inline DFFRTileInputs<SourceT> dffr_tile_inputs(STFTFresnel *fresnel, STFTDomain *stft,
                                               const KinematicsTable *kinematics,
                                               const typename SourceT::Model &model,
                                               int n_side_bins, int tdi_generation,
                                               bool linear_envelope, bool first_order_delays,
                                               double chirp_rate_floor,
                                               STFTDomain **stft_launch) {
  if (stft->tdi_type != TDI_XYZ || stft->num_channels != DFFR_NUM_CHANNELS)
    throw std::invalid_argument("dffr: the response computes the X, Y and Z channels; the domain "
                                "holds another TDI set.");
  // * The Fresnel object carries the window and the domain the grid; both must be one grid.
  if (fresnel->num_times != stft->num_times || fresnel->num_freqs != stft->num_freqs ||
      fresnel->t0 != stft->t0 || fresnel->f_min != stft->f_min || fresnel->dt != stft->dt ||
      fresnel->df != stft->df)
    throw std::invalid_argument(
        "dffr: the STFTFresnel and the STFTDomain describe different grids.");
  DFFRTileInputs<SourceT> inputs;
  inputs.settings = dffr_make_settings(*fresnel, linear_envelope, first_order_delays,
                                       tdi_generation, n_side_bins, chirp_rate_floor);
  dffr_check_table(inputs.settings, *kinematics, stft->t0);
  inputs.kinematics = *kinematics;
  inputs.model = model;
#ifdef __CUDACC__
  *stft_launch = stft_device_domain(stft);
#else
  *stft_launch = stft;
#endif
  return inputs;
}

// ===========================================================================
// The Fisher matrix
// ===========================================================================
//   F_ij = 4 df Re sum_{t,f,c,d} conj(dh_i,c) invC_cd dh_j,d,
// with dh_i the four-point central difference (two-point with easy_central_difference) of the
// template over parameter inds[i]: the estimator of stft_information_matrix_kernel
// (lat_stft_kernels.hh), with its parameter shifts, stencil weights, common support of the
// templates of one derivative, contraction and reduction. The output is exactly symmetric, and a
// step of zero or less freezes a parameter (zero row and column).
//
// Each stencil template, a variant, gets its own segment table through one scratch table, and
// keeps what its pixels read: its reference phasors and TDI weights, and its reception states
// unless the source model lists its parameter in reception_invariant_params. Those variants read
// the reception states of the unperturbed source, so a pixel evaluates their windowed sums once.
// * Each variant applies its own reference phasors and weights to its windowed sums in the order
// * of dffr_evaluate_pixel, so it equals the template that fill_global writes for the shifted
// * parameters bit for bit, and only the contraction separates this kernel from a reference built
// * from fill_global templates. For a galactic binary on 8 parameters a pixel evaluates 17 sets of
// * windowed sums (the unperturbed states and 4 for each of f0, fdot, lam and beta) for 32 variants.
// ! XYZ channels only, as the tile, so the contraction is the cross-channel one.
// * Rows per pass: the tables of every variant of one segment take about 16.5 kB on 8 parameters
// * and 19.2 kB on 9, so a pass holds 4 segments on the GPU (DFFR_FISHER_ROWS_PER_PASS), or 8
// * (its double), instead of the warp of the tile. Both are instantiated, and the A100 benchmarks
// * of the integration choose. On the CPU a block is one thread and a pass one segment.

#ifdef __CUDACC__
#define DFFR_FISHER_ROWS_PER_PASS 4
#define DFFR_FISHER_ROWS_PER_PASS_DOUBLE 8
#else
#define DFFR_FISHER_ROWS_PER_PASS 1
#define DFFR_FISHER_ROWS_PER_PASS_DOUBLE 1
#endif

// The bin-independent quantities of the reception states of one variant at slot
// row * 3 + spacecraft, under the names of DFFRSegmentTable, for dffr_reception_sums.
template <int Rows>
struct DFFRFisherReception {
  cmplx anchor_outer_plus[Rows * DFFR_NUM_SPACECRAFT];
  cmplx anchor_outer_minus[Rows * DFFR_NUM_SPACECRAFT];
  cmplx anchor_inner_plus[Rows * DFFR_NUM_SPACECRAFT];
  cmplx anchor_inner_minus[Rows * DFFR_NUM_SPACECRAFT];
  double effective_frequency[Rows * DFFR_NUM_SPACECRAFT];
  double chirp_rate[Rows * DFFR_NUM_SPACECRAFT];
  double scale_factor[Rows * DFFR_NUM_SPACECRAFT];
  double inverse_scale[Rows * DFFR_NUM_SPACECRAFT];
  double reciprocal_chirp[Rows * DFFR_NUM_SPACECRAFT];
  double moment_prefactor[Rows * DFFR_NUM_SPACECRAFT];
  int start_freq_ind[Rows];
};

// What a variant applies to the windowed sums of its reception states.
template <int Rows>
struct DFFRFisherAmplitude {
  cmplx reference_phasor[Rows * DFFR_NUM_SPACECRAFT];
  cmplx tdi_weights[Rows * DFFR_TDI_WEIGHTS];
};

// Which parameter each variant shifts and which reception states it reads, built on the host from
// inds and the steps and passed to the kernel by value. Variants run derivative by derivative, in
// stencil order: +eps, -eps, +2 eps, -2 eps.
struct DFFRFisherLayout {
  int num_derivs;
  int num_stencil;    // 4, or 2 with easy_central_difference
  int num_variants;   // num_stencil per derivative with a positive step
  int num_states;     // sets of reception states per pass
  int shared_states;  // the set of the unperturbed source, -1 when no variant reads it
  double step[STFT_FISHER_NDERIV_MAX];  // eps of each derivative; zero or less freezes it
  int first_variant[STFT_FISHER_NDERIV_MAX];
  int variant_derivative[STFT_FISHER_NTEMPLATE_MAX];
  int variant_param[STFT_FISHER_NTEMPLATE_MAX];
  double variant_shift[STFT_FISHER_NTEMPLATE_MAX];
  int variant_states[STFT_FISHER_NTEMPLATE_MAX];
  bool variant_writes_states[STFT_FISHER_NTEMPLATE_MAX];  // its build fills its set of states
};

// The layout of one launch, from host copies of inds and the steps. Host only.
// ! Raises std::invalid_argument for a derivative count or a parameter index out of range.
template <class SourceT>
inline DFFRFisherLayout dffr_information_layout(const int *inds, const double *param_eps,
                                                int nparams, int num_derivs,
                                                bool easy_central_difference) {
  if (num_derivs < 1 || num_derivs > STFT_FISHER_NDERIV_MAX)
    throw std::invalid_argument("dffr_information_matrix: num_derivs must lie in [1, " +
                                std::to_string(STFT_FISHER_NDERIV_MAX) + "], got " +
                                std::to_string(num_derivs) + ".");
  if (nparams < 1 || nparams > N_PARAMS_MAX)
    throw std::invalid_argument("dffr_information_matrix: nparams must lie in [1, " +
                                std::to_string(N_PARAMS_MAX) + "], got " +
                                std::to_string(nparams) + ".");
  DFFRFisherLayout layout;
  layout.num_derivs = num_derivs;
  layout.num_stencil = easy_central_difference ? 2 : 4;
  layout.num_variants = 0;
  layout.num_states = 0;
  layout.shared_states = -1;
  for (int derivative = 0; derivative < num_derivs; derivative += 1) {
    const int param = inds[derivative];
    if (param < 0 || param >= nparams)
      throw std::invalid_argument("dffr_information_matrix: inds must lie in [0, " +
                                  std::to_string(nparams) + "), got " + std::to_string(param) +
                                  ".");
    const double eps = param_eps[param];
    layout.step[derivative] = eps;
    layout.first_variant[derivative] = layout.num_variants;
    if (!(eps > 0.0)) continue;
    // * The shifts of stft_eval_block_information, which repeat the reference's p[ind] += eps,
    // * -= eps, += 2 eps and -= 2 eps exactly.
    const double shift[STFT_FISHER_NSTENCIL_MAX] = {eps, -eps, 2.0 * eps, -(2.0 * eps)};
    const bool invariant = ((SourceT::reception_invariant_params >> param) & 1u) != 0u;
    for (int point = 0; point < layout.num_stencil; point += 1) {
      const int variant = layout.num_variants;
      layout.num_variants += 1;
      layout.variant_derivative[variant] = derivative;
      layout.variant_param[variant] = param;
      layout.variant_shift[variant] = shift[point];
      if (invariant) {
        layout.variant_writes_states[variant] = (layout.shared_states < 0);
        if (layout.shared_states < 0) {
          layout.shared_states = layout.num_states;
          layout.num_states += 1;
        }
        layout.variant_states[variant] = layout.shared_states;
      } else {
        layout.variant_writes_states[variant] = true;
        layout.variant_states[variant] = layout.num_states;
        layout.num_states += 1;
      }
    }
  }
  return layout;
}

// Dynamic shared memory of the Fisher kernel: the scratch arena of the tile (source constants,
// settings, the segment table of the variant being built), the reception states and the amplitude
// parts, each section at a 16-byte boundary. After a source's last pass the final reduction stages
// its sums where the states and amplitude parts were, so that region holds at least those sums.
CUDA_CALLABLE_MEMBER inline size_t dffr_information_align(size_t bytes) {
  return (bytes + 15) & ~static_cast<size_t>(15);
}

template <class SourceT, int Rows>
CUDA_CALLABLE_MEMBER inline size_t dffr_information_states_offset() {
  return dffr_information_align(sizeof(typename DFFRTile<SourceT, Rows>::Arena));
}

template <class SourceT, int Rows>
CUDA_CALLABLE_MEMBER inline size_t dffr_information_amplitudes_offset(
    const DFFRFisherLayout &layout) {
  return dffr_information_states_offset<SourceT, Rows>() +
         dffr_information_align((size_t)layout.num_states * sizeof(DFFRFisherReception<Rows>));
}

template <class SourceT, int Rows>
inline size_t dffr_information_bytes(const DFFRFisherLayout &layout) {
  const size_t tables_end =
      dffr_information_amplitudes_offset<SourceT, Rows>(layout) +
      dffr_information_align((size_t)layout.num_variants * sizeof(DFFRFisherAmplitude<Rows>));
  const size_t staging_end =
      dffr_information_states_offset<SourceT, Rows>() +
      dffr_information_align((size_t)NUM_THREADS_HERE * layout.num_derivs * sizeof(double));
  return (tables_end > staging_end) ? tables_end : staging_end;
}

// Fill the reception states and amplitude parts of every variant for rows_here segments from
// row_base, and the lowest and highest carrier of each derivative's variants per row. Every
// thread of the block enters; it ends with a barrier.
template <class SourceT, int Rows>
CUDA_DEVICE inline void dffr_information_build(typename DFFRTile<SourceT, Rows>::Arena *scratch,
                                               DFFRFisherReception<Rows> *states,
                                               DFFRFisherAmplitude<Rows> *amplitudes,
                                               int *carrier_lo, int *carrier_hi,
                                               const DFFRTileInputs<SourceT> &inputs,
                                               const DFFRFisherLayout &layout,
                                               const double *params, int nparams, int row_base,
                                               int rows_here) {
  const DFFRSegmentTable<Rows> &table = scratch->table;
  for (int variant = 0; variant < layout.num_variants; variant += 1) {
    // * One thread loads the shifted source. The barrier after it also keeps this build from
    // * overwriting the scratch table while the copies of the previous variant still read it.
    if (THREAD_START_X == 0) {
      double shifted[N_PARAMS_MAX];
      for (int index = 0; index < nparams; index += 1) shifted[index] = params[index];
      const int param = layout.variant_param[variant];
      shifted[param] = params[param] + layout.variant_shift[variant];
      SourceT::load(&scratch->source, inputs.model, shifted);
    }
    CUDA_SYNC_THREADS;
    dffr_build_segment_table<SourceT, Rows>(scratch->table, inputs.settings, inputs.kinematics,
                                            scratch->source, row_base, rows_here);
    CUDA_SYNC_THREADS;

    DFFRFisherAmplitude<Rows> &amplitude = amplitudes[variant];
    const bool writes_states = layout.variant_writes_states[variant];
    DFFRFisherReception<Rows> &reception = states[layout.variant_states[variant]];
    // ? The first rows_here * 3 slots of the table are the states of rows 0 .. rows_here - 1.
    for (int slot = THREAD_START_X; slot < rows_here * DFFR_NUM_SPACECRAFT; slot += BLOCK_INCR_X) {
      amplitude.reference_phasor[slot] = table.reference_phasor[slot];
      if (writes_states) {
        reception.anchor_outer_plus[slot] = table.anchor_outer_plus[slot];
        reception.anchor_outer_minus[slot] = table.anchor_outer_minus[slot];
        reception.anchor_inner_plus[slot] = table.anchor_inner_plus[slot];
        reception.anchor_inner_minus[slot] = table.anchor_inner_minus[slot];
        reception.effective_frequency[slot] = table.effective_frequency[slot];
        reception.chirp_rate[slot] = table.chirp_rate[slot];
        reception.scale_factor[slot] = table.scale_factor[slot];
        reception.inverse_scale[slot] = table.inverse_scale[slot];
        reception.reciprocal_chirp[slot] = table.reciprocal_chirp[slot];
        reception.moment_prefactor[slot] = table.moment_prefactor[slot];
      }
    }
    for (int entry = THREAD_START_X; entry < rows_here * DFFR_TDI_WEIGHTS; entry += BLOCK_INCR_X)
      amplitude.tdi_weights[entry] = table.tdi_weights[entry];

    // * The same thread owns a row for every variant, so the carrier ranges need no atomics.
    const int derivative = layout.variant_derivative[variant];
    const bool first = (variant == layout.first_variant[derivative]);
    for (int row = THREAD_START_X; row < rows_here; row += BLOCK_INCR_X) {
      const int band_start = table.start_freq_ind[row];
      if (writes_states) reception.start_freq_ind[row] = band_start;
      const int carrier = band_start + inputs.settings.num_side_bins;
      const int slot = derivative * Rows + row;
      if (first || carrier < carrier_lo[slot]) carrier_lo[slot] = carrier;
      if (first || carrier > carrier_hi[slot]) carrier_hi[slot] = carrier;
    }
  }
  CUDA_SYNC_THREADS;
}

// The derivatives dh[d][c] of the three channels at one pixel of row `row`, on the common support
// of each derivative's variants and zero outside it. freq_j_here counts from the grid's f_min and
// frequency = f_min + freq_j_here df, as in the consumers.
template <int Rows>
CUDA_DEVICE inline void dffr_information_derivatives(
    const DFFRFisherReception<Rows> *states, const DFFRFisherAmplitude<Rows> *amplitudes,
    const int *carrier_lo, const int *carrier_hi, const DFFRSettings &settings,
    const DFFRFisherLayout &layout, int row, int freq_j_here, double frequency,
    cmplx dh[STFT_FISHER_NDERIV_MAX][DFFR_NUM_CHANNELS]) {
  const double bin_sign = (freq_j_here & 1) ? -1.0 : 1.0;
  const cmplx origin_phasor = bin_sign * settings.origin_phasor_base;

  // * The sums of the unperturbed states, formed at the first variant that reads them.
  bool shared_ready = false;
  cmplx shared_total[DFFR_NUM_SPACECRAFT], shared_moment[DFFR_NUM_SPACECRAFT];
  // * The band-bin phasors depend on a set's band start only, which differs between sets only
  // * where a shifted carrier crosses a bin.
  int phasor_band_start = 0;
  bool phasor_ready = false;
  DFFRBandBinPhasors band_bin_phasors;

  for (int derivative = 0; derivative < layout.num_derivs; derivative += 1) {
    const double step = layout.step[derivative];
    if (!(step > 0.0)) continue;
    const int support_lo = carrier_lo[derivative * Rows + row] - settings.num_side_bins;
    const int support_hi = carrier_hi[derivative * Rows + row] + settings.num_side_bins;
    const bool inside = (freq_j_here >= support_lo) && (freq_j_here <= support_hi);

    cmplx values[STFT_FISHER_NSTENCIL_MAX][DFFR_NUM_CHANNELS];
    for (int point = 0; point < layout.num_stencil; point += 1) {
      for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1)
        values[point][channel] = cmplx(0.0, 0.0);
      if (!inside) continue;
      const int variant = layout.first_variant[derivative] + point;
      const int set = layout.variant_states[variant];
      const bool shared = (set == layout.shared_states);

      cmplx total[DFFR_NUM_SPACECRAFT], moment[DFFR_NUM_SPACECRAFT];
      if (shared && shared_ready) {
        for (int spacecraft_index = 0; spacecraft_index < DFFR_NUM_SPACECRAFT;
             spacecraft_index += 1) {
          total[spacecraft_index] = shared_total[spacecraft_index];
          moment[spacecraft_index] = shared_moment[spacecraft_index];
        }
      } else {
        const DFFRFisherReception<Rows> &reception = states[set];
        const int band_start = reception.start_freq_ind[row];
        if (!phasor_ready || band_start != phasor_band_start) {
          band_bin_phasors = dffr_make_band_bin_phasors(settings, freq_j_here - band_start);
          phasor_band_start = band_start;
          phasor_ready = true;
        }
        // ! Not unrolled, as in dffr_evaluate_pixel: each iteration inlines a reception state.
        for (int spacecraft_index = 0; spacecraft_index < DFFR_NUM_SPACECRAFT;
             spacecraft_index += 1)
          dffr_reception_sums(reception, settings, row * DFFR_NUM_SPACECRAFT + spacecraft_index,
                              frequency, band_bin_phasors, &total[spacecraft_index],
                              &moment[spacecraft_index]);
        if (shared) {
          for (int spacecraft_index = 0; spacecraft_index < DFFR_NUM_SPACECRAFT;
               spacecraft_index += 1) {
            shared_total[spacecraft_index] = total[spacecraft_index];
            shared_moment[spacecraft_index] = moment[spacecraft_index];
          }
          shared_ready = true;
        }
      }

      // * The order of dffr_evaluate_reception_state and dffr_evaluate_pixel.
      const DFFRFisherAmplitude<Rows> &amplitude = amplitudes[variant];
      const cmplx *weights = amplitude.tdi_weights + row * DFFR_TDI_WEIGHTS;
      for (int spacecraft_index = 0; spacecraft_index < DFFR_NUM_SPACECRAFT;
           spacecraft_index += 1) {
        const cmplx overall = dffr_mul(
            amplitude.reference_phasor[row * DFFR_NUM_SPACECRAFT + spacecraft_index],
            origin_phasor);
        dffr_add_reception_state(weights, spacecraft_index,
                                 dffr_mul(overall, total[spacecraft_index]),
                                 dffr_mul(overall, moment[spacecraft_index]), values[point]);
      }
    }

    for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1) {
      if (layout.num_stencil == 2)
        dh[derivative][channel] = (values[0][channel] - values[1][channel]) / (2.0 * step);
      else
        dh[derivative][channel] = (-values[2][channel] + values[3][channel] +
                                   8.0 * (values[0][channel] - values[1][channel])) /
                                  (12.0 * step);
    }
  }
}

// The bands of a pass for StftBandBinMap, which reads them through a context.
template <int Rows>
struct DFFRFisherBands {
  typename StftBandBinMap<Rows>::Shared *shared;
  CUDA_DEVICE typename StftBandBinMap<Rows>::Shared *band_shared() const { return shared; }
};

// One block per source. Writes info_out[(bin * num_derivs + i) * num_derivs + j].
template <class SourceT, int Rows>
CUDA_KERNEL void dffr_information_matrix_kernel(double *info_out, DFFRTileInputs<SourceT> inputs,
                                                STFTDomain *stft, const double *params_all,
                                                const int *noise_index_all,
                                                DFFRFisherLayout layout, int num_bin, int nparams,
                                                const int *start_freq_inds,
                                                unsigned char *host_arena) {
  using Scratch = typename DFFRTile<SourceT, Rows>::Arena;
  using Map = StftBandBinMap<Rows>;

  CUDA_SHARED double params[N_PARAMS_MAX];
  CUDA_SHARED typename Map::Shared bands;
  // * Lowest and highest carrier of each derivative's variants per row of the pass.
  CUDA_SHARED int carrier_lo[STFT_FISHER_NDERIV_MAX * Rows];
  CUDA_SHARED int carrier_hi[STFT_FISHER_NDERIV_MAX * Rows];
#ifdef __CUDACC__
  const int thread_index = threadIdx.x;
#endif

  unsigned char *arena = stft_tile_arena(host_arena);
  Scratch *scratch = reinterpret_cast<Scratch *>(arena);
  DFFRFisherReception<Rows> *states = reinterpret_cast<DFFRFisherReception<Rows> *>(
      arena + dffr_information_states_offset<SourceT, Rows>());
  DFFRFisherAmplitude<Rows> *amplitudes = reinterpret_cast<DFFRFisherAmplitude<Rows> *>(
      arena + dffr_information_amplitudes_offset<SourceT, Rows>(layout));
  const DFFRSettings &settings = scratch->settings;
  const DFFRFisherBands<Rows> band_context{&bands};

  const int num_derivs = layout.num_derivs;
  const int num_times = inputs.settings.num_time_segments;
  const int num_freqs = stft->num_freqs;
  const double df = stft->df;
  const double f_min = stft->f_min;

  // * The pixels read the settings from shared memory, as the tile does, to keep one pointer.
  if (THREAD_START_X == 0) scratch->settings = inputs.settings;

  for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
    const int noise_index = noise_index_all[bin_i];
    // The window start belongs to the invC row this source is contracted against.
    const int start_j = (start_freq_inds == nullptr) ? 0 : start_freq_inds[noise_index];
    CUDA_SYNC_THREADS;
    for (int index = THREAD_START_X; index < nparams; index += BLOCK_INCR_X)
      params[index] = params_all[(size_t)bin_i * nparams + index];
    CUDA_SYNC_THREADS;

    // * This thread's share of every entry, upper triangle only.
    double fisher_sums[STFT_FISHER_NDERIV_MAX][STFT_FISHER_NDERIV_MAX];
    for (int row_derivative = 0; row_derivative < num_derivs; row_derivative += 1)
      for (int column_derivative = 0; column_derivative < num_derivs; column_derivative += 1)
        fisher_sums[row_derivative][column_derivative] = 0.0;

    for (int row_base = 0; row_base < num_times; row_base += Rows) {
      const int rows_here = (num_times - row_base < Rows) ? (num_times - row_base) : Rows;
      dffr_information_build<SourceT, Rows>(scratch, states, amplitudes, carrier_lo, carrier_hi,
                                            inputs, layout, params, nparams, row_base, rows_here);

      // * A row's band is the union of its derivatives' common supports.
      Map map(band_context, rows_here);
      for (int row = map.first_row(); row < rows_here; row += map.row_stride()) {
        int band_lo = 0, band_hi = -1;
        bool any_active = false;
        for (int derivative = 0; derivative < num_derivs; derivative += 1) {
          if (!(layout.step[derivative] > 0.0)) continue;
          const int lo = carrier_lo[derivative * Rows + row] - settings.num_side_bins;
          const int hi = carrier_hi[derivative * Rows + row] + settings.num_side_bins;
          if (!any_active || lo < band_lo) band_lo = lo;
          if (!any_active || hi > band_hi) band_hi = hi;
          any_active = true;
        }
        map.set_band(row, band_lo, band_hi);
      }
      map.finish_bands();

      for (int pixel = map.first_pixel(); pixel < map.end_pixel(); pixel += map.pixel_stride()) {
        int row, freq_j_here;
        if (!map.locate(pixel, &row, &freq_j_here)) continue;
        const int time_i = row_base + row;
        const int freq_j_local = freq_j_here - start_j;
        if ((freq_j_local < 0) || (freq_j_local > num_freqs - 1)) continue;
        const double freq_here = f_min + freq_j_here * df;

        cmplx dh[STFT_FISHER_NDERIV_MAX][DFFR_NUM_CHANNELS];  // derivative templates, per channel
        dffr_information_derivatives<Rows>(states, amplitudes, carrier_lo, carrier_hi, settings,
                                           layout, row, freq_j_here, freq_here, dh);

        // * The contraction of stft_eval_block_information on the cross-channel inverse CSD.
        cmplx invc[DFFR_NUM_CHANNELS][DFFR_NUM_CHANNELS];  // inverse CSD of the pixel
        for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1)
          for (int other_channel = 0; other_channel < DFFR_NUM_CHANNELS; other_channel += 1)
            invc[channel][other_channel] = stft->get_invC_cross_value(
                time_i, freq_j_local, channel, other_channel, noise_index);
        cmplx invc_dh[STFT_FISHER_NDERIV_MAX][DFFR_NUM_CHANNELS];  // invC dh, per channel
        for (int derivative = 0; derivative < num_derivs; derivative += 1) {
          if (!(layout.step[derivative] > 0.0)) continue;
          for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1)
            invc_dh[derivative][channel] = invc[channel][0] * dh[derivative][0] +
                                           invc[channel][1] * dh[derivative][1] +
                                           invc[channel][2] * dh[derivative][2];
        }
        // Re(conj(x) y) = x.re y.re + x.im y.im, accumulated on the upper triangle only.
        for (int row_derivative = 0; row_derivative < num_derivs; row_derivative += 1) {
          if (!(layout.step[row_derivative] > 0.0)) continue;
          for (int column_derivative = row_derivative; column_derivative < num_derivs;
               column_derivative += 1) {
            if (!(layout.step[column_derivative] > 0.0)) continue;
            double sum = 0.0;
            for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1)
              sum += dh[row_derivative][channel].real() *
                         invc_dh[column_derivative][channel].real() +
                     dh[row_derivative][channel].imag() *
                         invc_dh[column_derivative][channel].imag();
            fisher_sums[row_derivative][column_derivative] += sum;
          }
        }
      }
      map.finish_pass();
    }
    CUDA_SYNC_THREADS;

    const double scale = 4.0 * stft->diff_comp;
    double *info_source_out = &info_out[(size_t)bin_i * num_derivs * num_derivs];
#ifdef __CUDACC__
    double *reduction_staging =
        reinterpret_cast<double *>(arena + dffr_information_states_offset<SourceT, Rows>());
#endif
    for (int row_derivative = 0; row_derivative < num_derivs; row_derivative += 1) {
      const int width = num_derivs - row_derivative;
#ifdef __CUDACC__
      // * One row of the upper triangle at a time through the shared staging array.
      for (int offset = 0; offset < width; offset += 1)
        reduction_staging[thread_index * width + offset] =
            fisher_sums[row_derivative][row_derivative + offset];
      stft_block_reduce_vec(reduction_staging, width);
      if (thread_index == 0) {
        for (int offset = 0; offset < width; offset += 1) {
          const double value = scale * reduction_staging[offset];
          info_source_out[row_derivative * num_derivs + row_derivative + offset] = value;
          info_source_out[(row_derivative + offset) * num_derivs + row_derivative] = value;
        }
      }
      CUDA_SYNC_THREADS;
#else
      for (int offset = 0; offset < width; offset += 1) {
        const double value = scale * fisher_sums[row_derivative][row_derivative + offset];
        info_source_out[row_derivative * num_derivs + row_derivative + offset] = value;
        info_source_out[(row_derivative + offset) * num_derivs + row_derivative] = value;
      }
#endif
    }
  }
}

// Host launcher. On the GPU, inds and param_eps are device arrays, read back for the layout.
// ! Raises std::invalid_argument for inds outside [0, nparams), and std::runtime_error when the
// ! device cannot hold the launch.
template <class SourceT, int Rows>
inline void dffr_information_matrix(double *info_out, const DFFRTileInputs<SourceT> &inputs,
                                    STFTDomain *stft_launch, const double *params_all,
                                    const int *noise_index_all, const int *inds,
                                    const double *param_eps, int num_bin, int nparams,
                                    int num_derivs, bool easy_central_difference,
                                    const int *start_freq_inds) {
#ifdef __CUDACC__
  // ? Checked before the copies, which take their lengths from these counts.
  if (num_derivs < 1 || num_derivs > STFT_FISHER_NDERIV_MAX || nparams < 1 ||
      nparams > N_PARAMS_MAX)
    throw std::invalid_argument("dffr_information_matrix: num_derivs " +
                                std::to_string(num_derivs) + " or nparams " +
                                std::to_string(nparams) + " out of range.");
  std::vector<int> inds_host(num_derivs);
  std::vector<double> eps_host(nparams);
  gpuErrchk(cudaMemcpy(inds_host.data(), inds, num_derivs * sizeof(int), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(eps_host.data(), param_eps, nparams * sizeof(double),
                       cudaMemcpyDeviceToHost));
  const DFFRFisherLayout layout = dffr_information_layout<SourceT>(
      inds_host.data(), eps_host.data(), nparams, num_derivs, easy_central_difference);
#else
  const DFFRFisherLayout layout =
      dffr_information_layout<SourceT>(inds, param_eps, nparams, num_derivs,
                                       easy_central_difference);
#endif
  if (num_bin < 1) return;
  const size_t shared_bytes = dffr_information_bytes<SourceT, Rows>(layout);
#ifdef __CUDACC__
  stft_prepare_dynamic_shared(dffr_information_matrix_kernel<SourceT, Rows>, shared_bytes);
  dffr_information_matrix_kernel<SourceT, Rows><<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
      info_out, inputs, stft_launch, params_all, noise_index_all, layout, num_bin, nparams,
      start_freq_inds, nullptr);
  gpuErrchk(cudaGetLastError());
#else
  std::vector<unsigned char> host_arena(shared_bytes);
  dffr_information_matrix_kernel<SourceT, Rows>(info_out, inputs, stft_launch, params_all,
                                                noise_index_all, layout, num_bin, nparams,
                                                start_freq_inds, host_arena.data());
#endif
}

// ===========================================================================
// Diagnostics: the compact template and the segment table of a batch of sources
// ===========================================================================

// Per source and segment, everything the segment table build computes, for the stage checks.
struct DFFRSegmentTableOutput {
  double *effective_frequency;  // [source][segment][3]
  double *chirp_rate;           // [source][segment][3]
  cmplx *reference_phasor;      // [source][segment][3]
  cmplx *link_amplitude;        // [source][segment][6]
  cmplx *envelope_slope;        // [source][segment][6]
  cmplx *emission_phasor;       // [source][segment][6]
  double *emission_frequency;   // [source][segment][6]
  cmplx *tdi_weights;           // [source][segment][18]
  int *start_freq_ind;          // [source][segment]
};

// * The constants of one source, from its parameters; the table build that follows starts with a
// * barrier. Every thread of the block calls it.
template <class SourceT, int Rows>
CUDA_DEVICE inline void dffr_load_source(typename DFFRTile<SourceT, Rows>::Arena *arena,
                                         const DFFRTileInputs<SourceT> &inputs, double *params,
                                         const double *params_all, int bin_i, int nparams) {
  CUDA_SYNC_THREADS;
  for (int index = THREAD_START_X; index < nparams; index += BLOCK_INCR_X)
    params[index] = params_all[(size_t)bin_i * nparams + index];
  CUDA_SYNC_THREADS;
  if (THREAD_START_X == 0) SourceT::load(&arena->source, inputs.model, params);
}

// The compact template of every source: per segment the band of 2 n_side_bins + 1 bins from the
// segment's band start, template_out[((source * 3 + channel) * NT + segment) * band + band_bin],
// and the band starts, start_out[source * NT + segment]. Bins outside the grid hold zero. These
// are DFFR's pixel values, the amounts fill_global adds with a factor of one.
template <class SourceT, int Rows>
CUDA_KERNEL void dffr_fill_template_kernel(cmplx *template_out, int *start_out,
                                           DFFRTileInputs<SourceT> inputs, const double *params_all,
                                           int num_bin, int nparams, unsigned char *host_arena) {
  using Arena = typename DFFRTile<SourceT, Rows>::Arena;
  CUDA_SHARED double params[N_PARAMS_MAX];
  Arena *arena = reinterpret_cast<Arena *>(stft_tile_arena(host_arena));
  const DFFRSettings &settings = inputs.settings;
  const int num_times = settings.num_time_segments;
  const int band_width = 2 * settings.num_side_bins + 1;

  for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
    dffr_load_source<SourceT, Rows>(arena, inputs, params, params_all, bin_i, nparams);
    for (int row_base = 0; row_base < num_times; row_base += Rows) {
      const int rows_here = (num_times - row_base < Rows) ? (num_times - row_base) : Rows;
      CUDA_SYNC_THREADS;
      dffr_build_segment_table<SourceT, Rows>(arena->table, settings, inputs.kinematics,
                                              arena->source, row_base, rows_here);
      CUDA_SYNC_THREADS;

      for (int pixel = THREAD_START_X; pixel < rows_here * band_width; pixel += BLOCK_INCR_X) {
        const int band_bin = pixel / rows_here;
        const int row = pixel - band_bin * rows_here;
        const int segment_index = row_base + row;
        const int start_freq_ind = arena->table.start_freq_ind[row];
        const int bin_index = start_freq_ind + band_bin;
        if (band_bin == 0) start_out[(size_t)bin_i * num_times + segment_index] = start_freq_ind;

        cmplx channels[DFFR_NUM_CHANNELS] = {cmplx(0.0, 0.0), cmplx(0.0, 0.0), cmplx(0.0, 0.0)};
        if (bin_index >= 0 && bin_index < settings.num_bins) {
          const double frequency = settings.f_min + ((double)bin_index) * settings.df;
          dffr_evaluate_pixel(arena->table, settings, row, band_bin, bin_index, frequency,
                              channels);
        }
        const size_t stride = (size_t)num_times * band_width;
        const size_t base = (size_t)bin_i * DFFR_NUM_CHANNELS * stride;
        const size_t offset = (size_t)segment_index * band_width + band_bin;
        for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1)
          template_out[base + (size_t)channel * stride + offset] = channels[channel];
      }
    }
  }
}

// Copies of every bin-independent quantity of the segment table, for the stage checks.
template <class SourceT, int Rows>
CUDA_KERNEL void dffr_segment_table_kernel(DFFRSegmentTableOutput output,
                                           DFFRTileInputs<SourceT> inputs, const double *params_all,
                                           int num_bin, int nparams, unsigned char *host_arena) {
  using Arena = typename DFFRTile<SourceT, Rows>::Arena;
  CUDA_SHARED double params[N_PARAMS_MAX];
  Arena *arena = reinterpret_cast<Arena *>(stft_tile_arena(host_arena));
  const int num_times = inputs.settings.num_time_segments;

  for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
    dffr_load_source<SourceT, Rows>(arena, inputs, params, params_all, bin_i, nparams);
    for (int row_base = 0; row_base < num_times; row_base += Rows) {
      const int rows_here = (num_times - row_base < Rows) ? (num_times - row_base) : Rows;
      CUDA_SYNC_THREADS;
      dffr_build_segment_table<SourceT, Rows>(arena->table, inputs.settings, inputs.kinematics,
                                              arena->source, row_base, rows_here);
      CUDA_SYNC_THREADS;

      const DFFRSegmentTable<Rows> &table = arena->table;
      for (int row = THREAD_START_X; row < rows_here; row += BLOCK_INCR_X) {
        const size_t segment = (size_t)bin_i * num_times + row_base + row;
        for (int spacecraft_index = 0; spacecraft_index < DFFR_NUM_SPACECRAFT;
             spacecraft_index += 1) {
          const int slot = row * DFFR_NUM_SPACECRAFT + spacecraft_index;
          const size_t out = segment * DFFR_NUM_SPACECRAFT + spacecraft_index;
          output.effective_frequency[out] = table.effective_frequency[slot];
          output.chirp_rate[out] = table.chirp_rate[slot];
          output.reference_phasor[out] = table.reference_phasor[slot];
        }
        for (int link_index = 0; link_index < DFFR_NUM_LINKS; link_index += 1) {
          const int slot = row * DFFR_NUM_LINKS + link_index;
          const size_t out = segment * DFFR_NUM_LINKS + link_index;
          output.link_amplitude[out] = table.link_amplitude[slot];
          output.envelope_slope[out] = table.envelope_slope[slot];
          output.emission_phasor[out] = table.emission_phasor[slot];
          output.emission_frequency[out] = table.emission_frequency[slot];
        }
        for (int entry = 0; entry < DFFR_TDI_WEIGHTS; entry += 1)
          output.tdi_weights[segment * DFFR_TDI_WEIGHTS + entry] =
              table.tdi_weights[row * DFFR_TDI_WEIGHTS + entry];
        output.start_freq_ind[segment] = table.start_freq_ind[row];
      }
    }
  }
}

// Host launchers of the two diagnostics; they synchronise.
template <class SourceT, int Rows>
inline void dffr_fill_template(cmplx *template_out, int *start_out,
                               const DFFRTileInputs<SourceT> &inputs, const double *params_all,
                               int num_bin, int nparams) {
  if (num_bin < 1) return;
  const size_t shared_bytes = DFFRTile<SourceT, Rows>::table_bytes();
#ifdef __CUDACC__
  stft_prepare_dynamic_shared(dffr_fill_template_kernel<SourceT, Rows>, shared_bytes);
  dffr_fill_template_kernel<SourceT, Rows><<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
      template_out, start_out, inputs, params_all, num_bin, nparams, nullptr);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
#else
  std::vector<unsigned char> host_arena(shared_bytes);
  dffr_fill_template_kernel<SourceT, Rows>(template_out, start_out, inputs, params_all, num_bin,
                                           nparams, host_arena.data());
#endif
}

template <class SourceT, int Rows>
inline void dffr_segment_table(const DFFRSegmentTableOutput &output,
                               const DFFRTileInputs<SourceT> &inputs, const double *params_all,
                               int num_bin, int nparams) {
  if (num_bin < 1) return;
  const size_t shared_bytes = DFFRTile<SourceT, Rows>::table_bytes();
#ifdef __CUDACC__
  stft_prepare_dynamic_shared(dffr_segment_table_kernel<SourceT, Rows>, shared_bytes);
  dffr_segment_table_kernel<SourceT, Rows><<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
      output, inputs, params_all, num_bin, nparams, nullptr);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
#else
  std::vector<unsigned char> host_arena(shared_bytes);
  dffr_segment_table_kernel<SourceT, Rows>(output, inputs, params_all, num_bin, nparams,
                                           host_arena.data());
#endif
}

#endif  // __LAT_DFFR_KERNELS_HH__
