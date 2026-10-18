#ifndef __LAT_DFFR_KERNELS_HH__
#define __LAT_DFFR_KERNELS_HH__

// lat_dffr_kernels.hh -- the Direct Fast Fresnel Response (DFFR) of the STFT likelihood: settings,
// source contract, windowed chirp transform, TDI weights and segment table. Ported from the
// standalone package direct-fast-fresnel-response at commit 01530a5 (src/dffr/cutils/dffr.cu). The
// source-independent diagnostics of LAT are in lat_dffr_diagnostics.cu.
// docs/response/dffr/README.md describes the response and its integration.
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

#include "gbt_global.h"    // cmplx, gcmplx, CUDA_* macros, THREAD_START_X, BLOCK_INCR_X
#include "Detector.hpp"    // KinematicsTable
#include "domains.hpp"     // STFTFresnel
#include "lat_fresnel.hh"  // lat_fresnel_series, lat_fresnel_aux, LAT_FRESNEL_SEAM

#include <cmath>
#include <stdexcept>
#include <string>

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

CUDA_DEVICE
inline void dffr_evaluate_windowed_kernel(const DFFRSettings &settings,
                                          const DFFRQuadPhaseScalars &scalars,
                                          double effective_frequency, double frequency,
                                          const cmplx &outer_plus, const cmplx &outer_minus,
                                          const cmplx &inner_plus, const cmplx &inner_minus,
                                          const cmplx &overall, bool with_moment,
                                          cmplx *kernel_out, cmplx *moment_out) {
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

template <int Rows>
CUDA_DEVICE inline void dffr_evaluate_reception_state(const DFFRSegmentTable<Rows> &table,
                                                      const DFFRSettings &settings, int row,
                                                      int spacecraft_index, double frequency,
                                                      const DFFRBandBinPhasors &band_bin_phasors,
                                                      const cmplx &origin_phasor, cmplx *kernel_out,
                                                      cmplx *moment_out) {
  const int slot = row * DFFR_NUM_SPACECRAFT + spacecraft_index;

  DFFRQuadPhaseScalars scalars;
  scalars.chirp_rate = table.chirp_rate[slot];
  scalars.scale_factor = table.scale_factor[slot];
  scalars.inverse_scale = table.inverse_scale[slot];
  scalars.reciprocal_chirp = table.reciprocal_chirp[slot];
  scalars.moment_prefactor = table.moment_prefactor[slot];
  scalars.chirp_sign = (scalars.chirp_rate > 0.0) ? 1.0 : -1.0;

  const cmplx outer_plus = band_bin_phasors.outer_sign * table.anchor_outer_plus[slot];
  const cmplx outer_minus = band_bin_phasors.outer_sign * table.anchor_outer_minus[slot];
  const cmplx inner_plus = dffr_mul(table.anchor_inner_plus[slot], band_bin_phasors.inner);
  const cmplx inner_minus = dffr_mul(table.anchor_inner_minus[slot], band_bin_phasors.inner_conj);
  const cmplx overall = dffr_mul(table.reference_phasor[slot], origin_phasor);

  dffr_evaluate_windowed_kernel(settings, scalars, table.effective_frequency[slot], frequency,
                                outer_plus, outer_minus, inner_plus, inner_minus, overall, true,
                                kernel_out, moment_out);
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
#pragma unroll
    for (int channel = 0; channel < DFFR_NUM_CHANNELS; channel += 1) {
      const int entry = 6 * channel + 2 * spacecraft_index;
      channels[channel] = channels[channel] + dffr_mul(weights[entry], expansion_kernel) +
                          dffr_mul(weights[entry + 1], expansion_moment);
    }
  }
}

#endif  // __LAT_DFFR_KERNELS_HH__
