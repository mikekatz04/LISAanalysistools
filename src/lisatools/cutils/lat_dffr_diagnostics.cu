// lat_dffr_diagnostics.cu -- source-independent DFFR diagnostics for the tests of LAT: the bounded
// sub-interval integral, the windowed transform and the closed-form TDI coefficients, each at given
// inputs. Compiled into LAT's own module only; the source-dependent stages are checked through the
// GBGPU entry points.

#include "lat_dffr_kernels.hh"
#include "lat_dffr_diagnostics.hh"

#include <stdexcept>

#ifdef __CUDACC__
#define DFFR_DIAGNOSTIC_THREADS 128
#endif

CUDA_KERNEL
void dffr_sub_interval_kernel(cmplx *kernel_out, cmplx *moment_out, const double *lower_bounds,
                              const double *upper_bounds, const double *detunings,
                              const double *chirp_rates, double chirp_rate_floor,
                              int num_points) {
  for (int index = BLOCK_START_X * BLOCK_INCR_X + THREAD_START_X; index < num_points;
       index += BLOCK_INCR_X * GRID_INCR_X) {
    // ! The floor applies here as well: an exact zero would give NaN.
    const double chirp_rate = dffr_apply_chirp_rate_floor(chirp_rates[index], chirp_rate_floor);
    const DFFRQuadPhaseScalars scalars = dffr_make_quad_phase_scalars(chirp_rate);
    const double lower_bound = lower_bounds[index];
    const double upper_bound = upper_bounds[index];
    const double detuning = detunings[index];

    // * A bare sub-interval: the endpoint phasors carry no window weight and are formed directly.
    const cmplx rotation_lower = dffr_polar_unit(M_PI * chirp_rate * lower_bound * lower_bound +
                                                 DFFR_TWO_PI * detuning * lower_bound);
    const cmplx rotation_upper = dffr_polar_unit(M_PI * chirp_rate * upper_bound * upper_bound +
                                                 DFFR_TWO_PI * detuning * upper_bound);
    cmplx kernel_value(0.0, 0.0), moment_value(0.0, 0.0);
    dffr_accumulate_sub_interval(scalars, lower_bound, upper_bound, detuning,
                                 detuning * scalars.reciprocal_chirp, rotation_lower,
                                 rotation_upper, cmplx(1.0, 0.0), true, &kernel_value,
                                 &moment_value);
    kernel_out[index] = kernel_value;
    moment_out[index] = moment_value;
  }
}

CUDA_KERNEL
void dffr_windowed_kernel_kernel(cmplx *kernel_out, cmplx *moment_out, DFFRSettings settings,
                                 const double *effective_frequencies, const double *chirp_rates,
                                 const double *frequencies, int num_points) {
  for (int index = BLOCK_START_X * BLOCK_INCR_X + THREAD_START_X; index < num_points;
       index += BLOCK_INCR_X * GRID_INCR_X) {
    const double chirp_rate =
        dffr_apply_chirp_rate_floor(chirp_rates[index], settings.chirp_rate_floor);
    const DFFRQuadPhaseScalars scalars = dffr_make_quad_phase_scalars(chirp_rate);
    const double effective_frequency = effective_frequencies[index];
    const double frequency = frequencies[index];
    const double detuning = effective_frequency - frequency;

    // * At an arbitrary frequency the endpoint phasors are formed directly, not from per-segment
    // * anchors and per-band-bin factors.
    const double outer = settings.half_duration;
    const double inner = settings.plateau_half_width;
    const cmplx outer_plus =
        dffr_polar_unit(M_PI * chirp_rate * outer * outer + DFFR_TWO_PI * detuning * outer);
    const cmplx outer_minus =
        dffr_polar_unit(M_PI * chirp_rate * outer * outer - DFFR_TWO_PI * detuning * outer);
    const cmplx inner_plus =
        dffr_polar_unit(M_PI * chirp_rate * inner * inner + DFFR_TWO_PI * detuning * inner);
    const cmplx inner_minus =
        dffr_polar_unit(M_PI * chirp_rate * inner * inner - DFFR_TWO_PI * detuning * inner);
    const cmplx overall = dffr_polar_unit(-M_PI * frequency * settings.segment_duration);

    cmplx kernel_value, moment_value;
    dffr_evaluate_windowed_kernel(settings, scalars, effective_frequency, frequency, outer_plus,
                                  outer_minus, inner_plus, inner_minus, overall, true,
                                  &kernel_value, &moment_value);
    kernel_out[index] = kernel_value;
    moment_out[index] = moment_value;
  }
}

CUDA_KERNEL
void dffr_link_coefficients_kernel(cmplx *coefficients_out, KinematicsTable table,
                                   const double *carrier_frequencies, int tdi_generation) {
  for (int segment_index = BLOCK_START_X * BLOCK_INCR_X + THREAD_START_X;
       segment_index < table.num_time_segments; segment_index += BLOCK_INCR_X * GRID_INCR_X) {
    const double carrier_frequency = carrier_frequencies[segment_index];
    cmplx delay_phasor[DFFR_NUM_LINKS];
    for (int link_index = 0; link_index < DFFR_NUM_LINKS; link_index += 1) {
      const double light_travel_time =
          table.light_travel_time[link_index * table.num_time_segments + segment_index];
      delay_phasor[link_index] =
          dffr_polar_unit(-DFFR_TWO_PI * carrier_frequency * light_travel_time);
    }
    cmplx first[DFFR_NUM_LINKS];
    cmplx second[DFFR_NUM_LINKS];
    dffr_link_coefficients(delay_phasor, tdi_generation, first, second);

    cmplx *out = coefficients_out + (size_t)segment_index * DFFR_NUM_CHANNELS * DFFR_NUM_LINKS;
    for (int entry = 0; entry < DFFR_NUM_CHANNELS * DFFR_NUM_LINKS; entry += 1)
      out[entry] = cmplx(0.0, 0.0);
    for (int link_index = 0; link_index < DFFR_NUM_LINKS; link_index += 1) {
      out[dffr_link_channel_first(link_index) * DFFR_NUM_LINKS + link_index] = first[link_index];
      out[dffr_link_channel_second(link_index) * DFFR_NUM_LINKS + link_index] = second[link_index];
    }
  }
}

// One thread per point on the GPU; one serial loop on the CPU. The diagnostics synchronise.
#ifdef __CUDACC__
#define DFFR_DIAGNOSTIC_LAUNCH(kernel, num_points, ...)                                   \
  do {                                                                                     \
    const int num_blocks = ((num_points) + DFFR_DIAGNOSTIC_THREADS - 1) / DFFR_DIAGNOSTIC_THREADS; \
    kernel<<<num_blocks, DFFR_DIAGNOSTIC_THREADS>>>(__VA_ARGS__);                          \
    gpuErrchk(cudaGetLastError());                                                         \
    gpuErrchk(cudaDeviceSynchronize());                                                    \
  } while (0)
#else
#define DFFR_DIAGNOSTIC_LAUNCH(kernel, num_points, ...) kernel(__VA_ARGS__)
#endif

void dffr_sub_intervals_wrap(cmplx *kernel_out, cmplx *moment_out, const double *lower_bounds,
                             const double *upper_bounds, const double *detunings,
                             const double *chirp_rates, double chirp_rate_floor, int num_points) {
  if (!(chirp_rate_floor > 0.0))
    throw std::invalid_argument("dffr: chirp_rate_floor must be positive.");
  if (num_points < 1) return;
  DFFR_DIAGNOSTIC_LAUNCH(dffr_sub_interval_kernel, num_points, kernel_out, moment_out,
                         lower_bounds, upper_bounds, detunings, chirp_rates, chirp_rate_floor,
                         num_points);
}

void dffr_windowed_kernels_wrap(cmplx *kernel_out, cmplx *moment_out, const STFTFresnel *fresnel,
                                const double *effective_frequencies, const double *chirp_rates,
                                const double *frequencies, double chirp_rate_floor,
                                int num_points) {
  // ? The knobs other than the floor do not reach a single transform.
  const DFFRSettings settings =
      dffr_make_settings(*fresnel, false, false, 2, 0, chirp_rate_floor);
  if (num_points < 1) return;
  DFFR_DIAGNOSTIC_LAUNCH(dffr_windowed_kernel_kernel, num_points, kernel_out, moment_out,
                         settings, effective_frequencies, chirp_rates, frequencies, num_points);
}

void dffr_link_coefficients_wrap(cmplx *coefficients_out, const KinematicsTable *table,
                                 const double *carrier_frequencies, int tdi_generation) {
  if (tdi_generation != 1 && tdi_generation != 2)
    throw std::invalid_argument("dffr: tdi_generation must be 1 or 2.");
  DFFR_DIAGNOSTIC_LAUNCH(dffr_link_coefficients_kernel, table->num_time_segments,
                         coefficients_out, *table, carrier_frequencies, tdi_generation);
}
