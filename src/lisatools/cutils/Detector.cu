#include "Detector.hpp"
#include "gbt_global.h"
#include "stdio.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// ============================================================================
// Macro Definitions for CPU/GPU Compatibility
// ============================================================================
#if defined(__CUDACC__) || defined(__CUDA_COMPILATION__)
#define Orbits OrbitsGPU
#else
#define Orbits OrbitsCPU
#endif

// Thread block sizes
#define NUM_THREADS 64

CUDA_DEVICE
int Orbits::get_window(double t, double t0, double dt, int N) {
  int out = int((t - t0) / dt);
  if ((out < 0) || (out >= N))
    return -1;
  else
    return out;
}

CUDA_DEVICE
int Orbits::get_link_ind(int link) {
  if (link == 12)
    return 0;
  else if (link == 23)
    return 1;
  else if (link == 31)
    return 2;
  else if (link == 13)
    return 3;
  else if (link == 32)
    return 4;
  else if (link == 21)
    return 5;
  else {
#ifdef __CUDACC__
    // printf("BAD link ind. Must be 12, 23, 31, 13, 32, 21.");
#else
    std::ostringstream oss;
    oss << "BAD link ind. Must be 12, 23, 31, 13, 32, 21. Link ind entered is ("
        << link << ")." << std::endl;
    std::string var = oss.str();
    throw std::invalid_argument(var);
#endif  // __CUDACC__
  }
  return -1;
}

CUDA_DEVICE
int Orbits::get_sc_ind(int sc) {
  if (sc == 1)
    return 0;
  else if (sc == 2)
    return 1;
  else if (sc == 3)
    return 2;
  else {
#ifdef __CUDACC__
    // printf("BAD sc ind. Must be 1,2,3. %d\n", sc);
#else
    std::ostringstream oss;
    oss << "Bad sc ind. Must be 1,2,3. Input sc is " << sc << " " << std::endl;
    std::string var = oss.str();
    throw std::invalid_argument(var);
#endif  // __CUDACC__
  }
  return 0;
}

CUDA_DEVICE
double Orbits::interpolate(double t, double* in_arr, double t0, double dt,
                           int window, int major_ndim, int major_ind, int ndim,
                           int pos) {
  double up = in_arr[((window + 1) * major_ndim + major_ind) * ndim +
                     pos];  // down_ind * ndim + pos];
  double down = in_arr[(window * major_ndim + major_ind) * ndim + pos];

  // m *(x - x0) + y0
  double fin = ((up - down) / dt) * (t - (t0 + dt * window)) + down;
  // if ((ndim == 1))
  //     printf("%d %e %e %e %e \n", window, fin, down, up, (t - (dt *
  //     window)));

  return fin;
}

CUDA_DEVICE
void Orbits::get_normal_unit_vec_ptr(Vec* vec, double t, int link) {
  Vec _tmp = get_normal_unit_vec(t, link);
  vec->x = _tmp.x;
  vec->y = _tmp.y;
  vec->z = _tmp.z;
}

CUDA_DEVICE
Vec Orbits::get_normal_unit_vec(double t, int link) {
  int window = get_window(t, sc_t0, sc_dt, sc_N);
  if (window == -1) {
    // out of bounds
    return Vec(0.0, 0.0, 0.0);
  }

  int link_ind = get_link_ind(link);

  int up_ind = (window + 1) * nlinks + link_ind;
  int down_ind = window * nlinks + link_ind;

  // x (pos = 0) ndim = 3
  double x_out =
      interpolate(t, n_arr, sc_t0, sc_dt, window, nlinks, link_ind, 3, 0);
  // y (pos = 1)
  double y_out =
      interpolate(t, n_arr, sc_t0, sc_dt, window, nlinks, link_ind, 3, 1);
  // z (pos = 2)
  double z_out =
      interpolate(t, n_arr, sc_t0, sc_dt, window, nlinks, link_ind, 3, 2);

  return Vec(x_out, y_out, z_out);
}

CUDA_DEVICE
double Orbits::get_light_travel_time(double t, int link) {
  int window = get_window(t, ltt_t0, ltt_dt, ltt_N);
  if (window == -1) {
    // out of bounds
    return 0.0;
  }

  int link_ind = get_link_ind(link);
  int up_ind = (window + 1) * (nlinks + link_ind);
  int down_ind = window * (nlinks + link_ind);

  // x (pos = 0), ndim = 1
  double ltt_out =
      interpolate(t, ltt_arr, ltt_t0, ltt_dt, window, nlinks, link_ind, 1, 0);

  return ltt_out;
}

CUDA_DEVICE
Vec Orbits::get_pos(double t, int sc) {
  int window = get_window(t, sc_t0, sc_dt, sc_N);
  if (window == -1) {
    // out of bounds
    return Vec(0.0, 0.0, 0.0);
  }

  int sc_ind = get_sc_ind(sc);

  // x (pos = 0), ndim = 3
  double x_out =
      interpolate(t, x_arr, sc_t0, sc_dt, window, nspacecraft, sc_ind, 3, 0);
  // y (pos = 1), ndim = 3
  double y_out =
      interpolate(t, x_arr, sc_t0, sc_dt, window, nspacecraft, sc_ind, 3, 1);
  // z (pos = 2), ndim = 3
  double z_out =
      interpolate(t, x_arr, sc_t0, sc_dt, window, nspacecraft, sc_ind, 3, 2);
  return Vec(x_out, y_out, z_out);
}

CUDA_DEVICE
void Orbits::get_pos_ptr(Vec* vec, double t, int sc) {
  Vec _tmp = get_pos(t, sc);
  vec->x = _tmp.x;
  vec->y = _tmp.y;
  vec->z = _tmp.z;
}

// ============================================================================
// Five-point derivatives of the orbit tables
// ============================================================================

// * A step of whole cells puts the five samples at one position within their cells, so the
// * linear-interpolation error of the table cancels in the difference.
CUDA_CALLABLE_MEMBER
static inline double orbit_stencil_step(double dt) {
  double cells = floor(ORBIT_STENCIL_STEP / dt + 0.5);
  if (cells < 1.0) cells = 1.0;
  return cells * dt;
}

// Stencil centre clamped so that all five samples lie in [t0, t0 + (N - 2) dt]; false when the
// table is shorter than the stencil. The last cell is left out, as the base classes put the
// file's last sample there, off the grid.
CUDA_DEVICE
static inline bool orbit_stencil_centre(double* centre, double t, double t0, double dt, int N,
                                        double step) {
  const double lower = t0 + 2.0 * step;
  const double upper = t0 + ((double)(N - 2)) * dt - 2.0 * step;
  if (upper < lower) return false;
  *centre = (t < lower) ? lower : ((t > upper) ? upper : t);
  return true;
}

CUDA_DEVICE
Vec Orbits::get_velocity(double t, int sc) {
  const double step = orbit_stencil_step(sc_dt);
  double centre;
  if (!orbit_stencil_centre(&centre, t, sc_t0, sc_dt, sc_N, step)) return Vec(0.0, 0.0, 0.0);

  const Vec plus_two = get_pos(centre + 2.0 * step, sc);
  const Vec plus_one = get_pos(centre + step, sc);
  const Vec minus_one = get_pos(centre - step, sc);
  const Vec minus_two = get_pos(centre - 2.0 * step, sc);
  const double rate_scale = 1.0 / (12.0 * step);
  return Vec((-plus_two.x + 8.0 * plus_one.x - 8.0 * minus_one.x + minus_two.x) * rate_scale,
             (-plus_two.y + 8.0 * plus_one.y - 8.0 * minus_one.y + minus_two.y) * rate_scale,
             (-plus_two.z + 8.0 * plus_one.z - 8.0 * minus_one.z + minus_two.z) * rate_scale);
}

CUDA_DEVICE
Vec Orbits::get_acceleration(double t, int sc) {
  const double step = orbit_stencil_step(sc_dt);
  double centre;
  if (!orbit_stencil_centre(&centre, t, sc_t0, sc_dt, sc_N, step)) return Vec(0.0, 0.0, 0.0);

  const Vec plus_two = get_pos(centre + 2.0 * step, sc);
  const Vec plus_one = get_pos(centre + step, sc);
  const Vec at_centre = get_pos(centre, sc);
  const Vec minus_one = get_pos(centre - step, sc);
  const Vec minus_two = get_pos(centre - 2.0 * step, sc);
  const double acceleration_scale = 1.0 / (12.0 * step * step);
  return Vec((-plus_two.x + 16.0 * plus_one.x - 30.0 * at_centre.x + 16.0 * minus_one.x -
              minus_two.x) * acceleration_scale,
             (-plus_two.y + 16.0 * plus_one.y - 30.0 * at_centre.y + 16.0 * minus_one.y -
              minus_two.y) * acceleration_scale,
             (-plus_two.z + 16.0 * plus_one.z - 30.0 * at_centre.z + 16.0 * minus_one.z -
              minus_two.z) * acceleration_scale);
}

CUDA_DEVICE
double Orbits::get_ltt_rate(double t, int link) {
  const double step = orbit_stencil_step(ltt_dt);
  double centre;
  if (!orbit_stencil_centre(&centre, t, ltt_t0, ltt_dt, ltt_N, step)) return 0.0;

  const double plus_two = get_light_travel_time(centre + 2.0 * step, link);
  const double plus_one = get_light_travel_time(centre + step, link);
  const double minus_one = get_light_travel_time(centre - step, link);
  const double minus_two = get_light_travel_time(centre - 2.0 * step, link);
  return (-plus_two + 8.0 * plus_one - 8.0 * minus_one + minus_two) / (12.0 * step);
}

CUDA_DEVICE
double Orbits::get_ltt_acceleration(double t, int link) {
  const double step = orbit_stencil_step(ltt_dt);
  double centre;
  if (!orbit_stencil_centre(&centre, t, ltt_t0, ltt_dt, ltt_N, step)) return 0.0;

  const double plus_two = get_light_travel_time(centre + 2.0 * step, link);
  const double plus_one = get_light_travel_time(centre + step, link);
  const double at_centre = get_light_travel_time(centre, link);
  const double minus_one = get_light_travel_time(centre - step, link);
  const double minus_two = get_light_travel_time(centre - 2.0 * step, link);
  return (-plus_two + 16.0 * plus_one - 30.0 * at_centre + 16.0 * minus_one - minus_two) /
         (12.0 * step * step);
}

CUDA_KERNEL
void get_light_travel_time_kernel(double* ltt, double* t, int* link, int num,
                                  Orbits& orbits) {
  int start, end, increment;
#ifdef __CUDACC__
  start = blockIdx.x * blockDim.x + threadIdx.x;
  end = num;
  increment = gridDim.x * blockDim.x;
#else   // __CUDACC__
  start = 0;
  end = num;
  increment = 1;
#endif  // __CUDACC__

  for (int i = start; i < end; i += increment) {
    ltt[i] = orbits.get_light_travel_time(t[i], link[i]);
  }
}

void Orbits::get_light_travel_time_arr(double* ltt, double* t, int* link,
                                       int num) {
#ifdef __CUDACC__
  int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);

  // copy self to GPU
  Orbits* orbits_gpu;
  gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
  gpuErrchk(
      cudaMemcpy(orbits_gpu, this, sizeof(Orbits), cudaMemcpyHostToDevice));

  get_light_travel_time_kernel<<<num_blocks, NUM_THREADS>>>(ltt, t, link, num,
                                                            *orbits_gpu);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaFree(orbits_gpu));

#else  // __CUDACC__

  get_light_travel_time_kernel(ltt, t, link, num, *this);

#endif  // __CUDACC__
}

CUDA_KERNEL
void get_pos_kernel(double* pos_x, double* pos_y, double* pos_z, double* t,
                    int* sc, int num, Orbits& orbits) {
  int start, end, increment;
#ifdef __CUDACC__
  start = blockIdx.x * blockDim.x + threadIdx.x;
  end = num;
  increment = gridDim.x * blockDim.x;
#else   // __CUDACC__
  start = 0;
  end = num;
  increment = 1;
#endif  // __CUDACC__
  Vec _tmp(0.0, 0.0, 0.0);

  for (int i = start; i < end; i += increment) {
    _tmp = orbits.get_pos(t[i], sc[i]);
    pos_x[i] = _tmp.x;
    pos_y[i] = _tmp.y;
    pos_z[i] = _tmp.z;
  }
}

void Orbits::get_pos_arr(double* pos_x, double* pos_y, double* pos_z, double* t,
                         int* sc, int num) {
#ifdef __CUDACC__
  int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);

  // copy self to GPU
  Orbits* orbits_gpu;
  gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
  gpuErrchk(
      cudaMemcpy(orbits_gpu, this, sizeof(Orbits), cudaMemcpyHostToDevice));

  get_pos_kernel<<<num_blocks, NUM_THREADS>>>(pos_x, pos_y, pos_z, t, sc, num,
                                              *orbits_gpu);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaFree(orbits_gpu));

#else  // __CUDACC__

  get_pos_kernel(pos_x, pos_y, pos_z, t, sc, num, *this);

#endif  // __CUDACC__
}

CUDA_KERNEL
void get_normal_unit_vec_kernel(double* normal_unit_vec_x,
                                double* normal_unit_vec_y,
                                double* normal_unit_vec_z, double* t, int* link,
                                int num, Orbits& orbits) {
  int start, end, increment;
#ifdef __CUDACC__
  start = blockIdx.x * blockDim.x + threadIdx.x;
  end = num;
  increment = gridDim.x * blockDim.x;
#else   // __CUDACC__
  start = 0;
  end = num;
  increment = 1;
#endif  // __CUDACC__
  Vec _tmp(0.0, 0.0, 0.0);

  for (int i = start; i < end; i += increment) {
    _tmp = orbits.get_normal_unit_vec(t[i], link[i]);
    normal_unit_vec_x[i] = _tmp.x;
    normal_unit_vec_y[i] = _tmp.y;
    normal_unit_vec_z[i] = _tmp.z;
  }
}

void Orbits::get_normal_unit_vec_arr(double* normal_unit_vec_x,
                                     double* normal_unit_vec_y,
                                     double* normal_unit_vec_z, double* t,
                                     int* link, int num) {
#ifdef __CUDACC__
  int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);

  // copy self to GPU
  Orbits* orbits_gpu;
  gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
  gpuErrchk(
      cudaMemcpy(orbits_gpu, this, sizeof(Orbits), cudaMemcpyHostToDevice));

  get_normal_unit_vec_kernel<<<num_blocks, NUM_THREADS>>>(
      normal_unit_vec_x, normal_unit_vec_y, normal_unit_vec_z, t, link, num,
      *orbits_gpu);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaFree(orbits_gpu));

#else  // __CUDACC__

  get_normal_unit_vec_kernel(normal_unit_vec_x, normal_unit_vec_y,
                             normal_unit_vec_z, t, link, num, *this);

#endif  // __CUDACC__
}

// ============================================================================
// Kinematics table at the segment midpoints
// ============================================================================

// * Bound on every LISA light travel time [s]; it widens the position grid needed by the senders.
#define ORBIT_MAX_LIGHT_TRAVEL_TIME 20.0

// Stage one: light travel times and their first two derivatives at the midpoints.
CUDA_KERNEL
void kinematics_light_travel_kernel(KinematicsTable table, Orbits& orbits) {
  int start, increment;
#ifdef __CUDACC__
  start = blockIdx.x * blockDim.x + threadIdx.x;
  increment = gridDim.x * blockDim.x;
#else   // __CUDACC__
  start = 0;
  increment = 1;
#endif  // __CUDACC__
  const int num_time_segments = table.num_time_segments;
  const int total = num_time_segments * orbits.nlinks;
  for (int flat = start; flat < total; flat += increment) {
    const int link_index = flat / num_time_segments;
    const int segment_index = flat - link_index * num_time_segments;
    const double midpoint = table.segment_midpoint(segment_index);
    const int link = orbits.get_link_from_arr(link_index);
    table.light_travel_time[flat] = orbits.get_light_travel_time(midpoint, link);
    table.light_travel_time_rate[flat] = orbits.get_ltt_rate(midpoint, link);
    table.light_travel_time_acceleration[flat] = orbits.get_ltt_acceleration(midpoint, link);
  }
}

// Stage two: spacecraft position, velocity and acceleration at both ends of every link.
CUDA_KERNEL
void kinematics_spacecraft_state_kernel(KinematicsTable table, Orbits& orbits) {
  int start, increment;
#ifdef __CUDACC__
  start = blockIdx.x * blockDim.x + threadIdx.x;
  increment = gridDim.x * blockDim.x;
#else   // __CUDACC__
  start = 0;
  increment = 1;
#endif  // __CUDACC__
  const int num_time_segments = table.num_time_segments;

  // * Receivers: one state per spacecraft, shared by the two links that arrive there.
  const int receiver_total = num_time_segments * orbits.nspacecraft;
  for (int flat = start; flat < receiver_total; flat += increment) {
    const int spacecraft_index = flat / num_time_segments;
    const int segment_index = flat - spacecraft_index * num_time_segments;
    const double midpoint = table.segment_midpoint(segment_index);
    const Vec position = orbits.get_pos(midpoint, spacecraft_index + 1);
    const Vec velocity = orbits.get_velocity(midpoint, spacecraft_index + 1);
    const Vec acceleration = orbits.get_acceleration(midpoint, spacecraft_index + 1);
    const int out = spacecraft_index * 3 * num_time_segments + segment_index;
    table.position_receiver[out] = position.x;
    table.position_receiver[out + num_time_segments] = position.y;
    table.position_receiver[out + 2 * num_time_segments] = position.z;
    table.velocity_receiver[out] = velocity.x;
    table.velocity_receiver[out + num_time_segments] = velocity.y;
    table.velocity_receiver[out + 2 * num_time_segments] = velocity.z;
    table.acceleration_receiver[out] = acceleration.x;
    table.acceleration_receiver[out + num_time_segments] = acceleration.y;
    table.acceleration_receiver[out + 2 * num_time_segments] = acceleration.z;
  }

  // * Senders: at the emission time of their link, which stage one has filled.
  const int sender_total = num_time_segments * orbits.nlinks;
  for (int flat = start; flat < sender_total; flat += increment) {
    const int link_index = flat / num_time_segments;
    const int segment_index = flat - link_index * num_time_segments;
    const double emission_time =
        table.segment_midpoint(segment_index) - table.light_travel_time[flat];
    const int sender = orbits.get_sc_e_from_arr(link_index);
    const Vec position = orbits.get_pos(emission_time, sender);
    const Vec velocity = orbits.get_velocity(emission_time, sender);
    const Vec acceleration = orbits.get_acceleration(emission_time, sender);
    const int out = link_index * 3 * num_time_segments + segment_index;
    table.position_sender[out] = position.x;
    table.position_sender[out + num_time_segments] = position.y;
    table.position_sender[out + 2 * num_time_segments] = position.z;
    table.velocity_sender[out] = velocity.x;
    table.velocity_sender[out + num_time_segments] = velocity.y;
    table.velocity_sender[out + 2 * num_time_segments] = velocity.z;
    table.acceleration_sender[out] = acceleration.x;
    table.acceleration_sender[out + num_time_segments] = acceleration.y;
    table.acceleration_sender[out + 2 * num_time_segments] = acceleration.z;
  }
}

// Stage three: unit vector from the sender at emission to the receiver at reception, and its
// rate. The sender velocity carries (1 - dL/dt) from t_e = t - L(t), and only the part of the
// relative velocity across the link turns the unit vector.
CUDA_KERNEL
void kinematics_unit_direction_kernel(KinematicsTable table, Orbits& orbits) {
  int start, increment;
#ifdef __CUDACC__
  start = blockIdx.x * blockDim.x + threadIdx.x;
  increment = gridDim.x * blockDim.x;
#else   // __CUDACC__
  start = 0;
  increment = 1;
#endif  // __CUDACC__
  const int num_time_segments = table.num_time_segments;
  const int total = num_time_segments * orbits.nlinks;
  for (int flat = start; flat < total; flat += increment) {
    const int link_index = flat / num_time_segments;
    const int segment_index = flat - link_index * num_time_segments;
    const int receiver_index = orbits.get_sc_r_from_arr(link_index) - 1;
    const double emission_rate = 1.0 - table.light_travel_time_rate[flat];

    double separation[3], separation_rate[3];
    for (int xyz = 0; xyz < 3; xyz += 1) {
      const int receiver_slot = (receiver_index * 3 + xyz) * num_time_segments + segment_index;
      const int link_slot = (link_index * 3 + xyz) * num_time_segments + segment_index;
      separation[xyz] = table.position_receiver[receiver_slot] - table.position_sender[link_slot];
      separation_rate[xyz] = table.velocity_receiver[receiver_slot] -
                             table.velocity_sender[link_slot] * emission_rate;
    }

    const double distance = sqrt(separation[0] * separation[0] + separation[1] * separation[1] +
                                 separation[2] * separation[2]);
    const double inverse_distance = 1.0 / distance;
    double unit_direction[3];
    for (int xyz = 0; xyz < 3; xyz += 1) unit_direction[xyz] = separation[xyz] * inverse_distance;
    const double radial_rate = unit_direction[0] * separation_rate[0] +
                               unit_direction[1] * separation_rate[1] +
                               unit_direction[2] * separation_rate[2];

    for (int xyz = 0; xyz < 3; xyz += 1) {
      const int out = (link_index * 3 + xyz) * num_time_segments + segment_index;
      table.unit_direction[out] = unit_direction[xyz];
      table.unit_direction_rate[out] =
          (separation_rate[xyz] - unit_direction[xyz] * radial_rate) * inverse_distance;
    }
  }
}

// Raise when the times [first, last], widened by two stencil steps, leave [t0, t0 + (N - 2) dt].
static void check_stencil_coverage(const char* grid_name, double first, double last, double t0,
                                   double dt, int N) {
  const double margin = 2.0 * orbit_stencil_step(dt);
  const double needed_start = first - margin;
  const double needed_end = last + margin;
  const double table_end = t0 + ((double)(N - 2)) * dt;
  if ((needed_start < t0) || (needed_end > table_end)) {
    std::ostringstream message;
    message << std::fixed << std::setprecision(1) << "The kinematics table needs the "
            << grid_name << " grid on [" << needed_start << ", " << needed_end
            << "] s (segment midpoints plus a " << margin << " s stencil margin), but it covers ["
            << t0 << ", " << table_end << "] s. Shorten the STFT grid or extend the orbits.";
    throw std::invalid_argument(message.str());
  }
}

void Orbits::get_kinematics_table(KinematicsTable* table) {
  if (table->num_time_segments < 1)
    throw std::invalid_argument("The kinematics table needs at least one segment.");
  const double first = table->segment_midpoint(0);
  const double last = table->segment_midpoint(table->num_time_segments - 1);
  // * Senders are read at the emission time t - L(t), up to one light travel time earlier.
  check_stencil_coverage("position", first - ORBIT_MAX_LIGHT_TRAVEL_TIME, last, sc_t0, sc_dt,
                         sc_N);
  check_stencil_coverage("light travel time", first, last, ltt_t0, ltt_dt, ltt_N);

#ifdef __CUDACC__
  const int total = table->num_time_segments * nlinks;
  const int num_blocks = (total + NUM_THREADS - 1) / NUM_THREADS;

  // copy self to GPU
  Orbits* orbits_gpu;
  gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
  gpuErrchk(cudaMemcpy(orbits_gpu, this, sizeof(Orbits), cudaMemcpyHostToDevice));

  // ! Each stage reads what the previous one wrote, so the launches stay in order.
  kinematics_light_travel_kernel<<<num_blocks, NUM_THREADS>>>(*table, *orbits_gpu);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  kinematics_spacecraft_state_kernel<<<num_blocks, NUM_THREADS>>>(*table, *orbits_gpu);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  kinematics_unit_direction_kernel<<<num_blocks, NUM_THREADS>>>(*table, *orbits_gpu);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaFree(orbits_gpu));

#else  // __CUDACC__

  kinematics_light_travel_kernel(*table, *this);
  kinematics_spacecraft_state_kernel(*table, *this);
  kinematics_unit_direction_kernel(*table, *this);

#endif  // __CUDACC__
}