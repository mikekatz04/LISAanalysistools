#pragma once

#include "gbt_global.h"
#include <cmath>
#include <vector>

// CPU/GPU class-name aliasing (sprint-wide rule): these structs are compiled
// into both backend shared objects and are held by GalacticGridWrap /
// registered with nanobind, so the two builds must emit distinct C++ type
// names. Added at the 2026-06 stft_tof merge (rule (b): underlying classes
// held by wrappers need aliases too, not just the *Wrap layer).
#if defined(__CUDACC__) || defined(__CUDA_COMPILATION__)
#define GalacticGrid GalacticGridGPU
#define GalacticGridSetup GalacticGridSetupGPU
#else
#define GalacticGrid GalacticGridCPU
#define GalacticGridSetup GalacticGridSetupCPU
#endif

// ============================================================================
// Constants
// ============================================================================

#define SQRT3_GAL  1.7320508075688772935
#define PI2_GAL    (M_PI / 2.0)
#define OMEGA_LISA (2.0 * M_PI / 3.154e7)   // rad/s — matches galaxy_transfer.py


// ============================================================================
// GalacticGridSetup — host-side precomputation helper (unchanged)
// ============================================================================

struct GalacticGridSetup
{
    int N_lambda = 90;
    int N_beta   = 60;
    int N_quad   = 16;
    int N_sky    = 0;

    std::vector<double> lam_ecl;
    std::vector<double> beta_ecl;
    std::vector<double> cos_beta_ecl;
    std::vector<double> lam_gal;
    std::vector<double> beta_gal;
    std::vector<double> D_nodes;
    std::vector<double> quad_weights;
    std::vector<double> R_vals_quad;   // (N_quad * N_sky)
    std::vector<double> z_vals_quad;   // (N_quad * N_sky)

    void compute(int N_lambda_ = 90, int N_beta_ = 60);
    void print_summary() const;
};


// ============================================================================
// GalacticGrid
//
// Holds all precomputed galactic geometry on device.
//
// Lifecycle:
//   1. allocate_and_setup(...)      — upload quadrature geometry
//   2. initialize(R_d, z_d, times)  — compute fixed sky weights + R_avg
//      These two steps happen ONCE before inference starts.
//
// Per likelihood call:
//   3. compute_gal_covariance(...)  — broadcast R_avg * S_gal(f) per walker
//      OR: access R_avg directly from the likelihood kernel (preferred,
//          avoids storing the full (N_times, N_freqs, 6) tensor).
//
// R_d, z_d, alpha0, beta0 are NOT inferred — they are fixed at init.
// Only spectral parameters (Amp, alpha, f_1, f_knee, f_2) vary.
// ============================================================================

struct GalacticGrid {

    // ---- Quadrature geometry (device, not owned by caller) ----
    double *R_vals_quad;   // (N_quad * N_sky)
    double *z_vals_quad;   // (N_quad * N_sky)
    double *quad_weights;  // (N_quad,)
    double *cos_beta_ecl;  // (N_sky,)
    double *lam_ecl;       // (N_sky,)
    double *beta_ecl;      // (N_sky,)

    int N_quad;
    int N_sky;

    // ---- LISA orientation (fixed at init) ----
    double alpha0;
    double beta0;
    double t0;

    // ---- Sky weights — (N_sky,), computed once from R_d, z_d ----
    double *weights;

    // ---- Sky-averaged response — (N_times, 6), computed once ----
    // Layout: R_avg[t * 6 + k], k in [XX, XY, XZ, YY, YZ, ZZ]
    // This is what the likelihood kernel reads directly — no per-walker copy.
    double *R_avg;
    int N_times_alloc;

    // ---- Optional: full covariance for debugging / external inspection ----
    // Layout: R_gal_arr[(t * N_freqs + f) * 6 + k]
    // Not used inside psd_likelihood_xyz_kernel — kept for get_noise_covariance_arr.
    double *R_gal_arr;
    int N_freqs_alloc;

    bool initialized;

    // ------------------------------------------------------------------
    // Step 1: allocate GPU memory and upload quadrature geometry
    // ------------------------------------------------------------------
    void allocate_and_setup(
        const double *h_R_vals_quad,
        const double *h_z_vals_quad,
        const double *h_quad_weights,
        const double *h_cos_beta_ecl,
        const double *h_lam_ecl,
        const double *h_beta_ecl,
        int N_quad_in, int N_sky_in,
        double alpha0_in, double beta0_in,
        double t0_in,
        int N_times_in, int N_freqs_in
    );

    /** Free all device allocations. */
    void free_gpu();

    // ------------------------------------------------------------------
    // Step 2a: compute normalized sky weights (called by initialize)
    // ------------------------------------------------------------------
    void compute_sky_weights(double R_d, double z_d);

    // ------------------------------------------------------------------
    // Step 2b: compute sky-averaged response R_avg[t, k] (called by initialize)
    // ------------------------------------------------------------------
    void compute_sky_average(const double *d_times, int N_times);

    // ------------------------------------------------------------------
    // Step 2 (combined): call once before inference begins
    // ------------------------------------------------------------------
    /**
     * Compute and store sky weights + R_avg from fixed disk/orbit parameters.
     * Must be called after allocate_and_setup().
     *
     * @param R_d      Disk radial scale length [kpc]
     * @param z_d      Disk vertical scale height [kpc]
     * @param d_times  Device array of segment centre times (N_times,)
     * @param N_times  Number of time segments
     */
    void initialize(double R_d, double z_d,
                    const double *d_times, int N_times);

    // ------------------------------------------------------------------
    // Step 3: per-likelihood call (spectral params vary per walker)
    //
    // Computes R_gal_arr[(t * N_freqs + f) * 6 + k]
    //   = R_avg[t * 6 + k] * S_gal(f; Amp, alpha, f_1, f_knee, f_2)
    //
    // This uses a single set of spectral params (e.g. for diagnostics /
    // get_noise_covariance_arr).  Inside psd_likelihood_xyz_kernel the
    // per-walker S_gal is computed inline, so this method is NOT called
    // on the hot path.
    // ------------------------------------------------------------------
    void compute_gal_covariance(
        const double *d_freqs, int N_freqs, int N_times,
        double Amp, double alpha,
        double f_1, double f_knee, double f_2,
        double avg_d
    );
};


// ============================================================================
// Device functions — TDI angular response (defined in galactic_response.cu)
// ============================================================================

CUDA_DEVICE double galaxy_XX(double alpha0, double beta0, double theta_N, double phi_N, double delta_t);
CUDA_DEVICE double galaxy_XY(double alpha0, double beta0, double theta_N, double phi_N, double delta_t);
CUDA_DEVICE double galaxy_XZ(double alpha0, double beta0, double theta_N, double phi_N, double delta_t);
CUDA_DEVICE double galaxy_YY(double alpha0, double beta0, double theta_N, double phi_N, double delta_t);
CUDA_DEVICE double galaxy_YZ(double alpha0, double beta0, double theta_N, double phi_N, double delta_t);
CUDA_DEVICE double galaxy_ZZ(double alpha0, double beta0, double theta_N, double phi_N, double delta_t);