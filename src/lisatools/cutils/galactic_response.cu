#include "galactic_response.hpp"
#include "gbt_global.h"

#include <cmath>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>
#include <stdio.h>

// ============================================================================
// GPU error checking
// ============================================================================

#ifdef __CUDACC__
#ifndef gpuErrchk
#define gpuErrchk(ans) { gpuAssert_gal((ans), __FILE__, __LINE__); }
inline void gpuAssert_gal(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert (galactic): %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif
#endif

// ============================================================================
// Thread block sizes
// ============================================================================

#define GAL_THREADS_SKY  256
#define GAL_THREADS_SPEC 256

// ============================================================================
// GalacticGridSetup — host implementation
// ============================================================================

static const double GL16_NODES[16] = {
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318,
    -0.7554044083550030, -0.6178762444026438, -0.4580167776572274,
    -0.2816035507792589, -0.0950125098376374,  0.0950125098376374,
     0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
     0.7554044083550030,  0.8656312023878318,  0.9445750230732326,
     0.9894009349916499
};

static const double GL16_WEIGHTS[16] = {
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928,
    0.1246289712555339, 0.1495959888165767, 0.1691565193950025,
    0.1826034150449236, 0.1894506104550685, 0.1894506104550685,
    0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479,
    0.0271524594117541
};

static constexpr double GGS_R_SUN = 8.12;
static constexpr double GGS_Z_SUN = 0.03;
static constexpr double GGS_D_MAX = 20.0;

static const double GGS_ROT[3][3] = {
    { -5.48755604e-02, -9.93821362e-01, -9.64768044e-02 },
    {  4.94109428e-01, -1.10990888e-01,  8.62285855e-01 },
    { -8.67666149e-01, -3.51679084e-04,  4.97147192e-01 }
};

static inline void ecl_to_gal_sky(double lam_ecl, double beta_ecl,
                                    double &lam_gal, double &beta_gal)
{
    double cb = cos(beta_ecl);
    double x  = cos(lam_ecl) * cb;
    double y  = sin(lam_ecl) * cb;
    double z  = sin(beta_ecl);

    double xr = GGS_ROT[0][0]*x + GGS_ROT[0][1]*y + GGS_ROT[0][2]*z;
    double yr = GGS_ROT[1][0]*x + GGS_ROT[1][1]*y + GGS_ROT[1][2]*z;
    double zr = GGS_ROT[2][0]*x + GGS_ROT[2][1]*y + GGS_ROT[2][2]*z;

    double phi = atan2(yr, xr);
    if (phi < 0.0) phi += 2.0 * M_PI;

    lam_gal  = phi;
    beta_gal = atan2(zr, sqrt(xr*xr + yr*yr));
}

void GalacticGridSetup::compute(int N_lambda_, int N_beta_)
{
    N_lambda = N_lambda_;
    N_beta   = N_beta_;
    N_sky    = N_lambda * N_beta;
    N_quad   = 16;

    std::vector<double> lams(N_lambda), betas(N_beta);
    for (int il = 0; il < N_lambda; il++)
        lams[il]  = 2.0 * M_PI * il / N_lambda;
    for (int ib = 0; ib < N_beta; ib++)
        betas[ib] = -M_PI/2.0 + M_PI * ib / (N_beta - 1);

    lam_ecl.resize(N_sky);
    beta_ecl.resize(N_sky);
    cos_beta_ecl.resize(N_sky);
    lam_gal.resize(N_sky);
    beta_gal.resize(N_sky);

    for (int il = 0; il < N_lambda; il++) {
        for (int ib = 0; ib < N_beta; ib++) {
            int idx           = il * N_beta + ib;
            lam_ecl[idx]      = lams[il];
            beta_ecl[idx]     = betas[ib];
            cos_beta_ecl[idx] = cos(betas[ib]);
            ecl_to_gal_sky(lams[il], betas[ib], lam_gal[idx], beta_gal[idx]);
        }
    }

    D_nodes.resize(N_quad);
    quad_weights.resize(N_quad);
    for (int q = 0; q < N_quad; q++) {
        D_nodes[q]      = 0.5 * (GL16_NODES[q] + 1.0) * GGS_D_MAX;
        quad_weights[q] = 0.5 * GGS_D_MAX * GL16_WEIGHTS[q];
    }

    R_vals_quad.resize(N_quad * N_sky);
    z_vals_quad.resize(N_quad * N_sky);

    for (int q = 0; q < N_quad; q++) {
        double D = D_nodes[q];
        for (int i = 0; i < N_sky; i++) {
            double lg  = lam_gal[i];
            double bg  = beta_gal[i];
            double x = D * sin(lg) * cos(bg);
            double y = GGS_R_SUN - D * cos(lg) * cos(bg);
            double z = GGS_Z_SUN - D * sin(bg);
            R_vals_quad[q * N_sky + i] = sqrt(x*x + y*y);
            z_vals_quad[q * N_sky + i] = z;
        }
    }
}

void GalacticGridSetup::print_summary() const
{
    printf("GalacticGridSetup summary:\n");
    printf("  N_lambda=%d  N_beta=%d  N_sky=%d  N_quad=%d\n",
           N_lambda, N_beta, N_sky, N_quad);
    if (D_nodes.empty()) { printf("  (not yet computed)\n"); return; }
    printf("  D_nodes: [%.4f, %.4f] kpc\n", D_nodes.front(), D_nodes.back());
    double w_sum = 0.0; for (double w : quad_weights) w_sum += w;
    printf("  quad_weights sum = %.10f  (expect %.4f)\n", w_sum, GGS_D_MAX);
}

// ============================================================================
// Device helpers
// ============================================================================

static CUDA_DEVICE
double disk_density(double R, double z, double R_d, double z_d)
{
    return exp(-R / R_d) * exp(-fabs(z) / z_d);
}

static CUDA_DEVICE
double gal_spectral_model(double f, double Amp, double alpha,
                           double f_1, double f_knee, double f_2,
                           double avg_d)
{
    double omega_f  = 2.0 * M_PI * f;
    double x        = omega_f * avg_d;
    double transfer = 4.0 * (x * sin(x)) * (x * sin(x))
                    * 4.0 * sin(2.0 * x) * sin(2.0 * x);
    double psd_gal  = Amp * exp(-pow(f / f_1, alpha)) * pow(f, -7.0 / 3.0)
                    * 0.5 * (1.0 + tanh(-(f - f_knee) / f_2));
    return psd_gal * transfer;
}

// ============================================================================
// TDI response device functions (unchanged from original)
// ============================================================================
CUDA_DEVICE
double galaxy_XX(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double ot  = OMEGA_LISA * delta_t;
    double a0t = alpha0 + ot;           // alpha0 + omega*(t - t0)
    double pi2 = PI2_GAL;
    double s3  = SQRT3_GAL;

    double result = (-105.0 * (
        -12484.0
        + 1840.0  * cos(2*theta_N - pi2)
        + 148.0   * cos(4*theta_N - pi2)
        + 486.0   * cos(4*beta0 - 4*phi_N)
        + 81.0    * cos(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        + 81.0    * cos(4*(beta0 + theta_N - pi2 - phi_N))
        + 324.0   * cos(4*(a0t - beta0))
        + 630.0   * cos(4*(a0t - beta0 - theta_N - pi2))
        + 630.0   * cos(4*(a0t - beta0 + theta_N - pi2))
        - 5328.0  * cos(2*(a0t - phi_N))
        - 108.0   * cos(4*(a0t - phi_N))
        + 72.0    * cos(2*(a0t - 2*theta_N - pi2 - phi_N))
        + 2592.0  * cos(2*(a0t - theta_N - pi2 - phi_N))
        - 18.0    * cos(4*(a0t - theta_N - pi2 - phi_N))
        + 2592.0  * cos(2*(a0t + theta_N - pi2 - phi_N))
        - 18.0    * cos(4*(a0t + theta_N - pi2 - phi_N))
        + 72.0    * cos(2*(a0t + 2*theta_N - pi2 - phi_N))
        + 648.0   * cos(2*(a0t - 2*beta0 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        + 360.0   * cos(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        + 360.0   * cos(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        + 72.0    * cos(4*alpha0 - 2*theta_N - pi2 - 4*phi_N + 4*ot)
        + 72.0    * cos(4*alpha0 + 2*theta_N - pi2 - 4*phi_N + 4*ot)
        + 72.0    * cos(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        + 48.0    * cos(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 1.0     * cos(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        + 48.0    * cos(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        + 6.0     * cos(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        + 1.0     * cos(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
        + 80.0    * s3 * sin(a0t - 4*theta_N - pi2 - phi_N)
        - 3488.0  * s3 * sin(a0t - 2*theta_N - pi2 - phi_N)
        + 3488.0  * s3 * sin(a0t + 2*theta_N - pi2 - phi_N)
        - 80.0    * s3 * sin(a0t + 4*theta_N - pi2 - phi_N)
        + 216.0   * s3 * sin(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        - 432.0   * s3 * sin(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        + 432.0   * s3 * sin(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        - 216.0   * s3 * sin(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        + 48.0    * s3 * sin(3*a0t - 4*theta_N - pi2 - 3*phi_N - 2*ot)
        - 96.0    * s3 * sin(3*a0t - 2*theta_N - pi2 - 3*phi_N - 2*ot)
        + 96.0    * s3 * sin(3*a0t + 2*theta_N - pi2 - 3*phi_N - 2*ot)
        - 48.0    * s3 * sin(3*a0t + 4*theta_N - pi2 - 3*phi_N - 2*ot)
        - 504.0   * s3 * sin(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        - 144.0   * s3 * sin(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        + 144.0   * s3 * sin(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        + 504.0   * s3 * sin(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        + 168.0   * s3 * sin(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        + 48.0    * s3 * sin(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        - 48.0    * s3 * sin(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        - 168.0   * s3 * sin(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        - 8.0     * s3 * sin(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 16.0    * s3 * sin(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 16.0    * s3 * sin(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 8.0     * s3 * sin(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
    )) / 262144.0;

    return result;
}

// ----------------------------------------------------------------------------

CUDA_DEVICE
double galaxy_XY(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double ot  = OMEGA_LISA * delta_t;
    double a0t = alpha0 + ot; // alpha0 + omega*(t - t0)
    double pi2 = PI2_GAL;
    double s3  = SQRT3_GAL;

    double result = (105.0 * (
        -12484.0
        + 1840.0  * cos(2*theta_N - pi2)
        + 148.0   * cos(4*theta_N - pi2)
        + 486.0   * cos(4*beta0 - 4*phi_N)
        + 81.0    * cos(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        + 81.0    * cos(4*(beta0 + theta_N - pi2 - phi_N))
        + 324.0   * cos(4*(a0t - beta0))
        + 630.0   * cos(4*(a0t - beta0 - theta_N - pi2))
        + 630.0   * cos(4*(a0t - beta0 + theta_N - pi2))
        - 5328.0  * cos(2*(a0t - phi_N))
        - 108.0   * cos(4*(a0t - phi_N))
        + 72.0    * cos(2*(a0t - 2*theta_N - pi2 - phi_N))
        + 2592.0  * cos(2*(a0t - theta_N - pi2 - phi_N))
        - 18.0    * cos(4*(a0t - theta_N - pi2 - phi_N))
        + 2592.0  * cos(2*(a0t + theta_N - pi2 - phi_N))
        - 18.0    * cos(4*(a0t + theta_N - pi2 - phi_N))
        + 72.0    * cos(2*(a0t + 2*theta_N - pi2 - phi_N))
        + 648.0   * cos(2*(a0t - 2*beta0 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        - 648.0   * cos(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        + 1296.0  * cos(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        - 1296.0  * cos(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        + 648.0   * cos(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        + 1512.0  * cos(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        + 432.0   * cos(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        - 432.0   * cos(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        - 1512.0  * cos(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        + 360.0   * cos(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        + 360.0   * cos(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        + 72.0    * cos(4*alpha0 - 2*theta_N - pi2 - 4*phi_N + 4*ot)
        + 72.0    * cos(4*alpha0 + 2*theta_N - pi2 - 4*phi_N + 4*ot)
        - 504.0   * cos(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        - 144.0   * cos(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        + 144.0   * cos(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        + 504.0   * cos(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        + 72.0    * cos(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        + 48.0    * cos(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        + 24.0    * cos(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        - 48.0    * cos(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 48.0    * cos(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 24.0    * cos(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 1.0     * cos(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        + 48.0    * cos(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        + 6.0     * cos(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        + 1.0     * cos(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
        - 486.0   * s3 * sin(4*beta0 - 4*phi_N)
        - 81.0    * s3 * sin(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        + 324.0   * s3 * sin(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        + 324.0   * s3 * sin(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        - 81.0    * s3 * sin(4*(beta0 + theta_N - pi2 - phi_N))
        + 324.0   * s3 * sin(4*(a0t - beta0))
        + 630.0   * s3 * sin(4*(a0t - beta0 - theta_N - pi2))
        + 630.0   * s3 * sin(4*(a0t - beta0 + theta_N - pi2))
        + 80.0    * s3 * sin(a0t - 4*theta_N - pi2 - phi_N)
        - 3488.0  * s3 * sin(a0t - 2*theta_N - pi2 - phi_N)
        + 3488.0  * s3 * sin(a0t + 2*theta_N - pi2 - phi_N)
        - 80.0    * s3 * sin(a0t + 4*theta_N - pi2 - phi_N)
        + 648.0   * s3 * sin(2*(a0t - 2*beta0 + phi_N))
        - 756.0   * s3 * sin(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        + 432.0   * s3 * sin(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        + 432.0   * s3 * sin(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        - 756.0   * s3 * sin(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        + 216.0   * s3 * sin(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        - 432.0   * s3 * sin(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        + 432.0   * s3 * sin(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        - 216.0   * s3 * sin(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        + 48.0    * s3 * sin(3*a0t - 4*theta_N - pi2 - 3*phi_N - 2*ot)
        - 96.0    * s3 * sin(3*a0t - 2*theta_N - pi2 - 3*phi_N - 2*ot)
        + 96.0    * s3 * sin(3*a0t + 2*theta_N - pi2 - 3*phi_N - 2*ot)
        - 48.0    * s3 * sin(3*a0t + 4*theta_N - pi2 - 3*phi_N - 2*ot)
        - 504.0   * s3 * sin(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        - 144.0   * s3 * sin(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        + 144.0   * s3 * sin(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        + 504.0   * s3 * sin(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        + 360.0   * s3 * sin(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        + 360.0   * s3 * sin(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        + 168.0   * s3 * sin(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        + 48.0    * s3 * sin(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        - 48.0    * s3 * sin(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        - 168.0   * s3 * sin(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        + 72.0    * s3 * sin(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        - 84.0    * s3 * sin(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        + 48.0    * s3 * sin(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        - 84.0    * s3 * sin(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 8.0     * s3 * sin(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 16.0    * s3 * sin(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 16.0    * s3 * sin(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 8.0     * s3 * sin(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
        - 4.0     * s3 * sin(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 4.0     * s3 * sin(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 1.0     * s3 * sin(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        + 48.0    * s3 * sin(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        + 6.0     * s3 * sin(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        + 1.0     * s3 * sin(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
    )) / 524288.0;

    return result;
}

// ----------------------------------------------------------------------------

CUDA_DEVICE
double galaxy_YY(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double ot  = OMEGA_LISA * delta_t;
    double a0t = alpha0 + ot; // alpha0 + omega*(t - t0)
    double pi2 = PI2_GAL;
    double s3  = SQRT3_GAL;

    double result = (105.0 * (
        24968.0
        - 3680.0  * cos(2*theta_N - pi2)
        - 296.0   * cos(4*theta_N - pi2)
        + 486.0   * cos(4*beta0 - 4*phi_N)
        + 81.0    * cos(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        + 81.0    * cos(4*(beta0 + theta_N - pi2 - phi_N))
        + 324.0   * cos(4*(a0t - beta0))
        + 630.0   * cos(4*(a0t - beta0 - theta_N - pi2))
        + 630.0   * cos(4*(a0t - beta0 + theta_N - pi2))
        + 10656.0 * cos(2*(a0t - phi_N))
        + 216.0   * cos(4*(a0t - phi_N))
        - 144.0   * cos(2*(a0t - 2*theta_N - pi2 - phi_N))
        - 5184.0  * cos(2*(a0t - theta_N - pi2 - phi_N))
        + 36.0    * cos(4*(a0t - theta_N - pi2 - phi_N))
        - 5184.0  * cos(2*(a0t + theta_N - pi2 - phi_N))
        + 36.0    * cos(4*(a0t + theta_N - pi2 - phi_N))
        - 144.0   * cos(2*(a0t + 2*theta_N - pi2 - phi_N))
        + 648.0   * cos(2*(a0t - 2*beta0 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        + 648.0   * cos(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        - 1296.0  * cos(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        + 1296.0  * cos(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        - 648.0   * cos(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        - 1512.0  * cos(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        - 432.0   * cos(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        + 432.0   * cos(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        + 1512.0  * cos(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        + 360.0   * cos(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        + 360.0   * cos(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        - 144.0   * cos(4*alpha0 - 2*theta_N - pi2 - 4*phi_N + 4*ot)
        - 144.0   * cos(4*alpha0 + 2*theta_N - pi2 - 4*phi_N + 4*ot)
        + 504.0   * cos(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        + 144.0   * cos(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        - 144.0   * cos(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        - 504.0   * cos(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        + 72.0    * cos(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        + 48.0    * cos(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 24.0    * cos(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 48.0    * cos(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 48.0    * cos(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 24.0    * cos(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 1.0     * cos(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        + 48.0    * cos(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        + 6.0     * cos(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        + 1.0     * cos(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
        + 486.0   * s3 * sin(4*beta0 - 4*phi_N)
        + 81.0    * s3 * sin(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        - 324.0   * s3 * sin(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        - 324.0   * s3 * sin(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        + 81.0    * s3 * sin(4*(beta0 + theta_N - pi2 - phi_N))
        - 324.0   * s3 * sin(4*(a0t - beta0))
        - 630.0   * s3 * sin(4*(a0t - beta0 - theta_N - pi2))
        - 630.0   * s3 * sin(4*(a0t - beta0 + theta_N - pi2))
        - 160.0   * s3 * sin(a0t - 4*theta_N - pi2 - phi_N)
        + 6976.0  * s3 * sin(a0t - 2*theta_N - pi2 - phi_N)
        - 6976.0  * s3 * sin(a0t + 2*theta_N - pi2 - phi_N)
        + 160.0   * s3 * sin(a0t + 4*theta_N - pi2 - phi_N)
        - 648.0   * s3 * sin(2*(a0t - 2*beta0 + phi_N))
        + 756.0   * s3 * sin(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        - 432.0   * s3 * sin(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        - 432.0   * s3 * sin(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        + 756.0   * s3 * sin(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        + 216.0   * s3 * sin(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        - 432.0   * s3 * sin(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        + 432.0   * s3 * sin(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        - 216.0   * s3 * sin(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        - 96.0    * s3 * sin(3*a0t - 4*theta_N - pi2 - 3*phi_N - 2*ot)
        + 192.0   * s3 * sin(3*a0t - 2*theta_N - pi2 - 3*phi_N - 2*ot)
        - 192.0   * s3 * sin(3*a0t + 2*theta_N - pi2 - 3*phi_N - 2*ot)
        + 96.0    * s3 * sin(3*a0t + 4*theta_N - pi2 - 3*phi_N - 2*ot)
        - 504.0   * s3 * sin(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        - 144.0   * s3 * sin(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        + 144.0   * s3 * sin(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        + 504.0   * s3 * sin(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        - 360.0   * s3 * sin(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        - 360.0   * s3 * sin(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        + 168.0   * s3 * sin(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        + 48.0    * s3 * sin(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        - 48.0    * s3 * sin(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        - 168.0   * s3 * sin(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        - 72.0    * s3 * sin(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        + 84.0    * s3 * sin(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 48.0    * s3 * sin(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        + 84.0    * s3 * sin(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 8.0     * s3 * sin(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 16.0    * s3 * sin(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 16.0    * s3 * sin(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 8.0     * s3 * sin(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 4.0     * s3 * sin(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 4.0     * s3 * sin(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 1.0     * s3 * sin(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        - 48.0    * s3 * sin(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        - 6.0     * s3 * sin(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        - 1.0     * s3 * sin(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
    )) / 524288.0;

    return result;
}

// ----------------------------------------------------------------------------

CUDA_DEVICE
double galaxy_XZ(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double ot  = OMEGA_LISA * delta_t;
    double a0t = alpha0 + ot; // alpha0 + omega*(t - t0)
    double pi2 = PI2_GAL;
    double s3  = SQRT3_GAL;

    double result = (105.0 * (
        -12484.0
        + 1840.0  * cos(2*theta_N - pi2)
        + 148.0   * cos(4*theta_N - pi2)
        + 486.0   * cos(4*beta0 - 4*phi_N)
        + 81.0    * cos(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        + 81.0    * cos(4*(beta0 + theta_N - pi2 - phi_N))
        + 324.0   * cos(4*(a0t - beta0))
        + 630.0   * cos(4*(a0t - beta0 - theta_N - pi2))
        + 630.0   * cos(4*(a0t - beta0 + theta_N - pi2))
        - 5328.0  * cos(2*(a0t - phi_N))
        - 108.0   * cos(4*(a0t - phi_N))
        + 72.0    * cos(2*(a0t - 2*theta_N - pi2 - phi_N))
        + 2592.0  * cos(2*(a0t - theta_N - pi2 - phi_N))
        - 18.0    * cos(4*(a0t - theta_N - pi2 - phi_N))
        + 2592.0  * cos(2*(a0t + theta_N - pi2 - phi_N))
        - 18.0    * cos(4*(a0t + theta_N - pi2 - phi_N))
        + 72.0    * cos(2*(a0t + 2*theta_N - pi2 - phi_N))
        + 648.0   * cos(2*(a0t - 2*beta0 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        + 648.0   * cos(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        - 1296.0  * cos(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        + 1296.0  * cos(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        - 648.0   * cos(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        - 1512.0  * cos(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        - 432.0   * cos(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        + 432.0   * cos(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        + 1512.0  * cos(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        + 360.0   * cos(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        + 360.0   * cos(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        + 72.0    * cos(4*alpha0 - 2*theta_N - pi2 - 4*phi_N + 4*ot)
        + 72.0    * cos(4*alpha0 + 2*theta_N - pi2 - 4*phi_N + 4*ot)
        + 504.0   * cos(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        + 144.0   * cos(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        - 144.0   * cos(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        - 504.0   * cos(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        + 72.0    * cos(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        + 48.0    * cos(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 24.0    * cos(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 48.0    * cos(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 48.0    * cos(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 24.0    * cos(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 1.0     * cos(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        + 48.0    * cos(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        + 6.0     * cos(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        + 1.0     * cos(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
        + 486.0   * s3 * sin(4*beta0 - 4*phi_N)
        + 81.0    * s3 * sin(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        - 324.0   * s3 * sin(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        - 324.0   * s3 * sin(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        + 81.0    * s3 * sin(4*(beta0 + theta_N - pi2 - phi_N))
        - 324.0   * s3 * sin(4*(a0t - beta0))
        - 630.0   * s3 * sin(4*(a0t - beta0 - theta_N - pi2))
        - 630.0   * s3 * sin(4*(a0t - beta0 + theta_N - pi2))
        + 80.0    * s3 * sin(a0t - 4*theta_N - pi2 - phi_N)
        - 3488.0  * s3 * sin(a0t - 2*theta_N - pi2 - phi_N)
        + 3488.0  * s3 * sin(a0t + 2*theta_N - pi2 - phi_N)
        - 80.0    * s3 * sin(a0t + 4*theta_N - pi2 - phi_N)
        - 648.0   * s3 * sin(2*(a0t - 2*beta0 + phi_N))
        + 756.0   * s3 * sin(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        - 432.0   * s3 * sin(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        - 432.0   * s3 * sin(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        + 756.0   * s3 * sin(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        + 216.0   * s3 * sin(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        - 432.0   * s3 * sin(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        + 432.0   * s3 * sin(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        - 216.0   * s3 * sin(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        + 48.0    * s3 * sin(3*a0t - 4*theta_N - pi2 - 3*phi_N - 2*ot)
        - 96.0    * s3 * sin(3*a0t - 2*theta_N - pi2 - 3*phi_N - 2*ot)
        + 96.0    * s3 * sin(3*a0t + 2*theta_N - pi2 - 3*phi_N - 2*ot)
        - 48.0    * s3 * sin(3*a0t + 4*theta_N - pi2 - 3*phi_N - 2*ot)
        - 504.0   * s3 * sin(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        - 144.0   * s3 * sin(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        + 144.0   * s3 * sin(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        + 504.0   * s3 * sin(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        - 360.0   * s3 * sin(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        - 360.0   * s3 * sin(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        + 168.0   * s3 * sin(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        + 48.0    * s3 * sin(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        - 48.0    * s3 * sin(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        - 168.0   * s3 * sin(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        - 72.0    * s3 * sin(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        + 84.0    * s3 * sin(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 48.0    * s3 * sin(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        + 84.0    * s3 * sin(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 8.0     * s3 * sin(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 16.0    * s3 * sin(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 16.0    * s3 * sin(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 8.0     * s3 * sin(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 4.0     * s3 * sin(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 4.0     * s3 * sin(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 1.0     * s3 * sin(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        - 48.0    * s3 * sin(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        - 6.0     * s3 * sin(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        - 1.0     * s3 * sin(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
    )) / 524288.0;

    return result;
}

// ----------------------------------------------------------------------------

CUDA_DEVICE
double galaxy_YZ(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double ot  = OMEGA_LISA * delta_t;
    double a0t = alpha0 + ot; // alpha0 + omega*(t - t0)
    double pi2 = PI2_GAL;
    double s3  = SQRT3_GAL;

    double result = (-105.0 * (
        6242.0
        - 920.0   * cos(2*theta_N - pi2)
        - 74.0    * cos(4*theta_N - pi2)
        + 486.0   * cos(4*beta0 - 4*phi_N)
        + 81.0    * cos(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        - 324.0   * cos(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        + 81.0    * cos(4*(beta0 + theta_N - pi2 - phi_N))
        + 324.0   * cos(4*(a0t - beta0))
        + 630.0   * cos(4*(a0t - beta0 - theta_N - pi2))
        + 630.0   * cos(4*(a0t - beta0 + theta_N - pi2))
        + 2664.0  * cos(2*(a0t - phi_N))
        + 54.0    * cos(4*(a0t - phi_N))
        - 36.0    * cos(2*(a0t - 2*theta_N - pi2 - phi_N))
        - 1296.0  * cos(2*(a0t - theta_N - pi2 - phi_N))
        + 9.0     * cos(4*(a0t - theta_N - pi2 - phi_N))
        - 1296.0  * cos(2*(a0t + theta_N - pi2 - phi_N))
        + 9.0     * cos(4*(a0t + theta_N - pi2 - phi_N))
        - 36.0    * cos(2*(a0t + 2*theta_N - pi2 - phi_N))
        + 648.0   * cos(2*(a0t - 2*beta0 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        + 432.0   * cos(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        - 756.0   * cos(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        + 360.0   * cos(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        + 360.0   * cos(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        - 36.0    * cos(4*alpha0 - 2*theta_N - pi2 - 4*phi_N + 4*ot)
        - 36.0    * cos(4*alpha0 + 2*theta_N - pi2 - 4*phi_N + 4*ot)
        + 72.0    * cos(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        + 48.0    * cos(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        - 84.0    * cos(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 4.0     * cos(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 1.0     * cos(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        + 48.0    * cos(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        + 6.0     * cos(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        + 1.0     * cos(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
        - 40.0    * s3 * sin(a0t - 4*theta_N - pi2 - phi_N)
        + 1744.0  * s3 * sin(a0t - 2*theta_N - pi2 - phi_N)
        - 1744.0  * s3 * sin(a0t + 2*theta_N - pi2 - phi_N)
        + 40.0    * s3 * sin(a0t + 4*theta_N - pi2 - phi_N)
        + 216.0   * s3 * sin(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        - 432.0   * s3 * sin(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        + 432.0   * s3 * sin(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        - 216.0   * s3 * sin(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        - 24.0    * s3 * sin(3*a0t - 4*theta_N - pi2 - 3*phi_N - 2*ot)
        + 48.0    * s3 * sin(3*a0t - 2*theta_N - pi2 - 3*phi_N - 2*ot)
        - 48.0    * s3 * sin(3*a0t + 2*theta_N - pi2 - 3*phi_N - 2*ot)
        + 24.0    * s3 * sin(3*a0t + 4*theta_N - pi2 - 3*phi_N - 2*ot)
        - 504.0   * s3 * sin(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        - 144.0   * s3 * sin(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        + 144.0   * s3 * sin(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        + 504.0   * s3 * sin(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        + 168.0   * s3 * sin(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        + 48.0    * s3 * sin(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        - 48.0    * s3 * sin(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        - 168.0   * s3 * sin(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        - 8.0     * s3 * sin(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 16.0    * s3 * sin(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 16.0    * s3 * sin(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 8.0     * s3 * sin(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
    )) / 262144.0;

    return result;
}

// ----------------------------------------------------------------------------

CUDA_DEVICE
double galaxy_ZZ(double alpha0, double beta0, double theta_N, double phi_N, double delta_t)
{
    double ot  = OMEGA_LISA * delta_t;
    double a0t = alpha0 + ot; // alpha0 + omega*(t - t0)
    double pi2 = PI2_GAL;
    double s3  = SQRT3_GAL;

    double result = (-105.0 * (
        -24968.0
        + 3680.0  * cos(2*theta_N - pi2)
        + 296.0   * cos(4*theta_N - pi2)
        - 486.0   * cos(4*beta0 - 4*phi_N)
        - 81.0    * cos(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        + 324.0   * cos(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        + 324.0   * cos(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        - 81.0    * cos(4*(beta0 + theta_N - pi2 - phi_N))
        - 324.0   * cos(4*(a0t - beta0))
        - 630.0   * cos(4*(a0t - beta0 - theta_N - pi2))
        - 630.0   * cos(4*(a0t - beta0 + theta_N - pi2))
        - 10656.0 * cos(2*(a0t - phi_N))
        - 216.0   * cos(4*(a0t - phi_N))
        + 144.0   * cos(2*(a0t - 2*theta_N - pi2 - phi_N))
        + 5184.0  * cos(2*(a0t - theta_N - pi2 - phi_N))
        - 36.0    * cos(4*(a0t - theta_N - pi2 - phi_N))
        + 5184.0  * cos(2*(a0t + theta_N - pi2 - phi_N))
        - 36.0    * cos(4*(a0t + theta_N - pi2 - phi_N))
        + 144.0   * cos(2*(a0t + 2*theta_N - pi2 - phi_N))
        - 648.0   * cos(2*(a0t - 2*beta0 + phi_N))
        + 756.0   * cos(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        - 432.0   * cos(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        - 432.0   * cos(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        + 756.0   * cos(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        + 648.0   * cos(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        - 1296.0  * cos(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        + 1296.0  * cos(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        - 648.0   * cos(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        - 1512.0  * cos(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        - 432.0   * cos(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        + 432.0   * cos(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        + 1512.0  * cos(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        - 360.0   * cos(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        - 360.0   * cos(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        + 144.0   * cos(4*alpha0 - 2*theta_N - pi2 - 4*phi_N + 4*ot)
        + 144.0   * cos(4*alpha0 + 2*theta_N - pi2 - 4*phi_N + 4*ot)
        + 504.0   * cos(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        + 144.0   * cos(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        - 144.0   * cos(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        - 504.0   * cos(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        - 72.0    * cos(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        + 84.0    * cos(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 48.0    * cos(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        + 84.0    * cos(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 24.0    * cos(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 48.0    * cos(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 48.0    * cos(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 24.0    * cos(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 4.0     * cos(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 4.0     * cos(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 1.0     * cos(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        - 48.0    * cos(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        - 6.0     * cos(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        - 1.0     * cos(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
        + 486.0   * s3 * sin(4*beta0 - 4*phi_N)
        + 81.0    * s3 * sin(4*beta0 - 4*theta_N - pi2 - 4*phi_N)
        - 324.0   * s3 * sin(4*beta0 - 2*theta_N - pi2 - 4*phi_N)
        - 324.0   * s3 * sin(4*beta0 + 2*theta_N - pi2 - 4*phi_N)
        + 81.0    * s3 * sin(4*(beta0 + theta_N - pi2 - phi_N))
        - 324.0   * s3 * sin(4*(a0t - beta0))
        - 630.0   * s3 * sin(4*(a0t - beta0 - theta_N - pi2))
        - 630.0   * s3 * sin(4*(a0t - beta0 + theta_N - pi2))
        + 160.0   * s3 * sin(a0t - 4*theta_N - pi2 - phi_N)
        - 6976.0  * s3 * sin(a0t - 2*theta_N - pi2 - phi_N)
        + 6976.0  * s3 * sin(a0t + 2*theta_N - pi2 - phi_N)
        - 160.0   * s3 * sin(a0t + 4*theta_N - pi2 - phi_N)
        - 648.0   * s3 * sin(2*(a0t - 2*beta0 + phi_N))
        + 756.0   * s3 * sin(2*(a0t - 2*beta0 - 2*theta_N - pi2 + phi_N))
        - 432.0   * s3 * sin(2*(a0t - 2*beta0 - theta_N - pi2 + phi_N))
        - 432.0   * s3 * sin(2*(a0t - 2*beta0 + theta_N - pi2 + phi_N))
        + 756.0   * s3 * sin(2*(a0t - 2*beta0 + 2*theta_N - pi2 + phi_N))
        - 216.0   * s3 * sin(a0t - 4*beta0 - 4*theta_N - pi2 + 3*phi_N)
        + 432.0   * s3 * sin(a0t - 4*beta0 - 2*theta_N - pi2 + 3*phi_N)
        - 432.0   * s3 * sin(a0t - 4*beta0 + 2*theta_N - pi2 + 3*phi_N)
        + 216.0   * s3 * sin(a0t - 4*beta0 + 4*theta_N - pi2 + 3*phi_N)
        + 96.0    * s3 * sin(3*a0t - 4*theta_N - pi2 - 3*phi_N - 2*ot)
        - 192.0   * s3 * sin(3*a0t - 2*theta_N - pi2 - 3*phi_N - 2*ot)
        + 192.0   * s3 * sin(3*a0t + 2*theta_N - pi2 - 3*phi_N - 2*ot)
        - 96.0    * s3 * sin(3*a0t + 4*theta_N - pi2 - 3*phi_N - 2*ot)
        + 504.0   * s3 * sin(3*a0t - 4*beta0 - 4*theta_N - pi2 + phi_N - 2*ot)
        + 144.0   * s3 * sin(3*a0t - 4*beta0 - 2*theta_N - pi2 + phi_N - 2*ot)
        - 144.0   * s3 * sin(3*a0t - 4*beta0 + 2*theta_N - pi2 + phi_N - 2*ot)
        - 504.0   * s3 * sin(3*a0t - 4*beta0 + 4*theta_N - pi2 + phi_N - 2*ot)
        - 360.0   * s3 * sin(4*alpha0 - 4*beta0 - 2*theta_N - pi2 + 4*ot)
        - 360.0   * s3 * sin(4*alpha0 - 4*beta0 + 2*theta_N - pi2 + 4*ot)
        - 168.0   * s3 * sin(5*a0t - 4*beta0 - 4*theta_N - pi2 - phi_N - 4*ot)
        - 48.0    * s3 * sin(5*a0t - 4*beta0 - 2*theta_N - pi2 - phi_N - 4*ot)
        + 48.0    * s3 * sin(5*a0t - 4*beta0 + 2*theta_N - pi2 - phi_N - 4*ot)
        + 168.0   * s3 * sin(5*a0t - 4*beta0 + 4*theta_N - pi2 - phi_N - 4*ot)
        - 72.0    * s3 * sin(6*alpha0 - 4*beta0 - 2*phi_N + 6*ot)
        + 84.0    * s3 * sin(6*alpha0 - 4*beta0 - 4*theta_N - pi2 - 2*phi_N + 6*ot)
        - 48.0    * s3 * sin(6*alpha0 - 4*beta0 + 2*theta_N - pi2 - 2*phi_N + 6*ot)
        + 84.0    * s3 * sin(6*alpha0 - 4*beta0 + 4*theta_N - pi2 - 2*phi_N + 6*ot)
        + 8.0     * s3 * sin(7*a0t - 4*beta0 - 4*theta_N - pi2 - 3*phi_N - 6*ot)
        - 16.0    * s3 * sin(7*a0t - 4*beta0 - 2*theta_N - pi2 - 3*phi_N - 6*ot)
        + 16.0    * s3 * sin(7*a0t - 4*beta0 + 2*theta_N - pi2 - 3*phi_N - 6*ot)
        - 8.0     * s3 * sin(7*a0t - 4*beta0 + 4*theta_N - pi2 - 3*phi_N - 6*ot)
        + 4.0     * s3 * sin(8*alpha0 - 4*beta0 - 2*theta_N - pi2 - 4*phi_N + 8*ot)
        + 4.0     * s3 * sin(8*alpha0 - 4*beta0 + 2*theta_N - pi2 - 4*phi_N + 8*ot)
        - 1.0     * s3 * sin(8*alpha0 - 4*beta0 + 4*theta_N - pi2 - 4*phi_N + 8*ot)
        - 48.0    * s3 * sin(6*alpha0 - 2*(2*beta0 + theta_N - pi2 + phi_N - 3*ot))
        - 6.0     * s3 * sin(8*alpha0 - 4*(beta0 + phi_N - 2*ot))
        - 1.0     * s3 * sin(8*alpha0 - 4*(beta0 + theta_N - pi2 + phi_N - 2*ot))
    )) / 524288.0;

    return result;
}

// ============================================================================
// Kernel 1: compute normalized sky weights
// ============================================================================

CUDA_KERNEL
void compute_column_densities_kernel(
    double *col_density_out,
    const double *R_vals_quad,
    const double *z_vals_quad,
    const double *quad_weights,
    const double *cos_beta_ecl,
    double R_d, double z_d,
    int N_quad, int N_sky)
{
#ifdef __CUDACC__
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
#else
    int i = 0, stride = 1;
#endif
    for (; i < N_sky; i += stride)
    {
        double col = 0.0;
        for (int q = 0; q < N_quad; q++)
            col += quad_weights[q] * disk_density(
                       R_vals_quad[q * N_sky + i],
                       z_vals_quad[q * N_sky + i],
                       R_d, z_d);
        col_density_out[i] = col * cos_beta_ecl[i];
    }
}

CUDA_KERNEL
void normalize_weights_kernel(double *weights, double inv_total, int N_sky)
{
#ifdef __CUDACC__
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
#else
    int i = 0, stride = 1;
#endif
    for (; i < N_sky; i += stride)
        weights[i] *= inv_total;
}

// ============================================================================
// Kernel 2: sky-averaged response R_avg[t, k]
// Output layout: R_avg[t_idx * 6 + k], k in {XX,XY,XZ,YY,YZ,ZZ}
// Uses atomicAdd across sky-pixel blocks for a given time.
// ============================================================================

CUDA_KERNEL
void sky_average_response_kernel(
    double *R_avg_out,
    const double *weights,
    const double *lam_ecl,
    const double *beta_ecl,
    const double *times,
    double alpha0, double beta0, double t0,
    int N_times, int N_sky)
{
#ifdef __CUDACC__
    int t_idx = blockIdx.y;
    if (t_idx >= N_times) return;

    double delta_t = times[t_idx] - t0;

    __shared__ double s[6][GAL_THREADS_SKY];
    int tid = threadIdx.x;

    double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int i = blockIdx.x * blockDim.x + tid; i < N_sky; i += gridDim.x * blockDim.x)
    {
        double lam  = lam_ecl[i];
        double beta = beta_ecl[i];
        double w    = weights[i];

        acc[0] += w * galaxy_XX(alpha0, beta0, beta, lam, delta_t);
        acc[1] += w * galaxy_XY(alpha0, beta0, beta, lam, delta_t);
        acc[2] += w * galaxy_XZ(alpha0, beta0, beta, lam, delta_t);
        acc[3] += w * galaxy_YY(alpha0, beta0, beta, lam, delta_t);
        acc[4] += w * galaxy_YZ(alpha0, beta0, beta, lam, delta_t);
        acc[5] += w * galaxy_ZZ(alpha0, beta0, beta, lam, delta_t);
    }

    for (int k = 0; k < 6; k++) s[k][tid] = acc[k];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            for (int k = 0; k < 6; k++)
                s[k][tid] += s[k][tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        for (int k = 0; k < 6; k++)
            atomicAdd(&R_avg_out[t_idx * 6 + k], s[k][0]);

#else
    for (int t_idx = 0; t_idx < N_times; t_idx++)
    {
        double delta_t   = times[t_idx] - t0;
        double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < N_sky; i++)
        {
            double lam  = lam_ecl[i];
            double beta = beta_ecl[i];
            double w    = weights[i];
            acc[0] += w * galaxy_XX(alpha0, beta0, beta, lam, delta_t);
            acc[1] += w * galaxy_XY(alpha0, beta0, beta, lam, delta_t);
            acc[2] += w * galaxy_XZ(alpha0, beta0, beta, lam, delta_t);
            acc[3] += w * galaxy_YY(alpha0, beta0, beta, lam, delta_t);
            acc[4] += w * galaxy_YZ(alpha0, beta0, beta, lam, delta_t);
            acc[5] += w * galaxy_ZZ(alpha0, beta0, beta, lam, delta_t);
        }
        for (int k = 0; k < 6; k++)
            R_avg_out[t_idx * 6 + k] = acc[k];
    }
#endif
}

// ============================================================================
// Kernel 3: spectral broadcast
// R_gal_arr[(t * N_freqs + f) * 6 + k] = R_avg[t * 6 + k] * S_gal(f; params)
// Used only for diagnostic / get_noise_covariance_arr — NOT on hot path.
// ============================================================================

CUDA_KERNEL
void apply_spectral_model_kernel(
    double *R_gal_arr,
    const double *R_avg,
    const double *freqs,
    double Amp, double alpha, double f_1, double f_knee, double f_2,
    double avg_d,
    int N_times, int N_freqs)
{
#ifdef __CUDACC__
    int f_idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx    = blockIdx.y * blockDim.y + threadIdx.y;
    int f_stride = gridDim.x * blockDim.x;
    int t_stride = gridDim.y * blockDim.y;
#else
    int f_idx = 0, t_idx = 0, f_stride = 1, t_stride = 1;
#endif

    for (int t = t_idx; t < N_times; t += t_stride)
    for (int f = f_idx; f < N_freqs; f += f_stride)
    {
        double freq = freqs[f];
        if (freq <= 0.0) freq = freqs[1];

        double S = gal_spectral_model(freq, Amp, alpha, f_1, f_knee, f_2, avg_d);

        int out_base = (t * N_freqs + f) * 6;
        int avg_base =  t * 6;

        for (int k = 0; k < 6; k++)
            R_gal_arr[out_base + k] = R_avg[avg_base + k] * S;
    }
}

// ============================================================================
// GalacticGrid method implementations
// ============================================================================

void GalacticGrid::allocate_and_setup(
    const double *h_R_vals_quad,
    const double *h_z_vals_quad,
    const double *h_quad_weights,
    const double *h_cos_beta_ecl,
    const double *h_lam_ecl,
    const double *h_beta_ecl,
    int N_quad_in, int N_sky_in,
    double alpha0_in, double beta0_in,
    double t0_in,
    int N_times_in, int N_freqs_in)
{
    N_quad        = N_quad_in;
    N_sky         = N_sky_in;
    alpha0        = alpha0_in;
    beta0         = beta0_in;
    t0            = t0_in;
    N_times_alloc = N_times_in;
    N_freqs_alloc = N_freqs_in;
    initialized   = false;

#ifdef __CUDACC__
    gpuErrchk(cudaMalloc(&R_vals_quad,  N_quad * N_sky * sizeof(double)));
    gpuErrchk(cudaMalloc(&z_vals_quad,  N_quad * N_sky * sizeof(double)));
    gpuErrchk(cudaMalloc(&quad_weights, N_quad          * sizeof(double)));
    gpuErrchk(cudaMalloc(&cos_beta_ecl, N_sky           * sizeof(double)));
    gpuErrchk(cudaMalloc(&lam_ecl,      N_sky           * sizeof(double)));
    gpuErrchk(cudaMalloc(&beta_ecl,     N_sky           * sizeof(double)));
    gpuErrchk(cudaMalloc(&weights,      N_sky           * sizeof(double)));
    gpuErrchk(cudaMalloc(&R_avg,        N_times_in * 6  * sizeof(double)));
    gpuErrchk(cudaMalloc(&R_gal_arr,    N_times_in * N_freqs_in * 6 * sizeof(double)));

    gpuErrchk(cudaMemcpy(R_vals_quad,  h_R_vals_quad,  N_quad * N_sky * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(z_vals_quad,  h_z_vals_quad,  N_quad * N_sky * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(quad_weights, h_quad_weights, N_quad          * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cos_beta_ecl, h_cos_beta_ecl, N_sky           * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(lam_ecl,      h_lam_ecl,      N_sky           * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(beta_ecl,     h_beta_ecl,     N_sky           * sizeof(double), cudaMemcpyHostToDevice));
#else
    R_vals_quad  = const_cast<double*>(h_R_vals_quad);
    z_vals_quad  = const_cast<double*>(h_z_vals_quad);
    quad_weights = const_cast<double*>(h_quad_weights);
    cos_beta_ecl = const_cast<double*>(h_cos_beta_ecl);
    lam_ecl      = const_cast<double*>(h_lam_ecl);
    beta_ecl     = const_cast<double*>(h_beta_ecl);
    weights      = new double[N_sky];
    R_avg        = new double[N_times_in * 6];
    R_gal_arr    = new double[N_times_in * N_freqs_in * 6];
#endif

    initialized = true;
}

void GalacticGrid::free_gpu()
{
    if (!initialized) return;
#ifdef __CUDACC__
    gpuErrchk(cudaFree(R_vals_quad));
    gpuErrchk(cudaFree(z_vals_quad));
    gpuErrchk(cudaFree(quad_weights));
    gpuErrchk(cudaFree(cos_beta_ecl));
    gpuErrchk(cudaFree(lam_ecl));
    gpuErrchk(cudaFree(beta_ecl));
    gpuErrchk(cudaFree(weights));
    gpuErrchk(cudaFree(R_avg));
    gpuErrchk(cudaFree(R_gal_arr));
#else
    delete[] weights;
    delete[] R_avg;
    delete[] R_gal_arr;
#endif
    initialized = false;
}

void GalacticGrid::compute_sky_weights(double R_d, double z_d)
{
#ifdef __CUDACC__
    int n_blocks = (N_sky + GAL_THREADS_SKY - 1) / GAL_THREADS_SKY;
    compute_column_densities_kernel<<<n_blocks, GAL_THREADS_SKY>>>(
        weights, R_vals_quad, z_vals_quad, quad_weights, cos_beta_ecl,
        R_d, z_d, N_quad, N_sky);
    gpuErrchk(cudaGetLastError());

    double *h_weights = new double[N_sky];
    gpuErrchk(cudaMemcpy(h_weights, weights, N_sky * sizeof(double), cudaMemcpyDeviceToHost));
    double total = 0.0;
    for (int i = 0; i < N_sky; i++) total += h_weights[i];
    delete[] h_weights;

    double inv_total = (total > 0.0) ? 1.0 / total : 0.0;
    normalize_weights_kernel<<<n_blocks, GAL_THREADS_SKY>>>(weights, inv_total, N_sky);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    compute_column_densities_kernel(
        weights, R_vals_quad, z_vals_quad, quad_weights, cos_beta_ecl,
        R_d, z_d, N_quad, N_sky);
    double total = 0.0;
    for (int i = 0; i < N_sky; i++) total += weights[i];
    double inv_total = (total > 0.0) ? 1.0 / total : 0.0;
    normalize_weights_kernel(weights, inv_total, N_sky);
#endif
}

void GalacticGrid::compute_sky_average(const double *d_times, int N_times)
{
#ifdef __CUDACC__
    gpuErrchk(cudaMemset(R_avg, 0, N_times * 6 * sizeof(double)));
    int n_blocks_sky = (N_sky + GAL_THREADS_SKY - 1) / GAL_THREADS_SKY;
    dim3 grid(n_blocks_sky, N_times);
    sky_average_response_kernel<<<grid, GAL_THREADS_SKY>>>(
        R_avg, weights, lam_ecl, beta_ecl, d_times,
        alpha0, beta0, t0, N_times, N_sky);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    for (int i = 0; i < N_times * 6; i++) R_avg[i] = 0.0;
    sky_average_response_kernel(
        R_avg, weights, lam_ecl, beta_ecl, d_times,
        alpha0, beta0, t0, N_times, N_sky);
#endif
}

void GalacticGrid::initialize(double R_d, double z_d,
                               const double *d_times, int N_times)
{
    compute_sky_weights(R_d, z_d);
    compute_sky_average(d_times, N_times);
}

void GalacticGrid::compute_gal_covariance(
    const double *d_freqs, int N_freqs, int N_times,
    double Amp, double alpha,
    double f_1, double f_knee, double f_2,
    double avg_d)
{
#ifdef __CUDACC__
    dim3 block(16, 8);
    dim3 grid(
        (N_freqs + block.x - 1) / block.x,
        (N_times + block.y - 1) / block.y
    );
    apply_spectral_model_kernel<<<grid, block>>>(
        R_gal_arr, R_avg, d_freqs,
        Amp, alpha, f_1, f_knee, f_2,
        avg_d, N_times, N_freqs);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    apply_spectral_model_kernel(
        R_gal_arr, R_avg, d_freqs,
        Amp, alpha, f_1, f_knee, f_2,
        avg_d, N_times, N_freqs);
#endif
}
