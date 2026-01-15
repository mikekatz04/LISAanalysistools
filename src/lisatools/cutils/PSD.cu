#include "stdio.h"
#include "gbt_global.h"
#include "PSD.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

// ============================================================================
// Type Aliases
// ============================================================================
using cmplx = gcmplx::complex<double>;

#if defined(__CUDACC__) || defined(__CUDA_COMPILATION__)
#define XYZSensitivityMatrix XYZSensitivityMatrixGPU
#define NoiseLevels NoiseLevelsGPU

#else
#define XYZSensitivityMatrix XYZSensitivityMatrixCPU
#define NoiseLevels NoiseLevelsCPU

#endif

#define NUM_THREADS 64
#define NUM_THREADS_LIKE 256

// ============================================================================
// Helper Math Functions (static device functions)
// ============================================================================


/**
 * @brief Angular frequency from frequency in Hz.
 * @param f Frequency in Hz.
 * @return Angular frequency in rad/s.
 */
static CUDA_DEVICE
double omega(double f)
{   
    return 2.0 * M_PI * f;
}

/**
 * @brief Phase shift: delay times angular frequency.
 * @param d Delay in seconds.
 * @param f Frequency in Hz.
 * @return Phase in radians.
 */
static CUDA_DEVICE
double d_times_omega(double d, double f)
{   
    return d * omega(f);
}

/**
 * @brief Sine of phase shift for given delay and frequency.
 */
static CUDA_DEVICE
double s_wl(double d, double f)
{
    return sin(d_times_omega(d, f));
}

/**
 * @brief Cosine of phase shift for given delay and frequency.
 */
static CUDA_DEVICE
double c_wl(double d, double f)
{
    return cos(d_times_omega(d, f));
}

/**
 * @brief Cosine of double phase shift for given delay and frequency.
 */
static CUDA_DEVICE
double c_2wl(double d, double f)
{
    return cos(2.0 * d_times_omega(d, f));
}

/**
 * @brief Convert link ID (12, 23, 31, 13, 32, 21) to array index (0-5).
 * @param link Link identifier.
 * @return Array index, or -1 if invalid.
 */
static CUDA_DEVICE
int link_to_index(int link)
{
    switch (link) {
        case 12: return 0;
        case 23: return 1;
        case 31: return 2;
        case 13: return 3;
        case 32: return 4;
        case 21: return 5;
        default:
#ifndef __CUDACC__
            throw std::invalid_argument("Bad link ind. Must be 12, 23, 31, 13, 32, 21.");
#endif
            return -1;
    }
}

// ============================================================================
// LISA Noise Models
// ============================================================================

/**
 * @brief Compute LISA instrumental noise PSDs.
 * 
 * Computes the test mass (acceleration) noise and optical metrology system noise
 * power spectral densities at a given frequency.
 * 
 * @param[out] S_tm Test mass noise PSD.
 * @param[out] S_isi_oms OMS noise PSD.
 * @param[in] f Frequency in Hz.
 * @param[in] Soms_d_in OMS displacement noise amplitude (m/sqrt(Hz)).
 * @param[in] Sa_a_in Acceleration noise amplitude (m/s^2/sqrt(Hz)).
 * @param[in] return_relative_frequency If true, return in relative frequency units.
 */
CUDA_DEVICE
void NoiseLevels::get_testmass_noise(double *S_tm, double f, double Sa_a_in)
{
    double Sa_a = Sa_a_in * Sa_a_in * (1.0 + pow(f_knee_tm / f, 2)) * (1.0 + pow(f / f_break_tm, 4));
    double Sa_d = Sa_a * pow(2.0 * M_PI * f, -4.0);  // In displacement
    double Sa_nu = Sa_d * pow(2.0 * M_PI * f / Clight, 2);  // In relative frequency   
    if (return_relative_frequency) {
        *S_tm = Sa_nu;
    } else {
        *S_tm = Sa_d;
    }
}

CUDA_DEVICE
void NoiseLevels::get_isi_oms_noise(double *S_oms, double f, double Soms_d_in)
{
    double Soms_d = Soms_d_in * Soms_d_in * (1.0 + pow(f_knee_oms / f, 4));
    double Soms_nu = Soms_d * pow(2.0 * M_PI * f / Clight, 2);
    if (return_relative_frequency) {
        *S_oms = Soms_nu;
    } else {
        *S_oms = Soms_d;
    }
}
/**
 * @brief Galactic foreground confusion noise model.
 * 
 * @param f Frequency in Hz.
 * @param Amp Amplitude parameter.
 * @param alpha Spectral slope parameter.
 * @param slope_1 Exponential cutoff slope.
 * @param f_knee Knee frequency.
 * @param slope_2 Hyperbolic tangent slope.
 * @return Galactic foreground PSD.
 */
CUDA_DEVICE
void NoiseLevels::get_galactic_foreground(double *S_gal, double f, double Amp, double alpha, double slope_1, double f_knee, double slope_2)
{
    *S_gal = Amp * exp(-pow(f, alpha) * slope_1) * pow(f, -7.0/3.0) 
           * 0.5 * (1.0 + tanh(-(f - f_knee) * slope_2));
}

// ============================================================================
// 3x3 Hermitian Matrix Operations
// ============================================================================

/**
 * @brief Invert a 3x3 Hermitian matrix and compute its log determinant.
 * 
 * Uses cofactor expansion. Only the upper triangle is stored/computed
 * since the matrix is Hermitian (lower triangle = conj of upper).
 * 
 * @param[in] c00, c01, c02, c11, c12, c22 Input matrix upper triangle.
 * @param[out] i00, i01, i02, i11, i12, i22 Inverse matrix upper triangle.
 * @param[out] log_det Natural log of determinant.
 */
static CUDA_DEVICE
void invert_3x3_hermitian(
    double c00, cmplx c01, cmplx c02,
    double c11, cmplx c12,
    double c22,
    double &i00, cmplx &i01, cmplx &i02,
    double &i11, cmplx &i12,
    double &i22,
    double &log_det)
{
    // Cofactors for diagonal elements (real)
    double C00 = c11 * c22 - gcmplx::norm(c12);
    double C11 = c00 * c22 - gcmplx::norm(c02);
    double C22 = c00 * c11 - gcmplx::norm(c01);

    // Cofactors for off-diagonal elements (complex)
    cmplx C01 = c12 * gcmplx::conj(c02) - gcmplx::conj(c01) * c22;
    cmplx C02 = gcmplx::conj(c01) * gcmplx::conj(c12) - c11 * gcmplx::conj(c02);
    cmplx C12 = c01 * gcmplx::conj(c02) - c00 * gcmplx::conj(c12);

    // Determinant (real for Hermitian matrix)
    double det = c00 * C00 + (c01 * C01).real() + (c02 * C02).real();
    double inv_det = 1.0 / det;
    log_det = log(det);

    // Inverse = adjugate / det (adjugate = conjugate transpose of cofactor matrix)
    i00 = C00 * inv_det;
    i11 = C11 * inv_det;
    i22 = C22 * inv_det;
    i01 = gcmplx::conj(C01) * inv_det;
    i02 = gcmplx::conj(C02) * inv_det;
    i12 = gcmplx::conj(C12) * inv_det;
} 


/**
 * @brief Compute quadratic form d^H @ C^{-1} @ d for Hermitian 3x3 matrix.
 * 
 * C^{-1} is stored as upper triangle (Hermitian: lower = conj of upper).
 * Structure: [[i00, i01, i02], [conj(i01), i11, i12], [conj(i02), conj(i12), i22]]
 * 
 * @param d_X, d_Y, d_Z Complex data vector d.
 * @param i00, i01, i02, i11, i12, i22 Upper triangle of inverse covariance matrix.
 * @return Real-valued quadratic form (imaginary part cancels for Hermitian).
 */
CUDA_DEVICE
double quadratic_form(
    cmplx d_X, cmplx d_Y, cmplx d_Z,
    double i00, cmplx i01, cmplx i02,
    double i11, cmplx i12,
    double i22)
{
    // Compute C^{-1} @ d (Hermitian matrix times complex vector)
    // Row 0: i00*d_X + i01*d_Y + i02*d_Z
    // Row 1: conj(i01)*d_X + i11*d_Y + i12*d_Z
    // Row 2: conj(i02)*d_X + conj(i12)*d_Y + i22*d_Z
    cmplx inv_d_X = i00 * d_X + i01 * d_Y + i02 * d_Z;
    cmplx inv_d_Y = gcmplx::conj(i01) * d_X + i11 * d_Y + i12 * d_Z;
    cmplx inv_d_Z = gcmplx::conj(i02) * d_X + gcmplx::conj(i12) * d_Y + i22 * d_Z;
    
    // Compute d^H @ (C^{-1} @ d) = conj(d) . (C^{-1} @ d)
    // Result is real for Hermitian form
    double quad = (gcmplx::conj(d_X) * inv_d_X + 
                   gcmplx::conj(d_Y) * inv_d_Y + 
                   gcmplx::conj(d_Z) * inv_d_Z).real();
    
    return quad;
}


/**
 * @brief CUDA kernel for PSD likelihood over XYZ TDI channels.
 * 
 * Computes log-likelihood by summing over all time segments and frequencies.
 * 
 * Data layout (time-major):
 *   data_in[data_index][channel][time_idx * num_freqs + freq_idx]
 *   Flattened: data_in[(data_index * 3 + channel) * (num_times * num_freqs) + time_idx * num_freqs + freq_idx]
 * 
 * Loop order (time-major, same as data for coalesced access):
 *   idx = t_idx * num_freqs + f_idx
 *   Adjacent threads process adjacent (t,f) pairs, ensuring coalesced data access.
 * 
 * @param like_contrib Output partial likelihood contributions per block/PSD.
 * @param f_arr Frequency array of length num_freqs.
 * @param data_in Flattened complex TDI data: (num_datasets, 3, num_times * num_freqs).
 * @param data_index_all Maps each PSD to a dataset index, length num_psds.
 * @param time_index_all Array of time indices for orbit config lookup, length num_times.
 * @param num_freqs Number of frequency bins.
 * @param num_times Number of time segments.
 * @param df Frequency resolution.
 * @param num_psds Number of PSD configurations (batch size).
 */
CUDA_KERNEL void psd_likelihood_xyz_kernel(
    double *like_contrib, double *f_arr, cmplx *data_in,
    int *data_index_all, int *time_index_all,
    double *Soms_d_in_all, double *Sa_a_in_all,
    double *Amp_all, double *alpha_all, double *slope_1_all, double *f_knee_all, double *slope_2_all,
    double df, int num_freqs, int num_times, int num_psds, 
    XYZSensitivityMatrix &sensitivity_matrix)
{
    int tid;
#ifdef __CUDACC__ 
    tid = threadIdx.x;
#else
    tid = 0;
#endif

    // Per-thread variables
    double Soms_d_in, Sa_a_in, Amp, alpha, slope_1, f_knee, slope_2;
    double f;
    int data_index, time_index;
    cmplx d_X, d_Y, d_Z;
    
    // Covariance matrix elements (Upper triangle of 3x3 Hermitian)
    double c00, c11, c22;
    cmplx c01, c02, c12;
    // Inverse elements
    double i00, i11, i22, log_det;
    cmplx i01, i02, i12;

    int start_psd, incr_psd;
    int start_idx, incr_idx;
    
    // Total number of (time, frequency) pairs to sum over
    int total_tf_pairs = num_times * num_freqs;

#ifdef __CUDACC__
    start_psd = blockIdx.y;
    incr_psd = gridDim.y;
#else
    start_psd = 0;
    incr_psd = 1;

    // Allocate "shared" memory for CPU (just one thread)
    double like_vals[1];
#endif

#ifdef __CUDACC__
    CUDA_SHARED double like_vals[NUM_THREADS_LIKE];
#endif

    // Loop over PSDs
    for (int psd_i = start_psd; psd_i < num_psds; psd_i += incr_psd)
    {
        data_index = data_index_all[psd_i];

        // Noise parameters for this PSD
        Soms_d_in = Soms_d_in_all[psd_i];
        Sa_a_in = Sa_a_in_all[psd_i];
        Amp = Amp_all[psd_i];
        alpha = alpha_all[psd_i];
        slope_1 = slope_1_all[psd_i];
        f_knee = f_knee_all[psd_i];
        slope_2 = slope_2_all[psd_i];

        // Initialize reduction
        if (tid < NUM_THREADS_LIKE)
            like_vals[tid] = 0.0;
        
#ifdef __CUDACC__
        CUDA_SYNC_THREADS;
        start_idx = blockIdx.x * blockDim.x + threadIdx.x;
        incr_idx = blockDim.x * gridDim.x;
#else
        start_idx = 0;
        incr_idx = 1;
#endif

        // Loop over flattened (time, frequency) index
        // Layout: idx = t_idx * num_freqs + f_idx (time-major, matches data layout)
        // Adjacent threads access adjacent memory locations → coalesced
        for (int idx = start_idx; idx < total_tf_pairs; idx += incr_idx)
        {
            int t_idx = idx / num_freqs;
            int f_idx = idx % num_freqs;
            
            time_index = time_index_all[t_idx];
            f = f_arr[f_idx];

            // Get noise covariance matrix for this (time, frequency) pair
            sensitivity_matrix.get_noise_covariance(
                f, time_index,
                Soms_d_in, Sa_a_in,
                Amp, alpha, slope_1, f_knee, slope_2,
                &c00, &c01, &c02, &c11, &c12, &c22
            );

            // Invert C -> C^-1
            invert_3x3_hermitian(c00, c01, c02, c11, c12, c22, 
                                 i00, i01, i02, i11, i12, i22, log_det);
            
            // Get Data: layout matches loop (time-major)
            // data_in[(data_index * 3 + channel) * total_tf_pairs + idx]
            int base_idx = data_index * 3 * total_tf_pairs + idx;
            d_X = data_in[base_idx];
            d_Y = data_in[base_idx + total_tf_pairs];
            d_Z = data_in[base_idx + 2 * total_tf_pairs];

            // Compute Quadratic Form: d^H * C^-1 * d
            double Q = quadratic_form(d_X, d_Y, d_Z, i00, i01, i02, i11, i12, i22);
            
            // Likelihood Accumulation
            like_vals[tid] += -0.5 * (4.0 * df * Q + log_det);
        }
#ifdef __CUDACC__
        CUDA_SYNC_THREADS;

        // Block Reduction (tree-based for efficiency)
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                like_vals[tid] += like_vals[tid + s];
            }
            CUDA_SYNC_THREADS;
        }
#endif

        // Store result for this block/PSD
        if (tid == 0)
        {
#ifdef __CUDACC__
            like_contrib[psd_i * gridDim.x + blockIdx.x] = like_vals[0];
#else
            like_contrib[psd_i] = like_vals[0];
#endif
        }
#ifdef __CUDACC__
        CUDA_SYNC_THREADS;
#endif
    }
}

CUDA_KERNEL void like_sum_from_contrib(double *like_contrib_final, double *like_contrib, int num_blocks_per_psd, int num_psds)
{
    int tid;
    int psd_i;
    int incr_psd;
    
#ifdef __CUDACC__
    tid = threadIdx.x;
    psd_i = blockIdx.y; 
    incr_psd = gridDim.y;
    CUDA_SHARED double shared_sum[NUM_THREADS_LIKE];
#else
    tid = 0;
    psd_i = 0;
    incr_psd = 1;
    double shared_sum[1];
#endif

    for (int p = psd_i; p < num_psds; p += incr_psd)
    {
        double sum = 0.0;
        
#ifdef __CUDACC__
        for (int i = tid; i < num_blocks_per_psd; i += blockDim.x)
        {
             sum += like_contrib[p * num_blocks_per_psd + i];
        }
#else
        for (int i = 0; i < num_blocks_per_psd; i++)
        {
             sum += like_contrib[p * num_blocks_per_psd + i];
        }
#endif
        
        shared_sum[tid] = sum;
#ifdef __CUDACC__
        CUDA_SYNC_THREADS;
        
        for (unsigned int s = 1; s < blockDim.x; s *= 2) {
             if (tid % (2 * s) == 0) {
                 shared_sum[tid] += shared_sum[tid + s];
             }
             CUDA_SYNC_THREADS;
        }
#endif
        
        if (tid == 0) {
            like_contrib_final[p] = shared_sum[0];
        }
    }
}

void XYZSensitivityMatrix::psd_likelihood_wrap(
    double *like_contrib_final, double *f_arr, cmplx *data, 
    int *data_index_all, int *time_index_all,
    double *Soms_d_in_all, double *Sa_a_in_all, 
    double *Amp_all, double *alpha_all, double *slope_1_all, double *f_knee_all, double *slope_2_all, 
    double df, int num_freqs, int num_times, int num_psds)
{
    int total_tf_pairs = num_times * num_freqs;
    
#ifdef __CUDACC__
    double *like_contrib;
    int num_blocks = std::ceil((double)total_tf_pairs / NUM_THREADS_LIKE); // Blocks for (time, freq) coverage

    gpuErrchk(cudaMalloc(&like_contrib, num_psds * num_blocks * sizeof(double)));
    
    // Grid: X=blocks for (time,freq) pairs, Y=blocks for PSDs
    dim3 grid(num_blocks, std::min(num_psds, 65535), 1);
    
    XYZSensitivityMatrix *dev_ptr;
    gpuErrchk(cudaMalloc(&dev_ptr, sizeof(XYZSensitivityMatrix)));
    gpuErrchk(cudaMemcpy(dev_ptr, this, sizeof(XYZSensitivityMatrix), cudaMemcpyHostToDevice));

    psd_likelihood_xyz_kernel<<<grid, NUM_THREADS_LIKE>>>(
        like_contrib, f_arr, data, data_index_all, time_index_all,
        Soms_d_in_all, Sa_a_in_all,
        Amp_all, alpha_all, slope_1_all, f_knee_all, slope_2_all,
        df, num_freqs, num_times, num_psds, *dev_ptr);
        
    gpuErrchk(cudaGetLastError());
    
    // Reduction across blocks
    dim3 grid_reduc(1, std::min(num_psds, 65535), 1);
    like_sum_from_contrib<<<grid_reduc, NUM_THREADS_LIKE>>>(like_contrib_final, like_contrib, num_blocks, num_psds);
    
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(like_contrib));
    gpuErrchk(cudaFree(dev_ptr));
#else
    // CPU Fallback
    psd_likelihood_xyz_kernel(
        like_contrib_final, f_arr, data, data_index_all, time_index_all,
        Soms_d_in_all, Sa_a_in_all,
        Amp_all, alpha_all, slope_1_all, f_knee_all, slope_2_all,
        df, num_freqs, num_times, num_psds, *this);
#endif
}

CUDA_DEVICE
int XYZSensitivityMatrix::get_adjacent_mosa(int mosa)
{
    if (mosa == 12)
        return 13;
    else if (mosa == 23)
        return 21;
    else if (mosa == 31)
        return 32;
    return -1;
}

CUDA_DEVICE
cmplx XYZSensitivityMatrix::oms_xx_unequal_armlength(double f, double avg_d_ij, double avg_d_ik)
{
    cmplx _oms = 8.0 * (pow(s_wl(avg_d_ij, f), 2) + pow(s_wl(avg_d_ik, f), 2));

    if (generation == 2)
    {
        _oms = _oms * 4.0 * pow(s_wl(avg_d_ij + avg_d_ik, f), 2);
    }
    return _oms;
}

CUDA_DEVICE
cmplx XYZSensitivityMatrix::oms_xy_unequal_armlength(double f, double avg_d_ij, double avg_d_ik, double avg_d_jk, double delta_d_ij)
{
    cmplx _oms = -8.0 * (
        c_wl(avg_d_ij, f)
        * s_wl(avg_d_ik, f)
        * s_wl(avg_d_jk, f)
        * gcmplx::exp(cmplx(0.0, -1.0) * d_times_omega((avg_d_ik - avg_d_jk + 0.5 * delta_d_ij), f))
    );

    if (generation == 2)
    {
        _oms = _oms * (
            4.0
            * s_wl(avg_d_ij + avg_d_ik, f)
            * s_wl(avg_d_ij + avg_d_jk, f)
            * gcmplx::exp(cmplx(0.0, -1.0) * d_times_omega((avg_d_ik - avg_d_jk), f))
        );
    }
    return _oms;
}

CUDA_DEVICE
cmplx XYZSensitivityMatrix::tm_xx_unequal_armlength(double f, double avg_d_ij, double avg_d_ik)
{
    cmplx _tm = 8.0 * (
        pow(s_wl(avg_d_ij, f), 2) * (3 + c_2wl(avg_d_ik, f))
        + pow(s_wl(avg_d_ik, f), 2) * (3 + c_2wl(avg_d_ij, f))
    );
    if (generation == 2)
    {
        _tm *= 4.0 * pow(s_wl(avg_d_ij + avg_d_ik, f), 2);
    }
    return _tm;
}

CUDA_DEVICE
cmplx XYZSensitivityMatrix::tm_xy_unequal_armlength(double f, double avg_d_ij, double avg_d_ik, double avg_d_jk, double delta_d_ij)
{
    cmplx _tm = -32.0 * (
        c_wl(avg_d_ij, f)
        * s_wl(avg_d_ik, f)
        * s_wl(avg_d_jk, f)
        * gcmplx::exp(cmplx(0.0, -1.0) * d_times_omega((avg_d_ik - avg_d_jk + 0.5 * delta_d_ij), f))
    );  
    if (generation == 2)
    {
        _tm *= (
            4.0
            * s_wl(avg_d_ij + avg_d_ik, f)
            * s_wl(avg_d_ij + avg_d_jk, f)
            * gcmplx::exp(cmplx(0.0, -1.0) * d_times_omega((avg_d_ik - avg_d_jk), f))
        );
    }
    return _tm;
}

CUDA_DEVICE
void XYZSensitivityMatrix::get_noise_tfs(double f, cmplx *oms_xx, cmplx *oms_xy, cmplx *oms_xz, cmplx *oms_yy, cmplx *oms_yz, cmplx *oms_zz,
                                  cmplx *tm_xx, cmplx *tm_xy, cmplx *tm_xz, cmplx *tm_yy, cmplx *tm_yz, cmplx *tm_zz,
                                  int time_index)
{
    int index1, index2, index3;
    // Retrieve average and delta light travel times for the given time index
    double avg_d[6];
    double delta_d[6];
    for (int i = 0; i < 6; i++)
    {
        avg_d[i] = averaged_ltts_arr[time_index * n_links + i];
        delta_d[i] = delta_ltts_arr[time_index * n_links + i];
    }

    // Compute OMS noise transfer functions
    index1 = link_to_index(12);
    index2 = link_to_index(13);
    index3 =  link_to_index(23);
    *oms_xx = oms_xx_unequal_armlength(f, avg_d[index1], avg_d[index2]);
    *tm_xx = tm_xx_unequal_armlength(f, avg_d[index1], avg_d[index2]);
    *oms_xy = oms_xy_unequal_armlength(f, avg_d[index1], avg_d[index2], avg_d[index3], delta_d[index1]);
    *tm_xy = tm_xy_unequal_armlength(f, avg_d[index1], avg_d[index2], avg_d[index3], delta_d[index1]);

    index1 = link_to_index(23);
    index2 = link_to_index(21);
    index3 = link_to_index(13);
    *oms_yy = oms_xx_unequal_armlength(f, avg_d[index1], avg_d[index2]);
    *tm_yy = tm_xx_unequal_armlength(f, avg_d[index1], avg_d[index2]);
    *oms_yz = oms_xy_unequal_armlength(f, avg_d[index1], avg_d[index2], avg_d[index3], delta_d[index1]);
    *tm_yz = tm_xy_unequal_armlength(f, avg_d[index1], avg_d[index2], avg_d[index3], delta_d[index1]);
    
    index1 = link_to_index(31);
    index2 = link_to_index(32);
    index3 = link_to_index(12);
    *oms_zz = oms_xx_unequal_armlength(f, avg_d[index1], avg_d[index2]);
    *tm_zz = tm_xx_unequal_armlength(f, avg_d[index1], avg_d[index2]);
    *oms_xz = oms_xy_unequal_armlength(f, avg_d[index1], avg_d[index3], avg_d[index2], delta_d[index1]);    
    *tm_xz = tm_xy_unequal_armlength(f, avg_d[index1], avg_d[index3], avg_d[index2], delta_d[index1]);
}

// now, add a cuda kernel to compute all noise tfs at once for an array of frequencies and time indices
CUDA_KERNEL
void get_noise_tfs_kernel(double *frequencies, int *time_indices,
                          cmplx *oms_xx_arr, cmplx *oms_xy_arr, cmplx *oms_xz_arr,
                          cmplx *oms_yy_arr, cmplx *oms_yz_arr, cmplx *oms_zz_arr,
                          cmplx *tm_xx_arr, cmplx *tm_xy_arr, cmplx *tm_xz_arr,
                          cmplx *tm_yy_arr, cmplx *tm_yz_arr, cmplx *tm_zz_arr,
                          int num, XYZSensitivityMatrix &sensitivity_matrix)
{
    int start, end, increment;
    int start_time, end_time, increment_time;
#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num;
    increment = gridDim.x * blockDim.x;

    start_time = blockIdx.y * blockDim.y + threadIdx.y;
    end_time = sensitivity_matrix.n_times;
    increment_time = gridDim.y * blockDim.y;   
#else  // __CUDACC__
    start = 0;
    end = num;
    increment = 1;
    start_time = 0;
    end_time = sensitivity_matrix.n_times;
    increment_time = 1;
#endif // __CUDACC__

    for (int i = start; i < end; i += increment)
    {
        double f = frequencies[i];
        for (int j = start_time; j < end_time; j += increment_time)
        {
            int time_index = time_indices[j];

            sensitivity_matrix.get_noise_tfs(f,
                                             &oms_xx_arr[j * num + i], &oms_xy_arr[j * num + i], &oms_xz_arr[j * num + i],
                                             &oms_yy_arr[j * num + i], &oms_yz_arr[j * num + i], &oms_zz_arr[j * num + i],
                                             &tm_xx_arr[j * num + i], &tm_xy_arr[j * num + i], &tm_xz_arr[j * num + i],
                                             &tm_yy_arr[j * num + i], &tm_yz_arr[j * num + i], &tm_zz_arr[j * num + i],
                                             time_index);
        }
    }
}

void XYZSensitivityMatrix::get_noise_tfs_arr(double *freqs,
                          cmplx *oms_xx_arr, cmplx *oms_xy_arr, cmplx *oms_xz_arr,
                          cmplx *oms_yy_arr, cmplx *oms_yz_arr, cmplx *oms_zz_arr,
                          cmplx *tm_xx_arr, cmplx *tm_xy_arr, cmplx *tm_xz_arr,
                          cmplx *tm_yy_arr, cmplx *tm_yz_arr, cmplx *tm_zz_arr,
                          int num,
                          int *time_indices)
{
#ifdef __CUDACC__
    int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);
    // copy self to GPU
    XYZSensitivityMatrix *sensitivity_matrix_gpu;
    gpuErrchk(cudaMalloc(&sensitivity_matrix_gpu, sizeof(XYZSensitivityMatrix)));
    gpuErrchk(cudaMemcpy(sensitivity_matrix_gpu, this, sizeof(XYZSensitivityMatrix), cudaMemcpyHostToDevice));
    get_noise_tfs_kernel<<<num_blocks, NUM_THREADS>>>(freqs, time_indices,
                                                      oms_xx_arr, oms_xy_arr, oms_xz_arr,
                                                      oms_yy_arr, oms_yz_arr, oms_zz_arr,
                                                      tm_xx_arr, tm_xy_arr, tm_xz_arr,
                                                      tm_yy_arr, tm_yz_arr, tm_zz_arr,
                                                      num, *sensitivity_matrix_gpu);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(sensitivity_matrix_gpu));
#else // __CUDACC__
    get_noise_tfs_kernel(freqs, time_indices,
                        oms_xx_arr, oms_xy_arr, oms_xz_arr,
                        oms_yy_arr, oms_yz_arr, oms_zz_arr, 
                        tm_xx_arr, tm_xy_arr, tm_xz_arr,
                        tm_yy_arr, tm_yz_arr, tm_zz_arr,
                        num, *this);
#endif // __CUDACC__
}

// ============================================================================
// Noise Covariance Matrix Computation
// ============================================================================

CUDA_DEVICE
void XYZSensitivityMatrix::get_noise_covariance(
    double f, int time_index,
    double Soms_d_in, double Sa_a_in,
    double Amp, double alpha, double slope_1, double f_knee, double slope_2,
    double *c00, cmplx *c01, cmplx *c02,
    double *c11, cmplx *c12, double *c22)
{
    // Get noise transfer functions
    cmplx oms_xx, oms_xy, oms_xz, oms_yy, oms_yz, oms_zz;
    cmplx tm_xx, tm_xy, tm_xz, tm_yy, tm_yz, tm_zz;
    
    get_noise_tfs(f, 
                  &oms_xx, &oms_xy, &oms_xz, &oms_yy, &oms_yz, &oms_zz,
                  &tm_xx, &tm_xy, &tm_xz, &tm_yy, &tm_yz, &tm_zz, 
                  time_index);

    // Calculate Noise PSDs
    double S_tm, S_isi_oms, S_gal;
    noise_levels.get_testmass_noise(&S_tm, f, Sa_a_in);
    noise_levels.get_isi_oms_noise(&S_isi_oms, f, Soms_d_in);
    noise_levels.get_galactic_foreground(&S_gal, f, Amp, alpha, slope_1, f_knee, slope_2);

    // Build Covariance Matrix C (3x3 Hermitian, upper triangle)
    // Diagonal elements are real
    *c00 = (gcmplx::real(oms_xx) * S_isi_oms) + (gcmplx::real(tm_xx) * S_tm);
    *c11 = (gcmplx::real(oms_yy) * S_isi_oms) + (gcmplx::real(tm_yy) * S_tm);
    *c22 = (gcmplx::real(oms_zz) * S_isi_oms) + (gcmplx::real(tm_zz) * S_tm);
    
    // Off-diagonal elements are complex
    *c01 = oms_xy * S_isi_oms + tm_xy * S_tm;
    *c02 = oms_xz * S_isi_oms + tm_xz * S_tm;
    *c12 = oms_yz * S_isi_oms + tm_yz * S_tm;
}

CUDA_KERNEL
void get_noise_covariance_kernel(
    double *frequencies, int *time_indices,
    double Soms_d_in, double Sa_a_in,
    double Amp, double alpha, double slope_1, double f_knee, double slope_2,
    double *c00_arr, cmplx *c01_arr, cmplx *c02_arr,
    double *c11_arr, cmplx *c12_arr, double *c22_arr,
    int num_freqs, int num_times,
    XYZSensitivityMatrix &sensitivity_matrix)
{
    // Memory layout: output[t_idx * num_freqs + f_idx]
    // Frequencies are the fast-varying dimension for coalesced access
    
    int start_freq, end_freq, increment_freq;
    int start_time, end_time, increment_time;

#ifdef __CUDACC__
    // X dimension for frequencies (fast), Y dimension for times (slow)
    start_freq = blockIdx.x * blockDim.x + threadIdx.x;
    end_freq = num_freqs;
    increment_freq = gridDim.x * blockDim.x;

    start_time = blockIdx.y * blockDim.y + threadIdx.y;
    end_time = num_times;
    increment_time = gridDim.y * blockDim.y;
#else
    start_freq = 0;
    end_freq = num_freqs;
    increment_freq = 1;
    start_time = 0;
    end_time = num_times;
    increment_time = 1;
#endif

    for (int t_idx = start_time; t_idx < end_time; t_idx += increment_time)
    {
        int time_index = time_indices[t_idx];
        int base_out_idx = t_idx * num_freqs;

        for (int f_idx = start_freq; f_idx < end_freq; f_idx += increment_freq)
        {
            double f = frequencies[f_idx];
            int out_idx = base_out_idx + f_idx;

            sensitivity_matrix.get_noise_covariance(
                f, time_index,
                Soms_d_in, Sa_a_in,
                Amp, alpha, slope_1, f_knee, slope_2,
                &c00_arr[out_idx], &c01_arr[out_idx], &c02_arr[out_idx],
                &c11_arr[out_idx], &c12_arr[out_idx], &c22_arr[out_idx]
            );
        }
    }
}

void XYZSensitivityMatrix::get_noise_covariance_arr(
    double *freqs, int *time_indices,
    double Soms_d_in, double Sa_a_in,
    double Amp, double alpha, double slope_1, double f_knee, double slope_2,
    double *c00_arr, cmplx *c01_arr, cmplx *c02_arr,
    double *c11_arr, cmplx *c12_arr, double *c22_arr,
    int num_freqs, int num_times)
{
#ifdef __CUDACC__
    // 2D grid: X for frequencies (coalesced), Y for time indices
    // Use more threads in X for better coalescing
    dim3 block(32, 8);  // 32 threads in freq dimension for warp coalescing
    dim3 grid(
        (num_freqs + block.x - 1) / block.x,
        (num_times + block.y - 1) / block.y
    );
    
    // Copy self to GPU
    XYZSensitivityMatrix *sensitivity_matrix_gpu;
    gpuErrchk(cudaMalloc(&sensitivity_matrix_gpu, sizeof(XYZSensitivityMatrix)));
    gpuErrchk(cudaMemcpy(sensitivity_matrix_gpu, this, sizeof(XYZSensitivityMatrix), cudaMemcpyHostToDevice));
    
    get_noise_covariance_kernel<<<grid, block>>>(
        freqs, time_indices,
        Soms_d_in, Sa_a_in,
        Amp, alpha, slope_1, f_knee, slope_2,
        c00_arr, c01_arr, c02_arr,
        c11_arr, c12_arr, c22_arr,
        num_freqs, num_times,
        *sensitivity_matrix_gpu
    );
    
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(sensitivity_matrix_gpu));
#else
    get_noise_covariance_kernel(
        freqs, time_indices,
        Soms_d_in, Sa_a_in,
        Amp, alpha, slope_1, f_knee, slope_2,
        c00_arr, c01_arr, c02_arr,
        c11_arr, c12_arr, c22_arr,
        num_freqs, num_times,
        *this
    );
#endif
}

// ============================================================================
// Batch 3x3 Hermitian Matrix Inversion
// ============================================================================

CUDA_KERNEL
void get_inverse_logdet_kernel(
    double *c00_arr, cmplx *c01_arr, cmplx *c02_arr,
    double *c11_arr, cmplx *c12_arr, double *c22_arr,
    double *i00_arr, cmplx *i01_arr, cmplx *i02_arr,
    double *i11_arr, cmplx *i12_arr, double *i22_arr,
    double *log_det_arr,
    int num)
{
    int start, end, increment;

#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num;
    increment = gridDim.x * blockDim.x;
#else
    start = 0;
    end = num;
    increment = 1;
#endif

    for (int i = start; i < end; i += increment)
    {
        invert_3x3_hermitian(
            c00_arr[i], c01_arr[i], c02_arr[i],
            c11_arr[i], c12_arr[i], c22_arr[i],
            i00_arr[i], i01_arr[i], i02_arr[i],
            i11_arr[i], i12_arr[i], i22_arr[i],
            log_det_arr[i]
        );
    }
}

void XYZSensitivityMatrix::get_inverse_logdet_arr(
    double *c00_arr, cmplx *c01_arr, cmplx *c02_arr,
    double *c11_arr, cmplx *c12_arr, double *c22_arr,
    double *i00_arr, cmplx *i01_arr, cmplx *i02_arr,
    double *i11_arr, cmplx *i12_arr, double *i22_arr,
    double *log_det_arr,
    int num)
{
#ifdef __CUDACC__
    int num_blocks = (num + NUM_THREADS - 1) / NUM_THREADS;
    
    get_inverse_logdet_kernel<<<num_blocks, NUM_THREADS>>>(
        c00_arr, c01_arr, c02_arr,
        c11_arr, c12_arr, c22_arr,
        i00_arr, i01_arr, i02_arr,
        i11_arr, i12_arr, i22_arr,
        log_det_arr,
        num
    );
    
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    get_inverse_logdet_kernel(
        c00_arr, c01_arr, c02_arr,
        c11_arr, c12_arr, c22_arr,
        i00_arr, i01_arr, i02_arr,
        i11_arr, i12_arr, i22_arr,
        log_det_arr,
        num
    );
#endif
}

// ============================================================================
// pdf calculation from sangria psd file
// ============================================================================

#ifdef __CUDACC__
#define PDF_NUM_THREADS 32
#else
#define PDF_NUM_THREADS 1
#endif
#define PDF_NDIM 6

CUDA_KERNEL
void compute_logpdf(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js,
                    int num_points, int *start_index, int num_components)
{
    int start_index_here, end_index_here, component_here, j;
    CUDA_SHARED double point_here[PDF_NDIM];
    CUDA_SHARED double log_sum_arr[PDF_NUM_THREADS];
    CUDA_SHARED double max_log_sum_arr[PDF_NUM_THREADS];
    CUDA_SHARED double max_log_all;
    CUDA_SHARED double max_tmp;
    CUDA_SHARED double total_log_sum;
    CUDA_SHARED double current_log_sum;
    double mean_here[PDF_NDIM];
    double invcov_here[PDF_NDIM][PDF_NDIM];
    double mins_here[PDF_NDIM];
    double maxs_here[PDF_NDIM];
    double point_mapped[PDF_NDIM];
    double diff_from_mean[PDF_NDIM];
    double log_main_part, log_norm_factor, log_weighted_pdf;
    double det_here, log_J_here, weight_here, tmp;
    double kernel_sum = 0.0;

double A_Soms_d_val, A_Sa_a_val, E_Soms_d_val, E_Sa_a_val;
#ifdef __CUDACC__
    int start = blockIdx.x;
    int incr = gridDim.x;
    int tid = threadIdx.x;
    int start2 = threadIdx.x;
    int incr2 = blockDim.x;
#else   
    int start = 0;
    int incr = 1;
    int tid = 0;
    int start2 = 0;
    int incr2 = 1;
#endif

    
    for (int i = start; i < num_points; i += incr)
    {   
        if (tid == 0){total_log_sum = -1e300;}
        CUDA_SYNC_THREADS;
        for (int k = start2; k < PDF_NDIM; k += incr2)
        {
            point_here[k] = points[i * PDF_NDIM + k];
        }
        CUDA_SYNC_THREADS;

        start_index_here = start_index[i];
        end_index_here = start_index[i + 1];

        while (start_index_here < end_index_here)
        {
            CUDA_SYNC_THREADS;
            log_sum_arr[tid] = -1e300;
            max_log_sum_arr[tid] = -1e300;
            CUDA_SYNC_THREADS;

            j = start_index_here + tid;
            CUDA_SYNC_THREADS;
            if (j < end_index_here)
            {
                // make sure if threads are not used that they do not affect the sum
                component_here = component_index[j];
                for (int k = 0; k < PDF_NDIM; k += 1)
                {
                    mins_here[k] = mins[k * num_components + component_here];
                    maxs_here[k] = maxs[k * num_components + component_here];
                    mean_here[k] = means[k * num_components + component_here];
                    for (int l = 0; l < PDF_NDIM; l += 1)
                    {
                        invcov_here[k][l] = invcovs[(k * PDF_NDIM + l) * num_components + component_here];
                    }
                }
                det_here = dets[component_here];
                log_J_here = log_Js[component_here];
                weight_here = weights[component_here];
                for (int k = 0; k < PDF_NDIM; k += 1)
                {
                    point_mapped[k] = ((point_here[k] - mins_here[k]) / (maxs_here[k] - mins_here[k])) * 2. - 1.;
                    diff_from_mean[k] = point_mapped[k] - mean_here[k];
                    // if ((blockIdx.x == 0) && (tid == 0)) printf("%d %d %.10e %.10e\n", component_here, k, point_mapped[k],diff_from_mean[k]);
                }
                // calculate (x-mu)^T * invcov * (x-mu)
                kernel_sum = 0.0;
                for (int k = 0; k < PDF_NDIM; k += 1)
                {
                    tmp = 0.0;
                    for (int l = 0; l < PDF_NDIM; l += 1)
                    {
                        tmp += invcov_here[k][l] * diff_from_mean[l];
                    }
                    kernel_sum += diff_from_mean[k] * tmp;
                }
                log_main_part = -1./2. * kernel_sum;
                log_norm_factor = -(double(PDF_NDIM) / 2.) * log(2 * M_PI) - (1. / 2.) * log(det_here);
                log_weighted_pdf = log(weight_here) + log_norm_factor + log_main_part;

                log_sum_arr[tid] = log_weighted_pdf + log_J_here;
                max_log_sum_arr[tid] = log_weighted_pdf + log_J_here;
                // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e %.10e %.10e %.10e\n", component_here, log(weight_here), log(det_here), kernel_sum, -(double(PDF_NDIM) / 2.) * log(2 * M_PI));
                
            }
#ifdef __CUDACC__
            CUDA_SYNC_THREADS;
            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    max_log_sum_arr[tid] = max(max_log_sum_arr[tid], max_log_sum_arr[tid + s]);
                }
                CUDA_SYNC_THREADS;
            }
            CUDA_SYNC_THREADS;
#endif
            // store max in shared value
            if (tid == 0){max_log_all = max_log_sum_arr[tid];}
#ifdef __CUDACC__
            CUDA_SYNC_THREADS;
            // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e\n", component_here, max_log_all);
            
            // subtract max from every value and take exp
            log_sum_arr[tid] = exp(log_sum_arr[tid] - max_log_all);
            CUDA_SYNC_THREADS;
            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    log_sum_arr[tid] += log_sum_arr[tid + s];
                }
                CUDA_SYNC_THREADS;
            }
            CUDA_SYNC_THREADS;         

#endif
            // do it again to add next round if there
            if (tid == 0)
            {
                // finish up initial computation
                current_log_sum = max_log_all + log(log_sum_arr[0]);
                //if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e %.10e\n", component_here, current_log_sum, total_log_sum);

                // start new computation
                // get max
// TODO: make this better?
#ifdef __CUDACC__
                max_tmp = max(current_log_sum, total_log_sum);
#else
                max_tmp = std::max(current_log_sum, total_log_sum);
#endif
                // subtract max from all values and take exp
                current_log_sum = exp(current_log_sum - max_tmp);
                total_log_sum = exp(total_log_sum - max_tmp);
                // sum values, take log and add back max
                total_log_sum = max_tmp + log(current_log_sum + total_log_sum);
                // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e\n", component_here, total_log_sum);
            }             
            
            // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %d\n", start_index_here, end_index_here);
            CUDA_SYNC_THREADS;
            start_index_here += PDF_NUM_THREADS;
        }
        logpdf_out[i] = total_log_sum;
    }
}

void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim)
{
    if (ndim != PDF_NDIM){throw std::invalid_argument("ndim in does not equal NDIM_PDF in GPU code.");}

#ifdef __CUDACC__
    compute_logpdf<<<num_points, PDF_NUM_THREADS>>>(logpdf_out, component_index, points,
                    weights, mins, maxs, means, invcovs, dets, log_Js,
                    num_points, start_index, num_components);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    compute_logpdf(logpdf_out, component_index, points,
                    weights, mins, maxs, means, invcovs, dets, log_Js,
                    num_points, start_index, num_components);
#endif
}

// ============================================================================
// LEGACY PSD Likelihood Computation
// ============================================================================

const double lisaL = 2.5e9;           // LISA's arm meters
const double lisaLT = lisaL / Clight; // LISA's armn in sec

CUDA_DEVICE void lisanoises(double *Spm, double *Sop, double f, double Soms_d_in, double Sa_a_in, bool return_relative_frequency)
{
    double frq = f;
    // Acceleration noise
    // In acceleration
    double Sa_a = Sa_a_in * (1.0 + pow((0.4e-3 / frq), 2)) * (1.0 + pow((frq / 8e-3), 4));
    // In displacement
    double Sa_d = Sa_a * pow((2.0 * M_PI * frq), (-4.0));
    // In relative frequency unit
    double Sa_nu = Sa_d * pow((2.0 * M_PI * frq / Clight), 2);

    if (return_relative_frequency)
    {
        *Spm = Sa_nu;
    }
    else
    {
        *Spm = Sa_d;
    }

    // Optical Metrology System
    // In displacement
    double Soms_d = Soms_d_in * (1.0 + pow((2.0e-3 / f), 4));
    // In relative frequency unit
    double Soms_nu = Soms_d * pow((2.0 * M_PI * frq / Clight), 2);
    *Sop = Soms_nu;

    if (return_relative_frequency)
    {
        *Sop = Soms_nu;
    }
    else
    {
        *Sop = Soms_d;
    }

    // if ((threadIdx.x == 10) && (blockIdx.x == 0) && (blockIdx.y == 0))
    //     printf("%.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e \n", frq, Sa_a_in, Soms_d_in, Sa_a, Sa_d, Sa_nu, *Spm, Soms_d, Soms_nu, *Sop);
}

CUDA_DEVICE double SGal(double fr, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double Sgal_out = (Amp * exp(-(pow(fr, alpha)) * sl1) * (pow(fr, (-7.0 / 3.0))) * 0.5 * (1.0 + tanh(-(fr - kn) * sl2)));
    return Sgal_out;
}

CUDA_DEVICE double GalConf(double fr, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double Sgal_int = SGal(fr, Amp, alpha, sl1, kn, sl2);
    return Sgal_int;
}

CUDA_DEVICE double WDconfusionX(double f, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double x = 2.0 * M_PI * lisaLT * f;
    double t = 4.0 * pow(x, 2) * pow(sin(x), 2);

    double Sg_sens = GalConf(f, Amp, alpha, sl1, kn, sl2);

    // t = 4 * x**2 * xp.sin(x)**2 * (1.0 if obs == 'X' else 1.5)
    return t * Sg_sens;
}

CUDA_DEVICE double WDconfusionAE(double f, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double SgX = WDconfusionX(f, Amp, alpha, sl1, kn, sl2);
    return 1.5 * SgX;
}

CUDA_DEVICE double lisasens(const double f, const double Soms_d_in, const double Sa_a_in, const double Amp, const double alpha, const double sl1, const double kn, const double sl2)
{
    double x = 2.0 * M_PI * lisaLT * f;
    double Sa_d, Sop;
    bool return_relative_frequency = false;
    lisanoises(&Sa_d, &Sop, f, Soms_d_in, Sa_a_in, return_relative_frequency);

    double ALL_m = sqrt(4.0 * Sa_d + Sop);
    // Average the antenna response
    double AvResp = sqrt(5.);
    // Projection effect
    double Proj = 2.0 / sqrt(3.);
    // Approximative transfert function
    double f0 = 1.0 / (2.0 * lisaLT);
    double a = 0.41;
    double T = sqrt(1. + pow((f / (a * f0)), 2));
    double Sens = pow((AvResp * Proj * T * ALL_m / lisaL), 2);

    if (Amp > 0.0)
    {
        Sens += GalConf(f, Amp, alpha, sl1, kn, sl2);
    }

    return Sens;
}

CUDA_DEVICE double noisepsd_AE(const double f, const double Soms_d_in, const double Sa_a_in, const double Amp, const double alpha, const double sl1, const double kn, const double sl2)
{
    double x = 2.0 * M_PI * lisaLT * f;
    double Spm, Sop;
    bool return_relative_frequency = true;
    lisanoises(&Spm, &Sop, f, Soms_d_in, Sa_a_in, return_relative_frequency);

    double Sa = (8.0 * (sin(x) * sin(x)) * (2.0 * Spm * (3.0 + 2.0 * cos(x) + cos(2 * x)) + Sop * (2.0 + cos(x))));

    if (Amp > 0.0)
    {
        Sa += WDconfusionAE(f, Amp, alpha, sl1, kn, sl2);
    }

    return Sa;
    //,
}

CUDA_DEVICE
double get_full_like_value(double f, double df, cmplx d_A, cmplx d_E, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double A_Soms_d_val = A_Soms_d_in * A_Soms_d_in;
    double A_Sa_a_val = A_Sa_a_in * A_Sa_a_in;
    double E_Soms_d_val = E_Soms_d_in * E_Soms_d_in;
    double E_Sa_a_val = E_Sa_a_in * E_Sa_a_in;
    double Sn_A = noisepsd_AE(f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
    double Sn_E = noisepsd_AE(f, E_Soms_d_val, E_Sa_a_val, Amp, alpha, sl1, kn, sl2);

    double inner_product = (4.0 * ((gcmplx::conj(d_A) * d_A / Sn_A) + (gcmplx::conj(d_E) * d_E / Sn_E)).real() * df);
    return -1.0 / 2.0 * inner_product - (log(Sn_A) + log(Sn_E));
}



CUDA_KERNEL void psd_likelihood(double *like_contrib, double *f_arr, cmplx *data, int *data_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, double df, int data_length, int num_data, int num_psds)
{
    #ifdef __CUDACC__
    CUDA_SHARED double like_vals[NUM_THREADS_LIKE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    int data_index;
    double A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in, Amp, alpha, sl1, kn, sl2;
    cmplx d_A, d_E;
    double f, Sn_A, Sn_E;
    double inner_product;
    double A_Soms_d_val, A_Sa_a_val, E_Soms_d_val, E_Sa_a_val;
    for (int psd_i = blockIdx.y; psd_i < num_psds; psd_i += gridDim.y)
    {
        data_index = data_index_all[psd_i];

        A_Soms_d_in = A_Soms_d_in_all[psd_i];
        A_Sa_a_in = A_Sa_a_in_all[psd_i];
        E_Soms_d_in = E_Soms_d_in_all[psd_i];
        E_Sa_a_in = E_Sa_a_in_all[psd_i];
        Amp = Amp_all[psd_i];
        alpha = alpha_all[psd_i];
        sl1 = sl1_all[psd_i];
        kn = kn_all[psd_i];
        sl2 = sl2_all[psd_i];

        for (int i = threadIdx.x; i < NUM_THREADS_LIKE; i += blockDim.x)
        {
            like_vals[i] = 0.0;
        }
        CUDA_SYNC_THREADS;

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < data_length; i += blockDim.x * gridDim.x)
        {
            d_A = data[(data_index * 2 + 0) * data_length + i];
            d_E = data[(data_index * 2 + 1) * data_length + i];
            f = f_arr[i];
            if (f == 0.0)
            {
                f = df; // TODO switch this?
            }

            like_vals[tid] += get_full_like_value(f, df, d_A, d_E, A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in, Amp, alpha, sl1, kn, sl2);
        }
        CUDA_SYNC_THREADS;

        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                like_vals[tid] += like_vals[tid + s];
                // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
            }
            CUDA_SYNC_THREADS;
        }
        CUDA_SYNC_THREADS;

        if (tid == 0)
        {
            like_contrib[psd_i * num_blocks + bid] = like_vals[0];
        }
        CUDA_SYNC_THREADS;
    }
    #endif
}

void psd_likelihood_cpu(double *like_vals, double *f_arr, cmplx *data, int *data_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, double df, int data_length, int num_data, int num_psds)
{
    int data_index;
    double _tmp_like_val = 0.0;
    double A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in, Amp, alpha, sl1, kn, sl2;
    cmplx d_A, d_E;
    double f, Sn_A, Sn_E;
    double inner_product;
    double A_Soms_d_val, A_Sa_a_val, E_Soms_d_val, E_Sa_a_val;
    for (int psd_i = 0; psd_i < num_psds; psd_i += 1)
    {
        _tmp_like_val = 0.0;
        data_index = data_index_all[psd_i];
        A_Soms_d_in = A_Soms_d_in_all[psd_i];
        A_Sa_a_in = A_Sa_a_in_all[psd_i];
        E_Soms_d_in = E_Soms_d_in_all[psd_i];
        E_Sa_a_in = E_Sa_a_in_all[psd_i];
        Amp = Amp_all[psd_i];
        alpha = alpha_all[psd_i];
        sl1 = sl1_all[psd_i];
        kn = kn_all[psd_i];
        sl2 = sl2_all[psd_i];

        for (int i = 0; i < data_length; i += 1)
        {
            d_A = data[(data_index * 2 + 0) * data_length + i];
            d_E = data[(data_index * 2 + 1) * data_length + i];
            f = f_arr[i];
            if (f == 0.0)
            {
                f = df; // TODO switch this?
            }

            _tmp_like_val += get_full_like_value(f, df, d_A, d_E, A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in, Amp, alpha, sl1, kn, sl2);
        }
        like_vals[psd_i] = _tmp_like_val;
    }
}


void psd_likelihood_wrap(double *like_contrib_final, double *f_arr, cmplx *data, int *data_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                         double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, double df, int data_length, int num_data, int num_psds)
{
    #ifdef __CUDACC__
    double *like_contrib;

    int num_blocks = std::ceil((data_length + NUM_THREADS_LIKE - 1) / NUM_THREADS_LIKE);

    gpuErrchk(cudaMalloc(&like_contrib, num_psds * num_blocks * sizeof(double)));

    dim3 grid(num_blocks, num_psds, 1);

    psd_likelihood<<<grid, NUM_THREADS_LIKE>>>(like_contrib, f_arr, data, data_index_all, A_Soms_d_in_all, A_Sa_a_in_all, E_Soms_d_in_all, E_Sa_a_in_all,
                                               Amp_all, alpha_all, sl1_all, kn_all, sl2_all, df, data_length, num_data, num_psds);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    dim3 grid_gather(1, num_psds, 1);
    like_sum_from_contrib<<<grid_gather, NUM_THREADS_LIKE>>>(like_contrib_final, like_contrib, num_blocks, num_psds);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(like_contrib));
    #else
    psd_likelihood_cpu(like_contrib_final, f_arr, data, data_index_all, A_Soms_d_in_all, A_Sa_a_in_all, E_Soms_d_in_all, E_Sa_a_in_all,
                                               Amp_all, alpha_all, sl1_all, kn_all, sl2_all, df, data_length, num_data, num_psds);

    #endif
}


#define NUM_THREADS_LIKE 256
CUDA_KERNEL void get_psd_val(double *Sn_A_out, double *Sn_E_out, double *f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double sl1, double kn, double sl2, int num_f)
{
    int noise_index;
    double f, Sn_A, Sn_E;
    double A_Soms_d_val, A_Sa_a_val, E_Soms_d_val, E_Sa_a_val;
#ifdef __CUDACC__
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int incr = gridDim.x * blockDim.x;
#else   
    int start = 0;
    int incr = 1;
#endif
    for (int f_i = start; f_i < num_f; f_i += incr)
    {
        f = f_arr[f_i];
        
        A_Soms_d_val = A_Soms_d_in * A_Soms_d_in;
        A_Sa_a_val = A_Sa_a_in * A_Sa_a_in;
        E_Soms_d_val = E_Soms_d_in * E_Soms_d_in;
        E_Sa_a_val = E_Sa_a_in * E_Sa_a_in;
        Sn_A = noisepsd_AE(f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
        Sn_E = noisepsd_AE(f, E_Soms_d_val, E_Sa_a_val, Amp, alpha, sl1, kn, sl2);

        // if (Sn_A != Sn_A)
        // {
        //     printf("BADDDDD: %d %e %e %e %e %e %e %e %e\n", f_i, f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
        // }

        Sn_A_out[f_i] = Sn_A;
        Sn_E_out[f_i] = Sn_E;
    }
}

void get_psd_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, double A_Soms_d_in, double A_Sa_a_in, double E_Soms_d_in, double E_Sa_a_in,
                               double Amp, double alpha, double sl1, double kn, double sl2, int num_f)
{
    #ifdef __CUDACC__
    int num_blocks = std::ceil((num_f + NUM_THREADS_LIKE - 1) / NUM_THREADS_LIKE);

    get_psd_val<<<num_blocks, NUM_THREADS_LIKE>>>(Sn_A_out, Sn_E_out, f_arr, A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in,
                                               Amp, alpha, sl1, kn, sl2, num_f);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    #else
     get_psd_val(Sn_A_out, Sn_E_out, f_arr, A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in,
                                               Amp, alpha, sl1, kn, sl2, num_f);

    #endif
}