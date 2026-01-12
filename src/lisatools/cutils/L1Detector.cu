#include "stdio.h"
#include "gbt_global.h"
#include "L1Detector.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

#if defined(__CUDACC__) || defined(__CUDA_COMPILATION__)
#define L1Orbits L1OrbitsGPU
#define XYZSensitivityMatrix XYZSensitivityMatrixGPU
#else
#define L1Orbits L1OrbitsCPU
#define XYZSensitivityMatrix XYZSensitivityMatrixCPU
#endif
// TODO WHEN BACK FROM BREAK:
// SEPARATE OUT ANY FUNCTION THAT INCLUDES ANYTHING PYBIND RELATED INTO BINDING. INHERIT THE ORBITS CLASS INTO A WRAPPER CLASS THAT ADDS THE FUNCTIONS THAT SPECIFICALLY TAKE IN NUMPY AND CUPY ARRAYS 

CUDA_DEVICE
int L1Orbits::get_window(double t, double t0, double dt, int N)
{
    int out = int( (t - t0) / dt);
    if ((out < 0) || (out >= N))
        return -1;
    else
        return out;
}

CUDA_DEVICE
int L1Orbits::get_link_ind(int link)
{
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
    else
#ifdef __CUDACC__
        // printf("BAD link ind. Must be 12, 23, 31, 13, 32, 21.");
#else
        throw std::invalid_argument("Bad link ind. Must be 12, 23, 31, 13, 32, 21.");
#endif // __CUDACC__
    return -1;
}

CUDA_DEVICE
int L1Orbits::get_sc_ind(int sc)
{
    if (sc == 1)
        return 0;
    else if (sc == 2)
        return 1;
    else if (sc == 3)
        return 2;
    else
    {
#ifdef __CUDACC__
        // printf("BAD sc ind. Must be 1,2,3. %d\n", sc);
#else
        std::ostringstream oss;
        oss << "Bad sc ind. Must be 1,2,3. Input sc is " << sc << " " << std::endl;
        std::string var = oss.str();
        throw std::invalid_argument(var);
#endif // __CUDACC__
    }
    return 0;
}

CUDA_DEVICE
double L1Orbits::interpolate(double t, double *in_arr, double t0, double dt, int window, int major_ndim, int major_ind, int ndim, int pos)
{
    double up = in_arr[((window + 1) * major_ndim + major_ind) * ndim + pos]; // down_ind * ndim + pos];
    double down = in_arr[(window * major_ndim + major_ind) * ndim + pos];

    // m *(x - x0) + y0
    double fin = ((up - down) / dt) * (t - (t0 + dt * window)) + down;
    // if ((ndim == 1))
    //     printf("%d %e %e %e %e \n", window, fin, down, up, (t - (dt * window)));

    return fin;
}

CUDA_DEVICE
void L1Orbits::get_normal_unit_vec_ptr(Vec *vec, double t, int link)
{
    Vec _tmp = get_normal_unit_vec(t, link);
    vec->x = _tmp.x;
    vec->y = _tmp.y;
    vec->z = _tmp.z;
}

CUDA_DEVICE
Vec L1Orbits::get_normal_unit_vec(double t, int link)
{
    int window = get_window(t, sc_t0, sc_dt, sc_N);
    if (window == -1)
    {
        // out of bounds
        return Vec(0.0, 0.0, 0.0);
    }

    int link_ind = get_link_ind(link);

    int up_ind = (window + 1) * nlinks + link_ind;
    int down_ind = window * nlinks + link_ind;

    // x (pos = 0) ndim = 3
    double x_out = interpolate(t, n_arr, sc_t0, sc_dt, window, nlinks, link_ind, 3, 0);
    // y (pos = 1)
    double y_out = interpolate(t, n_arr, sc_t0, sc_dt, window, nlinks, link_ind, 3, 1);
    // z (pos = 2)
    double z_out = interpolate(t, n_arr, sc_t0, sc_dt, window, nlinks, link_ind, 3, 2);

    return Vec(x_out, y_out, z_out);
}

CUDA_DEVICE
double L1Orbits::get_light_travel_time(double t, int link)
{
    int window = get_window(t, ltt_t0, ltt_dt, ltt_N);
    if (window == -1)
    {
        // out of bounds
        return 0.0;
    }

    int link_ind = get_link_ind(link);
    int up_ind = (window + 1) * (nlinks + link_ind);
    int down_ind = window * (nlinks + link_ind);

    // x (pos = 0), ndim = 1
    double ltt_out = interpolate(t, ltt_arr, ltt_t0, ltt_dt, window, nlinks, link_ind, 1, 0);

    return ltt_out;
}

CUDA_DEVICE
Vec L1Orbits::get_pos(double t, int sc)
{
    int window = get_window(t, sc_t0, sc_dt, sc_N);
    if (window == -1)
    {
        // out of bounds
        return Vec(0.0, 0.0, 0.0);
    }

    int sc_ind = get_sc_ind(sc);

    // x (pos = 0), ndim = 3
    double x_out = interpolate(t, x_arr, sc_t0, sc_dt, window, nspacecraft, sc_ind, 3, 0);
    // y (pos = 1), ndim = 3
    double y_out = interpolate(t, x_arr, sc_t0, sc_dt, window, nspacecraft, sc_ind, 3, 1);
    // z (pos = 2), ndim = 3
    double z_out = interpolate(t, x_arr, sc_t0, sc_dt, window, nspacecraft, sc_ind, 3, 2);
    return Vec(x_out, y_out, z_out);
}

CUDA_DEVICE
void L1Orbits::get_pos_ptr(Vec *vec, double t, int sc)
{
    Vec _tmp = get_pos(t, sc);
    vec->x = _tmp.x;
    vec->y = _tmp.y;
    vec->z = _tmp.z;
}

#define NUM_THREADS 64


CUDA_KERNEL
void get_light_travel_time_kernel(double *ltt, double *t, int *link, int num, L1Orbits &orbits)
{
    int start, end, increment;
#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num;
    increment = gridDim.x * blockDim.x;
#else  // __CUDACC__
    start = 0;
    end = num;
    increment = 1;
#endif // __CUDACC__

    for (int i = start; i < end; i += increment)
    {
        ltt[i] = orbits.get_light_travel_time(t[i], link[i]);
    }
}


void L1Orbits::get_light_travel_time_arr(double *ltt, double *t, int *link, int num)
{
#ifdef __CUDACC__
    int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);

    // copy self to GPU
    L1Orbits *orbits_gpu;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(L1Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, this, sizeof(L1Orbits), cudaMemcpyHostToDevice));

    get_light_travel_time_kernel<<<num_blocks, NUM_THREADS>>>(ltt, t, link, num, *orbits_gpu);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(orbits_gpu));

#else // __CUDACC__

    get_light_travel_time_kernel(ltt, t, link, num, *this);

#endif // __CUDACC__
}


CUDA_KERNEL
void get_pos_kernel(double *pos_x, double *pos_y, double *pos_z, double *t, int *sc, int num, L1Orbits &orbits)
{
    int start, end, increment;
#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num;
    increment = gridDim.x * blockDim.x;
#else  // __CUDACC__
    start = 0;
    end = num;
    increment = 1;
#endif // __CUDACC__
    Vec _tmp(0.0, 0.0, 0.0);

    for (int i = start; i < end; i += increment)
    {
        _tmp = orbits.get_pos(t[i], sc[i]);
        pos_x[i] = _tmp.x;
        pos_y[i] = _tmp.y;
        pos_z[i] = _tmp.z;
    }
}


void L1Orbits::get_pos_arr(double *pos_x, double *pos_y, double *pos_z, double *t, int *sc, int num)
{
#ifdef __CUDACC__
    int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);

    // copy self to GPU
    L1Orbits *orbits_gpu;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(L1Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, this, sizeof(L1Orbits), cudaMemcpyHostToDevice));

    get_pos_kernel<<<num_blocks, NUM_THREADS>>>(pos_x, pos_y, pos_z, t, sc, num, *orbits_gpu);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(orbits_gpu));

#else // __CUDACC__

    get_pos_kernel(pos_x, pos_y, pos_z, t, sc, num, *this);

#endif // __CUDACC__
}


CUDA_KERNEL
void get_normal_unit_vec_kernel(double *normal_unit_vec_x, double *normal_unit_vec_y, double *normal_unit_vec_z, double *t, int *link, int num, L1Orbits &orbits)
{
    int start, end, increment;
#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num;
    increment = gridDim.x * blockDim.x;
#else  // __CUDACC__
    start = 0;
    end = num;
    increment = 1;
#endif // __CUDACC__
    Vec _tmp(0.0, 0.0, 0.0);

    for (int i = start; i < end; i += increment)
    {
        _tmp = orbits.get_normal_unit_vec(t[i], link[i]);
        normal_unit_vec_x[i] = _tmp.x;
        normal_unit_vec_y[i] = _tmp.y;
        normal_unit_vec_z[i] = _tmp.z;
    }
}

void L1Orbits::get_normal_unit_vec_arr(double *normal_unit_vec_x, double *normal_unit_vec_y, double *normal_unit_vec_z, double *t, int *link, int num)
{
#ifdef __CUDACC__
    int num_blocks = std::ceil((num + NUM_THREADS - 1) / NUM_THREADS);

    // copy self to GPU
    L1Orbits *orbits_gpu;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(L1Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, this, sizeof(L1Orbits), cudaMemcpyHostToDevice));

    get_normal_unit_vec_kernel<<<num_blocks, NUM_THREADS>>>(normal_unit_vec_x, normal_unit_vec_y, normal_unit_vec_z, t, link, num, *orbits_gpu);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(orbits_gpu));

#else // __CUDACC__

    get_normal_unit_vec_kernel(normal_unit_vec_x, normal_unit_vec_y, normal_unit_vec_z, t, link, num, *this);

#endif // __CUDACC__
}

static CUDA_DEVICE
double omega(double f)
{   
    return 2.0 * M_PI * f;
}

static CUDA_DEVICE
double d_times_omega(double d, double f)
// multiplies a delay d (in seconds) by angular frequency omega (in rad/s) given frequency f (in Hz)
{   
    return d * omega(f);
}

static CUDA_DEVICE
double s_wl(double d, double f)
// compute the sine of the wavelength phase shift for delay d (in seconds) and frequency f (in Hz)
{
    return sin(d_times_omega(d, f));
}

static CUDA_DEVICE
double c_wl(double d, double f)
// compute the cosine of the wavelength phase shift for delay d (in seconds) and frequency f (in Hz)
{
    return cos(d_times_omega(d, f));
}

static CUDA_DEVICE
double s_2wl(double d, double f)
// compute the sine of the wavelength phase shift for delay d (in seconds) and frequency f (in Hz)
{
    return sin(2 * d_times_omega(d, f));
}

static CUDA_DEVICE
double c_2wl(double d, double f)
// compute the sine of the wavelength phase shift for delay d (in seconds) and frequency f (in Hz)
{
    return cos(2 * d_times_omega(d, f));
}

static CUDA_DEVICE
void invert_3x3_hermitian(
    double c00, gcmplx::complex<double> c01, gcmplx::complex<double> c02,
    double c11, gcmplx::complex<double> c12,
    double c22,
    double &i00, gcmplx::complex<double> &i01, gcmplx::complex<double> &i02,
    double &i11, gcmplx::complex<double> &i12,
    double &i22,
    double &log_det)
{
    // Cofactors (upper triangle)
    // C00 = c11*c22 - |c12|^2
    double C00 = c11 * c22 - gcmplx::norm(c12);
    // C11 = c00*c22 - |c02|^2
    double C11 = c00 * c22 - gcmplx::norm(c02);
    // C22 = c00*c11 - |c01|^2
    double C22 = c00 * c11 - gcmplx::norm(c01);

    // C01 = c12*conj(c02) - conj(c01)*c22 (Calculated as M10 earlier, need C01 for determinant? No, usually expand on row 0)
    // Det = c00*C00 + c01*C01 + c02*C02
    // C01 = c12*conj(c02) - conj(c01)*c22
    gcmplx::complex<double> C01 = c12 * gcmplx::conj(c02) - gcmplx::conj(c01) * c22;
    // C02 = conj(c01)*conj(c12) - c11*conj(c02)
    gcmplx::complex<double> C02 = gcmplx::conj(c01) * gcmplx::conj(c12) - c11 * gcmplx::conj(c02);

    // Det calculation (Expansion along row 0)
    // c00 is real. C00 is real.
    // c01*C01 + c02*C02 + c00*C00
    // Since matrix is Hermitian, Det is real.
    double det = c00 * C00 + (c01 * C01).real() + (c02 * C02).real();
    double inv_det = 1.0 / det;
    log_det = log(det);

    // Elements of Inverse (Upper Triangle) = Cofactor / Det.
    // Note: Inv_ij = Cofactor_ji / Det.
    // Since Hermitian, Inv_01 = Cofactor_10 / Det = Conj(Cofactor_01) / Det.
    
    i00 = C00 * inv_det;
    i11 = C11 * inv_det;
    i22 = C22 * inv_det;

    i01 = gcmplx::conj(C01) * inv_det;
    i02 = gcmplx::conj(C02) * inv_det;
    
    // i12 = Inv_12 = Conj(C12)/Det
    // C12 = c01*conj(c02) - c00*conj(c12)
    gcmplx::complex<double> C12 = c01 * gcmplx::conj(c02) - c00 * gcmplx::conj(c12);
    i12 = gcmplx::conj(C12) * inv_det;
}

#define NUM_THREADS_LIKE 256

CUDA_KERNEL void psd_likelihood_xyz_kernel(
    double *like_contrib, double *f_arr, gcmplx::complex<double> *data_in,
    int *data_index_all, int *time_index_all,
    double *Soms_d_in_all, double *Sa_a_in_all,
    double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all,
    double df, int data_length, int num_psds, XYZSensitivityMatrix &sensitivity_matrix)
{
    CUDA_SHARED double like_vals[NUM_THREADS_LIKE];
    int tid = threadIdx.x;
    
    // Per-thread variables
    double Soms_d_in, Sa_a_in, Amp, alpha, sl1, kn, sl2;
    double f, S_tm, S_isi_oms, S_gal;
    int data_index, time_index;
    gcmplx::complex<double> d_X, d_Y, d_Z;
    
    // Matrix elements (Upper triangle)
    gcmplx::complex<double> oms_xx, oms_xy, oms_xz, oms_yy, oms_yz, oms_zz;
    gcmplx::complex<double> tm_xx, tm_xy, tm_xz, tm_yy, tm_yz, tm_zz;
    double c00, c11, c22;
    gcmplx::complex<double> c01, c02, c12;
    // Inverse elements
    double i00, i11, i22, log_det;
    gcmplx::complex<double> i01, i02, i12;
    double quad_form;

    // Loop over PSDs (one block y per PSD chunk logic, or just grid stride)
    // tmp.cu uses: for (int psd_i = blockIdx.y; psd_i < num_psds; psd_i += gridDim.y)
    for (int psd_i = blockIdx.y; psd_i < num_psds; psd_i += gridDim.y)
    {
        data_index = data_index_all[psd_i];
        time_index = time_index_all[psd_i];

        Soms_d_in = Soms_d_in_all[psd_i];
        Sa_a_in = Sa_a_in_all[psd_i];
        Amp = Amp_all[psd_i];
        alpha = alpha_all[psd_i];
        sl1 = sl1_all[psd_i];
        kn = kn_all[psd_i];
        sl2 = sl2_all[psd_i];

        // Initialize reduction
        like_vals[tid] = 0.0;
        CUDA_SYNC_THREADS;

        // Loop over frequencies
        // stride: blockDim.x * gridDim.x (dim x covers frequencies)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < data_length; i += blockDim.x * gridDim.x)
        {
            f = f_arr[i];

            // Get noise transfer functions
            sensitivity_matrix.get_noise_tfs(f, 
                                             &oms_xx, &oms_xy, &oms_xz, &oms_yy, &oms_yz, &oms_zz,
                                             &tm_xx, &tm_xy, &tm_xz, &tm_yy, &tm_yz, &tm_zz, 
                                             time_index);

            // Calculate Noise PSDs
            lisanoises(&S_tm, &S_isi_oms, f, Soms_d_in, Sa_a_in, true);
            S_gal = SGal(f, Amp, alpha, sl1, kn, sl2);

            // Build Covariance Matrix C (3x3 Hermitian)
            // C = (coeff_oms * S_oms) + (coeff_tm * S_tm)
            // Diagonal elements are real because oms_xx/tm_xx are "real-ish"? 
            // sensitivity_matrix.oms_xx... returns complex but is real on diagonal physically?
            // Actually, XYZSensitivityMatrix methods return complex.
            // Assumption: |TF|^2 * S is the component.
            // Diagonal: C_XX = (|oms_xx|^2 * S_isi) + (|tm_xx|^2 * S_tm) + S_gal
            c00 = (gcmplx::norm(oms_xx) * S_isi_oms) + (gcmplx::norm(tm_xx) * S_tm) + S_gal;
            c11 = (gcmplx::norm(oms_yy) * S_isi_oms) + (gcmplx::norm(tm_yy) * S_tm) + S_gal;
            c22 = (gcmplx::norm(oms_zz) * S_isi_oms) + (gcmplx::norm(tm_zz) * S_tm) + S_gal;

            // Off-diagonal: C_XY = (oms_xx * conj(oms_xy))? No.
            // It's Cov(X, Y) = < (H_x_oms n_oms + ...) (H_y_oms n_oms + ...)* >
            // = H_x_oms H_y_oms* S_oms + ...
            // Note: oms_xy from get_noise_tfs IS the cross term? 
            // In `get_noise_tfs`, `oms_xy` is `oms_xy_unequal_armlength`.
            // As determined earlier, these seem to be the transfer functions H.
            // Wait, for off-diagonal, `oms_xy` return from `get_noise_tfs` is likely the `H` for that channel combination? 
            // NO. `oms_xy_unequal_armlength` calculates the *cross-spectral density coefficient*.
            // Look at `XYZSensitivityMatrix::oms_xy_unequal_armlength`:
            // `_oms = -8.0 * (c_wl * s_wl * s_wl * exp(...))`
            // This is the Cross Spectrum term directly (normalized by S_oms).
            // So C_xy = oms_xy (complex) * S_isi_oms + tm_xy * S_tm.
            
            c01 = oms_xy * S_isi_oms + tm_xy * S_tm;
            c02 = oms_xz * S_isi_oms + tm_xz * S_tm;
            c12 = oms_yz * S_isi_oms + tm_yz * S_tm;

            // Invert C -> C^-1
            invert_3x3_hermitian(c00, c01, c02, c11, c12, c22, 
                                 i00, i01, i02, i11, i12, i22, log_det);
            
            // Get Data
            // data array: (num_psds, 3, data_length)? Or linearized differently?
            // tmp.py says: linearized (num_streams * 3 * data_length)
            // data_index points to the start for a stream?
            // Let's assume data layout: [data_index][3][frequency]
            int base_idx = (data_index * 3 * data_length) + i;
            d_X = data_in[base_idx];
            d_Y = data_in[base_idx + data_length];
            d_Z = data_in[base_idx + 2 * data_length];

            // Compute Quadratic Form: d^H * C^-1 * d
            gcmplx::complex<double> termX = i00 * d_X + i01 * d_Y + i02 * d_Z;
            gcmplx::complex<double> termY = gcmplx::conj(i01) * d_X + i11 * d_Y + i12 * d_Z; // i10 = conj(i01)
            gcmplx::complex<double> termZ = gcmplx::conj(i02) * d_X + gcmplx::conj(i12) * d_Y + i22 * d_Z; // i20=conj(i02), i21=conj(i12)

            // d^H * (terms)
            double Q = (gcmplx::conj(d_X) * termX + gcmplx::conj(d_Y) * termY + gcmplx::conj(d_Z) * termZ).real();
            
            // Likelihood Accumulation
            // -0.5 * (4 * df * Q + log_det + const)
            // log_det is sum of log eigenvalues -> log(det)
            // const term usually ignored or N*log(2pi)
            // tmp.cu uses: `-1.0/2.0 * inner_product - log_det`
            // where inner_product = 4 * real(...) * df.
            
            like_vals[tid] += -0.5 * (4.0 * df * Q + log_det);
        }
        CUDA_SYNC_THREADS;

        // Block Reduction
        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                like_vals[tid] += like_vals[tid + s];
            }
            CUDA_SYNC_THREADS;
        }

        // Store result for this block/PSD
        if (tid == 0)
        {
            // like_contrib needs to handle multiple blocks per PSD if needed.
            // But here grid y is PSDs, grid x is frequency blocks.
            // Index: psd_i * gridDim.x + blockIdx.x
            like_contrib[psd_i * gridDim.x + blockIdx.x] = like_vals[0];
        }
        CUDA_SYNC_THREADS;
    }
}

CUDA_KERNEL void like_sum_from_contrib(double *like_contrib_final, double *like_contrib, int num_blocks_per_psd, int num_psds)
{
    // Sums up the partial results from blocks for each PSD.
    // One block (y) per PSD? Or one thread per PSD?
    // If num_psds is large, use grid.
    
    int tid = threadIdx.x;
    int psd_i = blockIdx.x * blockDim.x + threadIdx.x; // if 1D grid
    
    // tmp.cu implementation uses shared mem reduction again.
    // "grid_gather(1, num_psds, 1)" -> num_psds blocks in Y.
    // loop over psd_i.
    
    for (int p = blockIdx.y; p < num_psds; p += gridDim.y)
    {
        double sum = 0.0;
        for (int i = tid; i < num_blocks_per_psd; i += blockDim.x)
        {
             sum += like_contrib[p * num_blocks_per_psd + i];
        }
        
        // Parallel reduction in shared mem? 
        // Or just straightforward sum if num_blocks is small?
        // Let's use Warp/Block reduction for speed.
        
        static __shared__ double shared_sum[NUM_THREADS_LIKE];
        shared_sum[tid] = sum;
        CUDA_SYNC_THREADS;
        
        for (unsigned int s = 1; s < blockDim.x; s *= 2) {
             if (tid % (2 * s) == 0) {
                 shared_sum[tid] += shared_sum[tid + s];
             }
             CUDA_SYNC_THREADS;
        }
        
        if (tid == 0) {
            like_contrib_final[p] = shared_sum[0];
        }
    }
}

void XYZSensitivityMatrix::psd_likelihood_wrap(
    double *like_contrib_final, double *f_arr, gcmplx::complex<double> *data, 
    int *data_index_all, int *time_index_all,
    double *Soms_d_in_all, double *Sa_a_in_all, 
    double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, 
    double df, int data_length, int num_psds)
{
#ifdef __CUDACC__
    double *like_contrib;
    int num_blocks = std::ceil((double)data_length / NUM_THREADS_LIKE); // Blocks for frequency coverage

    // Allocation not shown (assume managed or passed pointer? No, must allocate intermediate)
    gpuErrchk(cudaMalloc(&like_contrib, num_psds * num_blocks * sizeof(double)));
    
    // Grid: X=blocks for freq, Y=blocks for PSDs (one block per PSD group?)
    // If num_psds is large, we iterate in kernel.
    // Let's use Y=num_psds (limited by 65535 on old cards, but modern are fine. If large, need loop)
    // Safe to use smaller grid and loop.
    dim3 grid(num_blocks, std::min(num_psds, 65535), 1);
    
    // Copy/Management of `this` pointer to device handled? 
    // `XYZSensitivityMatrix` is a class. We need to pass the device object reference.
    // This wrapper is likely a HOST method of the class. 
    // But the KERNEL needs the DEVICE object.
    // Usually we copy `*this` to device and pass that.
    
    XYZSensitivityMatrix *dev_ptr;
    gpuErrchk(cudaMalloc(&dev_ptr, sizeof(XYZSensitivityMatrix)));
    gpuErrchk(cudaMemcpy(dev_ptr, this, sizeof(XYZSensitivityMatrix), cudaMemcpyHostToDevice));

    psd_likelihood_xyz_kernel<<<grid, NUM_THREADS_LIKE>>>(
        like_contrib, f_arr, data, data_index_all, time_index_all,
        Soms_d_in_all, Sa_a_in_all,
        Amp_all, alpha_all, sl1_all, kn_all, sl2_all,
        df, data_length, num_psds, *dev_ptr);
        
    gpuErrchk(cudaGetLastError());
    
    // Reduction
    dim3 grid_reduc(1, std::min(num_psds, 65535), 1);
    like_sum_from_contrib<<<grid_reduc, NUM_THREADS_LIKE>>>(like_contrib_final, like_contrib, num_blocks, num_psds);
    
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(like_contrib));
    gpuErrchk(cudaFree(dev_ptr));
#else
    // CPU Fallback ...
#endif
}


static CUDA_DEVICE 
double get_cholesky_decomposition(gcmplx::complex<double> *c_arr, gcmplx::complex<double> xx_term, gcmplx::complex<double> xy_term, gcmplx::complex<double> xz_term,
                                     gcmplx::complex<double> yy_term, gcmplx::complex<double> yz_term, gcmplx::complex<double> zz_term)
{
    // Cholesky decomposition of 3x3 Hermitian matrix
    // [ xx, xy, xz ]
    // [ yx, yy, yz ]
    // [ zx, zy, zz ]
    // where yx = conj(xy), zx = conj(xz), zy = conj(yz)

    // Compute L matrix elements
    gcmplx::complex<double> l11 = gcmplx::sqrt(xx_term);
    gcmplx::complex<double> l21 = xy_term / l11;
    gcmplx::complex<double> l31 = xz_term / l11;

    gcmplx::complex<double> l22 = gcmplx::sqrt(yy_term - l21 * gcmplx::conj(l21));
    gcmplx::complex<double> l32 = (yz_term - l31 * gcmplx::conj(l21)) / l22;

    gcmplx::complex<double> l33 = gcmplx::sqrt(zz_term - l31 * gcmplx::conj(l31) - l32 * gcmplx::conj(l32));

    // now compute the inverse of L
    gcmplx::complex<double> w11 = 1.0 / l11;
    gcmplx::complex<double> w21 = -l21 / (l11 * l22);
    gcmplx::complex<double> w31 = (-l31 * l22 + l21 * l32) / (l11 * l22 * l33);
    gcmplx::complex<double> w22 = 1.0 / l22;
    gcmplx::complex<double> w32 = -l32 / (l22 * l33);
    gcmplx::complex<double> w33 = 1.0 / l33;

    // C^{-1} = L^{-T} @ L^{-1} (only compute unique elements)

    gcmplx::complex<double> c00 = w11 * w11 + w21 * w21 + w31 * w31;
    gcmplx::complex<double> c01 = w21 * w22 + w31 * w32;
    gcmplx::complex<double> c02 = w31 * w33;
    gcmplx::complex<double> c11 = w22 * w22 + w32 * w32;
    gcmplx::complex<double> c12 = w32 * w33;
    gcmplx::complex<double> c22 = w33 * w33;

    // Store in c_arr
    c_arr[0] = c00;
    c_arr[1] = c01;
    c_arr[2] = c02;
    c_arr[3] = c11;
    c_arr[4] = c12;
    c_arr[5] = c22;

    // Compute log determinant
    double log_det = 2.0 * (gcmplx::log(l11).real() + gcmplx::log(l22).real() + gcmplx::log(l33).real());
    return log_det;
}



CUDA_DEVICE
void lisanoises(double *S_tm, double *S_isi_oms, double f, double Soms_d_in, double Sa_a_in, bool return_relative_frequency)
{
    // Acceleration noise
    // In acceleration
    double Sa_a = Sa_a_in * (1.0 + pow((0.4e-3 / f), 2)) * (1.0 + pow((f / 8e-3), 4));
    // In displacement
    double Sa_d = Sa_a * pow((2.0 * M_PI * f), (-4.0));
    // In relative frequency unit
    double Sa_nu = Sa_d * pow((2.0 * M_PI * f / Clight), 2);

    if (return_relative_frequency)
    {
        *S_tm = Sa_nu;
    }
    else
    {
        *S_tm = Sa_d;
    }

    // Optical Metrology System
    // In displacement
    double Soms_d = Soms_d_in * (1.0 + pow((2.0e-3 / f), 4));
    // In relative frequency unit
    double Soms_nu = Soms_d * pow((2.0 * M_PI * f / Clight), 2);
    *S_isi_oms = Soms_nu;

    if (return_relative_frequency)
    {
        *S_isi_oms = Soms_nu;
    }
    else
    {
        *S_isi_oms = Soms_d;
    }
}

CUDA_DEVICE
double SGal(double f, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double Sgal_out = (Amp * exp(-(pow(f, alpha)) * sl1) * (pow(f, (-7.0 / 3.0))) * 0.5 * (1.0 + tanh(-(f - kn) * sl2)));
    return Sgal_out;
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
}

CUDA_DEVICE
gcmplx::complex<double> XYZSensitivityMatrix::oms_xx_unequal_armlength(double f, double avg_d_ij, double avg_d_ik)
{
    gcmplx::complex<double> _oms = 8.0 * (pow(s_wl(avg_d_ij, f), 2) + pow(s_wl(avg_d_ik, f), 2));

    if (generation == 2)
    {
        _oms = _oms * 4.0 * pow(s_wl(avg_d_ij + avg_d_ik, f), 2);
    }
    return _oms;
}

CUDA_DEVICE
gcmplx::complex<double> XYZSensitivityMatrix::oms_xy_unequal_armlength(double f, double avg_d_ij, double avg_d_ik, double avg_d_jk, double delta_d_ij)
{
    gcmplx::complex<double> _oms = -8.0 * (
        c_wl(avg_d_ij, f)
        * s_wl(avg_d_ik, f)
        * s_wl(avg_d_jk, f)
        * gcmplx::exp(gcmplx::complex<double>(0.0, -1.0) * d_times_omega((avg_d_ik - avg_d_jk + 0.5 * delta_d_ij), f))
    );

    if (generation == 2)
    {
        _oms = _oms * (
            4.0
            * s_wl(avg_d_ij + avg_d_ik, f)
            * s_wl(avg_d_ij + avg_d_jk, f)
            * gcmplx::exp(gcmplx::complex<double>(0.0, -1.0) * d_times_omega((avg_d_ik - avg_d_jk), f))
        );
    }
    return _oms;
}

CUDA_DEVICE
gcmplx::complex<double> XYZSensitivityMatrix::tm_xx_unequal_armlength(double f, double avg_d_ij, double avg_d_ik)
{
    gcmplx::complex<double> _tm = 8.0 * (
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
gcmplx::complex<double> XYZSensitivityMatrix::tm_xy_unequal_armlength(double f, double avg_d_ij, double avg_d_ik, double avg_d_jk, double delta_d_ij)
{
    gcmplx::complex<double> _tm = -32.0 * (
        c_wl(avg_d_ij, f)
        * s_wl(avg_d_ik, f)
        * s_wl(avg_d_jk, f)
        * gcmplx::exp(gcmplx::complex<double>(0.0, -1.0) * d_times_omega((avg_d_ik - avg_d_jk + 0.5 * delta_d_ij), f))
    );  
    if (generation == 2)
    {
        _tm *= (
            4.0
            * s_wl(avg_d_ij + avg_d_ik, f)
            * s_wl(avg_d_ij + avg_d_jk, f)
            * gcmplx::exp(gcmplx::complex<double>(0.0, -1.0) * d_times_omega((avg_d_ik - avg_d_jk), f))
        );
    }
    return _tm;
}

CUDA_DEVICE
gcmplx::complex<double> XYZSensitivityMatrix::build_noise_matrix_element(double S_tm, double S_isi_oms, gcmplx::complex<double> oms_tf, gcmplx::complex<double> tm_tf)
{
    gcmplx::complex<double> noise_matrix_element = (S_tm * tm_tf) + (S_isi_oms * oms_tf);
    return noise_matrix_element;
}

CUDA_DEVICE
void XYZSensitivityMatrix::get_noise_tfs(double f, gcmplx::complex<double> *oms_xx, gcmplx::complex<double> *oms_xy, gcmplx::complex<double> *oms_xz, gcmplx::complex<double> *oms_yy, gcmplx::complex<double> *oms_yz, gcmplx::complex<double> *oms_zz,
                                  gcmplx::complex<double> *tm_xx, gcmplx::complex<double> *tm_xy, gcmplx::complex<double> *tm_xz, gcmplx::complex<double> *tm_yy, gcmplx::complex<double> *tm_yz, gcmplx::complex<double> *tm_zz,
                                  int time_index)
{
    // Retrieve average and delta light travel times for the given time index
    double avg_d[6];
    double delta_d[6];
    for (int i = 0; i < 6; i++)
    {
        avg_d[i] = averaged_ltts_arr[time_index * n_links + i];
        delta_d[i] = delta_ltts_arr[time_index * n_links + i];
    }

    // Compute OMS noise transfer functions
    *oms_xx = oms_xx_unequal_armlength(f, avg_d[0], avg_d[1]);
    *oms_xy = oms_xy_unequal_armlength(f, avg_d[0], avg_d[1], avg_d[2], delta_d[0]);
    *oms_xz = oms_xy_unequal_armlength(f, avg_d[0], avg_d[2], avg_d[1], delta_d[0]); // Assuming symmetry
    *oms_yy = oms_xx_unequal_armlength(f, avg_d[1], avg_d[2]);
    *oms_yz = oms_xy_unequal_armlength(f, avg_d[1], avg_d[2], avg_d[0], delta_d[1]); // Assuming symmetry
    *oms_zz = oms_xx_unequal_armlength(f, avg_d[2], avg_d[0]);

    // Compute TM noise transfer functions
    *tm_xx = tm_xx_unequal_armlength(f, avg_d[0], avg_d[1]);
    *tm_xy = tm_xy_unequal_armlength(f, avg_d[0], avg_d[1], avg_d[2], delta_d[0]);
    *tm_xz = tm_xy_unequal_armlength(f, avg_d[0], avg_d[2], avg_d[1], delta_d[0]); // Assuming symmetry
    *tm_yy = tm_xx_unequal_armlength(f, avg_d[1], avg_d[2]);
    *tm_yz = tm_xy_unequal_armlength(f, avg_d[1], avg_d[2], avg_d[0], delta_d[1]); // Assuming symmetry
    *tm_zz = tm_xx_unequal_armlength(f, avg_d[2], avg_d[0]);
}

// now, add a cuda kernel to compute all noise tfs at once for an array of frequencies and time indices
CUDA_KERNEL
void get_noise_tfs_kernel(double *frequencies, int *time_indices,
                          gcmplx::complex<double> *oms_xx_arr, gcmplx::complex<double> *oms_xy_arr, gcmplx::complex<double> *oms_xz_arr,
                          gcmplx::complex<double> *oms_yy_arr, gcmplx::complex<double> *oms_yz_arr, gcmplx::complex<double> *oms_zz_arr,
                          gcmplx::complex<double> *tm_xx_arr, gcmplx::complex<double> *tm_xy_arr, gcmplx::complex<double> *tm_xz_arr,
                          gcmplx::complex<double> *tm_yy_arr, gcmplx::complex<double> *tm_yz_arr, gcmplx::complex<double> *tm_zz_arr,
                          int num, XYZSensitivityMatrix &sensitivity_matrix)
{
    int start, end, increment;
#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num;
    increment = gridDim.x * blockDim.x;
#else  // __CUDACC__
    start = 0;
    end = num;
    increment = 1;
#endif // __CUDACC__

    for (int i = start; i < end; i += increment)
    {
        double f = frequencies[i];
        int time_index = time_indices[i];

        sensitivity_matrix.get_noise_tfs(f,
                                         &oms_xx_arr[i], &oms_xy_arr[i], &oms_xz_arr[i],
                                         &oms_yy_arr[i], &oms_yz_arr[i], &oms_zz_arr[i],
                                         &tm_xx_arr[i], &tm_xy_arr[i], &tm_xz_arr[i],
                                         &tm_yy_arr[i], &tm_yz_arr[i], &tm_zz_arr[i],
                                         time_index);
    }
}

void XYZSensitivityMatrix::get_noise_tfs_arr(double *freqs,
                          gcmplx::complex<double> *oms_xx_arr, gcmplx::complex<double> *oms_xy_arr, gcmplx::complex<double> *oms_xz_arr,
                          gcmplx::complex<double> *oms_yy_arr, gcmplx::complex<double> *oms_yz_arr, gcmplx::complex<double> *oms_zz_arr,
                          gcmplx::complex<double> *tm_xx_arr, gcmplx::complex<double> *tm_xy_arr, gcmplx::complex<double> *tm_xz_arr,
                          gcmplx::complex<double> *tm_yy_arr, gcmplx::complex<double> *tm_yz_arr, gcmplx::complex<double> *tm_zz_arr,
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