#include <stdio.h>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include "global.hpp"
#include "cuda_complex.hpp"


const double lisaL = 2.5e9;           // LISA's arm meters
const double lisaLT = lisaL / Clight; // LISA's armn in sec

__device__ void lisanoises(double *Spm, double *Sop, double f, double Soms_d_in, double Sa_a_in, bool return_relative_frequency)
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

__device__ double SGal(double fr, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double Sgal_out = (Amp * exp(-(pow(fr, alpha)) * sl1) * (pow(fr, (-7.0 / 3.0))) * 0.5 * (1.0 + tanh(-(fr - kn) * sl2)));
    return Sgal_out;
}

__device__ double GalConf(double fr, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double Sgal_int = SGal(fr, Amp, alpha, sl1, kn, sl2);
    return Sgal_int;
}

__device__ double WDconfusionX(double f, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double x = 2.0 * M_PI * lisaLT * f;
    double t = 4.0 * pow(x, 2) * pow(sin(x), 2);

    double Sg_sens = GalConf(f, Amp, alpha, sl1, kn, sl2);

    // t = 4 * x**2 * xp.sin(x)**2 * (1.0 if obs == 'X' else 1.5)
    return t * Sg_sens;
}

__device__ double WDconfusionAE(double f, double Amp, double alpha, double sl1, double kn, double sl2)
{
    double SgX = WDconfusionX(f, Amp, alpha, sl1, kn, sl2);
    return 1.5 * SgX;
}

__device__ double lisasens(const double f, const double Soms_d_in, const double Sa_a_in, const double Amp, const double alpha, const double sl1, const double kn, const double sl2)
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

__device__ double noisepsd_AE(const double f, const double Soms_d_in, const double Sa_a_in, const double Amp, const double alpha, const double sl1, const double kn, const double sl2)
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


#define NUM_THREADS_LIKE 256
__global__ void psd_likelihood(double *like_contrib, double *f_arr, cmplx *data, int *data_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, double df, int data_length, int num_data, int num_psds)
{
    __shared__ double like_vals[NUM_THREADS_LIKE];
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
        __syncthreads();

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < data_length; i += blockDim.x * gridDim.x)
        {
            d_A = data[(data_index * 2 + 0) * data_length + i];
            d_E = data[(data_index * 2 + 1) * data_length + i];
            f = f_arr[i];
            if (f == 0.0)
            {
                f = df; // TODO switch this?
            }

            A_Soms_d_val = A_Soms_d_in * A_Soms_d_in;
            A_Sa_a_val = A_Sa_a_in * A_Sa_a_in;
            E_Soms_d_val = E_Soms_d_in * E_Soms_d_in;
            E_Sa_a_val = E_Sa_a_in * E_Sa_a_in;
            Sn_A = noisepsd_AE(f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
            Sn_E = noisepsd_AE(f, E_Soms_d_val, E_Sa_a_val, Amp, alpha, sl1, kn, sl2);

            inner_product = (4.0 * ((gcmplx::conj(d_A) * d_A / Sn_A) + (gcmplx::conj(d_E) * d_E / Sn_E)).real() * df);
            like_vals[tid] += -1.0 / 2.0 * inner_product - (log(Sn_A) + log(Sn_E));
            // if ((psd_i == 0) && (i > 400000) && (i < 400020)) printf("%d %.12e %.12e %.12e %.12e %.12e %.12e \n", i, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), Sn_A, Sn_E);
        }
        __syncthreads();

        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                like_vals[tid] += like_vals[tid + s];
                // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
            }
            __syncthreads();
        }
        __syncthreads();

        if (tid == 0)
        {
            like_contrib[psd_i * num_blocks + bid] = like_vals[0];
        }
        __syncthreads();
    }
}

#define NUM_THREADS_LIKE 256
__global__ void like_sum_from_contrib(double *like_contrib_final, double *like_contrib, int num_blocks_orig, int num_psds)
{
    __shared__ double like_vals[NUM_THREADS_LIKE];
    int tid = threadIdx.x;

    for (int psd_i = blockIdx.y; psd_i < num_psds; psd_i += gridDim.y)
    {
        for (int i = threadIdx.x; i < NUM_THREADS_LIKE; i += blockDim.x)
        {
            like_vals[i] = 0.0;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < num_blocks_orig; i += blockDim.x)
        {
            like_vals[tid] += like_contrib[psd_i * num_blocks_orig + i];
        }
        __syncthreads();

        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                like_vals[tid] += like_vals[tid + s];
                // if ((bin_i == 1) && (blockIdx.x == 0) && (channel_i == 0) && (s == 1))
            }
            __syncthreads();
        }
        __syncthreads();

        if (tid == 0)
        {
            like_contrib_final[psd_i] = like_vals[0];
        }
        __syncthreads();
    }
}

void psd_likelihood_wrap(double *like_contrib_final, double *f_arr, cmplx *data, int *data_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                         double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, double df, int data_length, int num_data, int num_psds)
{
    double *like_contrib;

    int num_blocks = std::ceil((data_length + NUM_THREADS_LIKE - 1) / NUM_THREADS_LIKE);

    CUDA_CHECK_AND_EXIT(cudaMalloc(&like_contrib, num_psds * num_blocks * sizeof(double)));

    dim3 grid(num_blocks, num_psds, 1);

    psd_likelihood<<<grid, NUM_THREADS_LIKE>>>(like_contrib, f_arr, data, data_index_all, A_Soms_d_in_all, A_Sa_a_in_all, E_Soms_d_in_all, E_Sa_a_in_all,
                                               Amp_all, alpha_all, sl1_all, kn_all, sl2_all, df, data_length, num_data, num_psds);

    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    dim3 grid_gather(1, num_psds, 1);
    like_sum_from_contrib<<<grid_gather, NUM_THREADS_LIKE>>>(like_contrib_final, like_contrib, num_blocks, num_psds);
    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    CUDA_CHECK_AND_EXIT(cudaFree(like_contrib));
}

#define PDF_NUM_THREADS 32
#define PDF_NDIM 8

__global__
void compute_logpdf(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js,
                    int num_points, int *start_index, int num_components)
{
    int start_index_here, end_index_here, component_here, j;
    __shared__ double point_here[PDF_NDIM];
    __shared__ double log_sum_arr[PDF_NUM_THREADS];
    __shared__ double max_log_sum_arr[PDF_NUM_THREADS];
    __shared__ double max_log_all;
    __shared__ double max_tmp;
    __shared__ double total_log_sum;
    __shared__ double current_log_sum;
    double mean_here[PDF_NDIM];
    double invcov_here[PDF_NDIM][PDF_NDIM];
    double mins_here[PDF_NDIM];
    double maxs_here[PDF_NDIM];
    double point_mapped[PDF_NDIM];
    double diff_from_mean[PDF_NDIM];
    double log_main_part, log_norm_factor, log_weighted_pdf;
    double det_here, log_J_here, weight_here, tmp;
    double kernel_sum = 0.0;
    int tid = threadIdx.x;
    
    for (int i = blockIdx.x; i < num_points; i += gridDim.x)
    {   
        if (tid == 0){total_log_sum = -1e300;}
        __syncthreads();
        for (int k = threadIdx.x; k < PDF_NDIM; k += blockDim.x)
        {
            point_here[k] = points[i * PDF_NDIM + k];
        }
        __syncthreads();

        start_index_here = start_index[i];
        end_index_here = start_index[i + 1];

        while (start_index_here < end_index_here)
        {
            __syncthreads();
            log_sum_arr[tid] = -1e300;
            max_log_sum_arr[tid] = -1e300;
            __syncthreads();

            j = start_index_here + tid;
            __syncthreads();
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
                log_norm_factor = (double(PDF_NDIM) / 2.) * log(2 * M_PI) + (1. / 2.) * log(det_here);
                log_weighted_pdf = log(weight_here) + log_norm_factor + log_main_part;

                log_sum_arr[tid] = log_weighted_pdf + log_J_here;
                max_log_sum_arr[tid] = log_weighted_pdf + log_J_here;
                // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e\n", component_here, log_weighted_pdf);
                
            }
            __syncthreads();
            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    max_log_sum_arr[tid] = max(max_log_sum_arr[tid], max_log_sum_arr[tid + s]);
                }
                __syncthreads();
            }
            __syncthreads();
            // store max in shared value
            if (tid == 0){max_log_all = max_log_sum_arr[tid];}
            __syncthreads();
            // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e\n", component_here, max_log_all);
            
            // subtract max from every value and take exp
            log_sum_arr[tid] = exp(log_sum_arr[tid] - max_log_all);
            __syncthreads();
            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    log_sum_arr[tid] += log_sum_arr[tid + s];
                }
                __syncthreads();
            }
            __syncthreads();
            // do it again to add next round if there
            if (tid == 0)
            {
                // finish up initial computation
                current_log_sum = max_log_all + log(log_sum_arr[0]);
                //if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e %.10e\n", component_here, current_log_sum, total_log_sum);

                // start new computation
                // get max
                max_tmp = max(current_log_sum, total_log_sum);
                // subtract max from all values and take exp
                current_log_sum = exp(current_log_sum - max_tmp);
                total_log_sum = exp(total_log_sum - max_tmp);
                // sum values, take log and add back max
                total_log_sum = max_tmp + log(current_log_sum + total_log_sum);
                // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %.10e\n", component_here, total_log_sum);
            }             
            start_index_here += PDF_NUM_THREADS;
            // if ((blockIdx.x == 0) && (tid == 0)) printf("%d, %d\n", start_index_here, end_index_here);
            __syncthreads();
        }
        logpdf_out[i] = total_log_sum;
    }
}

void compute_logpdf_wrap(double *logpdf_out, int *component_index, double *points,
                    double *weights, double *mins, double *maxs, double *means, double *invcovs, double *dets, double *log_Js, 
                    int num_points, int *start_index, int num_components, int ndim)
{
    if (ndim != PDF_NDIM){throw std::invalid_argument("ndim in does not equal NDIM_PDF in GPU code.");}

    compute_logpdf<<<num_points, PDF_NUM_THREADS>>>(logpdf_out, component_index, points,
                    weights, mins, maxs, means, invcovs, dets, log_Js,
                    num_points, start_index, num_components);
    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
}


#define NUM_THREADS_LIKE 256
__global__ void get_psd_val(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    int noise_index;
    double A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in, Amp, alpha, sl1, kn, sl2;
    double f, Sn_A, Sn_E;
    double A_Soms_d_val, A_Sa_a_val, E_Soms_d_val, E_Sa_a_val;
    for (int f_i = blockIdx.x * blockDim.x + threadIdx.x; f_i < num_f; f_i += gridDim.x * blockDim.x)
    {
        noise_index = noise_index_all[f_i];

        A_Soms_d_in = A_Soms_d_in_all[noise_index];
        A_Sa_a_in = A_Sa_a_in_all[noise_index];
        E_Soms_d_in = E_Soms_d_in_all[noise_index];
        E_Sa_a_in = E_Sa_a_in_all[noise_index];
        Amp = Amp_all[noise_index];
        alpha = alpha_all[noise_index];
        sl1 = sl1_all[noise_index];
        kn = kn_all[noise_index];
        sl2 = sl2_all[noise_index];
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

void get_psd_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f)
{

    int num_blocks = std::ceil((num_f + NUM_THREADS_LIKE - 1) / NUM_THREADS_LIKE);

    get_psd_val<<<num_blocks, NUM_THREADS_LIKE>>>(Sn_A_out, Sn_E_out, f_arr, noise_index_all, A_Soms_d_in_all, A_Sa_a_in_all, E_Soms_d_in_all, E_Sa_a_in_all,
                                               Amp_all, alpha_all, sl1_all, kn_all, sl2_all, num_f);

    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
}




#define NUM_THREADS_LIKE 256
__global__ void get_lisasens_val(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    int noise_index;
    double A_Soms_d_in, A_Sa_a_in, E_Soms_d_in, E_Sa_a_in, Amp, alpha, sl1, kn, sl2;
    double f, Sn_A, Sn_E;
    double A_Soms_d_val, A_Sa_a_val, E_Soms_d_val, E_Sa_a_val;
    for (int f_i = blockIdx.x * blockDim.x + threadIdx.x; f_i < num_f; f_i += gridDim.x * blockDim.x)
    {
        noise_index = noise_index_all[f_i];

        A_Soms_d_in = A_Soms_d_in_all[noise_index];
        A_Sa_a_in = A_Sa_a_in_all[noise_index];
        E_Soms_d_in = E_Soms_d_in_all[noise_index];
        E_Sa_a_in = E_Sa_a_in_all[noise_index];
        Amp = Amp_all[noise_index];
        alpha = alpha_all[noise_index];
        sl1 = sl1_all[noise_index];
        kn = kn_all[noise_index];
        sl2 = sl2_all[noise_index];
        f = f_arr[f_i];
        
        A_Soms_d_val = A_Soms_d_in * A_Soms_d_in;
        A_Sa_a_val = A_Sa_a_in * A_Sa_a_in;
        E_Soms_d_val = E_Soms_d_in * E_Soms_d_in;
        E_Sa_a_val = E_Sa_a_in * E_Sa_a_in;
        Sn_A = lisasens(f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
        Sn_E = lisasens(f, E_Soms_d_val, E_Sa_a_val, Amp, alpha, sl1, kn, sl2);

        // if (Sn_A != Sn_A)
        // {
        //     printf("BADDDDD: %d %e %e %e %e %e %e %e %e\n", f_i, f, A_Soms_d_val, A_Sa_a_val, Amp, alpha, sl1, kn, sl2);
        // }

        Sn_A_out[f_i] = Sn_A;
        Sn_E_out[f_i] = Sn_E;
    }
}

void get_lisasens_val_wrap(double *Sn_A_out, double *Sn_E_out, double *f_arr, int *noise_index_all, double *A_Soms_d_in_all, double *A_Sa_a_in_all, double *E_Soms_d_in_all, double *E_Sa_a_in_all,
                               double *Amp_all, double *alpha_all, double *sl1_all, double *kn_all, double *sl2_all, int num_f)
{

    int num_blocks = std::ceil((num_f + NUM_THREADS_LIKE - 1) / NUM_THREADS_LIKE);

    get_lisasens_val<<<num_blocks, NUM_THREADS_LIKE>>>(Sn_A_out, Sn_E_out, f_arr, noise_index_all, A_Soms_d_in_all, A_Sa_a_in_all, E_Soms_d_in_all, E_Sa_a_in_all,
                                               Amp_all, alpha_all, sl1_all, kn_all, sl2_all, num_f);

    cudaDeviceSynchronize();
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
}




#define NUM_THREADS_LIKE 64

// __global__ void specialty_piece_wise_likelihoods(
//     double *lnL,
//     cmplx *data,
//     double *noise,
//     int *data_index,
//     int *noise_index,
//     int *start_inds,
//     int *lengths,
//     double df,
//     int num_parts,
//     int start_freq_ind,
//     int data_length,
//     int tdi_channel_setup,
//     int num_data, 
//     int num_noise)
// {
//     using complex_type = cmplx;

//     int tid = threadIdx.x;
//     __shared__ double lnL_tmp_for_sum[NUM_THREADS_LIKE];

//     int nchannels = 3;
//     if (tdi_channel_setup == TDI_CHANNEL_SETUP_AE) nchannels = 2;
//     for (int i = threadIdx.x; i < NUM_THREADS_LIKE; i += blockDim.x)
//     {
//         lnL_tmp_for_sum[i] = 0.0;
//     }
//     __syncthreads();

//     cmplx tmp1;
//     int data_ind, noise_ind, start_ind, length;

//     int jj = 0;
//     // example::io<FFT>::load_to_smem(this_block_data, shared_mem);

//     cmplx d, h;
//     cmplx _ignore_this = 0.0;
//     cmplx _ignore_this_2 = 0.0;
//     double n;
//     for (int part_i = blockIdx.x; part_i < num_parts; part_i += gridDim.x)
//     {

//         data_ind = data_index[part_i];
//         noise_ind = noise_index[part_i];
//         start_ind = start_inds[part_i];
//         length = lengths[part_i];

//         tmp1 = 0.0;
//         for (int i = threadIdx.x; i < length; i += blockDim.x)
//         {
//             jj = i + start_ind - start_freq_ind;
//             // d_A = data_A[data_ind * data_length + jj];
//             // d_E = data_E[data_ind * data_length + jj];
//             // n_A = noise_A[noise_ind * data_length + jj];
//             // n_E = noise_E[noise_ind * data_length + jj];
//             add_inner_product_contribution(
//                 &tmp1, &_ignore_this, &_ignore_this_2,
//                 data, data, 
//                 jj, jj, 
//                 ARRAY_TYPE_DATA, ARRAY_TYPE_DATA,
//                 noise, noise_ind, jj,
//                 data_ind, tdi_channel_setup, data_length, -1,
//                 num_data, num_noise
//             );

//             // if (part_i == 0)
//             //{
//             //     printf("check vals %d %d %d %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", i, jj, start_ind, part_i, d_A.real(), d_A.imag(), d_E.real(), d_E.imag(), n_A, n_E, df);
//             // }
//             // tmp1 += (gcmplx::conj(d_A) * d_A / n_A + gcmplx::conj(d_E) * d_E / n_E).real();
//         }
//         __syncthreads();
//         lnL_tmp_for_sum[tid] = tmp1.real();

//         __syncthreads();
//         if (tid == 0)
//         {
//             lnL[part_i] = -1. / 2. * (4.0 * df * lnL_tmp_for_sum[0]);
//         }

//         __syncthreads();
//         for (unsigned int s = 1; s < blockDim.x; s *= 2)
//         {
//             if (tid % (2 * s) == 0)
//             {
//                 lnL_tmp_for_sum[tid] += lnL_tmp_for_sum[tid + s];
//             }
//             __syncthreads();
//         }
//         __syncthreads();

//         if (tid == 0)
//         {
//             lnL[part_i] = -1. / 2. * (4.0 * df * lnL_tmp_for_sum[0]);
//         }
//         __syncthreads();

//         // example::io<FFT>::store_from_smem(shared_mem, this_block_data);
//     }
//     //
// }

// void specialty_piece_wise_likelihoods_wrap(
//     double *lnL,
//     cmplx *data,
//     double *noise,
//     int *data_index,
//     int *noise_index,
//     int *start_inds,
//     int *lengths,
//     double df,
//     int num_parts,
//     int start_freq_ind,
//     int data_length,
//     int tdi_channel_setup,
//     bool do_synchronize,
//     int num_data, 
//     int num_noise)
// {
//     if (num_parts == 0)
//     {
//         printf("num_parts is 0\n");
//         return;
//     }
//     specialty_piece_wise_likelihoods<<<num_parts, NUM_THREADS_LIKE>>>(
//         lnL,
//         data,
//         noise,
//         data_index,
//         noise_index,
//         start_inds,
//         lengths,
//         df,
//         num_parts,
//         start_freq_ind,
//         data_length,
//         tdi_channel_setup,
//         num_data, 
//         num_noise);

//     CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

//     if (do_synchronize)
//         CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
// }
