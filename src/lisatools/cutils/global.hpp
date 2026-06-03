#ifndef __GLOBAL_HPP__
#define __GLOBAL_HPP__

// LAT-specific global header. Most of what used to live here -- the
// cuda_complex.hpp include, the `cmplx` typedef, CUDA_KERNEL / CUDA_DEVICE /
// CUDA_SHARED / CUDA_SYNCTHREADS macros, the gpuErrchk macro -- is now owned
// by GPUBackendTools' gbt_global.h. We pick those up via that header so
// there is one sprint-wide copy. Only LAT-specific symbols (Clight,
// CUDA_CHECK_AND_EXIT) remain here.

#include "gbt_global.h"

// gbt_global.h does not provide CUDA_SYNCTHREADS (only CUDA_SYNC_THREADS).
// Keep the old spelling here as a backwards-compatibility alias for LAT
// code that uses the no-underscore form.
#ifndef CUDA_SYNCTHREADS
#define CUDA_SYNCTHREADS CUDA_SYNC_THREADS
#endif

#define Clight 299792458.

#ifdef __CUDACC__

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                throw std::invalid_argument(cudaGetErrorString(status));                                    \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT

#endif // __CUDACC__

#endif // __GLOBAL_HPP__
