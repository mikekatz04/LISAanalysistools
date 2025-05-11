#ifndef __GLOBAL_HPP__
#define __GLOBAL_HPP__

#include "cuda_complex.hpp"

typedef gcmplx::complex<double> cmplx;
#define Clight 299792458.


#ifdef __CUDACC__
#define CUDA_KERNEL __global__
#define CUDA_DEVICE __device__

#else // __CUDACC__
#define CUDA_KERNEL 
#define CUDA_DEVICE  

#endif // __CUDACC__

#ifdef __CUDACC__
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// TODO: copied from GBGPU constants file. Need to merge all the constants.

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                    throw std::invalid_argument(cudaGetErrorString(status) );                \
                /*std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);*/                                                                          \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT


#endif //__CUDACC__


#endif // __GLOBAL_HPP__