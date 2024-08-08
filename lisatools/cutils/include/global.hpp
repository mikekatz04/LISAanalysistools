#ifndef __GLOBAL_HPP__
#define __GLOBAL_HPP__

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

#endif


#endif // __GLOBAL_HPP__