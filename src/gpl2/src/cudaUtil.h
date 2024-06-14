#pragma once

#include <cuda.h>
#include <cufft.h>
#include <stdio.h>

#define PI (3.141592653589793238462643383279502884197169)

#ifndef CUDA_CHECK
#define CUDA_CHECK(status) __CUDA_CHECK(status, __FILE__, __LINE__)
#endif

inline __device__ __host__ bool isPowerOf2(int val)
{
  return val && (val & (val - 1)) == 0;
}

inline void __CUDA_CHECK(cudaError_t status, const char* file, const int line)
{
  if (status != cudaSuccess) {
    fprintf(stderr,
            "[CUDA-ERROR] Error %s at line %d in file %s\n",
            cudaGetErrorString(status),
            line,
            file);
    exit(status);
  }
}

inline __host__ __device__ int INDEX(const int hid, const int wid, const int N)
{
  return (hid * N + wid);
}

inline __host__ __device__ cufftDoubleComplex complexMul(const cufftDoubleComplex& x,
                                                const cufftDoubleComplex& y)
{
  cufftDoubleComplex res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = x.x * y.y + x.y * y.x;
  return res;
}

inline __host__ __device__ cufftComplex complexMul(const cufftComplex& x,
                                          const cufftComplex& y)
{
  cufftComplex res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = x.x * y.y + x.y * y.x;
  return res;
}

inline __host__ __device__ cufftDoubleReal RealPartOfMul(const cufftDoubleComplex& x,
                                                const cufftDoubleComplex& y)
{
  return x.x * y.x - x.y * y.y;
}

inline __host__ __device__ cufftReal RealPartOfMul(const cufftComplex& x,
                                          const cufftComplex& y)
{
  return x.x * y.x - x.y * y.y;
}

inline __host__ __device__ cufftDoubleReal
ImaginaryPartOfMul(const cufftDoubleComplex& x, const cufftDoubleComplex& y)
{
  return x.x * y.y + x.y * y.x;
}

inline __host__ __device__ cufftReal ImaginaryPartOfMul(const cufftComplex& x,
                                               const cufftComplex& y)
{
  return x.x * y.y + x.y * y.x;
}

inline __host__ __device__ cufftDoubleComplex complexAdd(const cufftDoubleComplex& x,
                                                const cufftDoubleComplex& y)
{
  cufftDoubleComplex res;
  res.x = x.x + y.x;
  res.y = x.y + y.y;
  return res;
}

inline __host__ __device__ cufftComplex complexAdd(const cufftComplex& x,
                                          const cufftComplex& y)
{
  cufftComplex res;
  res.x = x.x + y.x;
  res.y = x.y + y.y;
  return res;
}

inline __host__ __device__ cufftDoubleComplex
complexSubtract(const cufftDoubleComplex& x, const cufftDoubleComplex& y)
{
  cufftDoubleComplex res;
  res.x = x.x - y.x;
  res.y = x.y - y.y;
  return res;
}

inline __host__ __device__ cufftComplex complexSubtract(const cufftComplex& x,
                                               const cufftComplex& y)
{
  cufftComplex res;
  res.x = x.x - y.x;
  res.y = x.y - y.y;
  return res;
}

inline __host__ __device__ cufftDoubleComplex complexConj(const cufftDoubleComplex& x)
{
  cufftDoubleComplex res;
  res.x = x.x;
  res.y = -x.y;
  return res;
}

inline __host__ __device__ cufftComplex complexConj(const cufftComplex& x)
{
  cufftComplex res;
  res.x = x.x;
  res.y = -x.y;
  return res;
}

inline __host__ __device__ cufftDoubleComplex complexMulConj(const cufftDoubleComplex& x,
                                                    const cufftDoubleComplex& y)
{
  cufftDoubleComplex res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = -(x.x * y.y + x.y * y.x);
  return res;
}

inline __host__ __device__ cufftComplex complexMulConj(const cufftComplex& x,
                                              const cufftComplex& y)
{
  cufftComplex res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = -(x.x * y.y + x.y * y.x);
  return res;
}
