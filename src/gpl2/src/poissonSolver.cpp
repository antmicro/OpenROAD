///////////////////////////////////////////////////////////////////////////
//
// BSD 3-Clause License
//
// Copyright (c) 2023, Google LLC
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// The density force is calculated by solving the Poisson equation.
// It is originally developed by the graduate student Jaekyung Kim
// (jkim97@postech.ac.kr) at Pohang University of Science and Technology
// (POSTECH), then modified by our UCSD team. We thank Jaekyung Kim for his
// contribution.
//
//
///////////////////////////////////////////////////////////////////////////////

#include "poissonSolver.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <stdio.h>

#include <cmath>
#include <memory>

#include <stdio.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

#include "poissonSolver.h"
#include "util.h"


namespace gpl2 {

PoissonSolver::PoissonSolver()
    : binCntX_(0),
      binCntY_(0),
      binSizeX_(0),
      binSizeY_(0),
      d_expkN_(nullptr),
      d_expkM_(nullptr),
      d_expkNForInverse_(nullptr),
      d_expkMForInverse_(nullptr),
      d_expkMN1_(nullptr),
      d_expkMN2_(nullptr),
      d_binDensity_(nullptr),
      d_auv_(nullptr),
      d_potential_(nullptr),
      d_efX_(nullptr),
      d_efY_(nullptr),
      d_workSpaceReal1_(nullptr),
      d_workSpaceReal2_(nullptr),
      d_workSpaceReal3_(nullptr),
      d_workSpaceComplex_(nullptr),
      d_inputForX_(nullptr),
      d_inputForY_(nullptr)
{
}

PoissonSolver::PoissonSolver(int binCntX,
                             int binCntY,
                             int binSizeX,
                             int binSizeY)
    : PoissonSolver()
{
  binCntX_ = binCntX;
  binCntY_ = binCntY;
  binSizeX_ = binSizeX;
  binSizeY_ = binSizeY;

  printf("[PoissonSolver] Start Initialization!\n");

  initCUDAKernel();

  printf("[PoissonSolver] Initialization Succeed!\n");
}

PoissonSolver::~PoissonSolver()
{
  freeCUDAKernel();
}

__host__ __device__ void divideByWSquare(const int wID,
                                const int hID,
                                const int binCntX,
                                const int binCntY,
                                const int binSizeX,
                                const int binSizeY,
                                cufftReal* input)
{
  if (wID < binCntX && hID < binCntY) {
    int binID = wID + hID * binCntX;

    if (hID == 0 && wID == 0)
      input[binID] = 0.0;
    else {
      float denom1 = (2.0 * float(FFT_PI) * wID) / binCntX;
      float denom2
          = (2.0 * float(FFT_PI) * hID) / binCntY * binSizeY / binSizeX;

      input[binID] /= (denom1 * denom1 + denom2 * denom2);
    }
  }
}

void PoissonSolver::solvePoissonPotential(const float* binDensity,
                                          float* potential)
{
  // Step #1. Compute Coefficient (a_uv)
  dct_2d_fft(binCntY_,
             binCntX_,
             plan_,
             d_expkM_,
             d_expkN_,
             binDensity,
             d_workSpaceReal1_,
             d_workSpaceComplex_,
             d_auv_);

  // Step #2. Divide by (w_u^2 + w_v^2)
  auto binCntX = binCntX_, binCntY = binCntY_, binSizeX = binSizeX_, binSizeY = binSizeY_;
  auto d_auv = d_auv_;
  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {binCntX_, binCntY_}),
  KOKKOS_LAMBDA (const int wID, const int hID) {
    divideByWSquare(hID, wID, binCntX, binCntY, binSizeX, binSizeY, d_auv);
  });

  // Step #3. Compute Potential
  idct_2d_fft(binCntY_,
              binCntX_,
              planInverse_,
              d_expkMForInverse_,
              d_expkNForInverse_,
              d_expkMN1_,
              d_expkMN2_,
              d_auv_,
              d_workSpaceComplex_,
              d_workSpaceReal1_,
              potential);
}

void PoissonSolver::solvePoisson(const float* binDensity,
                                 float* potential,
                                 float* electroForceX,
                                 float* electroForceY)
{
  // Step #1. Compute Coefficient (a_uv)
  dct_2d_fft(binCntY_,
             binCntX_,
             plan_,
             d_expkM_,
             d_expkN_,
             binDensity,
             d_workSpaceReal1_,
             d_workSpaceComplex_,
             d_auv_);

  // Step #2. Divide by (w_u^2 + w_v^2)
  auto binCntX = binCntX_, binCntY = binCntY_, binSizeX = binSizeX_, binSizeY = binSizeY_;
  auto d_auv = d_auv_;
  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {binCntX_, binCntY_}),
  KOKKOS_LAMBDA (const int wID, const int hID) {
    divideByWSquare(hID, wID, binCntX, binCntY, binSizeX, binSizeY, d_auv);
  });

  // Step #3. Compute Potential
  idct_2d_fft(binCntY_,
              binCntX_,
              planInverse_,
              d_expkMForInverse_,
              d_expkNForInverse_,
              d_expkMN1_,
              d_expkMN2_,
              d_auv_,
              d_workSpaceComplex_,
              d_workSpaceReal1_,
              potential);

  // Step #4. Multiply w_u , w_v
  auto d_inputForX = d_inputForX_, d_inputForY = d_inputForY_;
  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {binCntX_, binCntY_}),
  KOKKOS_LAMBDA (const int wID, const int hID) {
    int binID = wID + hID * binCntX;

    float w_u = (2.0 * float(FFT_PI) * wID) / binCntX;
    float w_v = (2.0 * float(FFT_PI) * hID) / binCntY * binSizeY / binSizeX;

    d_inputForX[binID] = w_u * d_auv[binID];
    d_inputForY[binID] = w_v * d_auv[binID];
  });

  // Step #5. Compute ElectroForceX
  idxst_idct(binCntY_,
             binCntX_,
             planInverse_,
             d_expkMForInverse_,
             d_expkNForInverse_,
             d_expkMN1_,
             d_expkMN2_,
             d_inputForX_,
             d_workSpaceReal1_,
             d_workSpaceComplex_,
             d_workSpaceReal2_,
             d_workSpaceReal3_,
             electroForceX);

  // Step #6. Compute ElectroForceY
  idct_idxst(binCntY_,
             binCntX_,
             planInverse_,
             d_expkMForInverse_,
             d_expkNForInverse_,
             d_expkMN1_,
             d_expkMN2_,
             d_inputForY_,
             d_workSpaceReal1_,
             d_workSpaceComplex_,
             d_workSpaceReal2_,
             d_workSpaceReal3_,
             electroForceY);

  cudaDeviceSynchronize();
}

void PoissonSolver::initCUDAKernel()
{
  CUDA_CHECK(cudaMalloc((void**) &d_binDensity_,
                        binCntX_ * binCntY_ * sizeof(cufftReal)));

  CUDA_CHECK(
      cudaMalloc((void**) &d_auv_, binCntX_ * binCntY_ * sizeof(cufftReal)));

  CUDA_CHECK(cudaMalloc((void**) &d_potential_,
                        binCntX_ * binCntY_ * sizeof(cufftReal)));

  CUDA_CHECK(
      cudaMalloc((void**) &d_efX_, binCntX_ * binCntY_ * sizeof(cufftReal)));

  CUDA_CHECK(
      cudaMalloc((void**) &d_efY_, binCntX_ * binCntY_ * sizeof(cufftReal)));

  CUDA_CHECK(cudaMalloc((void**) &d_workSpaceReal1_,
                        binCntX_ * binCntY_ * sizeof(cufftReal)));

  CUDA_CHECK(cudaMalloc((void**) &d_workSpaceReal2_,
                        binCntX_ * binCntY_ * sizeof(cufftReal)));

  CUDA_CHECK(cudaMalloc((void**) &d_workSpaceReal3_,
                        binCntX_ * binCntY_ * sizeof(cufftReal)));

  CUDA_CHECK(cudaMalloc((void**) &d_workSpaceComplex_,
                        (binCntX_ / 2 + 1) * binCntY_ * sizeof(cufftComplex)));

  // expk
  // For DCT2D
  CUDA_CHECK(cudaMalloc((void**) &d_expkM_,
                        (binCntY_ / 2 + 1) * sizeof(cufftComplex)));

  CUDA_CHECK(cudaMalloc((void**) &d_expkN_,
                        (binCntX_ / 2 + 1) * sizeof(cufftComplex)));

  // For IDCT2D & IDXST_IDCT & IDCT_IDXST
  CUDA_CHECK(cudaMalloc((void**) &d_expkMForInverse_,
                        (binCntY_) * sizeof(cufftComplex)));

  CUDA_CHECK(cudaMalloc((void**) &d_expkNForInverse_,
                        (binCntX_ / 2 + 1) * sizeof(cufftComplex)));

  CUDA_CHECK(cudaMalloc((void**) &d_expkMN1_,
                        (binCntX_ + binCntY_) * sizeof(cufftComplex)));

  CUDA_CHECK(cudaMalloc((void**) &d_expkMN2_,
                        (binCntX_ + binCntY_) * sizeof(cufftComplex)));

  // For Input For IDXST_IDCT & IDCT_IDXST
  CUDA_CHECK(cudaMalloc((void**) &d_inputForX_,
                        binCntX_ * binCntY_ * sizeof(cufftReal)));

  CUDA_CHECK(cudaMalloc((void**) &d_inputForY_,
                        binCntX_ * binCntY_ * sizeof(cufftReal)));

  auto M = binCntY_, N = binCntX_;
  auto expkM = d_expkM_, expkN = d_expkN_;
  Kokkos::parallel_for(std::max(binCntX_, binCntY_), KOKKOS_LAMBDA (const int tID) {
    if (tID <= M / 2) {
      int hID = tID;
      cufftComplex W_h_4M = make_float2(Kokkos::cosf((float) PI * hID / (2 * M)),
                                        -Kokkos::sinf((float) PI * hID / (M * 2)));
      expkM[hID] = W_h_4M;
    }
    if (tID <= N / 2) {
      int wid = tID;
      cufftComplex W_w_4N = make_float2(Kokkos::cosf((float) PI * wid / (2 * N)),
                                        -Kokkos::sinf((float) PI * wid / (N * 2)));
      expkN[wid] = W_w_4N;
    }
  });

  auto expkMForInverse = d_expkMForInverse_, expkNForInverse = d_expkNForInverse_;
  auto expkMN_1 = d_expkMN1_, expkMN_2 = d_expkMN2_;
  Kokkos::parallel_for(std::max(binCntX_, binCntY_), KOKKOS_LAMBDA (const int tid) {
      if (tid < M) {
      int hid = tid;
      cufftComplex W_h_4M = make_float2(Kokkos::cosf((float) PI * hid / (2 * M)),
                                        -Kokkos::sinf((float) PI * hid / (M * 2)));
      expkMForInverse[hid] = W_h_4M;
      // expkMN_1
      cufftComplex W_h_4M_offset
          = make_float2(Kokkos::cosf((float) PI * (hid + M) / (2 * M)),
                        -Kokkos::sinf((float) PI * (hid + M) / (M * 2)));
      expkMN_1[hid] = W_h_4M;
      expkMN_1[hid + M] = W_h_4M_offset;

      // expkMN_2
      W_h_4M = make_float2(-Kokkos::sinf((float) PI * (hid - (N - 1)) / (M * 2)),
                           -Kokkos::cosf((float) PI * (hid - (N - 1)) / (2 * M)));

      W_h_4M_offset
          = make_float2(-Kokkos::sinf((float) PI * (hid - (N - 1) + M) / (M * 2)),
                        -Kokkos::cosf((float) PI * (hid - (N - 1) + M) / (2 * M)));
      expkMN_2[hid] = W_h_4M;
      expkMN_2[hid + M] = W_h_4M_offset;
    }
    if (tid <= N / 2) {
      int wid = tid;
      cufftComplex W_w_4N = make_float2(Kokkos::cosf((float) PI * wid / (2 * N)),
                                        -Kokkos::sinf((float) PI * wid / (N * 2)));
      expkNForInverse[wid] = W_w_4N;
    }
  }); 

  cufftPlan2d(&plan_, binCntY_, binCntX_, CUFFT_R2C);
  cufftPlan2d(&planInverse_, binCntY_, binCntX_, CUFFT_C2R);
}

void PoissonSolver::freeCUDAKernel()
{
  CUDA_CHECK(cudaFree(d_binDensity_));
  CUDA_CHECK(cudaFree(d_auv_));
  CUDA_CHECK(cudaFree(d_potential_));

  CUDA_CHECK(cudaFree(d_efX_));
  CUDA_CHECK(cudaFree(d_efY_));

  CUDA_CHECK(cudaFree(d_workSpaceReal1_));
  CUDA_CHECK(cudaFree(d_workSpaceReal2_));
  CUDA_CHECK(cudaFree(d_workSpaceReal3_));

  CUDA_CHECK(cudaFree(d_workSpaceComplex_));

  CUDA_CHECK(cudaFree(d_expkN_));
  CUDA_CHECK(cudaFree(d_expkM_));

  CUDA_CHECK(cudaFree(d_expkNForInverse_));
  CUDA_CHECK(cudaFree(d_expkMForInverse_));

  CUDA_CHECK(cudaFree(d_expkMN1_));
  CUDA_CHECK(cudaFree(d_expkMN2_));

  CUDA_CHECK(cudaFree(d_inputForX_));
  CUDA_CHECK(cudaFree(d_inputForY_));

  cufftDestroy(plan_);
  cufftDestroy(planInverse_);
}

};  // namespace gpl2
