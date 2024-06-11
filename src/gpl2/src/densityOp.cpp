///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2018-2023, The Regents of the University of California
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
// ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

#include "densityOp.h"

#include <Kokkos_Core.hpp>

#include <memory>

#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include "placerBase.h"
#include "placerObjects.h"

namespace gpl2 {

//////////////////////////////////////////////////////////////
// Class DensityOp

DensityOp::DensityOp()
    : pb_(nullptr),
      fft_(nullptr),
      logger_(nullptr),
      // Bin information
      numBins_(0),
      binCntX_(0),
      binCntY_(0),
      binSizeX_(0),
      binSizeY_(0),
      // region information
      coreLx_(0),
      coreLy_(0),
      coreUx_(0),
      coreUy_(0),
      // bin pointers
      dBinLxPtr_(nullptr),
      dBinLyPtr_(nullptr),
      dBinUxPtr_(nullptr),
      dBinUyPtr_(nullptr),
      dBinNonPlaceAreaPtr_(nullptr),
      dBinInstPlacedAreaPtr_(nullptr),
      dBinFillerAreaPtr_(nullptr),
      dBinScaledAreaPtr_(nullptr),
      dBinOverflowAreaPtr_(nullptr),
      dBinDensityPtr_(nullptr),
      dBinElectroPhiPtr_(nullptr),
      dBinElectroForceXPtr_(nullptr),
      dBinElectroForceYPtr_(nullptr),
      // instance information
      numInsts_(0),
      sumOverflow_(0),
      dGCellDensityWidthPtr_(nullptr),
      dGCellDensityHeightPtr_(nullptr),
      dGCellDCxPtr_(nullptr),
      dGCellDCyPtr_(nullptr),
      dGCellDensityScalePtr_(nullptr),
      dGCellIsFillerPtr_(nullptr)
{
}

DensityOp::DensityOp(PlacerBase* pb) : DensityOp()
{
  pb_ = pb;
  logger_ = pb_->logger();
  logger_->report("[DensityOp] Start Initialization.");

  numBins_ = pb_->numBins();
  binCntX_ = pb_->binCntX();
  binCntY_ = pb_->binCntY();
  binSizeX_ = pb_->binSizeX();
  binSizeY_ = pb_->binSizeY();

  coreLx_ = pb_->coreLx();
  coreLy_ = pb_->coreLy();
  coreUx_ = pb_->coreUx();
  coreUy_ = pb_->coreUy();

  // placeable insts + filler insts
  numInsts_ = pb_->numInsts();

  // Initialize fft structure based on bins
  fft_ = std::make_unique<PoissonSolver>(
      binCntX_, binCntY_, binSizeX_, binSizeY_);

  initCUDAKernel();
  logger_->report("[DensityOp] Initialization Succeed.");
}

DensityOp::~DensityOp()
{
  freeCUDAKernel();
}

/////////////////////////////////////////////////////////
// Class Density Op

void DensityOp::initCUDAKernel()
{
  // Initialize the bin related information
  // Copy from host to device
  thrust::host_vector<int> hBinLx(numBins_);
  thrust::host_vector<int> hBinLy(numBins_);
  thrust::host_vector<int> hBinUx(numBins_);
  thrust::host_vector<int> hBinUy(numBins_);
  thrust::host_vector<int64_t_cu> hBinNonPlaceArea(numBins_);
  thrust::host_vector<float> hBinScaledArea(numBins_);
  thrust::host_vector<float> hBinTargetDensity(numBins_);

  int binIdx = 0;
  for (auto& bin : pb_->bins()) {
    hBinLx[binIdx] = bin->lx();
    hBinLy[binIdx] = bin->ly();
    hBinUx[binIdx] = bin->ux();
    hBinUy[binIdx] = bin->uy();
    hBinNonPlaceArea[binIdx] = bin->nonPlaceArea();
    hBinScaledArea[binIdx] = bin->area() * bin->targetDensity();
    hBinTargetDensity[binIdx] = bin->targetDensity();
    binIdx++;
  }

  // allocate memory on device side
  dBinLxPtr_ = setThrustVector<int>(numBins_, dBinLx_);
  dBinLyPtr_ = setThrustVector<int>(numBins_, dBinLy_);
  dBinUxPtr_ = setThrustVector<int>(numBins_, dBinUx_);
  dBinUyPtr_ = setThrustVector<int>(numBins_, dBinUy_);
  dBinTargetDensityPtr_ = setThrustVector<float>(numBins_, dBinTargetDensity_);

  dBinNonPlaceAreaPtr_
      = setThrustVector<int64_t_cu>(numBins_, dBinNonPlaceArea_);
  dBinInstPlacedAreaPtr_
      = setThrustVector<int64_t_cu>(numBins_, dBinInstPlacedArea_);
  dBinFillerAreaPtr_ = setThrustVector<int64_t_cu>(numBins_, dBinFillerArea_);
  dBinScaledAreaPtr_ = setThrustVector<float>(numBins_, dBinScaledArea_);
  dBinOverflowAreaPtr_ = setThrustVector<float>(numBins_, dBinOverflowArea_);

  dBinDensityPtr_ = setThrustVector<float>(numBins_, dBinDensity_);
  dBinElectroPhiPtr_ = setThrustVector<float>(numBins_, dBinElectroPhi_);
  dBinElectroForceXPtr_ = setThrustVector<float>(numBins_, dBinElectroForceX_);
  dBinElectroForceYPtr_ = setThrustVector<float>(numBins_, dBinElectroForceY_);

  // copy from host to device
  thrust::copy(hBinLx.begin(), hBinLx.end(), dBinLx_.begin());
  thrust::copy(hBinLy.begin(), hBinLy.end(), dBinLy_.begin());
  thrust::copy(hBinUx.begin(), hBinUx.end(), dBinUx_.begin());
  thrust::copy(hBinUy.begin(), hBinUy.end(), dBinUy_.begin());
  thrust::copy(hBinNonPlaceArea.begin(),
               hBinNonPlaceArea.end(),
               dBinNonPlaceArea_.begin());
  thrust::copy(
      hBinScaledArea.begin(), hBinScaledArea.end(), dBinScaledArea_.begin());
  thrust::copy(hBinTargetDensity.begin(),
               hBinTargetDensity.end(),
               dBinTargetDensity_.begin());

  // Initialize the instance related information
  thrust::host_vector<int> hGCellDensityWidth(numInsts_);
  thrust::host_vector<int> hGCellDensityHeight(numInsts_);
  thrust::host_vector<float> hGCellDensityScale(numInsts_);
  thrust::host_vector<bool> hGCellIsFiller(numInsts_);
  thrust::host_vector<bool> hGCellIsMacro(numInsts_);

  int instIdx = 0;
  for (auto& inst : pb_->insts()) {
    hGCellDensityWidth[instIdx] = inst->dDx();
    hGCellDensityHeight[instIdx] = inst->dDy();
    hGCellDensityScale[instIdx] = inst->densityScale();
    hGCellIsFiller[instIdx] = inst->isFiller();
    hGCellIsMacro[instIdx] = inst->isMacro();
    instIdx++;
  }

  // allocate memory on device side
  dGCellDensityWidthPtr_ = setThrustVector<int>(numInsts_, dGCellDensityWidth_);
  dGCellDensityHeightPtr_
      = setThrustVector<int>(numInsts_, dGCellDensityHeight_);
  dGCellDensityScalePtr_
      = setThrustVector<float>(numInsts_, dGCellDensityScale_);
  dGCellIsFillerPtr_ = setThrustVector<bool>(numInsts_, dGCellIsFiller_);
  dGCellIsMacroPtr_ = setThrustVector<bool>(numInsts_, dGCellIsMacro_);

  dGCellDCxPtr_ = setThrustVector<int>(numInsts_, dGCellDCx_);
  dGCellDCyPtr_ = setThrustVector<int>(numInsts_, dGCellDCy_);

  // copy from host to device
  thrust::copy(hGCellDensityWidth.begin(),
               hGCellDensityWidth.end(),
               dGCellDensityWidth_.begin());
  thrust::copy(hGCellDensityHeight.begin(),
               hGCellDensityHeight.end(),
               dGCellDensityHeight_.begin());
  thrust::copy(hGCellDensityScale.begin(),
               hGCellDensityScale.end(),
               dGCellDensityScale_.begin());
  thrust::copy(
      hGCellIsFiller.begin(), hGCellIsFiller.end(), dGCellIsFiller_.begin());
  thrust::copy(
      hGCellIsMacro.begin(), hGCellIsMacro.end(), dGCellIsMacro_.begin());
}

void DensityOp::freeCUDAKernel()
{
  // since we use thrust::device_vector,
  // we don't need to free the memory explicitly
  pb_ = nullptr;
  fft_ = nullptr;
  logger_ = nullptr;

  dBinLxPtr_ = nullptr;
  dBinLyPtr_ = nullptr;
  dBinUxPtr_ = nullptr;
  dBinUyPtr_ = nullptr;
  dBinTargetDensityPtr_ = nullptr;

  dBinNonPlaceAreaPtr_ = nullptr;
  dBinInstPlacedAreaPtr_ = nullptr;
  dBinFillerAreaPtr_ = nullptr;
  dBinScaledAreaPtr_ = nullptr;
  dBinOverflowAreaPtr_ = nullptr;

  dBinDensityPtr_ = nullptr;
  dBinElectroPhiPtr_ = nullptr;
  dBinElectroForceXPtr_ = nullptr;
  dBinElectroForceYPtr_ = nullptr;

  dGCellDensityWidthPtr_ = nullptr;
  dGCellDensityHeightPtr_ = nullptr;
  dGCellDensityScalePtr_ = nullptr;
  dGCellIsFillerPtr_ = nullptr;
  dGCellIsMacroPtr_ = nullptr;
}

__host__ __device__ inline IntRect getMinMaxIdxXY(const int numBins,
                                         const int binSizeX,
                                         const int binSizeY,
                                         const int binCntX,
                                         const int binCntY,
                                         const int coreLx,
                                         const int coreLy,
                                         const int coreUx,
                                         const int coreUy,
                                         const int instDCx,
                                         const int instDCy,
                                         const float instDDx,
                                         const float instDDy)
{
  IntRect binRect;

  const float lx = instDCx - instDDx / 2;
  const float ly = instDCy - instDDy / 2;
  const float ux = instDCx + instDDx / 2;
  const float uy = instDCy + instDDy / 2;

  int minIdxX = (int) floor((lx - coreLx) / binSizeX);
  int minIdxY = (int) floor((ly - coreLy) / binSizeY);
  int maxIdxX = (int) ceil((ux - coreLx) / binSizeX);
  int maxIdxY = (int) ceil((uy - coreLy) / binSizeY);

  binRect.lx = max(minIdxX, 0);
  binRect.ly = max(minIdxY, 0);
  binRect.ux = min(maxIdxX, binCntX);
  binRect.uy = min(maxIdxY, binCntY);

  return binRect;
}

// Utility functions
__host__ __device__ inline float getOverlapWidth(const float& instDLx,
                                        const float& instDUx,
                                        const float& binLx,
                                        const float& binUx)
{
  if (instDUx <= binLx || instDLx >= binUx) {
    return 0.0;
  } else {
    return min(instDUx, binUx) - max(instDLx, binLx);
  }
}

void DensityOp::updateDensityForceBin()
{
  // Step 1: Initialize the bin density information
  auto dBinInstPlacedAreaPtr = dBinInstPlacedAreaPtr_, dBinFillerAreaPtr = dBinFillerAreaPtr_;
  Kokkos::parallel_for(numBins_, KOKKOS_LAMBDA (const int binIdx) {
    dBinInstPlacedAreaPtr[binIdx] = 0;
    dBinFillerAreaPtr[binIdx] = 0;
  });

  // Step 2: compute the overlap between bin and instance
  auto numBins = numBins_, binSizeX = binSizeX_, binSizeY = binSizeY_, binCntX = binCntX_, binCntY = binCntY_,
      coreLx = coreLx_, coreLy = coreLy_, coreUx = coreUx_, coreUy = coreUy_;
  auto dGCellDCxPtr = dGCellDCxPtr_, dGCellDCyPtr = dGCellDCyPtr_, dGCellDensityWidthPtr = dGCellDensityWidthPtr_,
       dGCellDensityHeightPtr = dGCellDensityHeightPtr_, dBinLxPtr = dBinLxPtr_, dBinLyPtr = dBinLyPtr_,
       dBinUxPtr = dBinUxPtr_, dBinUyPtr = dBinUyPtr_;
  auto dGCellDensityScalePtr = dGCellDensityScalePtr_, dBinTargetDensityPtr = dBinTargetDensityPtr_;
  auto dGCellIsFillerPtr = dGCellIsFillerPtr_, dGCellIsMacroPtr = dGCellIsMacroPtr_;
  Kokkos::parallel_for(numBins_, KOKKOS_LAMBDA (const int instIdx) {
    IntRect binRect = getMinMaxIdxXY(numBins,
                                     binSizeX,
                                     binSizeY,
                                     binCntX,
                                     binCntY,
                                     coreLx,
                                     coreLy,
                                     coreUx,
                                     coreUy,
                                     dGCellDCxPtr[instIdx],
                                     dGCellDCyPtr[instIdx],
                                     dGCellDensityWidthPtr[instIdx],
                                     dGCellDensityHeightPtr[instIdx]);

    for (int i = binRect.lx; i < binRect.ux; i++) {
      for (int j = binRect.ly; j < binRect.uy; j++) {
        const int binIdx = j * binCntX + i;
        const float instDLx
            = dGCellDCxPtr[instIdx] - dGCellDensityWidthPtr[instIdx] / 2;
        const float instDLy
            = dGCellDCyPtr[instIdx] - dGCellDensityHeightPtr[instIdx] / 2;
        const float instDUx
            = dGCellDCxPtr[instIdx] + dGCellDensityWidthPtr[instIdx] / 2;
        const float instDUy
            = dGCellDCyPtr[instIdx] + dGCellDensityHeightPtr[instIdx] / 2;
        const float overlapWidth = getOverlapWidth(
            instDLx, instDUx, dBinLxPtr[binIdx], dBinUxPtr[binIdx]);
        const float overlapHeight = getOverlapWidth(
            instDLy, instDUy, dBinLyPtr[binIdx], dBinUyPtr[binIdx]);
        float overlapArea
            = overlapWidth * overlapHeight * dGCellDensityScalePtr[instIdx];
        // Atomic addition is used to safely update each bin's value in the
        // global grid array to account for the area occupied by the instance.
        // This ensures that updates from different threads don’t interfere with
        // each other, providing a correct total even when multiple threads
        // update the same bin simultaneously.
        if (dGCellIsFillerPtr[instIdx]) {
          Kokkos::atomic_add(&dBinFillerAreaPtr[binIdx],
                    static_cast<int64_t_cu>(overlapArea));
        } else {
          if (dGCellIsMacroPtr[instIdx] == true) {
            overlapArea = overlapArea * dBinTargetDensityPtr[binIdx];
          }
          Kokkos::atomic_add(&dBinInstPlacedAreaPtr[binIdx],
                    static_cast<int64_t_cu>(overlapArea));
        }
      }
    }
  });

  // Step 3: update overflow
  auto dBinDensityPtr = dBinDensityPtr_, dBinOverflowAreaPtr = dBinOverflowAreaPtr_;
  auto dBinNonPlaceAreaPtr = dBinNonPlaceAreaPtr_;
  auto dBinScaledAreaPtr = dBinScaledAreaPtr_;
  Kokkos::parallel_for(numBins_, KOKKOS_LAMBDA (const int binIdx) {
  dBinDensityPtr[binIdx]
        = (static_cast<float>(dBinNonPlaceAreaPtr[binIdx])
           + static_cast<float>(dBinInstPlacedAreaPtr[binIdx])
           + static_cast<float>(dBinFillerAreaPtr[binIdx]))
          / dBinScaledAreaPtr[binIdx];

    dBinOverflowAreaPtr[binIdx]
        = max(0.0,
              static_cast<float>(dBinInstPlacedAreaPtr[binIdx])
                  + static_cast<float>(dBinNonPlaceAreaPtr[binIdx])
                  - dBinScaledAreaPtr[binIdx]);
  });

  sumOverflow_ = thrust::reduce(dBinOverflowArea_.begin(),
                                dBinOverflowArea_.end(),
                                0.0,
                                thrust::plus<float>());

  // Step 4: solve the poisson equation
  fft_->solvePoisson(dBinDensityPtr_,
                     dBinElectroPhiPtr_,
                     dBinElectroForceXPtr_,
                     dBinElectroForceYPtr_);
}

void DensityOp::getDensityGradient(float* densityGradientX,
                                   float* densityGradientY)
{
  // Step 5: Compute electro force for each instance
  auto numBins = numBins_, binSizeX = binSizeX_, binSizeY = binSizeY_, binCntX = binCntX_, binCntY = binCntY_,
      coreLx = coreLx_, coreLy = coreLy_, coreUx = coreUx_, coreUy = coreUy_;
  auto dGCellDCxPtr = dGCellDCxPtr_, dGCellDCyPtr = dGCellDCyPtr_, dGCellDensityWidthPtr = dGCellDensityWidthPtr_,
       dGCellDensityHeightPtr = dGCellDensityHeightPtr_, dBinLxPtr = dBinLxPtr_, dBinLyPtr = dBinLyPtr_,
       dBinUxPtr = dBinUxPtr_, dBinUyPtr = dBinUyPtr_;
  auto dGCellDensityScalePtr = dGCellDensityScalePtr_;
  auto dBinElectroForceXPtr = dBinElectroForceXPtr_, dBinElectroForceYPtr = dBinElectroForceYPtr_;
  Kokkos::parallel_for(numBins_, KOKKOS_LAMBDA (const int instIdx) {
    IntRect binRect = getMinMaxIdxXY(numBins,
                                     binSizeX,
                                     binSizeY,
                                     binCntX,
                                     binCntY,
                                     coreLx,
                                     coreLy,
                                     coreUx,
                                     coreUy,
                                     dGCellDCxPtr[instIdx],
                                     dGCellDCyPtr[instIdx],
                                     dGCellDensityWidthPtr[instIdx],
                                     dGCellDensityHeightPtr[instIdx]);

    float electroForceSumX = 0.0;
    float electroForceSumY = 0.0;

    for (int i = binRect.lx; i < binRect.ux; i++) {
      for (int j = binRect.ly; j < binRect.uy; j++) {
        const int binIdx = j * binCntX + i;
        const float instDLx
            = dGCellDCxPtr[instIdx] - dGCellDensityWidthPtr[instIdx] / 2;
        const float instDLy
            = dGCellDCyPtr[instIdx] - dGCellDensityHeightPtr[instIdx] / 2;
        const float instDUx
            = dGCellDCxPtr[instIdx] + dGCellDensityWidthPtr[instIdx] / 2;
        const float instDUy
            = dGCellDCyPtr[instIdx] + dGCellDensityHeightPtr[instIdx] / 2;
        const float overlapWidth = getOverlapWidth(
            instDLx, instDUx, dBinLxPtr[binIdx], dBinUxPtr[binIdx]);
        const float overlapHeight = getOverlapWidth(
            instDLy, instDUy, dBinLyPtr[binIdx], dBinUyPtr[binIdx]);
        const float overlapArea
            = overlapWidth * overlapHeight * dGCellDensityScalePtr[instIdx];
        electroForceSumX += 0.5 * overlapArea * dBinElectroForceXPtr[binIdx];
        electroForceSumY += 0.5 * overlapArea * dBinElectroForceYPtr[binIdx];
      }
    }

    densityGradientX[instIdx] = electroForceSumX;
    densityGradientY[instIdx] = electroForceSumY;
  });
}

void DensityOp::updateGCellLocation(const int* instDCx, const int* instDCy)
{
  auto dGCellDCxPtr = dGCellDCxPtr_, dGCellDCyPtr = dGCellDCyPtr_;
  Kokkos::parallel_for(numInsts_, KOKKOS_LAMBDA (const int instIdx) {
    dGCellDCxPtr[instIdx] = instDCx[instIdx];
    dGCellDCyPtr[instIdx] = instDCy[instIdx];
  });
}

}  // namespace gpl2
