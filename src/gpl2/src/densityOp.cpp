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
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <memory>

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
      // instance information
      numInsts_(0),
      sumOverflow_(0)
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

  initDeviceMemory();
  logger_->report("[DensityOp] Initialization Succeed.");
}

/////////////////////////////////////////////////////////
// Class Density Op

void DensityOp::initDeviceMemory()
{
  // Initialize the bin related information
  // allocate memory on device side
  dBinLx_ = Kokkos::View<int*>("BinLx", numBins_);
  auto hBinLx = Kokkos::create_mirror_view(dBinLx_);
  dBinLy_ = Kokkos::View<int*>("BinLy", numBins_);
  auto hBinLy = Kokkos::create_mirror_view(dBinLy_);
  dBinUx_ = Kokkos::View<int*>("BinUx", numBins_);
  auto hBinUx = Kokkos::create_mirror_view(dBinUx_);
  dBinUy_ = Kokkos::View<int*>("BinUy", numBins_);
  auto hBinUy = Kokkos::create_mirror_view(dBinUy_);
  dBinTargetDensity_ = Kokkos::View<float*>("BinTargetDensity", numBins_);
  auto hBinTargetDensity = Kokkos::create_mirror_view(dBinTargetDensity_);

  dBinNonPlaceArea_ = Kokkos::View<int64_t_cu*>("BinNonPlaceArea", numBins_);
  auto hBinNonPlaceArea = Kokkos::create_mirror_view(dBinNonPlaceArea_);
  dBinInstPlacedArea_ = Kokkos::View<int64_t_cu*>("BinInstPlacedArea", numBins_);
  auto hBinInstPlacedArea = Kokkos::create_mirror_view(dBinInstPlacedArea_);
  dBinFillerArea_ = Kokkos::View<int64_t_cu*>("BinFillerArea", numBins_);
  auto hBinFillerArea = Kokkos::create_mirror_view(dBinFillerArea_);
  dBinScaledArea_ = Kokkos::View<float*>("BinScaledArea", numBins_);
  auto hBinScaledArea = Kokkos::create_mirror_view(dBinScaledArea_);
  dBinOverflowArea_ = Kokkos::View<float*>("BinOverflowArea", numBins_);
  auto hBinOverflowArea = Kokkos::create_mirror_view(dBinOverflowArea_);

  dBinDensity_ = Kokkos::View<float*>("BinDensity", numBins_);
  auto hBinDensity = Kokkos::create_mirror_view(dBinDensity_);
  dBinElectroPhi_ = Kokkos::View<float*>("BinElectroPhi", numBins_);
  auto hBinElectroPhi = Kokkos::create_mirror_view(dBinElectroPhi_);
  dBinElectroForceX_ = Kokkos::View<float*>("BinElectroForceX", numBins_);
  auto hBinElectroForceX = Kokkos::create_mirror_view(dBinElectroForceX_);
  dBinElectroForceY_ = Kokkos::View<float*>("BinElectroForceY", numBins_);
  auto hBinElectroForceY = Kokkos::create_mirror_view(dBinElectroForceY_);

  // initialize
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

  // copy from host to device
  Kokkos::deep_copy(dBinLx_, hBinLx);
  Kokkos::deep_copy(dBinLy_, hBinLy);
  Kokkos::deep_copy(dBinUx_, hBinUx);
  Kokkos::deep_copy(dBinUy_, hBinUy);
  Kokkos::deep_copy(dBinNonPlaceArea_, hBinNonPlaceArea);
  Kokkos::deep_copy(dBinScaledArea_, hBinScaledArea);
  Kokkos::deep_copy(dBinTargetDensity_, hBinTargetDensity);

  // Initialize the instance related information
  // allocate memory on device side
  dGCellDensityWidth_ = Kokkos::View<int*>("GCellDensityWidth", numInsts_);
  auto hGCellDensityWidth = Kokkos::create_mirror_view(dGCellDensityWidth_);
  dGCellDensityHeight_ = Kokkos::View<int*>("GCellDensityHeight", numInsts_);
  auto hGCellDensityHeight = Kokkos::create_mirror_view(dGCellDensityHeight_);
  dGCellDensityScale_ = Kokkos::View<float*>("GCellDensityScale", numInsts_);
  auto hGCellDensityScale = Kokkos::create_mirror_view(dGCellDensityScale_);
  dGCellIsFiller_ = Kokkos::View<bool*>("GCellIsFiller", numInsts_);
  auto hGCellIsFiller = Kokkos::create_mirror_view(dGCellIsFiller_);
  dGCellIsMacro_ = Kokkos::View<bool*>("GCellIsMacro", numInsts_);
  auto hGCellIsMacro = Kokkos::create_mirror_view(dGCellIsMacro_);

  dGCellDCx_ = Kokkos::View<int*>("GCellDCx", numInsts_);
  auto hGCellDCx = Kokkos::create_mirror_view(dGCellDCx_);
  dGCellDCy_ = Kokkos::View<int*>("GCellDCy", numInsts_);
  auto hGCellDCy = Kokkos::create_mirror_view(dGCellDCy_);

  // initialize
  int instIdx = 0;
  for (auto& inst : pb_->insts()) {
    hGCellDensityWidth[instIdx] = inst->dDx();
    hGCellDensityHeight[instIdx] = inst->dDy();
    hGCellDensityScale[instIdx] = inst->densityScale();
    hGCellIsFiller[instIdx] = inst->isFiller();
    hGCellIsMacro[instIdx] = inst->isMacro();
    instIdx++;
  }

  // copy from host to device
  Kokkos::deep_copy(dGCellDensityWidth_, hGCellDensityWidth);
  Kokkos::deep_copy(dGCellDensityHeight_, hGCellDensityHeight);
  Kokkos::deep_copy(dGCellDensityScale_, hGCellDensityScale);
  Kokkos::deep_copy(dGCellIsFiller_, hGCellIsFiller);
  Kokkos::deep_copy(dGCellIsMacro_, hGCellIsMacro);
}

DEVICE_FUNC inline IntRect getMinMaxIdxXY(const int numBins,
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
DEVICE_FUNC inline float getOverlapWidth(const float& instDLx,
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
  auto dBinInstPlacedArea = dBinInstPlacedArea_, dBinFillerArea = dBinFillerArea_;
  Kokkos::parallel_for(numBins_, KOKKOS_LAMBDA (const int binIdx) {
    dBinInstPlacedArea[binIdx] = 0;
    dBinFillerArea[binIdx] = 0;
  });

  // Step 2: compute the overlap between bin and instance
  auto numBins = numBins_, binSizeX = binSizeX_, binSizeY = binSizeY_, binCntX = binCntX_, binCntY = binCntY_,
      coreLx = coreLx_, coreLy = coreLy_, coreUx = coreUx_, coreUy = coreUy_;
  auto dGCellDCx = dGCellDCx_, dGCellDCy = dGCellDCy_, dGCellDensityWidth = dGCellDensityWidth_,
       dGCellDensityHeight = dGCellDensityHeight_, dBinLx = dBinLx_, dBinLy = dBinLy_,
       dBinUx = dBinUx_, dBinUy = dBinUy_;
  auto dGCellDensityScale = dGCellDensityScale_, dBinTargetDensity = dBinTargetDensity_;
  auto dGCellIsFiller = dGCellIsFiller_, dGCellIsMacro = dGCellIsMacro_;
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
                                     dGCellDCx[instIdx],
                                     dGCellDCy[instIdx],
                                     dGCellDensityWidth[instIdx],
                                     dGCellDensityHeight[instIdx]);

    for (int i = binRect.lx; i < binRect.ux; i++) {
      for (int j = binRect.ly; j < binRect.uy; j++) {
        const int binIdx = j * binCntX + i;
        const float instDLx
            = dGCellDCx[instIdx] - dGCellDensityWidth[instIdx] / 2;
        const float instDLy
            = dGCellDCy[instIdx] - dGCellDensityHeight[instIdx] / 2;
        const float instDUx
            = dGCellDCx[instIdx] + dGCellDensityWidth[instIdx] / 2;
        const float instDUy
            = dGCellDCy[instIdx] + dGCellDensityHeight[instIdx] / 2;
        const float overlapWidth = getOverlapWidth(
            instDLx, instDUx, dBinLx[binIdx], dBinUx[binIdx]);
        const float overlapHeight = getOverlapWidth(
            instDLy, instDUy, dBinLy[binIdx], dBinUy[binIdx]);
        float overlapArea
            = overlapWidth * overlapHeight * dGCellDensityScale[instIdx];
        // Atomic addition is used to safely update each bin's value in the
        // global grid array to account for the area occupied by the instance.
        // This ensures that updates from different threads don’t interfere with
        // each other, providing a correct total even when multiple threads
        // update the same bin simultaneously.
        if (dGCellIsFiller[instIdx]) {
          Kokkos::atomic_add(&dBinFillerArea[binIdx],
                    static_cast<int64_t_cu>(overlapArea));
        } else {
          if (dGCellIsMacro[instIdx]) {
            overlapArea = overlapArea * dBinTargetDensity[binIdx];
          }
          Kokkos::atomic_add(&dBinInstPlacedArea[binIdx],
                    static_cast<int64_t_cu>(overlapArea));
        }
      }
    }
  });

  // Step 3: update overflow
  auto dBinDensity = dBinDensity_, dBinOverflowArea = dBinOverflowArea_;
  auto dBinNonPlaceArea = dBinNonPlaceArea_;
  auto dBinScaledArea = dBinScaledArea_;
  Kokkos::parallel_for(numBins_, KOKKOS_LAMBDA (const int binIdx) {
    dBinDensity[binIdx]
        = (static_cast<float>(dBinNonPlaceArea[binIdx])
           + static_cast<float>(dBinInstPlacedArea[binIdx])
           + static_cast<float>(dBinFillerArea[binIdx]))
          / dBinScaledArea[binIdx];

    dBinOverflowArea[binIdx]
        = max(0.0,
              static_cast<float>(dBinInstPlacedArea[binIdx])
                  + static_cast<float>(dBinNonPlaceArea[binIdx])
                  - dBinScaledArea[binIdx]);
  });

  auto begin = thrust::device_ptr<float>(dBinOverflowArea_.data());
  auto end = begin + dBinOverflowArea_.size();
  sumOverflow_ = thrust::reduce(begin, end, 0.0, thrust::plus<float>());

  // Step 4: solve the poisson equation
  fft_->solvePoisson(dBinDensity_.data(),
                     dBinElectroPhi_.data(),
                     dBinElectroForceX_.data(),
                     dBinElectroForceY_.data());
}

void DensityOp::getDensityGradient(float* densityGradientX,
                                   float* densityGradientY)
{
  // Step 5: Compute electro force for each instance
  auto numBins = numBins_, binSizeX = binSizeX_, binSizeY = binSizeY_, binCntX = binCntX_, binCntY = binCntY_,
      coreLx = coreLx_, coreLy = coreLy_, coreUx = coreUx_, coreUy = coreUy_;
  auto dGCellDCx = dGCellDCx_, dGCellDCy = dGCellDCy_, dGCellDensityWidth = dGCellDensityWidth_,
       dGCellDensityHeight = dGCellDensityHeight_, dBinLx = dBinLx_, dBinLy = dBinLy_,
       dBinUx = dBinUx_, dBinUy = dBinUy_;
  auto dGCellDensityScale = dGCellDensityScale_;
  auto dBinElectroForceX = dBinElectroForceX_, dBinElectroForceY = dBinElectroForceY_;
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
                                     dGCellDCx[instIdx],
                                     dGCellDCy[instIdx],
                                     dGCellDensityWidth[instIdx],
                                     dGCellDensityHeight[instIdx]);

    float electroForceSumX = 0.0;
    float electroForceSumY = 0.0;

    for (int i = binRect.lx; i < binRect.ux; i++) {
      for (int j = binRect.ly; j < binRect.uy; j++) {
        const int binIdx = j * binCntX + i;
        const float instDLx
            = dGCellDCx[instIdx] - dGCellDensityWidth[instIdx] / 2;
        const float instDLy
            = dGCellDCy[instIdx] - dGCellDensityHeight[instIdx] / 2;
        const float instDUx
            = dGCellDCx[instIdx] + dGCellDensityWidth[instIdx] / 2;
        const float instDUy
            = dGCellDCy[instIdx] + dGCellDensityHeight[instIdx] / 2;
        const float overlapWidth = getOverlapWidth(
            instDLx, instDUx, dBinLx[binIdx], dBinUx[binIdx]);
        const float overlapHeight = getOverlapWidth(
            instDLy, instDUy, dBinLy[binIdx], dBinUy[binIdx]);
        const float overlapArea
            = overlapWidth * overlapHeight * dGCellDensityScale[instIdx];
        electroForceSumX += 0.5 * overlapArea * dBinElectroForceX[binIdx];
        electroForceSumY += 0.5 * overlapArea * dBinElectroForceY[binIdx];
      }
    }

    densityGradientX[instIdx] = electroForceSumX;
    densityGradientY[instIdx] = electroForceSumY;
  });
}

void DensityOp::updateGCellLocation(const int* instDCx, const int* instDCy)
{
  auto dGCellDCx = dGCellDCx_, dGCellDCy = dGCellDCy_;
  Kokkos::parallel_for(numInsts_, KOKKOS_LAMBDA (const int instIdx) {
    dGCellDCx[instIdx] = instDCx[instIdx];
    dGCellDCy[instIdx] = instDCy[instIdx];
  });
}

}  // namespace gpl2
