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
///////////////////////////////////////////////////////////////////////////////

#include "wirelengthOp.h"

#include "placerBase.h"
#include "placerObjects.h"

#include <Kokkos_Core.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <climits>
#include <cmath>
#include <vector>

namespace gpl2 {

//////////////////////////////////////////////////////////////
// Class WirelengthOp

WirelengthOp::WirelengthOp()
    : pbc_(nullptr),
      logger_(nullptr),
      numInsts_(0),
      numPins_(0),
      numNets_(0),
      numPlaceInsts_(0),
      dInstPinIdxPtr_(nullptr),
      dInstPinPosPtr_(nullptr),
      dPinInstIdPtr_(nullptr),
      dNetPinIdxPtr_(nullptr),
      dNetPinPosPtr_(nullptr),
      dPinNetIdPtr_(nullptr),
      // pin information
      dPinXPtr_(nullptr),
      dPinYPtr_(nullptr),
      dPinOffsetXPtr_(nullptr),
      dPinOffsetYPtr_(nullptr),
      dPinGradXPtr_(nullptr),
      dPinGradYPtr_(nullptr),
      dPinAPosXPtr_(nullptr),
      dPinAPosYPtr_(nullptr),
      dPinANegXPtr_(nullptr),
      dPinANegYPtr_(nullptr),
      // net information
      dNetWidthPtr_(nullptr),
      dNetHeightPtr_(nullptr),
      dNetLxPtr_(nullptr),
      dNetLyPtr_(nullptr),
      dNetUxPtr_(nullptr),
      dNetUyPtr_(nullptr),
      dNetWeightPtr_(nullptr),
      dNetVirtualWeightPtr_(nullptr),
      dNetBPosXPtr_(nullptr),
      dNetBPosYPtr_(nullptr),
      dNetBNegXPtr_(nullptr),
      dNetBNegYPtr_(nullptr),
      dNetCPosXPtr_(nullptr),
      dNetCPosYPtr_(nullptr),
      dNetCNegXPtr_(nullptr),
      dNetCNegYPtr_(nullptr)
{
}

WirelengthOp::WirelengthOp(PlacerBaseCommon* pbc) : WirelengthOp()
{
  pbc_ = pbc;
  logger_ = pbc_->logger();
  logger_->report("[WirelengthOp] Start Initialization.");

  // placeable instances + fixed instances
  numInsts_ = pbc_->numInsts();
  numPlaceInsts_ = pbc_->numPlaceInsts();
  numPins_ = pbc_->numPins();
  numNets_ = pbc_->numNets();

  initCUDAKernel();
  logger_->report("[WirelengthOp] Initialization Succeed.");
}

WirelengthOp::~WirelengthOp()
{
  freeCUDAKernel();
}

/////////////////////////////////////////////////////////
// Class WirelengthOp
void WirelengthOp::initCUDAKernel()
{
  // Initialize related information
  std::vector<int> hInstPinIdx;
  thrust::host_vector<int> hInstPinPos(numInsts_ + 1);
  thrust::host_vector<int> hPinInstId(numPins_);

  std::vector<int> hNetPinIdx;
  thrust::host_vector<int> hNetPinPos(numNets_ + 1);
  thrust::host_vector<int> hPinNetId(numPins_);

  thrust::host_vector<float> hNetWeight(numNets_);
  thrust::host_vector<float> hNetVirtualWeight(numNets_);

  int pinIdx = 0;
  for (auto pin : pbc_->pins()) {
    hPinInstId[pinIdx] = pin->instId();
    hPinNetId[pinIdx] = pin->netId();
    pinIdx++;
  }

  int instIdx = 0;
  hInstPinPos[0] = 0;
  for (auto& inst : pbc_->insts()) {
    for (auto& pin : inst->pins()) {
      hInstPinIdx.push_back(pin->pinId());
    }
    hInstPinPos[instIdx + 1] = hInstPinPos[instIdx] + inst->numPins();
    instIdx++;
  }

  int netIdx = 0;
  hNetPinPos[0] = 0;
  for (auto& net : pbc_->nets()) {
    for (auto& pin : net->pins()) {
      hNetPinIdx.push_back(pin->pinId());
    }

    hNetWeight[netIdx] = net->weight();
    hNetVirtualWeight[netIdx] = net->virtualWeight();
    hNetPinPos[netIdx + 1] = hNetPinPos[netIdx] + net->numPins();
    netIdx++;
  }

  // Allocate memory on the device side
  dInstPinIdxPtr_ = setThrustVector<int>(hInstPinIdx.size(), dInstPinIdx_);
  dInstPinPosPtr_ = setThrustVector<int>(numInsts_ + 1, dInstPinPos_);
  dPinInstIdPtr_ = setThrustVector<int>(numPins_, dPinInstId_);

  dNetPinIdxPtr_ = setThrustVector<int>(hNetPinIdx.size(), dNetPinIdx_);
  dNetWeightPtr_ = setThrustVector<float>(numNets_, dNetWeight_);
  dNetVirtualWeightPtr_ = setThrustVector<float>(numNets_, dNetVirtualWeight_);
  dNetPinPosPtr_ = setThrustVector<int>(numNets_ + 1, dNetPinPos_);
  dPinNetIdPtr_ = setThrustVector<int>(numPins_, dPinNetId_);

  // copy from host to device
  thrust::copy(hInstPinIdx.begin(), hInstPinIdx.end(), dInstPinIdx_.begin());
  thrust::copy(hInstPinPos.begin(), hInstPinPos.end(), dInstPinPos_.begin());
  thrust::copy(hPinInstId.begin(), hPinInstId.end(), dPinInstId_.begin());

  thrust::copy(hNetWeight.begin(), hNetWeight.end(), dNetWeight_.begin());
  thrust::copy(hNetVirtualWeight.begin(),
               hNetVirtualWeight.end(),
               dNetVirtualWeight_.begin());

  thrust::copy(hNetPinIdx.begin(), hNetPinIdx.end(), dNetPinIdx_.begin());
  thrust::copy(hNetPinPos.begin(), hNetPinPos.end(), dNetPinPos_.begin());
  thrust::copy(hPinNetId.begin(), hPinNetId.end(), dPinNetId_.begin());

  // Check the pin information
  thrust::host_vector<int> hPinX(numPins_);
  thrust::host_vector<int> hPinY(numPins_);
  thrust::host_vector<int> hPinOffsetX(numPins_);
  thrust::host_vector<int> hPinOffsetY(numPins_);

  // This is for fixed instances
  for (auto& pin : pbc_->pins()) {
    const int pinId = pin->pinId();
    hPinX[pinId] = pin->cx();
    hPinY[pinId] = pin->cy();
    hPinOffsetX[pinId] = pin->offsetCx();
    hPinOffsetY[pinId] = pin->offsetCy();
  }

  // allocate memory on the device side
  dPinXPtr_ = setThrustVector<int>(numPins_, dPinX_);
  dPinYPtr_ = setThrustVector<int>(numPins_, dPinY_);
  dPinOffsetXPtr_ = setThrustVector<int>(numPins_, dPinOffsetX_);
  dPinOffsetYPtr_ = setThrustVector<int>(numPins_, dPinOffsetY_);
  dPinGradXPtr_ = setThrustVector<float>(numPins_, dPinGradX_);
  dPinGradYPtr_ = setThrustVector<float>(numPins_, dPinGradY_);

  dPinAPosXPtr_ = setThrustVector<float>(numPins_, dPinAPosX_);
  dPinANegXPtr_ = setThrustVector<float>(numPins_, dPinANegX_);
  dPinAPosYPtr_ = setThrustVector<float>(numPins_, dPinAPosY_);
  dPinANegYPtr_ = setThrustVector<float>(numPins_, dPinANegY_);
  dNetBPosXPtr_ = setThrustVector<float>(numNets_, dNetBPosX_);
  dNetBNegXPtr_ = setThrustVector<float>(numNets_, dNetBNegX_);
  dNetBPosYPtr_ = setThrustVector<float>(numNets_, dNetBPosY_);
  dNetBNegYPtr_ = setThrustVector<float>(numNets_, dNetBNegY_);
  dNetCPosXPtr_ = setThrustVector<float>(numNets_, dNetCPosX_);
  dNetCNegXPtr_ = setThrustVector<float>(numNets_, dNetCNegX_);
  dNetCPosYPtr_ = setThrustVector<float>(numNets_, dNetCPosY_);
  dNetCNegYPtr_ = setThrustVector<float>(numNets_, dNetCNegY_);

  dNetLxPtr_ = setThrustVector<int>(numNets_, dNetLx_);
  dNetLyPtr_ = setThrustVector<int>(numNets_, dNetLy_);
  dNetUxPtr_ = setThrustVector<int>(numNets_, dNetUx_);
  dNetUyPtr_ = setThrustVector<int>(numNets_, dNetUy_);
  dNetWidthPtr_ = setThrustVector<int>(numNets_, dNetWidth_);
  dNetHeightPtr_ = setThrustVector<int>(numNets_, dNetHeight_);

  // copy from host to device
  thrust::copy(hPinX.begin(), hPinX.end(), dPinX_.begin());
  thrust::copy(hPinY.begin(), hPinY.end(), dPinY_.begin());
  thrust::copy(hPinOffsetX.begin(), hPinOffsetX.end(), dPinOffsetX_.begin());
  thrust::copy(hPinOffsetY.begin(), hPinOffsetY.end(), dPinOffsetY_.begin());
}

void WirelengthOp::freeCUDAKernel()
{
  numInsts_ = 0;
  numPins_ = 0;
  numNets_ = 0;
  numPlaceInsts_ = 0;

  pbc_ = nullptr;
  logger_ = nullptr;

  dInstPinIdxPtr_ = nullptr;
  dInstPinPosPtr_ = nullptr;
  dPinInstIdPtr_ = nullptr;

  dNetPinIdxPtr_ = nullptr;
  dNetPinPosPtr_ = nullptr;
  dPinNetIdPtr_ = nullptr;

  dPinXPtr_ = nullptr;
  dPinYPtr_ = nullptr;
  dPinOffsetXPtr_ = nullptr;
  dPinOffsetYPtr_ = nullptr;
  dPinGradXPtr_ = nullptr;
  dPinGradYPtr_ = nullptr;
  dPinAPosXPtr_ = nullptr;
  dPinANegXPtr_ = nullptr;
  dPinAPosYPtr_ = nullptr;
  dPinANegYPtr_ = nullptr;

  dNetWidthPtr_ = nullptr;
  dNetHeightPtr_ = nullptr;
  dNetLxPtr_ = nullptr;
  dNetLyPtr_ = nullptr;
  dNetUxPtr_ = nullptr;
  dNetUyPtr_ = nullptr;
  dNetWeightPtr_ = nullptr;
  dNetVirtualWeightPtr_ = nullptr;
  dNetBPosXPtr_ = nullptr;
  dNetBNegXPtr_ = nullptr;
  dNetBPosYPtr_ = nullptr;
  dNetBNegYPtr_ = nullptr;
  dNetCPosXPtr_ = nullptr;
  dNetCNegXPtr_ = nullptr;
  dNetCPosYPtr_ = nullptr;
  dNetCNegYPtr_ = nullptr;
}

void WirelengthOp::updatePinLocation(const int* instDCx, const int* instDCy)
{
  auto dInstPinPosPtr = dInstPinPosPtr_, dInstPinIdxPtr = dInstPinIdxPtr_, dPinXPtr = dPinXPtr_, dPinYPtr = dPinYPtr_,
       dPinOffsetXPtr = dPinOffsetXPtr_, dPinOffsetYPtr = dPinOffsetYPtr_;
  Kokkos::parallel_for(numPlaceInsts_, KOKKOS_LAMBDA (const int instId) {
    const int pinStart = dInstPinPosPtr[instId];
    const int pinEnd = dInstPinPosPtr[instId + 1];
    const float instDCxVal = instDCx[instId];
    const float instDCyVal = instDCy[instId];
    for (int pinId = pinStart; pinId < pinEnd; ++pinId) {
      const int pinIdx = dInstPinIdxPtr[pinId];
      dPinXPtr[pinIdx] = instDCxVal + dPinOffsetXPtr[pinIdx];
      dPinYPtr[pinIdx] = instDCyVal + dPinOffsetYPtr[pinIdx];
    }
  });

  auto dNetPinPosPtr = dNetPinPosPtr_, dNetPinIdxPtr = dNetPinIdxPtr_, dNetLxPtr = dNetLxPtr_, dNetLyPtr = dNetLyPtr_,
       dNetUxPtr = dNetUxPtr_, dNetUyPtr = dNetUyPtr_, dNetWidthPtr = dNetWidthPtr_, dNetHeightPtr = dNetHeightPtr_;
  Kokkos::parallel_for(numNets_, KOKKOS_LAMBDA (const int netId) {
    const int pinStart = dNetPinPosPtr[netId];
    const int pinEnd = dNetPinPosPtr[netId + 1];
    int netLx = INT_MAX;
    int netLy = INT_MAX;
    int netUx = 0;
    int netUy = 0;
    for (int pinId = pinStart; pinId < pinEnd; ++pinId) {
      const int pinIdx = dNetPinIdxPtr[pinId];
      const int pinX = dPinXPtr[pinIdx];
      const int pinY = dPinYPtr[pinIdx];
      netLx = min(netLx, pinX);
      netLy = min(netLy, pinY);
      netUx = max(netUx, pinX);
      netUy = max(netUy, pinY);
    }

    if (netLx > netUx || netLy > netUy) {
      netLx = 0;
      netUx = 0;
      netLy = 0;
      netUy = 0;
    }

    dNetLxPtr[netId] = netLx;
    dNetLyPtr[netId] = netLy;
    dNetUxPtr[netId] = netUx;
    dNetUyPtr[netId] = netUy;
    dNetWidthPtr[netId] = netUx - netLx;
    dNetHeightPtr[netId] = netUy - netLy;
  });
}

struct TypeConvertor
{
  __host__ __device__ int64_t operator()(const int& x) const
  {
    return static_cast<int64_t>(x);
  }
};

int64_t WirelengthOp::computeHPWL()
{
  int64_t hpwl = 0;
  hpwl = thrust::transform_reduce(dNetWidth_.begin(),
                                  dNetWidth_.end(),
                                  TypeConvertor(),
                                  hpwl,
                                  thrust::plus<int64_t>());

  hpwl = thrust::transform_reduce(dNetHeight_.begin(),
                                  dNetHeight_.end(),
                                  TypeConvertor(),
                                  hpwl,
                                  thrust::plus<int64_t>());
  return hpwl;
}

struct WeightHPWLFunctor
{
  float virtualWeightFactor_;

  WeightHPWLFunctor(float virtualWeightFactor)
      : virtualWeightFactor_(virtualWeightFactor)
  {
  }

  __host__ __device__ int64_t
  operator()(const thrust::tuple<int, int, float, float>& t) const
  {
    const int width = thrust::get<0>(t);
    const int height = thrust::get<1>(t);
    const float weight = thrust::get<2>(t);
    const float virtualWeight = thrust::get<3>(t);
    const float sumWeight = weight + virtualWeight * virtualWeightFactor_;
    return static_cast<int64_t>(sumWeight * (width + height));
  }
};

int64_t WirelengthOp::computeWeightedHPWL(float virtualWeightFactor)
{
  int64_t hpwl = 0;
  hpwl = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(dNetWidth_.begin(),
                                                   dNetHeight_.begin(),
                                                   dNetWeight_.begin(),
                                                   dNetVirtualWeight_.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(dNetWidth_.end(),
                                                   dNetHeight_.end(),
                                                   dNetWeight_.end(),
                                                   dNetVirtualWeight_.end())),
      WeightHPWLFunctor(virtualWeightFactor),
      hpwl,
      thrust::plus<int64_t>());

  return hpwl;
}

void WirelengthOp::computeWireLengthForce(const float wlCoeffX,
                                          const float wlCoeffY,
                                          const float virtualWeightFactor,
                                          float* wirelengthForceX,
                                          float* wirelengthForceY)
{
  auto dPinNetIdPtr = dPinNetIdPtr_, dPinXPtr = dPinXPtr_, dPinYPtr = dPinYPtr_, dNetUxPtr = dNetUxPtr_,
       dNetLxPtr = dNetLxPtr_, dNetUyPtr = dNetUyPtr_, dNetLyPtr = dNetLyPtr_;
  auto dPinAPosXPtr = dPinAPosXPtr_, dPinANegXPtr = dPinANegXPtr_, dPinAPosYPtr = dPinAPosYPtr_,
       dPinANegYPtr = dPinANegYPtr_;
  Kokkos::parallel_for(numPins_, KOKKOS_LAMBDA (const int pinId) {
    const int netId = dPinNetIdPtr[pinId];
    dPinAPosXPtr[pinId] = expf(wlCoeffX * (dPinXPtr[pinId] - dNetUxPtr[netId]));
    dPinANegXPtr[pinId]
        = expf(-1.0 * wlCoeffX * (dPinXPtr[pinId] - dNetLxPtr[netId]));
    dPinAPosYPtr[pinId] = expf(wlCoeffY * (dPinYPtr[pinId] - dNetUyPtr[netId]));
    dPinANegYPtr[pinId]
        = expf(-1.0 * wlCoeffY * (dPinYPtr[pinId] - dNetLyPtr[netId]));
  });

  auto dNetPinPosPtr = dNetPinPosPtr_, dNetPinIdxPtr = dNetPinIdxPtr_, dPinX = dPinXPtr_, dPinY = dPinYPtr_;
  auto dNetBPosXPtr = dNetBPosXPtr_, dNetBNegXPtr = dNetBNegXPtr_, dNetBPosYPtr = dNetBPosYPtr_,
       dNetBNegYPtr = dNetBNegYPtr_, dNetCPosXPtr = dNetCPosXPtr_, dNetCNegXPtr = dNetCNegXPtr_,
       dNetCPosYPtr = dNetCPosYPtr_, dNetCNegYPtr = dNetCNegYPtr_;
  Kokkos::parallel_for(numNets_, KOKKOS_LAMBDA (const int netId) {
    const int pinStart = dNetPinPosPtr[netId];
    const int pinEnd = dNetPinPosPtr[netId + 1];
    float bPosX = 0.0;
    float bNegX = 0.0;
    float bPosY = 0.0;
    float bNegY = 0.0;

    float cPosX = 0.0;
    float cNegX = 0.0;
    float cPosY = 0.0;
    float cNegY = 0.0;

    for (int pinId = pinStart; pinId < pinEnd; ++pinId) {
      const int pinIdx = dNetPinIdxPtr[pinId];
      bPosX += dPinAPosXPtr[pinIdx];
      bNegX += dPinANegXPtr[pinIdx];
      bPosY += dPinAPosYPtr[pinIdx];
      bNegY += dPinANegYPtr[pinIdx];

      cPosX += dPinXPtr[pinIdx] * dPinAPosXPtr[pinIdx];
      cNegX += dPinXPtr[pinIdx] * dPinANegXPtr[pinIdx];
      cPosY += dPinYPtr[pinIdx] * dPinAPosYPtr[pinIdx];
      cNegY += dPinYPtr[pinIdx] * dPinANegYPtr[pinIdx];
    }

    dNetBPosXPtr[netId] = bPosX;
    dNetBNegXPtr[netId] = bNegX;
    dNetBPosYPtr[netId] = bPosY;
    dNetBNegYPtr[netId] = bNegY;

    dNetCPosXPtr[netId] = cPosX;
    dNetCNegXPtr[netId] = cNegX;
    dNetCPosYPtr[netId] = cPosY;
    dNetCNegYPtr[netId] = cNegY;
  });

  auto dPinGradXPtr = dPinGradXPtr_, dPinGradYPtr = dPinGradYPtr_;
  Kokkos::parallel_for(numPlaceInsts_, KOKKOS_LAMBDA (const int pinIdx) {
    const int netId = dPinNetIdPtr[pinIdx];

    // TODO:  if we need to remove high-fanout nets,
    // we can remove it here

    float netBNegX2 = dNetBNegXPtr[netId] * dNetBNegXPtr[netId];
    float netBPosX2 = dNetBPosXPtr[netId] * dNetBPosXPtr[netId];
    float netBNegY2 = dNetBNegYPtr[netId] * dNetBNegYPtr[netId];
    float netBPosY2 = dNetBPosYPtr[netId] * dNetBPosYPtr[netId];

    float pinXWlCoeffX = dPinXPtr[pinIdx] * wlCoeffX;
    float pinYWlCoeffY = dPinYPtr[pinIdx] * wlCoeffY;

    dPinGradXPtr[pinIdx] = ((1.0f - pinXWlCoeffX) * dNetBNegXPtr[netId]
                            + wlCoeffX * dNetCNegXPtr[netId])
                               * dPinANegXPtr[pinIdx] / netBNegX2
                           - ((1.0f + pinXWlCoeffX) * dNetBPosXPtr[netId]
                              - wlCoeffX * dNetCPosXPtr[netId])
                                 * dPinAPosXPtr[pinIdx] / netBPosX2;

    dPinGradYPtr[pinIdx] = ((1.0f - pinYWlCoeffY) * dNetBNegYPtr[netId]
                            + wlCoeffY * dNetCNegYPtr[netId])
                               * dPinANegYPtr[pinIdx] / netBNegY2
                           - ((1.0f + pinYWlCoeffY) * dNetBPosYPtr[netId]
                              - wlCoeffY * dNetCPosYPtr[netId])
                                 * dPinAPosYPtr[pinIdx] / netBPosY2;
  });

  // get the force on each instance
  auto dInstPinPosPtr = dInstPinPosPtr_, dInstPinIdxPtr = dInstPinIdxPtr_;
  auto dNetWeightPtr = dNetWeightPtr_, dNetVirtualWeightPtr = dNetVirtualWeightPtr_;
  Kokkos::parallel_for(numPlaceInsts_, KOKKOS_LAMBDA (const int instId) {
    const int pinStart = dInstPinPosPtr[instId];
    const int pinEnd = dInstPinPosPtr[instId + 1];
    float wlGradX = 0.0;
    float wlGradY = 0.0;
    for (int pinId = pinStart; pinId < pinEnd; ++pinId) {
      const int pinIdx = dInstPinIdxPtr[pinId];
      const int netId = dPinNetIdPtr[pinIdx];
      const float weight = dNetWeightPtr[netId]
                           + dNetVirtualWeightPtr[netId] * virtualWeightFactor;
      wlGradX += dPinGradXPtr[pinIdx] * weight;
      wlGradY += dPinGradYPtr[pinIdx] * weight;
    }

    wirelengthForceX[instId] = wlGradX;
    wirelengthForceY[instId] = wlGradY;
  });
}

}  // namespace gpl2
