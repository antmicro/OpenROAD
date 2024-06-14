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
      dInstPinIdx_(nullptr),
      dInstPinPos_(nullptr),
      dPinInstId_(nullptr),
      dNetPinIdx_(nullptr),
      dNetPinPos_(nullptr),
      dPinNetId_(nullptr),
      // pin information
      dPinX_(nullptr),
      dPinY_(nullptr),
      dPinOffsetX_(nullptr),
      dPinOffsetY_(nullptr),
      dPinGradX_(nullptr),
      dPinGradY_(nullptr),
      dPinAPosX_(nullptr),
      dPinAPosY_(nullptr),
      dPinANegX_(nullptr),
      dPinANegY_(nullptr),
      // net information
      dNetWidth_(nullptr),
      dNetHeight_(nullptr),
      dNetLx_(nullptr),
      dNetLy_(nullptr),
      dNetUx_(nullptr),
      dNetUy_(nullptr),
      dNetWeight_(nullptr),
      dNetVirtualWeight_(nullptr),
      dNetBPosX_(nullptr),
      dNetBPosY_(nullptr),
      dNetBNegX_(nullptr),
      dNetBNegY_(nullptr),
      dNetCPosX_(nullptr),
      dNetCPosY_(nullptr),
      dNetCNegX_(nullptr),
      dNetCNegY_(nullptr)
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

  initDeviceMemory();
  logger_->report("[WirelengthOp] Initialization Succeed.");
}

/////////////////////////////////////////////////////////
// Class WirelengthOp
void WirelengthOp::initDeviceMemory()
{
  size_t instPinCount = 0;
  size_t netPinCount = 0;
  for (auto& inst : pbc_->insts()) {
    instPinCount += inst->numPins();
  }
  for (auto& net : pbc_->nets()) {
    netPinCount += net->numPins();
  }

  // Allocate memory on the device side
  dInstPinIdx_ = Kokkos::View<int*>("InstPinIdx", instPinCount);
  auto hInstPinIdx = Kokkos::create_mirror_view(dInstPinIdx_);
  dInstPinPos_ = Kokkos::View<int*>("InstPinPos", numInsts_ + 1);
  auto hInstPinPos = Kokkos::create_mirror_view(dInstPinPos_);
  dPinInstId_ = Kokkos::View<int*>("PinInstId", numPins_);
  auto hPinInstId = Kokkos::create_mirror_view(dPinInstId_);

  dNetPinIdx_ = Kokkos::View<int*>("NetPinIdx", netPinCount);
  auto hNetPinIdx = Kokkos::create_mirror_view(dNetPinIdx_);
  dNetWeight_ = Kokkos::View<float*>("NetWeight", numNets_);
  auto hNetWeight = Kokkos::create_mirror_view(dNetWeight_);
  dNetVirtualWeight_ = Kokkos::View<float*>("NetVirtualWeight", numNets_);
  auto hNetVirtualWeight = Kokkos::create_mirror_view(dNetVirtualWeight_);
  dNetPinPos_ = Kokkos::View<int*>("NetPinPos", numNets_ + 1);
  auto hNetPinPos = Kokkos::create_mirror_view(dNetPinPos_);
  dPinNetId_ = Kokkos::View<int*>("PinNetId", numPins_);
  auto hPinNetId = Kokkos::create_mirror_view(dPinNetId_);

  // Initialize related information
  int pinIdx = 0;
  for (auto pin : pbc_->pins()) {
    hPinInstId[pinIdx] = pin->instId();
    hPinNetId[pinIdx] = pin->netId();
    pinIdx++;
  }

  int instIdx = 0;
  int instPinIdx = 0;
  hInstPinPos[0] = 0;
  for (auto& inst : pbc_->insts()) {
    for (auto& pin : inst->pins()) {
      hInstPinIdx[instPinIdx++] = pin->pinId();
    }
    hInstPinPos[instIdx + 1] = hInstPinPos[instIdx] + inst->numPins();
    instIdx++;
  }

  int netIdx = 0;
  int netPinIdx = 0;
  hNetPinPos[0] = 0;
  for (auto& net : pbc_->nets()) {
    for (auto& pin : net->pins()) {
      hNetPinIdx[netPinIdx++] = pin->pinId();
    }

    hNetWeight[netIdx] = net->weight();
    hNetVirtualWeight[netIdx] = net->virtualWeight();
    hNetPinPos[netIdx + 1] = hNetPinPos[netIdx] + net->numPins();
    netIdx++;
  }

  // copy from host to device
  Kokkos::deep_copy(dInstPinIdx_, hInstPinIdx);
  Kokkos::deep_copy(dInstPinPos_, hInstPinPos);
  Kokkos::deep_copy(dPinInstId_, hPinInstId);

  Kokkos::deep_copy(dNetWeight_, hNetWeight);
  Kokkos::deep_copy(dNetVirtualWeight_, hNetVirtualWeight);

  Kokkos::deep_copy(dNetPinIdx_, hNetPinIdx);
  Kokkos::deep_copy(dNetPinPos_, hNetPinPos);
  Kokkos::deep_copy(dPinNetId_, hPinNetId);

  // allocate memory on the device side
  dPinX_ = Kokkos::View<int*>("PinX", numPins_);
  auto hPinX = Kokkos::create_mirror_view(dPinX_);
  dPinY_ = Kokkos::View<int*>("PinY", numPins_);
  auto hPinY = Kokkos::create_mirror_view(dPinY_);
  dPinOffsetX_ = Kokkos::View<int*>("PinOffsetX", numPins_);
  auto hPinOffsetX = Kokkos::create_mirror_view(dPinOffsetX_);
  dPinOffsetY_ = Kokkos::View<int*>("PinOffsetY", numPins_);
  auto hPinOffsetY = Kokkos::create_mirror_view(dPinOffsetY_);
  dPinGradX_ = Kokkos::View<float*>("PinGradX", numPins_);
  auto hPinGradX = Kokkos::create_mirror_view(dPinGradX_);
  dPinGradY_ = Kokkos::View<float*>("PinGradY", numPins_);
  auto hPinGradY = Kokkos::create_mirror_view(dPinGradY_);

  dPinAPosX_ = Kokkos::View<float*>("PinAPosX", numPins_);
  auto hPinAPosX = Kokkos::create_mirror_view(dPinAPosX_);
  dPinANegX_ = Kokkos::View<float*>("PinANegX", numPins_);
  auto hPinANegX = Kokkos::create_mirror_view(dPinANegX_);
  dPinAPosY_ = Kokkos::View<float*>("PinAPosY", numPins_);
  auto hPinAPosY = Kokkos::create_mirror_view(dPinAPosY_);
  dPinANegY_ = Kokkos::View<float*>("PinANegY", numPins_);
  auto hPinANegY = Kokkos::create_mirror_view(dPinANegY_);
  dNetBPosX_ = Kokkos::View<float*>("NetBPosX", numNets_);
  auto hNetBPosX = Kokkos::create_mirror_view(dNetBPosX_);
  dNetBNegX_ = Kokkos::View<float*>("NetBNegX", numNets_);
  auto hNetBNegX = Kokkos::create_mirror_view(dNetBNegX_);
  dNetBPosY_ = Kokkos::View<float*>("NetBPosY", numNets_);
  auto hNetBPosY = Kokkos::create_mirror_view(dNetBPosY_);
  dNetBNegY_ = Kokkos::View<float*>("NetBNegY", numNets_);
  auto hNetBNegY = Kokkos::create_mirror_view(dNetBNegY_);
  dNetCPosX_ = Kokkos::View<float*>("NetCPosX", numNets_);
  auto hNetCPosX = Kokkos::create_mirror_view(dNetCPosX_);
  dNetCNegX_ = Kokkos::View<float*>("NetCNegX", numNets_);
  auto hNetCNegX = Kokkos::create_mirror_view(dNetCNegX_);
  dNetCPosY_ = Kokkos::View<float*>("NetCPosY", numNets_);
  auto hNetCPosY = Kokkos::create_mirror_view(dNetCPosY_);
  dNetCNegY_ = Kokkos::View<float*>("NetCNegY", numNets_);
  auto hNetCNegY = Kokkos::create_mirror_view(dNetCNegY_);

  dNetLx_ = Kokkos::View<int*>("NetLx", numNets_);
  auto hNetLx = Kokkos::create_mirror_view(dNetLx_);
  dNetLy_ = Kokkos::View<int*>("NetLy", numNets_);
  auto hNetLy = Kokkos::create_mirror_view(dNetLy_);
  dNetUx_ = Kokkos::View<int*>("NetUx", numNets_);
  auto hNetUx = Kokkos::create_mirror_view(dNetUx_);
  dNetUy_ = Kokkos::View<int*>("NetUy", numNets_);
  auto hNetUy = Kokkos::create_mirror_view(dNetUy_);
  dNetWidth_ = Kokkos::View<int*>("NetWidth", numNets_);
  auto hNetWidth = Kokkos::create_mirror_view(dNetWidth_);
  dNetHeight_ = Kokkos::View<int*>("NetHeight", numNets_);
  auto hNetHeight = Kokkos::create_mirror_view(dNetHeight_);

  // This is for fixed instances
  for (auto& pin : pbc_->pins()) {
    const int pinId = pin->pinId();
    hPinX[pinId] = pin->cx();
    hPinY[pinId] = pin->cy();
    hPinOffsetX[pinId] = pin->offsetCx();
    hPinOffsetY[pinId] = pin->offsetCy();
  }

  // copy from host to device
  Kokkos::deep_copy(dPinX_, hPinX);
  Kokkos::deep_copy(dPinY_, hPinY);
  Kokkos::deep_copy(dPinOffsetX_, hPinOffsetX);
  Kokkos::deep_copy(dPinOffsetY_, hPinOffsetY);
}

void WirelengthOp::updatePinLocation(const int* instDCx, const int* instDCy)
{
  auto dInstPinPos = dInstPinPos_, dInstPinIdx = dInstPinIdx_, dPinX = dPinX_, dPinY = dPinY_,
       dPinOffsetX = dPinOffsetX_, dPinOffsetY = dPinOffsetY_;
  Kokkos::parallel_for(numPlaceInsts_, KOKKOS_LAMBDA (const int instId) {
    const int pinStart = dInstPinPos[instId];
    const int pinEnd = dInstPinPos[instId + 1];
    const float instDCxVal = instDCx[instId];
    const float instDCyVal = instDCy[instId];
    for (int pinId = pinStart; pinId < pinEnd; ++pinId) {
      const int pinIdx = dInstPinIdx[pinId];
      dPinX[pinIdx] = instDCxVal + dPinOffsetX[pinIdx];
      dPinY[pinIdx] = instDCyVal + dPinOffsetY[pinIdx];
    }
  });

  auto dNetPinPos = dNetPinPos_, dNetPinIdx = dNetPinIdx_, dNetLx = dNetLx_, dNetLy = dNetLy_,
       dNetUx = dNetUx_, dNetUy = dNetUy_, dNetWidth = dNetWidth_, dNetHeight = dNetHeight_;
  Kokkos::parallel_for(numNets_, KOKKOS_LAMBDA (const int netId) {
    const int pinStart = dNetPinPos[netId];
    const int pinEnd = dNetPinPos[netId + 1];
    int netLx = INT_MAX;
    int netLy = INT_MAX;
    int netUx = 0;
    int netUy = 0;
    for (int pinId = pinStart; pinId < pinEnd; ++pinId) {
      const int pinIdx = dNetPinIdx[pinId];
      const int pinX = dPinX[pinIdx];
      const int pinY = dPinY[pinIdx];
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

    dNetLx[netId] = netLx;
    dNetLy[netId] = netLy;
    dNetUx[netId] = netUx;
    dNetUy[netId] = netUy;
    dNetWidth[netId] = netUx - netLx;
    dNetHeight[netId] = netUy - netLy;
  });
}

struct TypeConvertor
{
  DEVICE_FUNC int64_t operator()(const int& x) const
  {
    return static_cast<int64_t>(x);
  }
};

int64_t WirelengthOp::computeHPWL()
{
  int64_t hpwl = 0;
  auto dNetWidth = dNetWidth_, dNetHeight = dNetHeight_;
  Kokkos::parallel_reduce(numNets_, KOKKOS_LAMBDA (const int i, int64_t& hpwl) {
    hpwl += dNetWidth[i] + dNetHeight[i];
  }, hpwl);
  Kokkos::fence();

  return hpwl;
}

int64_t WirelengthOp::computeWeightedHPWL(float virtualWeightFactor)
{
  int64_t hpwl = 0;
  auto dNetWidth = dNetWidth_, dNetHeight = dNetHeight_;
  auto dNetWeight = dNetWeight_, dNetVirtualWeight = dNetVirtualWeight_;
  Kokkos::parallel_reduce(numNets_, KOKKOS_LAMBDA (const int i, int64_t& hpwl) {
    hpwl += (dNetWeight[i] + dNetVirtualWeight[i] * virtualWeightFactor) * (dNetWidth[i] + dNetHeight[i]);
  }, hpwl);
  Kokkos::fence();

  return hpwl;
}

void WirelengthOp::computeWireLengthForce(const float wlCoeffX,
                                          const float wlCoeffY,
                                          const float virtualWeightFactor,
                                          float* wirelengthForceX,
                                          float* wirelengthForceY)
{
  auto dPinNetId = dPinNetId_, dPinX = dPinX_, dPinY = dPinY_, dNetUx = dNetUx_,
       dNetLx = dNetLx_, dNetUy = dNetUy_, dNetLy = dNetLy_;
  auto dPinAPosX = dPinAPosX_, dPinANegX = dPinANegX_, dPinAPosY = dPinAPosY_,
       dPinANegY = dPinANegY_;
  Kokkos::parallel_for(numPins_, KOKKOS_LAMBDA (const int pinId) {
    const int netId = dPinNetId[pinId];
    dPinAPosX[pinId] = expf(wlCoeffX * (dPinX[pinId] - dNetUx[netId]));
    dPinANegX[pinId]
        = expf(-1.0 * wlCoeffX * (dPinX[pinId] - dNetLx[netId]));
    dPinAPosY[pinId] = expf(wlCoeffY * (dPinY[pinId] - dNetUy[netId]));
    dPinANegY[pinId]
        = expf(-1.0 * wlCoeffY * (dPinY[pinId] - dNetLy[netId]));
  });

  auto dNetPinPos = dNetPinPos_, dNetPinIdx = dNetPinIdx_;
  auto dNetBPosX = dNetBPosX_, dNetBNegX = dNetBNegX_, dNetBPosY = dNetBPosY_,
       dNetBNegY = dNetBNegY_, dNetCPosX = dNetCPosX_, dNetCNegX = dNetCNegX_,
       dNetCPosY = dNetCPosY_, dNetCNegY = dNetCNegY_;
  Kokkos::parallel_for(numNets_, KOKKOS_LAMBDA (const int netId) {
    const int pinStart = dNetPinPos[netId];
    const int pinEnd = dNetPinPos[netId + 1];
    float bPosX = 0.0;
    float bNegX = 0.0;
    float bPosY = 0.0;
    float bNegY = 0.0;

    float cPosX = 0.0;
    float cNegX = 0.0;
    float cPosY = 0.0;
    float cNegY = 0.0;

    for (int pinId = pinStart; pinId < pinEnd; ++pinId) {
      const int pinIdx = dNetPinIdx[pinId];
      bPosX += dPinAPosX[pinIdx];
      bNegX += dPinANegX[pinIdx];
      bPosY += dPinAPosY[pinIdx];
      bNegY += dPinANegY[pinIdx];

      cPosX += dPinX[pinIdx] * dPinAPosX[pinIdx];
      cNegX += dPinX[pinIdx] * dPinANegX[pinIdx];
      cPosY += dPinY[pinIdx] * dPinAPosY[pinIdx];
      cNegY += dPinY[pinIdx] * dPinANegY[pinIdx];
    }

    dNetBPosX[netId] = bPosX;
    dNetBNegX[netId] = bNegX;
    dNetBPosY[netId] = bPosY;
    dNetBNegY[netId] = bNegY;

    dNetCPosX[netId] = cPosX;
    dNetCNegX[netId] = cNegX;
    dNetCPosY[netId] = cPosY;
    dNetCNegY[netId] = cNegY;
  });

  auto dPinGradX = dPinGradX_, dPinGradY = dPinGradY_;
  Kokkos::parallel_for(numPlaceInsts_, KOKKOS_LAMBDA (const int pinIdx) {
    const int netId = dPinNetId[pinIdx];

    // TODO:  if we need to remove high-fanout nets,
    // we can remove it here

    float netBNegX2 = dNetBNegX[netId] * dNetBNegX[netId];
    float netBPosX2 = dNetBPosX[netId] * dNetBPosX[netId];
    float netBNegY2 = dNetBNegY[netId] * dNetBNegY[netId];
    float netBPosY2 = dNetBPosY[netId] * dNetBPosY[netId];

    float pinXWlCoeffX = dPinX[pinIdx] * wlCoeffX;
    float pinYWlCoeffY = dPinY[pinIdx] * wlCoeffY;

    dPinGradX[pinIdx] = ((1.0f - pinXWlCoeffX) * dNetBNegX[netId]
                            + wlCoeffX * dNetCNegX[netId])
                               * dPinANegX[pinIdx] / netBNegX2
                           - ((1.0f + pinXWlCoeffX) * dNetBPosX[netId]
                              - wlCoeffX * dNetCPosX[netId])
                                 * dPinAPosX[pinIdx] / netBPosX2;

    dPinGradY[pinIdx] = ((1.0f - pinYWlCoeffY) * dNetBNegY[netId]
                            + wlCoeffY * dNetCNegY[netId])
                               * dPinANegY[pinIdx] / netBNegY2
                           - ((1.0f + pinYWlCoeffY) * dNetBPosY[netId]
                              - wlCoeffY * dNetCPosY[netId])
                                 * dPinAPosY[pinIdx] / netBPosY2;
  });

  // get the force on each instance
  auto dInstPinPos = dInstPinPos_, dInstPinIdx = dInstPinIdx_;
  auto dNetWeight = dNetWeight_, dNetVirtualWeight = dNetVirtualWeight_;
  Kokkos::parallel_for(numPlaceInsts_, KOKKOS_LAMBDA (const int instId) {
    const int pinStart = dInstPinPos[instId];
    const int pinEnd = dInstPinPos[instId + 1];
    float wlGradX = 0.0;
    float wlGradY = 0.0;
    for (int pinId = pinStart; pinId < pinEnd; ++pinId) {
      const int pinIdx = dInstPinIdx[pinId];
      const int netId = dPinNetId[pinIdx];
      const float weight = dNetWeight[netId]
                           + dNetVirtualWeight[netId] * virtualWeightFactor;
      wlGradX += dPinGradX[pinIdx] * weight;
      wlGradY += dPinGradY[pinIdx] * weight;
    }

    wirelengthForceX[instId] = wlGradX;
    wirelengthForceY[instId] = wlGradY;
  });
}

}  // namespace gpl2
