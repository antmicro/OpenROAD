///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2018-2020, The Regents of the University of California
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

#include "timingBase.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "nesterovBase.h"
#include "placerBase.h"
#include "rsz/Resizer.hh"
#include "sta/Fuzzy.hh"
#include "utl/Logger.h"

namespace gpl {

using utl::GPL;

// TimingBase
TimingBase::TimingBase() = default;

TimingBase::TimingBase(std::shared_ptr<NesterovBaseCommon> nbc,
                       rsz::Resizer* rs,
                       utl::Logger* log)
    : TimingBase()
{
  rs_ = rs;
  nbc_ = std::move(nbc);
  log_ = log;
}

void TimingBase::initTimingOverflowChk()
{
  timingOverflowChk_.clear();
  timingOverflowChk_.resize(timingNetWeightOverflow_.size(), false);
}

bool TimingBase::isTimingNetWeightOverflow(float overflow)
{
  int intOverflow = std::round(overflow * 100);
  // exception case handling
  if (timingNetWeightOverflow_.empty()
      || intOverflow > timingNetWeightOverflow_[0]) {
    return false;
  }

  bool needTdRun = false;
  for (int i = 0; i < timingNetWeightOverflow_.size(); i++) {
    if (timingNetWeightOverflow_[i] > intOverflow) {
      if (!timingOverflowChk_[i]) {
        timingOverflowChk_[i] = true;
        needTdRun = true;
      }
      continue;
    }
    return needTdRun;
  }
  return needTdRun;
}

void TimingBase::addTimingNetWeightOverflow(int overflow)
{
  std::vector<int>::iterator it = std::find(timingNetWeightOverflow_.begin(),
                                            timingNetWeightOverflow_.end(),
                                            overflow);

  // only push overflow when the overflow is not in vector.
  if (it == timingNetWeightOverflow_.end()) {
    timingNetWeightOverflow_.push_back(overflow);
  }

  // do sort in reverse order
  std::sort(timingNetWeightOverflow_.begin(),
            timingNetWeightOverflow_.end(),
            std::greater<int>());
}

void TimingBase::setTimingNetWeightOverflows(std::vector<int>& overflows)
{
  // sort by decreasing order
  std::sort(overflows.begin(), overflows.end(), std::greater<int>());
  for (auto& overflow : overflows) {
    addTimingNetWeightOverflow(overflow);
  }
  initTimingOverflowChk();
}

void TimingBase::deleteTimingNetWeightOverflow(int overflow)
{
  std::vector<int>::iterator it = std::find(timingNetWeightOverflow_.begin(),
                                            timingNetWeightOverflow_.end(),
                                            overflow);
  // only erase overflow when the overflow is in vector.
  if (it != timingNetWeightOverflow_.end()) {
    timingNetWeightOverflow_.erase(it);
  }
}

void TimingBase::clearTimingNetWeightOverflow()
{
  timingNetWeightOverflow_.clear();
}

size_t TimingBase::getTimingNetWeightOverflowSize() const
{
  return timingNetWeightOverflow_.size();
}

void TimingBase::setTimingNetWeightMax(float max)
{
  net_weight_max_ = max;
}

bool TimingBase::updateGNetWeights(float overflow)
{
  auto it = nbc_->gNets().begin();
  const auto end = nbc_->gNets().end();
  auto netSlacks
      = rs_->findResizeSlacks([&it, end]() -> std::optional<rsz::NetGNet> {
          while (it != end) {
            auto gNet = *it++;
            gNet->setTimingWeight(1.0);
            if (gNet->gPins().size() > 1) {
              return rsz::NetGNet{gNet->net()->dbNet(), gNet};
            }
          }
          return {};
        });

  if (netSlacks.empty()) {
    log_->warn(GPL, 114, "No net slacks found. Timing-driven mode disabled.");
    return false;
  }

  // min/max slack for worst nets
  auto slack_min = netSlacks.front().slack;
  auto slack_max = netSlacks.back().slack;

  log_->info(GPL, 100, "worst slack {:.3g}", slack_min);

  if (sta::fuzzyInf(slack_min)) {
    log_->warn(GPL, 102, "No slacks found. Timing-driven mode disabled.");
    return false;
  }

  if (slack_max == slack_min) {
    log_->info(GPL, 101, "All slacks are the same. Skipping weighting");
    return false;
  }

  for (auto& netSlack : netSlacks) {
    auto gNet = netSlack.net;
    auto net_slack = netSlack.slack;
    // weight(min_slack) = net_weight_max_
    // weight(max_slack) = 1
    const float weight = 1
                         + (net_weight_max_ - 1) * (slack_max - net_slack)
                               / (slack_max - slack_min);
    gNet->setTimingWeight(weight);
    debugPrint(log_,
               GPL,
               "timing",
               1,
               "net:{} slack:{} weight:{}",
               gNet->net()->dbNet()->getConstName(),
               net_slack,
               gNet->totalWeight());
  }

  log_->info(GPL, 103, "Weighted {} nets.", netSlacks.size());
  return true;
}

}  // namespace gpl
