// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-2025, The OpenROAD Authors

#pragma once

#include "cut/abc_library_factory.h"
#include "db_sta/dbSta.hh"
#include "resynthesis_strategy.h"
#include "rsz/Resizer.hh"
#include "sta/Corner.hh"
#include "sta/Delay.hh"
#include "utl/Logger.h"
#include "utl/unique_name.h"

namespace rmp {

using GiaOp = std::function<void(abc::Gia_Man_t*&)>;

class NoopStrategy : public ResynthesisStrategy
{
 public:
  explicit NoopStrategy(sta::Corner* corner, sta::Slack slack_threshold)
      : corner_(corner), slack_threshold_(slack_threshold)
  {
  }
  void OptimizeDesign(sta::dbSta* sta,
                      utl::UniqueName& name_generator,
                      rsz::Resizer* resizer,
                      utl::Logger* logger) override;

 private:
  sta::Corner* corner_;
  sta::Slack slack_threshold_;
};

}  // namespace rmp
