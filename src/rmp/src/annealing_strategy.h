// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-2025, The OpenROAD Authors

#pragma once

#include <optional>
#include <random>

#include "db_sta/dbSta.hh"
#include "resynthesis_strategy.h"
#include "rsz/Resizer.hh"
#include "sta/Corner.hh"
#include "sta/Delay.hh"
#include "utl/Logger.h"
#include "utl/unique_name.h"

namespace rmp {

class AnnealingStrategy : public ResynthesisStrategy
{
 public:
  explicit AnnealingStrategy(sta::Corner* corner,
                             sta::Slack slack_threshold,
                             std::optional<std::mt19937::result_type> seed,
                             std::optional<float> temperature,
                             unsigned iterations,
                             std::optional<unsigned> revert_after,
                             unsigned initial_ops)
      : corner_(corner),
        slack_threshold_(slack_threshold),
        temperature_(temperature),
        iterations_(iterations),
        revert_after_(revert_after),
        initial_ops_(initial_ops)
  {
    if (seed) {
      random_.seed(*seed);
    }
  }
  void OptimizeDesign(sta::dbSta* sta,
                      utl::UniqueName& name_generator,
                      rsz::Resizer* resizer,
                      utl::Logger* logger) override;

 private:
  sta::Corner* corner_;
  sta::Slack slack_threshold_;
  std::optional<float> temperature_;
  unsigned iterations_;
  std::optional<unsigned> revert_after_;
  unsigned initial_ops_;
  std::mt19937 random_;
};

}  // namespace rmp
