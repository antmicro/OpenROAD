// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-2025, The OpenROAD Authors

#pragma once

#include <random>

#include "db_sta/dbSta.hh"
#include "gia.h"
#include "resynthesis_strategy.h"
#include "sta/Corner.hh"
#include "utl/Logger.h"
#include "utl/unique_name.h"

namespace rmp {

class GeneticStrategy : public ResynthesisStrategy
{
 public:
  explicit GeneticStrategy(sta::Corner* corner,
                           sta::Slack slack_threshold,
                           std::optional<std::mt19937::result_type> seed,
                           unsigned pop_size,
                           float mut_prob,
                           float cross_prob,
                           unsigned tourn_size,
                           float tourn_prob,
                           unsigned iterations,
                           unsigned initial_ops)
      : corner_(corner),
        slack_threshold_(slack_threshold),
        pop_size_(pop_size),
        mut_prob_(mut_prob),
        cross_prob_(cross_prob),
        tourn_size_(tourn_size),
        tourn_prob_(tourn_prob),
        iterations_(iterations),
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
  unsigned pop_size_;
  float mut_prob_;
  float cross_prob_;
  unsigned tourn_size_;
  float tourn_prob_;
  unsigned iterations_;
  unsigned initial_ops_;
  std::mt19937 random_;
};

}  // namespace rmp
