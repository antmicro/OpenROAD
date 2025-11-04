// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-2025, The OpenROAD Authors

#pragma once

#include <random>

#include "cut/abc_library_factory.h"
#include "db_sta/dbSta.hh"
#include "resynthesis_strategy.h"
#include "sta/Corner.hh"
#include "utl/Logger.h"
#include "utl/unique_name.h"

namespace rmp {

using GiaOp = std::function<void(abc::Gia_Man_t*&)>;

class GeneticAlgorithm : public ResynthesisStrategy
{
 public:
  explicit GeneticAlgorithm(sta::Corner* corner,
                            sta::Slack slack_threshold,
                            std::optional<std::mt19937::result_type> seed,
                            unsigned pop_size,
                            unsigned mut_size,
                            unsigned cross_size,
                            unsigned tourn_size,
                            float tourn_prob,
                            unsigned iterations,
                            unsigned initial_ops)
      : corner_(corner),
        slack_threshold_(slack_threshold),
        pop_size_(pop_size),
        mut_size_(mut_size),
        cross_size_(cross_size),
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
  void RunGia(sta::dbSta* sta,
              const std::vector<sta::Vertex*>& candidate_vertices,
              cut::AbcLibrary& abc_library,
              const std::vector<GiaOp>& gia_ops,
              size_t resize_iters,
              utl::UniqueName& name_generator,
              utl::Logger* logger);

 private:
  sta::Corner* corner_;
  sta::Slack slack_threshold_;
  unsigned pop_size_;
  unsigned mut_size_;
  unsigned cross_size_;
  unsigned tourn_size_;
  float tourn_prob_;
  unsigned iterations_;
  unsigned initial_ops_;
  std::mt19937 random_;
};

}  // namespace rmp
