// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-2025, The OpenROAD Authors

#include "annealing_strategy.h"

#include <fcntl.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "cut/abc_library_factory.h"
#include "db_sta/dbNetwork.hh"
#include "db_sta/dbSta.hh"
#include "gia.h"
#include "odb/db.h"
#include "rsz/Resizer.hh"
#include "sta/Delay.hh"
#include "sta/Graph.hh"
#include "sta/MinMax.hh"
#include "utils.h"
#include "utl/Logger.h"
#include "utl/deleter.h"
#include "utl/unique_name.h"

namespace rmp {
using utl::RMP;

void AnnealingStrategy::OptimizeDesign(sta::dbSta* sta,
                                       utl::UniqueName& name_generator,
                                       rsz::Resizer* resizer,
                                       utl::Logger* logger)
{
  sta->ensureGraph();
  sta->ensureLevelized();
  sta->searchPreamble();
  sta->ensureClkNetwork();
  auto block = sta->db()->getChip()->getBlock();

  auto candidate_vertices = GetEndpoints(sta, resizer, slack_threshold_);
  if (candidate_vertices.empty()) {
    logger->info(utl::RMP,
                 51,
                 "All endpoints have slack above threshold, nothing to do.");
    return;
  }

  cut::AbcLibraryFactory factory(logger);
  factory.AddDbSta(sta);
  factory.AddResizer(resizer);
  factory.SetCorner(corner_);
  cut::AbcLibrary abc_library = factory.Build();

  std::vector<GiaOp> all_ops = GiaOps(logger);

  // Computes a random neighbor of a given GIA op list
  const auto neighbor = [&](std::vector<GiaOp> ops) {
    enum Move
    {
      ADD,
      REMOVE,
      SWAP,
      COUNT
    };
    Move move = ADD;
    if (ops.size() > 1) {
      move = Move(random_() % (COUNT));
    }
    switch (move) {
      case ADD: {
        debugPrint(logger, RMP, "annealing", 2, "Adding a new GIA operation");
        size_t i = random_() % (ops.size() + 1);
        size_t j = random_() % all_ops.size();
        ops.insert(ops.begin() + i, all_ops[j]);
      } break;
      case REMOVE: {
        debugPrint(logger, RMP, "annealing", 2, "Removing a GIA operation");
        size_t i = random_() % ops.size();
        ops.erase(ops.begin() + i);
      } break;
      case SWAP: {
        debugPrint(
            logger, RMP, "annealing", 2, "Swapping adjacent GIA operations");
        size_t i = random_() % (ops.size() - 1);
        std::swap(ops[i], ops[i + 1]);
      } break;
      case COUNT:
        // unreachable
        std::abort();
    }
    return ops;
  };

  // Initial solution and slack
  debugPrint(logger,
             RMP,
             "annealing",
             1,
             "Generating and evaluating the initial solution");
  std::vector<GiaOp> ops;
  ops.reserve(initial_ops_);
  for (size_t i = 0; i < initial_ops_; i++) {
    ops.push_back(all_ops[random_() % all_ops.size()]);
  }

  // The magic numbers are defaults from abc/src/base/abci/abc.c
  const size_t SEARCH_RESIZE_ITERS = 100;
  const size_t FINAL_RESIZE_ITERS = 1000;

  odb::dbDatabase::beginEco(block);

  RunGia(sta,
         candidate_vertices,
         abc_library,
         ops,
         SEARCH_RESIZE_ITERS,
         name_generator,
         logger);

  odb::dbDatabase::endEco(block);

  float worst_slack;
  sta::Vertex* worst_vertex;
  sta->worstSlack(corner_, sta::MinMax::max(), worst_slack, worst_vertex);

  odb::dbDatabase::undoEco(block);

  if (!temperature_) {
    sta::Delay required = sta->vertexRequired(worst_vertex, sta::MinMax::max());
    temperature_ = required;
  }

  logger->info(RMP, 52, "Resynthesis: starting simulated annealing");
  logger->info(RMP,
               53,
               "Initial temperature: {}, worst slack: {}",
               *temperature_,
               worst_slack);

  float best_worst_slack = worst_slack;
  auto best_ops = ops;
  size_t worse_iters = 0;

  for (unsigned i = 0; i < iterations_; i++) {
    float current_temp
        = *temperature_ * (static_cast<float>(iterations_ - i) / iterations_);

    if (revert_after_ && worse_iters >= *revert_after_) {
      logger->info(RMP, 57, "Reverting to the best found solution");
      ops = best_ops;
      worst_slack = best_worst_slack;
      worse_iters = 0;
    }

    if ((i + 1) % 10 == 0) {
      logger->info(RMP,
                   54,
                   "Iteration: {}, temperature: {}, best worst slack: {}",
                   i + 1,
                   current_temp,
                   best_worst_slack);
    } else {
      debugPrint(logger,
                 RMP,
                 "annealing",
                 1,
                 "Iteration: {}, temperature: {}, best worst slack: {}",
                 i + 1,
                 current_temp,
                 best_worst_slack);
    }

    odb::dbDatabase::beginEco(block);

    auto new_ops = neighbor(ops);
    RunGia(sta,
           candidate_vertices,
           abc_library,
           new_ops,
           SEARCH_RESIZE_ITERS,
           name_generator,
           logger);

    odb::dbDatabase::endEco(block);

    float worst_slack_new;
    sta->worstSlack(corner_, sta::MinMax::max(), worst_slack_new, worst_vertex);

    odb::dbDatabase::undoEco(block);

    if (worst_slack_new < best_worst_slack) {
      worse_iters++;
    } else {
      worse_iters = 0;
    }

    if (worst_slack_new < worst_slack) {
      float accept_prob
          = current_temp == 0
                ? 0
                : std::exp((worst_slack_new - worst_slack) / current_temp);
      debugPrint(
          logger,
          RMP,
          "annealing",
          1,
          "Current worst slack: {}, new: {}, accepting new ABC script with "
          "probability {}",
          worst_slack,
          worst_slack_new,
          accept_prob);
      if (std::uniform_real_distribution<float>(0, 1)(random_) < accept_prob) {
        debugPrint(logger,
                   RMP,
                   "annealing",
                   1,
                   "Accepting new ABC script with worse slack");
      } else {
        debugPrint(logger,
                   RMP,
                   "annealing",
                   1,
                   "Rejecting new ABC script with worse slack");
        continue;
      }
    } else {
      debugPrint(logger,
                 RMP,
                 "annealing",
                 1,

                 "Current worst slack: {}, new: {}, accepting new ABC script",
                 worst_slack,
                 worst_slack_new);
    }

    ops = std::move(new_ops);
    worst_slack = worst_slack_new;

    if (worst_slack > best_worst_slack) {
      best_worst_slack = worst_slack;
      best_ops = ops;
    }
  }

  logger->info(
      RMP, 55, "Resynthesis: End of simulated annealing, applying operations");
  logger->info(RMP, 56, "Resynthesis: Applying ABC operations");

  // Apply the ops
  RunGia(sta,
         candidate_vertices,
         abc_library,
         best_ops,
         FINAL_RESIZE_ITERS,
         name_generator,
         logger);
}

}  // namespace rmp
