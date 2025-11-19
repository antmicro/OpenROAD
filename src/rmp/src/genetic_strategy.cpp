// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-2025, The OpenROAD Authors

#include "genetic_strategy.h"

#include <fcntl.h>

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "cut/abc_library_factory.h"
#include "db_sta/dbNetwork.hh"
#include "db_sta/dbSta.hh"
#include "gia.h"
#include "rsz/Resizer.hh"
#include "sta/Graph.hh"
#include "sta/MinMax.hh"
#include "sta/Search.hh"
#include "utils.h"
#include "utl/Logger.h"
#include "utl/deleter.h"
#include "utl/unique_name.h"

// The magic numbers are defaults from abc/src/base/abci/abc.c
const size_t SEARCH_RESIZE_ITERS = 100;
const size_t FINAL_RESIZE_ITERS = 1000;

namespace rmp {
using utl::RMP;

using Solution = std::vector<size_t>;

struct SolutionSlack
{
  Solution solution;
  float worst_slack = -100000;
  bool computed_slack = false;
  std::string toString() const
  {
    std::ostringstream resStream;
    resStream << '['
              << (solution.size() > 0 ? std::to_string(solution[0]) : "");
    for (int i = 1; i < solution.size(); i++) {
      resStream << ", " << std::to_string(solution[i]);
    }
    resStream << "], worst slack: ";
    if (computed_slack) {
      resStream << worst_slack;
    } else {
      resStream << "not computed";
    }
    return resStream.str();
  }
};

std::vector<GiaOp> getSolutionOps(const Solution& sol,
                                  const std::vector<GiaOp>& all_ops)
{
  std::vector<GiaOp> solOps;
  solOps.reserve(sol.size());
  for (int i = 0; i < sol.size(); i++) {
    solOps.push_back(all_ops[sol[i]]);
  }
  return solOps;
}

void evaluateSolution(SolutionSlack& sol_slack,
                      const std::vector<sta::Vertex*>& candidate_vertices,
                      cut::AbcLibrary& abc_library,
                      sta::Corner* corner,
                      sta::dbSta* sta,
                      utl::UniqueName& name_generator,
                      utl::Logger* logger,
                      const std::vector<GiaOp>& all_ops)
{
  auto block = sta->db()->getChip()->getBlock();
  odb::dbDatabase::beginEco(block);

  RunGia(sta,
         candidate_vertices,
         abc_library,
         getSolutionOps(sol_slack.solution, all_ops),
         SEARCH_RESIZE_ITERS,
         name_generator,
         logger);

  odb::dbDatabase::endEco(block);

  sta::Vertex* worst_vertex_placeholder;
  sta->worstSlack(corner,
                  sta::MinMax::max(),
                  sol_slack.worst_slack,
                  worst_vertex_placeholder);
  sol_slack.computed_slack = true;

  odb::dbDatabase::undoEco(block);
}

float getWorstSlack(sta::dbSta* sta, sta::Corner* corner)
{
  float worst_slack;
  sta::Vertex* worst_vertex_placeholder;
  sta->worstSlack(
      corner, sta::MinMax::max(), worst_slack, worst_vertex_placeholder);
  return worst_slack;
}

void removeDuplicates(std::vector<SolutionSlack>& population,
                      utl::Logger* logger)
{
  struct HashVector
  {
    size_t operator()(const Solution& sol) const
    {
      std::size_t res = 0;
      for (const auto& item : sol) {
        res += item;
      }
      return res;
    }
  };
  std::unordered_set<Solution, HashVector> taken;
  population.erase(
      std::remove_if(
          population.begin(),
          population.end(),
          [&taken, logger](const SolutionSlack& s) {
            if (!taken.insert(s.solution).second) {
              debugPrint(
                  logger, RMP, "genetic", 2, "Removing: " + s.toString());
              return true;
            } else {
              debugPrint(logger, RMP, "genetic", 2, "Keeping: " + s.toString());
              return false;
            }
          }),
      population.end());
}

void GeneticStrategy::OptimizeDesign(sta::dbSta* sta,
                                     utl::UniqueName& name_generator,
                                     rsz::Resizer* resizer,
                                     utl::Logger* logger)
{
  sta->ensureGraph();
  sta->ensureLevelized();
  sta->searchPreamble();
  sta->ensureClkNetwork();

  auto candidate_vertices = GetEndpoints(sta, resizer, slack_threshold_);
  if (candidate_vertices.empty()) {
    logger->info(utl::RMP,
                 58,
                 "All endpoints have slack above threshold, nothing to do.");
    return;
  }

  cut::AbcLibraryFactory factory(logger);
  factory.AddDbSta(sta);
  factory.AddResizer(resizer);
  factory.SetCorner(corner_);
  cut::AbcLibrary abc_library = factory.Build();

  std::vector<GiaOp> all_ops = GiaOps(logger);

  // Computes a random neighbor of a given solution
  const auto neighbor = [&](Solution sol) {
    enum Move
    {
      ADD,
      REMOVE,
      SWAP,
      COUNT
    };
    Move move = ADD;
    if (sol.size() > 1) {
      move = Move(random_() % (COUNT));
    }
    switch (move) {
      case ADD: {
        debugPrint(logger, RMP, "annealing", 2, "Adding a new GIA operation");
        size_t i = random_() % (sol.size() + 1);
        size_t j = random_() % all_ops.size();
        sol.insert(sol.begin() + i, j);
      } break;
      case REMOVE: {
        debugPrint(logger, RMP, "annealing", 2, "Removing a GIA operation");
        size_t i = random_() % sol.size();
        sol.erase(sol.begin() + i);
      } break;
      case SWAP: {
        debugPrint(
            logger, RMP, "annealing", 2, "Swapping adjacent GIA operations");
        size_t i = random_() % (sol.size() - 1);
        std::swap(sol[i], sol[i + 1]);
      } break;
      case COUNT:
        // unreachable
        std::abort();
    }
    return sol;
  };

  // Initial solution and slack
  debugPrint(logger,
             RMP,
             "population",
             1,
             "Generating and evaluating the initial population");
  std::vector<SolutionSlack> population(pop_size_);
  for (auto& ind : population) {
    ind.solution.reserve(initial_ops_);
    for (size_t i = 0; i < initial_ops_; i++) {
      ind.solution.push_back(random_() % all_ops.size());
    }
  }

  logger->info(RMP,
               62,
               "Resynthesis: starting genetic algorithm, Worst slack is {}",
               getWorstSlack(sta, corner_));

  for (unsigned i = 0; i < pop_size_; i++) {
    evaluateSolution(population[i],
                     candidate_vertices,
                     abc_library,
                     corner_,
                     sta,
                     name_generator,
                     logger,
                     all_ops);
    debugPrint(logger, RMP, "genetic", 1, population[i].toString());
  }

  for (unsigned i = 0; i < iterations_; i++) {
    logger->info(RMP, 65, "Resynthesis: Iteration {} of genetic algorithm", i);
    unsigned generation_size = population.size();
    // Crossover
    unsigned cross_size = std::max<unsigned>(cross_prob_ * generation_size, 1);
    for (unsigned j = 0; j < cross_size; j++) {
      auto rand1 = random_() % generation_size;
      auto rand2 = random_() % generation_size;
      if (rand1 == rand2) {
        continue;
      }
      Solution& parent1_sol = population[rand1].solution;
      Solution& parent2_sol = population[rand2].solution;
      Solution child_sol(parent1_sol.begin(),
                         parent1_sol.begin() + parent1_sol.size() / 2);
      child_sol.insert(child_sol.end(),
                       parent2_sol.begin() + parent2_sol.size() / 2,
                       parent2_sol.end());
      SolutionSlack child_sol_slack;
      child_sol_slack.solution = std::move(child_sol);
      population.push_back(std::move(child_sol_slack));
    }
    // Mutations
    unsigned mut_size = std::max<unsigned>(mut_prob_ * generation_size, 1);
    for (unsigned j = 0; j < mut_size; j++) {
      SolutionSlack sol_slack;
      auto rand = random_() % generation_size;
      sol_slack.solution = neighbor(population[rand].solution);
      population.push_back(std::move(sol_slack));
    }
    removeDuplicates(population, logger);
    // Evaluation
    for (auto& sol_slack : population) {
      if (sol_slack.computed_slack) {
        continue;
      }
      evaluateSolution(sol_slack,
                       candidate_vertices,
                       abc_library,
                       corner_,
                       sta,
                       name_generator,
                       logger,
                       all_ops);
    }
    // Selection
    std::sort(
        population.begin(), population.end(), [](const auto& a, const auto& b) {
          return a.worst_slack > b.worst_slack;
        });
    std::vector<SolutionSlack> newPopulation;
    newPopulation.reserve(pop_size_);
    for (int j = 0; j < pop_size_; j++) {
      std::vector<size_t> tournament(tourn_size_);
      std::generate_n(tournament.begin(), tourn_size_, [&]() {
        return random_() % population.size();
      });
      std::sort(tournament.begin(), tournament.end());
      tournament.erase(std::unique(tournament.begin(), tournament.end()),
                       tournament.end());
      std::bernoulli_distribution bern_dist{tourn_prob_};
      for (int k = 0; k < tournament.size(); k++) {
        if (bern_dist(random_)) {
          newPopulation.push_back(population[tournament[k]]);
          break;
        }
      }
    }
    removeDuplicates(newPopulation, logger);
    population = newPopulation;

    for (int j = 0; j < population.size(); j++) {
      debugPrint(logger, RMP, "genetic", 1, population[j].toString());
    }
  }

  logger->info(
      RMP, 59, "Resynthesis: End of genetic algorithm, applying operations");
  logger->info(RMP, 63, "Resynthesis: Applying ABC operations");

  auto best_it = std::max_element(
      population.begin(), population.end(), [](const auto& a, const auto& b) {
        return a.worst_slack < b.worst_slack;
      });
  logger->info(RMP,
               66,
               "Resynthesis: Best result is of individual {}: {}",
               std::distance(population.begin(), best_it),
               best_it->worst_slack);
  // Apply the ops
  RunGia(sta,
         candidate_vertices,
         abc_library,
         getSolutionOps(best_it->solution, all_ops),
         FINAL_RESIZE_ITERS,
         name_generator,
         logger);
  logger->info(
      RMP, 67, "Resynthesis: Worst slack is {}", getWorstSlack(sta, corner_));
}
}  // namespace rmp
