// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-2025, The OpenROAD Authors

#include "genetic_algorithm.h"
#include "annealing_strategy.h"

#include <fcntl.h>

#include <algorithm>
#include <vector>

#include "aig/gia/gia.h"
#include "aig/gia/giaAig.h"
#include "base/abc/abc.h"
#include "base/main/main.h"
#include "cut/abc_library_factory.h"
#include "cut/logic_cut.h"
#include "cut/logic_extractor.h"
#include "db_sta/dbNetwork.hh"
#include "db_sta/dbSta.hh"
#include "map/if/if.h"
#include "map/scl/sclSize.h"
#include "misc/vec/vecPtr.h"
#include "proof/dch/dch.h"
#include "rsz/Resizer.hh"
#include "sta/Graph.hh"
#include "sta/GraphDelayCalc.hh"
#include "sta/MinMax.hh"
#include "sta/Search.hh"
#include "utils.h"
#include "utl/Logger.h"
#include "utl/deleter.h"
#include "utl/unique_name.h"

// The magic numbers are defaults from abc/src/base/abci/abc.c
const size_t SEARCH_RESIZE_ITERS = 100;
const size_t FINAL_RESIZE_ITERS = 1000;

namespace abc {
extern Abc_Ntk_t* Abc_NtkFromAigPhase(Aig_Man_t* pMan);
extern Abc_Ntk_t* Abc_NtkFromCellMappedGia(Gia_Man_t* p, int fUseBuffs);
extern Abc_Ntk_t* Abc_NtkFromDarChoices(Abc_Ntk_t* pNtkOld, Aig_Man_t* pMan);
extern Abc_Ntk_t* Abc_NtkFromMappedGia(Gia_Man_t* p,
                                       int fFindEnables,
                                       int fUseBuffs);
extern Abc_Ntk_t* Abc_NtkMap(Abc_Ntk_t* pNtk,
                             Mio_Library_t* userLib,
                             double DelayTarget,
                             double AreaMulti,
                             double DelayMulti,
                             float LogFan,
                             float Slew,
                             float Gain,
                             int nGatesMin,
                             int fRecovery,
                             int fSwitching,
                             int fSkipFanout,
                             int fUseProfile,
                             int fUseBuffs,
                             int fVerbose);
extern Aig_Man_t* Abc_NtkToDar(Abc_Ntk_t* pNtk, int fExors, int fRegisters);
extern Aig_Man_t* Abc_NtkToDarChoices(Abc_Ntk_t* pNtk);
extern Gia_Man_t* Gia_ManAigSynch2(Gia_Man_t* p,
                                   void* pPars,
                                   int nLutSize,
                                   int nRelaxRatio);
extern Gia_Man_t* Gia_ManCheckFalse(Gia_Man_t* p,
                                    int nSlackMax,
                                    int nTimeOut,
                                    int fVerbose,
                                    int fVeryVerbose);
extern Vec_Ptr_t* Abc_NtkCollectCiNames(Abc_Ntk_t* pNtk);
extern Vec_Ptr_t* Abc_NtkCollectCoNames(Abc_Ntk_t* pNtk);
extern void Abc_NtkRedirectCiCo(Abc_Ntk_t* pNtk);
}  // namespace abc
namespace rmp {
using utl::RMP;

class SuppressStdout
{
#ifndef _WIN32
 public:
  SuppressStdout()
  {
    // This is a hack to suppress excessive logs from ABC
    // Redirects stdout to /dev/null, preserves original stdout
    fflush(stdout);
    saved_stdout_fd = dup(1);
    int dev_null_fd = open("/dev/null", O_WRONLY);
    dup2(dev_null_fd, 1);
    close(dev_null_fd);
  }

  ~SuppressStdout()
  {
    // Restore stdout
    fflush(stdout);
    dup2(saved_stdout_fd, 1);
    close(saved_stdout_fd);
  }

 private:
  int saved_stdout_fd;
#endif
};

static void replaceGia(abc::Gia_Man_t*& gia, abc::Gia_Man_t* new_gia)
{
  if (gia == new_gia) {
    return;
  }
  if (gia->vNamesIn && !new_gia->vNamesIn) {
    std::swap(gia->vNamesIn, new_gia->vNamesIn);
  }
  if (gia->vNamesOut && !new_gia->vNamesOut) {
    std::swap(gia->vNamesOut, new_gia->vNamesOut);
  }
  if (gia->vNamesNode && !new_gia->vNamesNode) {
    std::swap(gia->vNamesNode, new_gia->vNamesNode);
  }
  abc::Gia_ManStop(gia);
  gia = new_gia;
}

struct SolutionSlack
{
  std::vector<GiaOp> solution;
  float worst_slack = -100000;
  bool computed_slack = false;
};

void evaluateSolution(SolutionSlack& sol_slack, const std::vector<sta::Vertex*>& candidate_vertices,
                      cut::AbcLibrary& abc_library, sta::Corner* corner, sta::dbSta* sta,
                      utl::UniqueName& name_generator, utl::Logger* logger) {
  auto block = sta->db()->getChip()->getBlock();
  odb::dbDatabase::beginEco(block);

  AnnealingStrategy::RunGia(sta,
                            candidate_vertices,
                            abc_library,
                            sol_slack.solution,
                            SEARCH_RESIZE_ITERS,
                            name_generator,
                            logger);

  odb::dbDatabase::endEco(block);

  sta::Vertex* worst_vertex_placeholder;
  sta->worstSlack(corner, sta::MinMax::max(), sol_slack.worst_slack, worst_vertex_placeholder);
  sol_slack.computed_slack = true;

  odb::dbDatabase::undoEco(block);
}

void GeneticAlgorithm::OptimizeDesign(sta::dbSta* sta,
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

  // GIA ops as lambdas
  // All the magic numbers are defaults from abc/src/base/abci/abc.c
  // Or from the ORFS qbc_speed script
  std::vector<GiaOp> all_ops
      = {[&](auto& gia) {
           // &st
           debugPrint(logger, RMP, "annealing", 1, "Starting rehash");
           replaceGia(gia, Gia_ManRehash(gia, false));
         },

         [&](auto& gia) {
           // &dch
           if (!gia->pReprs) {
             debugPrint(logger,
                        RMP,
                        "annealing",
                        1,
                        "Computing choices before equiv reduce");
             abc::Dch_Pars_t pars = {};
             Dch_ManSetDefaultParams(&pars);
             replaceGia(gia, Gia_ManPerformDch(gia, &pars));
           }
           debugPrint(logger, RMP, "annealing", 1, "Starting equiv reduce");
           replaceGia(gia, Gia_ManEquivReduce(gia, true, false, false, false));
         },

         [&](auto& gia) {
           // &syn2
           debugPrint(logger, RMP, "annealing", 1, "Starting syn2");
           replaceGia(gia,
                      Gia_ManAigSyn2(gia, false, true, 0, 20, 0, false, false));
         },

         [&](auto& gia) {
           // &syn3
           debugPrint(logger, RMP, "annealing", 1, "Starting syn3");
           replaceGia(gia, Gia_ManAigSyn3(gia, false, false));
         },

         [&](auto& gia) {
           // &syn4
           debugPrint(logger, RMP, "annealing", 1, "Starting syn4");
           replaceGia(gia, Gia_ManAigSyn4(gia, false, false));
         },

         [&](auto& gia) {
           // &retime
           debugPrint(logger, RMP, "annealing", 1, "Starting retime");
           replaceGia(gia, Gia_ManRetimeForward(gia, 100, false));
         },

         [&](auto& gia) {
           // &dc2
           debugPrint(logger, RMP, "annealing", 1, "Starting heavy rewriting");
           replaceGia(gia, Gia_ManCompress2(gia, true, false));
         },

         [&](auto& gia) {
           // &b
           debugPrint(logger, RMP, "annealing", 1, "Starting &b");
           replaceGia(
               gia, Gia_ManAreaBalance(gia, false, ABC_INFINITY, false, false));
         },

         [&](auto& gia) {
           // &b -d
           debugPrint(logger, RMP, "annealing", 1, "Starting &b -d");
           replaceGia(gia, Gia_ManBalance(gia, false, false, false));
         },

         [&](auto& gia) {
           // &false
           debugPrint(
               logger, RMP, "annealing", 1, "Starting false path elimination");
           SuppressStdout nostdout;
           replaceGia(gia, Gia_ManCheckFalse(gia, 0, 0, false, false));
         },

         [&](auto& gia) {
           // &reduce
           if (!gia->pReprs) {
             debugPrint(logger,
                        RMP,
                        "annealing",
                        1,
                        "Computing choices before equiv reduce");
             abc::Dch_Pars_t pars = {};
             Dch_ManSetDefaultParams(&pars);
             replaceGia(gia, Gia_ManPerformDch(gia, &pars));
           }
           debugPrint(
               logger, RMP, "annealing", 1, "Starting equiv reduce and remap");
           replaceGia(gia, Gia_ManEquivReduceAndRemap(gia, true, false));
         },

         [&](auto& gia) {
           // &if -g -K 6
           if (Gia_ManHasMapping(gia)) {
             debugPrint(logger,
                        RMP,
                        "annealing",
                        1,
                        "GIA has mapping - rehashing before mapping");
             replaceGia(gia, Gia_ManRehash(gia, false));
           }
           abc::If_Par_t pars = {};
           Gia_ManSetIfParsDefault(&pars);
           pars.fDelayOpt = true;
           pars.nLutSize = 6;
           pars.fTruth = true;
           pars.fCutMin = true;
           pars.fExpRed = false;
           debugPrint(logger, RMP, "annealing", 1, "Starting SOP balancing");
           replaceGia(gia, Gia_ManPerformMapping(gia, &pars));
         },

         [&](auto& gia) {
           // &synch2
           abc::Dch_Pars_t pars = {};
           Dch_ManSetDefaultParams(&pars);
           pars.nBTLimit = 100;
           debugPrint(logger, RMP, "annealing", 1, "Starting synch2");
           replaceGia(gia, Gia_ManAigSynch2(gia, &pars, 6, 20));
         }};
  /* Some ABC functions/commands that could be used, but crash in some
     permutations:
      * &nf. Call it like this:
            namespace abc {
            extern Gia_Man_t* Nf_ManPerformMapping(Gia_Man_t* pGia,
                                                   Jf_Par_t* pPars);
            }
            abc::Jf_Par_t pars = {};
            Nf_ManSetDefaultPars(&pars);
            new_gia = Nf_ManPerformMapping(gia, &pars);
        It crashes on a null pointer due to a missing time manager. We can make
        the time manager:
            gia->pManTime = abc::Tim_ManStart(Gia_ManCiNum(new_gia),
                                              Gia_ManCoNum(new_gia));
        But then, an assert deep in &nf fails.
      * &dsd. Call it like this:
            namespace abc {
            extern Gia_Man_t* Gia_ManCollapseTest(Gia_Man_t* p, int fVerbose);
            }
            new_gia = Gia_ManCollapseTest(gia, false);
        An assert fails.

      Some functions/commands don't actually exist:
      * &resub
      * &reshape, &reshape -a
      These are just stubs that return null.
   */

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
             "population",
             1,
             "Generating and evaluating the initial population");
  population_size_ = 4;
  std::vector<SolutionSlack> population(population_size_);
  for (auto& ind : population) {
    ind.solution.reserve(initial_ops_);
    for (size_t i = 0; i < initial_ops_; i++) {
      ind.solution.push_back(all_ops[random_() % all_ops.size()]);
    }
  }

  logger->info(RMP, 62, "Resynthesis: starting genetic algorithm");

  for (unsigned i = 0; i < population_size_; i++) {
    evaluateSolution(population[i], candidate_vertices, abc_library, corner_, sta, name_generator, logger);
    logger->info(RMP,
                 60,
                 "Individual: {}, worst slack: {}",
                 i,
                 population[i].worst_slack);
  }

  unsigned int crossover_count = population_size_;
  unsigned int iterations = 10;
  for (unsigned i = 0; i < iterations; i++) {
    // Crossover
    for (unsigned j = 0; j < crossover_count; j++) {
      auto rand1 = random_() % population_size_;
      auto rand2 = random_() % population_size_;
      if (rand1 == rand2) continue;
      std::vector<GiaOp>& parent1_sol = population[rand1].solution;
      std::vector<GiaOp>& parent2_sol = population[rand2].solution;
      std::vector<GiaOp> child_sol(parent1_sol.begin(), parent1_sol.begin() + parent1_sol.size() / 2);
      child_sol.insert(child_sol.end(), parent2_sol.begin() + parent2_sol.size() / 2, parent2_sol.end());
      SolutionSlack child_sol_slack;
      child_sol_slack.solution = std::move(child_sol);
      population.push_back(std::move(child_sol_slack));
    }
    // Mutations
    for (unsigned j = 0; j < population_size_; j++) {
      SolutionSlack sol_slack;
      sol_slack.solution = neighbor(population[j].solution);
      population.push_back(std::move(sol_slack));
    }
    // Evaluation
    for (auto& sol_slack : population) {
      if (sol_slack.computed_slack) continue;
      evaluateSolution(sol_slack, candidate_vertices, abc_library, corner_, sta, name_generator, logger);
    }
    // Selection
    std::nth_element(population.begin(), population.begin() + population_size_, population.end(),
                     [](const auto& a, const auto& b) { return a.worst_slack > b.worst_slack;});
    population.resize(population_size_);
  }

  logger->info(
      RMP, 59, "Resynthesis: End of genetic algorithm, applying operations");
  logger->info(RMP, 63, "Resynthesis: Applying ABC operations");

  auto best_it = std::max_element(population.begin(), population.end(),
                                  [](const auto& a, const auto& b) { return a.worst_slack < b.worst_slack;});
  // Apply the ops
  AnnealingStrategy::RunGia(sta,
                            candidate_vertices,
                            abc_library,
                            best_it->solution,
                            FINAL_RESIZE_ITERS,
                            name_generator,
                            logger);
}
}  // namespace rmp
