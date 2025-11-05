// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-2025, The OpenROAD Authors

#include "strash_strategy.h"

#include <fcntl.h>

#include <cassert>
#include <vector>

#include "base/abc/abc.h"
#include "cut/abc_library_factory.h"
#include "cut/logic_cut.h"
#include "cut/logic_extractor.h"
#include "db_sta/dbNetwork.hh"
#include "db_sta/dbSta.hh"
#include "odb/db.h"
#include "rsz/Resizer.hh"
#include "sta/Graph.hh"
#include "sta/GraphDelayCalc.hh"
#include "sta/Search.hh"
#include "utils.h"
#include "utl/Logger.h"
#include "utl/deleter.h"
#include "utl/unique_name.h"

namespace abc {
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
extern void Abc_FrameSetLibGen(void* pLib);
extern void Abc_FrameSetDrivingCell(char* pName);

extern void Abc_NtkCecSat(Abc_Ntk_t* pNtk1,
                          Abc_Ntk_t* pNtk2,
                          int nConfLimit,
                          int nInsLimit);
extern void Abc_NtkCecFraig(Abc_Ntk_t* pNtk1,
                            Abc_Ntk_t* pNtk2,
                            int nSeconds,
                            int fVerbose);
extern void Abc_NtkCecFraigPart(Abc_Ntk_t* pNtk1,
                                Abc_Ntk_t* pNtk2,
                                int nSeconds,
                                int nPartSize,
                                int fVerbose);
extern void Abc_NtkCecFraigPartAuto(Abc_Ntk_t* pNtk1,
                                    Abc_Ntk_t* pNtk2,
                                    int nSeconds,
                                    int fVerbose);
}  // namespace abc

namespace rmp {

void StrashStrategy::OptimizeDesign(sta::dbSta* sta,
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
                 60,
                 "All endpoints have slack above threshold, nothing to do.");
    return;
  }

  cut::AbcLibraryFactory factory(logger);
  factory.AddDbSta(sta);
  factory.AddResizer(resizer);
  factory.SetCorner(corner_);
  cut::AbcLibrary abc_library = factory.Build();

  sta::dbNetwork* network = sta->getDbNetwork();

  // Disable incremental timing.
  sta->graphDelayCalc()->delaysInvalid();
  sta->search()->arrivalsInvalid();
  sta->search()->endpointsInvalid();

  cut::LogicExtractorFactory logic_extractor(sta, logger);
  for (sta::Vertex* negative_endpoint : candidate_vertices) {
    logic_extractor.AppendEndpoint(negative_endpoint);
  }

  cut::LogicCut cut = logic_extractor.BuildLogicCut(abc_library);

  utl::UniquePtrWithDeleter<abc::Abc_Ntk_t> mapped_abc_network
      = cut.BuildMappedAbcNetwork(abc_library, network, logger);

  utl::UniquePtrWithDeleter<abc::Abc_Ntk_t> current_network(
      abc::Abc_NtkToLogic(
          const_cast<abc::Abc_Ntk_t*>(mapped_abc_network.get())),
      &abc::Abc_NtkDelete);

  auto library = static_cast<abc::Mio_Library_t*>(mapped_abc_network->pManFunc);
  // Install library for NtkMap
  abc::Abc_FrameSetLibGen(library);
  abc::Mio_Gate_t* buffer_cell = abc::Mio_LibraryReadBuf(library);
  if (!buffer_cell) {
    logger->error(
        utl::RMP,
        59,
        "Cannot find buffer cell in abc library (ensure buffer "
        "exists in your PDK), if it does please report this internal error.");
  }
  abc::Abc_FrameSetDrivingCell(strdup(abc::Mio_GateReadName(buffer_cell)));

  current_network = WrapUnique(abc::Abc_NtkStrash(current_network.get(),
                                                  /*fAllNodes=*/false,
                                                  /*fCleanup=*/true,
                                                  /*fRecord=*/false));

  current_network = WrapUnique(abc::Abc_NtkMap(current_network.get(),
                                               nullptr,
                                               /*DelayTarget=*/1.0,
                                               /*AreaMulti=*/0.0,
                                               /*DelayMulti=*/2.5,
                                               /*LogFan=*/0.0,
                                               /*Slew=*/0.0,
                                               /*Gain=*/250.0,
                                               /*nGatesMin=*/0,
                                               /*fRecovery=*/true,
                                               /*fSwitching=*/false,
                                               /*fSkipFanout=*/false,
                                               /*fUseProfile=*/false,
                                               /*fUseBuffs=*/false,
                                               /*fVerbose=*/false));

  current_network = WrapUnique(abc::Abc_NtkToNetlist(current_network.get()));
  cut.InsertMappedAbcNetwork(
      current_network.get(), abc_library, network, name_generator, logger);
}
}  // namespace rmp
