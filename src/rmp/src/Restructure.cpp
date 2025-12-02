// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019-2025, The OpenROAD Authors

#include "rmp/Restructure.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "aig/gia/giaAig.h"
#include "annealing_strategy.h"
#include "base/abc/abc.h"
#include "base/main/abcapis.h"
#include "cut/abc_init.h"
#include "cut/abc_library_factory.h"
#include "cut/blif.h"
#include "cut/logic_cut.h"
#include "cut/logic_extractor.h"
#include "db_sta/dbNetwork.hh"
#include "db_sta/dbSta.hh"
#define FMT_CONSTEVAL
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#pragma clang diagnostic ignored "-Wbitfield-constant-conversion"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wdeprecated-builtins"
#pragma clang diagnostic ignored "-Wreturn-type"
#include "lorina/blif.hpp"
#include "lorina/genlib.hpp"
#include "mockturtle/algorithms/emap.hpp"
#include "mockturtle/io/genlib_reader.hpp"
#include "mockturtle/io/write_verilog.hpp"
#include "mockturtle/networks/aig.hpp"
#include "mockturtle/networks/block.hpp"
#include "mockturtle/views/cell_view.hpp"
#include "mockturtle/views/topo_view.hpp"
#pragma clang diagnostic pop
#undef FMT_CONSEVAL
#include "odb/db.h"
#include "ord/OpenRoad.hh"
#include "rsz/Resizer.hh"
#include "sta/Delay.hh"
#include "sta/Graph.hh"
#include "sta/GraphDelayCalc.hh"
#include "sta/Liberty.hh"
#include "sta/Network.hh"
#include "sta/NetworkClass.hh"
#include "sta/Path.hh"
#include "sta/PathEnd.hh"
#include "sta/PathExpanded.hh"
#include "sta/PatternMatch.hh"
#include "sta/PortDirection.hh"
#include "sta/Sdc.hh"
#include "sta/Search.hh"
#include "sta/Sta.hh"
#include "utils.h"
#include "utl/Logger.h"
#include "zero_slack_strategy.h"

namespace abc {
extern void Abc_FrameSetLibGen(void* pLib);
extern Aig_Man_t* Abc_NtkToDar(Abc_Ntk_t* pNtk, int fExors, int fRegisters);
}  // namespace abc

using utl::RMP;
using namespace abc;
using cut::Blif;

namespace rmp {

Restructure::Restructure(utl::Logger* logger,
                         sta::dbSta* open_sta,
                         odb::dbDatabase* db,
                         rsz::Resizer* resizer,
                         est::EstimateParasitics* estimate_parasitics)
{
  logger_ = logger;
  db_ = db;
  open_sta_ = open_sta;
  resizer_ = resizer;
  estimate_parasitics_ = estimate_parasitics;

  cut::abcInit();
}

void Restructure::deleteComponents()
{
}

Restructure::~Restructure()
{
  deleteComponents();
}

void Restructure::reset()
{
  lib_file_names_.clear();
  path_insts_.clear();
}

void Restructure::resynth(sta::Corner* corner)
{
  ZeroSlackStrategy zero_slack_strategy(corner);
  zero_slack_strategy.OptimizeDesign(
      open_sta_, name_generator_, resizer_, logger_);
}

void Restructure::resynthAnnealing(sta::Corner* corner)
{
  AnnealingStrategy annealing_strategy(corner,
                                       slack_threshold_,
                                       annealing_seed_,
                                       annealing_temp_,
                                       annealing_iters_,
                                       annealing_revert_after_,
                                       annealing_init_ops_);
  annealing_strategy.OptimizeDesign(
      open_sta_, name_generator_, resizer_, logger_);
}

void Restructure::run(char* liberty_file_name,
                      float slack_threshold,
                      unsigned max_depth,
                      char* workdir_name,
                      char* abc_logfile)
{
  reset();
  block_ = db_->getChip()->getBlock();
  if (!block_) {
    return;
  }

  logfile_ = abc_logfile;
  sta::Slack worst_slack = slack_threshold;

  lib_file_names_.emplace_back(liberty_file_name);
  work_dir_name_ = workdir_name;
  work_dir_name_ = work_dir_name_ + "/";

  if (is_area_mode_) {  // Only in area mode
    removeConstCells();
  }

  getBlob(max_depth);

  if (path_insts_.size()) {
    runABC();

    postABC(worst_slack);
  }
}

void Restructure::getBlob(unsigned max_depth)
{
  open_sta_->ensureGraph();
  open_sta_->ensureLevelized();
  open_sta_->searchPreamble();

  sta::PinSet ends(open_sta_->getDbNetwork());

  getEndPoints(ends, is_area_mode_, max_depth);
  if (ends.size()) {
    sta::PinSet boundary_points = !is_area_mode_
                                      ? resizer_->findFanins(ends)
                                      : resizer_->findFaninFanouts(ends);
    // fanin_fanouts.insert(ends.begin(), ends.end()); // Add seq cells
    logger_->report("Found {} pins in extracted logic.",
                    boundary_points.size());
    for (const sta::Pin* pin : boundary_points) {
      odb::dbITerm* term = nullptr;
      odb::dbBTerm* port = nullptr;
      odb::dbModITerm* moditerm = nullptr;
      open_sta_->getDbNetwork()->staToDb(pin, term, port, moditerm);
      if (term && !term->getInst()->getMaster()->isBlock()) {
        path_insts_.insert(term->getInst());
      }
    }
    logger_->report("Found {} instances for restructuring.",
                    path_insts_.size());
  }
}

void Restructure::runABC()
{
  const std::string prefix
      = work_dir_name_ + std::string(block_->getConstName());
  input_blif_file_name_ = prefix + "_crit_path.blif";
  std::vector<std::string> files_to_remove;

  debugPrint(logger_,
             utl::RMP,
             "remap",
             1,
             "Constants before remap {}",
             countConsts(block_));

  Blif blif_(
      logger_, open_sta_, locell_, loport_, hicell_, hiport_, ++blif_call_id_);
  blif_.setReplaceableInstances(path_insts_);
  blif_.writeBlif(input_blif_file_name_.c_str(), !is_area_mode_);
  debugPrint(
      logger_, RMP, "remap", 1, "Writing blif file {}", input_blif_file_name_);
  files_to_remove.emplace_back(input_blif_file_name_);

  // abc optimization
  std::vector<Mode> modes;
  std::vector<pid_t> child_proc;

  if (is_area_mode_) {
    // Area Mode
    modes = {Mode::AREA_1, Mode::AREA_2, Mode::AREA_3};
  } else {
    // Delay Mode
    modes = {Mode::DELAY_1, Mode::DELAY_2, Mode::DELAY_3, Mode::DELAY_4};
  }

  child_proc.resize(modes.size(), 0);

  std::string best_blif;
  int best_inst_count = std::numeric_limits<int>::max();
  float best_delay_gain = std::numeric_limits<float>::max();

  debugPrint(
      logger_, RMP, "remap", 1, "Running ABC with {} modes.", modes.size());

  for (size_t curr_mode_idx = 0; curr_mode_idx < modes.size();
       curr_mode_idx++) {
    output_blif_file_name_
        = prefix + std::to_string(curr_mode_idx) + "_crit_path_out.blif";

    opt_mode_ = modes[curr_mode_idx];

    const std::string abc_script_file
        = prefix + std::to_string(curr_mode_idx) + "ord_abc_script.tcl";
    if (logfile_ == "") {
      logfile_ = prefix + "abc.log";
    }

    debugPrint(logger_,
               RMP,
               "remap",
               1,
               "Writing ABC script file {}.",
               abc_script_file);

    if (writeAbcScript(abc_script_file)) {
      // call linked abc
      Abc_Start();
      Abc_Frame_t* abc_frame = Abc_FrameGetGlobalFrame();
      const std::string command = "source " + abc_script_file;
      child_proc[curr_mode_idx]
          = Cmd_CommandExecute(abc_frame, command.c_str());
      if (child_proc[curr_mode_idx]) {
        logger_->error(RMP, 6, "Error executing ABC command {}.", command);
        return;
      }
      Abc_Stop();
      // exit linked abc
      files_to_remove.emplace_back(abc_script_file);
    }
  }  // end modes

  // Inspect ABC results to choose blif with least instance count
  for (int curr_mode_idx = 0; curr_mode_idx < modes.size(); curr_mode_idx++) {
    // Skip failed ABC runs
    if (child_proc[curr_mode_idx] != 0) {
      continue;
    }

    output_blif_file_name_
        = prefix + std::to_string(curr_mode_idx) + "_crit_path_out.blif";
    const std::string abc_log_name = logfile_ + std::to_string(curr_mode_idx);

    int level_gain = 0;
    float delay = std::numeric_limits<float>::max();
    int num_instances = 0;
    bool success = readAbcLog(abc_log_name, level_gain, delay);
    if (success) {
      success
          = blif_.inspectBlif(output_blif_file_name_.c_str(), num_instances);
      logger_->report(
          "Optimized to {} instances in iteration {} with max path depth "
          "decrease of {}, delay of {}.",
          num_instances,
          curr_mode_idx,
          level_gain,
          delay);

      if (success) {
        if (is_area_mode_) {
          if (num_instances < best_inst_count) {
            best_inst_count = num_instances;
            best_blif = output_blif_file_name_;
          }
        } else {
          // Using only DELAY_4 for delay based gain since other modes not
          // showing good gains
          if (modes[curr_mode_idx] == Mode::DELAY_4) {
            best_delay_gain = delay;
            best_blif = output_blif_file_name_;
          }
        }
      }
    }
    files_to_remove.emplace_back(output_blif_file_name_);
  }

  if (best_inst_count < std::numeric_limits<int>::max()
      || best_delay_gain < std::numeric_limits<float>::max()) {
    // read back netlist
    debugPrint(logger_, RMP, "remap", 1, "Reading blif file {}.", best_blif);
    blif_.readBlif(best_blif.c_str(), block_);
    debugPrint(logger_,
               utl::RMP,
               "remap",
               1,
               "Number constants after restructure {}.",
               countConsts(block_));
  } else {
    logger_->info(
        RMP, 4, "All re-synthesis runs discarded, keeping original netlist.");
  }

  for (const auto& file_to_remove : files_to_remove) {
    if (!logger_->debugCheck(RMP, "remap", 1)) {
      std::error_code err;
      if (std::filesystem::remove(file_to_remove, err); err) {
        logger_->error(RMP, 11, "Fail to remove file {}", file_to_remove);
      }
    }
  }
}

void Restructure::postABC(float worst_slack)
{
  // Leave the parasitics up to date.
  estimate_parasitics_->estimateWireParasitics();
}
void Restructure::getEndPoints(sta::PinSet& ends,
                               bool area_mode,
                               unsigned max_depth)
{
  auto sta_state = open_sta_->search();
  sta::VertexSet* end_points = sta_state->endpoints();
  std::size_t path_found = end_points->size();
  logger_->report("Number of paths for restructure are {}", path_found);
  for (auto& end_point : *end_points) {
    if (!is_area_mode_) {
      sta::Path* path
          = open_sta_->vertexWorstSlackPath(end_point, sta::MinMax::max());
      sta::PathExpanded expanded(path, open_sta_);
      // Members in expanded include gate output and net so divide by 2
      logger_->report("Found path of depth {}", expanded.size() / 2);
      if (expanded.size() / 2 > max_depth) {
        ends.insert(end_point->pin());
        // Use only one end point to limit blob size for timing
        break;
      }
    } else {
      ends.insert(end_point->pin());
    }
  }

  // unconstrained end points
  if (is_area_mode_) {
    auto errors = open_sta_->checkTiming(false /*no_input_delay*/,
                                         false /*no_output_delay*/,
                                         false /*reg_multiple_clks*/,
                                         true /*reg_no_clks*/,
                                         true /*unconstrained_endpoints*/,
                                         false /*loops*/,
                                         false /*generated_clks*/);
    debugPrint(logger_, RMP, "remap", 1, "Size of errors = {}", errors.size());
    if (!errors.empty() && errors[0]->size() > 1) {
      sta::CheckError* error = errors[0];
      bool first = true;
      for (auto pinName : *error) {
        debugPrint(logger_, RMP, "remap", 1, "Unconstrained pin: {}", pinName);
        if (!first && open_sta_->getDbNetwork()->findPin(pinName)) {
          ends.insert(open_sta_->getDbNetwork()->findPin(pinName));
        }
        first = false;
      }
    }
    if (errors.size() > 1 && errors[1]->size() > 1) {
      sta::CheckError* error = errors[1];
      bool first = true;
      for (auto pinName : *error) {
        debugPrint(logger_, RMP, "remap", 1, "Unclocked pin: {}", pinName);
        if (!first && open_sta_->getDbNetwork()->findPin(pinName)) {
          ends.insert(open_sta_->getDbNetwork()->findPin(pinName));
        }
        first = false;
      }
    }
  }
  logger_->report("Found {} end points for restructure", ends.size());
}

int Restructure::countConsts(odb::dbBlock* top_block)
{
  int const_nets = 0;
  for (auto block_net : top_block->getNets()) {
    if (block_net->getSigType().isSupply()) {
      const_nets++;
    }
  }

  return const_nets;
}

void Restructure::removeConstCells()
{
  if (!hicell_.size() || !locell_.size()) {
    return;
  }

  odb::dbMaster* hicell_master = nullptr;
  odb::dbMTerm* hiterm = nullptr;
  odb::dbMaster* locell_master = nullptr;
  odb::dbMTerm* loterm = nullptr;

  for (auto&& lib : block_->getDb()->getLibs()) {
    hicell_master = lib->findMaster(hicell_.c_str());

    locell_master = lib->findMaster(locell_.c_str());
    if (locell_master && hicell_master) {
      break;
    }
  }
  if (!hicell_master || !locell_master) {
    return;
  }

  hiterm = hicell_master->findMTerm(hiport_.c_str());
  loterm = locell_master->findMTerm(loport_.c_str());
  if (!hiterm || !loterm) {
    return;
  }

  open_sta_->clearLogicConstants();
  open_sta_->findLogicConstants();
  std::set<odb::dbInst*> constInsts;
  int const_cnt = 1;
  for (auto inst : block_->getInsts()) {
    int outputs = 0;
    int const_outputs = 0;
    auto master = inst->getMaster();
    sta::LibertyCell* cell = open_sta_->getDbNetwork()->libertyCell(
        open_sta_->getDbNetwork()->dbToSta(master));
    if (cell->hasSequentials()) {
      continue;
    }

    for (auto&& iterm : inst->getITerms()) {
      if (iterm->getSigType() == odb::dbSigType::POWER
          || iterm->getSigType() == odb::dbSigType::GROUND) {
        continue;
      }

      if (iterm->getIoType() != odb::dbIoType::OUTPUT) {
        continue;
      }
      outputs++;
      auto pin = open_sta_->getDbNetwork()->dbToSta(iterm);
      sta::LogicValue pinVal = open_sta_->simLogicValue(pin);
      if (pinVal == sta::LogicValue::one || pinVal == sta::LogicValue::zero) {
        odb::dbNet* net = iterm->getNet();
        if (net) {
          odb::dbMaster* const_master = (pinVal == sta::LogicValue::one)
                                            ? hicell_master
                                            : locell_master;
          odb::dbMTerm* const_port
              = (pinVal == sta::LogicValue::one) ? hiterm : loterm;
          std::string inst_name = "rmp_const_" + std::to_string(const_cnt);
          debugPrint(logger_,
                     RMP,
                     "remap",
                     2,
                     "Adding cell {} inst {} for {}",
                     const_master->getName(),
                     inst_name,
                     inst->getName());
          auto new_inst
              = odb::dbInst::create(block_, const_master, inst_name.c_str());
          if (new_inst) {
            iterm->disconnect();
            new_inst->getITerm(const_port)->connect(net);
          } else {
            logger_->warn(RMP, 9, "Could not create instance {}.", inst_name);
          }
        }
        const_outputs++;
        const_cnt++;
      }
    }
    if (outputs > 0 && outputs == const_outputs) {
      constInsts.insert(inst);
    }
  }
  open_sta_->clearLogicConstants();

  debugPrint(
      logger_, RMP, "remap", 2, "Removing {} instances...", constInsts.size());

  for (auto inst : constInsts) {
    removeConstCell(inst);
  }
  logger_->report("Removed {} instances with constant outputs.",
                  constInsts.size());
}

void Restructure::removeConstCell(odb::dbInst* inst)
{
  for (auto iterm : inst->getITerms()) {
    iterm->disconnect();
  }
  odb::dbInst::destroy(inst);
}

bool Restructure::writeAbcScript(std::string file_name)
{
  std::ofstream script(file_name.c_str());

  if (!script.is_open()) {
    logger_->error(RMP, 3, "Cannot open file {} for writing.", file_name);
    return false;
  }

  for (const auto& lib_name : lib_file_names_) {
    // abc read_lib prints verbose by default, -v toggles to off to avoid read
    // time being printed
    std::string read_lib_str = "read_lib -v " + lib_name + "\n";
    script << read_lib_str;
  }

  script << "read_blif -n " << input_blif_file_name_ << std::endl;

  if (logger_->debugCheck(RMP, "remap", 1)) {
    script << "write_verilog " << input_blif_file_name_ + std::string(".v")
           << std::endl;
  }

  writeOptCommands(script);

  script << "write_blif " << output_blif_file_name_ << std::endl;

  if (logger_->debugCheck(RMP, "remap", 1)) {
    script << "write_verilog " << output_blif_file_name_ + std::string(".v")
           << std::endl;
  }

  script.close();

  return true;
}

void Restructure::writeOptCommands(std::ofstream& script)
{
  std::string choice
      = "alias choice \"fraig_store; resyn2; fraig_store; resyn2; fraig_store; "
        "fraig_restore\"";
  std::string choice2
      = "alias choice2 \"fraig_store; balance; fraig_store; resyn2; "
        "fraig_store; resyn2; fraig_store; resyn2; fraig_store; "
        "fraig_restore\"";
  script << "bdd; sop" << std::endl;

  script << "alias resyn2 \"balance; rewrite; refactor; balance; rewrite; "
            "rewrite -z; balance; refactor -z; rewrite -z; balance\""
         << std::endl;
  script << choice << std::endl;
  script << choice2 << std::endl;

  if (opt_mode_ == Mode::AREA_3) {
    script << "choice2" << std::endl;  // << "scleanup" << std::endl;
  } else {
    script << "resyn2" << std::endl;  // << "scleanup" << std::endl;
  }

  switch (opt_mode_) {
    case Mode::DELAY_1: {
      script << "map -D 0.01 -A 0.9 -B 0.2 -M 0 -p" << std::endl;
      script << "buffer -p -c" << std::endl;
      break;
    }
    case Mode::DELAY_2: {
      script << "choice" << std::endl;
      script << "map -D 0.01 -A 0.9 -B 0.2 -M 0 -p" << std::endl;
      script << "choice" << std::endl;
      script << "map -D 0.01" << std::endl;
      script << "buffer -p -c" << std::endl << "topo" << std::endl;
      break;
    }
    case Mode::DELAY_3: {
      script << "choice2" << std::endl;
      script << "map -D 0.01 -A 0.9 -B 0.2 -M 0 -p" << std::endl;
      script << "choice2" << std::endl;
      script << "map -D 0.01" << std::endl;
      script << "buffer -p -c" << std::endl << "topo" << std::endl;
      break;
    }
    case Mode::DELAY_4: {
      script << "choice2" << std::endl;
      script << "amap -F 20 -A 20 -C 5000 -Q 0.1 -m" << std::endl;
      script << "choice2" << std::endl;
      script << "map -D 0.01 -A 0.9 -B 0.2 -M 0 -p" << std::endl;
      script << "buffer -p -c" << std::endl;
      break;
    }
    case Mode::AREA_2:
    case Mode::AREA_3: {
      script << "choice2" << std::endl;
      script << "amap -m -Q 0.1 -F 20 -A 20 -C 5000" << std::endl;
      script << "choice2" << std::endl;
      script << "amap -m -Q 0.1 -F 20 -A 20 -C 5000" << std::endl;
      break;
    }
    case Mode::AREA_1:
    default: {
      script << "choice2" << std::endl;
      script << "amap -m -Q 0.1 -F 20 -A 20 -C 5000" << std::endl;
      break;
    }
  }
}

void Restructure::setMode(const char* mode_name)
{
  is_area_mode_ = true;

  if (!strcmp(mode_name, "timing")) {
    is_area_mode_ = false;
    opt_mode_ = Mode::DELAY_1;
  } else if (!strcmp(mode_name, "area")) {
    opt_mode_ = Mode::AREA_1;
  } else {
    logger_->warn(RMP, 10, "Mode {} not recognized.", mode_name);
  }
}

void Restructure::setTieHiPort(sta::LibertyPort* tieHiPort)
{
  if (tieHiPort) {
    hicell_ = tieHiPort->libertyCell()->name();
    hiport_ = tieHiPort->name();
  }
}

void Restructure::setTieLoPort(sta::LibertyPort* tieLoPort)
{
  if (tieLoPort) {
    locell_ = tieLoPort->libertyCell()->name();
    loport_ = tieLoPort->name();
  }
}

bool Restructure::readAbcLog(std::string abc_file_name,
                             int& level_gain,
                             float& final_delay)
{
  std::ifstream abc_file(abc_file_name);
  if (abc_file.bad()) {
    logger_->error(RMP, 2, "cannot open file {}", abc_file_name);
    return false;
  }
  debugPrint(
      logger_, utl::RMP, "remap", 1, "Reading ABC log {}.", abc_file_name);
  std::string buf;
  const char delimiter = ' ';
  bool status = true;
  std::vector<double> level;
  std::vector<float> delay;

  // read the file line by line
  while (std::getline(abc_file, buf)) {
    // convert the line in to stream:
    std::istringstream ss(buf);
    std::vector<std::string> tokens;

    // read the line, word by word
    while (std::getline(ss, buf, delimiter)) {
      tokens.push_back(buf);
    }

    if (!tokens.empty() && tokens[0] == "Error:") {
      status = false;
      logger_->warn(RMP,
                    5,
                    "ABC run failed, see log file {} for details.",
                    abc_file_name);
      break;
    }
    if (tokens.size() > 7 && tokens[tokens.size() - 3] == "lev"
        && tokens[tokens.size() - 2] == "=") {
      level.emplace_back(std::stoi(tokens[tokens.size() - 1]));
    }
    if (tokens.size() > 7) {
      std::string prev_token;
      for (std::string token : tokens) {
        if (prev_token == "delay" && token.at(0) == '=') {
          std::string delay_str = token;
          if (delay_str.size() > 1) {
            delay_str.erase(
                delay_str.begin());  // remove first char which is '='
            delay.emplace_back(std::stof(delay_str));
          }
          break;
        }
        prev_token = std::move(token);
      }
    }
  }

  if (level.size() > 1) {
    level_gain = level[0] - level[level.size() - 1];
  }
  if (delay.size() > 0) {
    final_delay = delay[delay.size() - 1];  // last value in file
  }
  return status;
}

static mockturtle::aig_network abc_to_mockturtle_aig(Aig_Man_t* pMan)
{
  using aig_ntk = mockturtle::aig_network;
  aig_ntk ntk;

  // Map ABC nodes to mockturtle signals
  std::unordered_map<Aig_Obj_t*, aig_ntk::signal> node_to_sig;

  // Constant 0
  auto const0 = ntk.get_constant(false);
  node_to_sig[Aig_ManConst0(pMan)] = const0;

  // Primary inputs (CIs)
  Aig_Obj_t* pObj;
  int i;

  Aig_ManForEachCi(pMan, pObj, i)
  {
    auto s = ntk.create_pi();
    node_to_sig[pObj] = s;
  }

  // Internal AND nodes (AIG nodes) in topological order
  Aig_ManForEachNode(pMan, pObj, i)
  {
    Aig_Obj_t* pF0 = Aig_ObjFanin0(pObj);
    Aig_Obj_t* pF1 = Aig_ObjFanin1(pObj);

    // get signals for fanins
    auto s0 = node_to_sig.at(pF0);
    auto s1 = node_to_sig.at(pF1);

    // apply input polarities (complemented edges)
    if (Aig_ObjFaninC0(pObj)) {
      s0 = ntk.create_not(s0);
    }
    if (Aig_ObjFaninC1(pObj)) {
      s1 = ntk.create_not(s1);
    }

    auto s = ntk.create_and(s0, s1);
    node_to_sig[pObj] = s;
  }

  // Primary outputs (COs)
  Aig_ManForEachCo(pMan, pObj, i)
  {
    Aig_Obj_t* pF = Aig_ObjFanin0(pObj);
    auto s = node_to_sig.at(pF);
    if (Aig_ObjFaninC0(pObj)) {
      s = ntk.create_not(s);
    }
    ntk.create_po(s);
  }

  return ntk;
}

using BlockNtk = mockturtle::cell_view<mockturtle::block_network>;
using Node = BlockNtk::node;
using Signal = BlockNtk::signal;

// Mapping info for one cell instance
struct CellMapping
{
  std::string master_name;              // dbMaster / Liberty cell name
  std::vector<std::string> input_pins;  // in fanin order (from gate::pins)
};

static odb::dbMaster* findMasterOrDie(odb::dbLib* lib, const std::string& name)
{
  if (!lib) {
    throw std::runtime_error("findMasterOrDie: dbLib is null");
  }
  odb::dbMaster* master = lib->findMaster(name.c_str());
  if (!master) {
    std::ostringstream oss;
    oss << "Cannot find dbMaster \"" << name << "\" in dbLib \""
        << lib->getName() << "\"";
    throw std::runtime_error(oss.str());
  }
  return master;
}

static std::vector<odb::dbMTerm*> getSignalOutputs(odb::dbMaster* master)
{
  std::vector<odb::dbMTerm*> outs;

  for (auto* mterm : master->getMTerms()) {
    auto io = mterm->getIoType();
    auto sig = mterm->getSigType();

    if (sig == odb::dbSigType::POWER || sig == odb::dbSigType::GROUND) {
      continue;
    }
    if (io == odb::dbIoType::INPUT) {
      continue;
    }
    // Accept OUTPUT, INOUT, FEEDTHRU here as "outputs"
    outs.push_back(mterm);
  }

  auto by_name = [](odb::dbMTerm* a, odb::dbMTerm* b) {
    return std::strcmp(a->getName().c_str(), b->getName().c_str()) < 0;
  };
  std::sort(outs.begin(), outs.end(), by_name);

  return outs;
}

static CellMapping map_cell_from_standard_cell(const BlockNtk& ntk,
                                               const Node& n,
                                               utl::Logger* logger)
{
  CellMapping m;

  const auto& sc = ntk.get_cell(n);  // cell_view API
  m.master_name = sc.name;           // must match Liberty/LEF cell name

  // logger->report("mapping cell {}", sc.name);

  if (sc.gates.empty()) {
    std::ostringstream oss;
    oss << "standard_cell \"" << sc.name << "\" has no gates (node "
        << ntk.node_to_index(n) << ")";
    throw std::runtime_error(oss.str());
  }

  const auto& first_gate = sc.gates.front();
  for (const auto& p : first_gate.pins) {
    m.input_pins.push_back(p.name);
  }

  return m;
}

static void import_block_network_to_db(sta::dbSta* sta,
                                       const BlockNtk& ntk_raw,
                                       odb::dbLib* lib,
                                       const std::string& block_name,
                                       utl::Logger* logger)
{
  const int N = 1150;
  mockturtle::topo_view<BlockNtk> ntk{ntk_raw};

  odb::dbChip* chip = sta->db()->getChip();

  odb::dbBlock* block = chip->getBlock();

  for (auto* inst : block->getInsts()) {
    odb::dbInst::destroy(inst);
  }

  for (auto* net : block->getNets()) {
    odb::dbNet::destroy(net);
  }

  for (auto* bterm : block->getBTerms()) {
    odb::dbBTerm::destroy(bterm);
  }

  const auto num_nodes = ntk.size();
  std::vector<std::vector<odb::dbNet*>> node_out_nets(num_nodes);

  // Const nets (for constant fanins)
  odb::dbNet* const0_net = nullptr;
  odb::dbNet* const1_net = nullptr;

  auto ensure_const_net = [&](bool value) -> odb::dbNet* {
    const char* net_name = value ? "CONST1" : "CONST0";
    odb::dbNet*& cache = value ? const1_net : const0_net;

    if (cache) {
      return cache;
    }

    odb::dbNet* net = block->findNet(net_name);
    if (!net) {
      net = odb::dbNet::create(block, net_name);
    }
    if (!net) {
      std::ostringstream oss;
      oss << "Failed to create or find const net " << net_name;
      throw std::runtime_error(oss.str());
    }

    cache = net;
    return net;
  };

  auto node_index = [&](const Node& n) -> std::size_t {
    return static_cast<std::size_t>(ntk.node_to_index(n));
  };

  // Primary inputs: nets + BTerms
  logger->report("primary inputs");
  {
    uint32_t pi_idx = 0;
    ntk.foreach_pi([&](const Node& n) {
      const auto idx = node_index(n);
      std::string name = "pi_" + std::to_string(pi_idx++);

      odb::dbNet* net = odb::dbNet::create(block, name.c_str());
      if (!net) {
        std::ostringstream oss;
        oss << "Failed to create PI net " << name;
        throw std::runtime_error(oss.str());
      }

      odb::dbBTerm* bt = odb::dbBTerm::create(net, name.c_str());
      bt->setSigType(odb::dbSigType::SIGNAL);
      bt->setIoType(odb::dbIoType::INPUT);

      node_out_nets[idx].resize(1);
      node_out_nets[idx][0] = net;

      // logger->report("\t{}", name);
    });
    logger->report("\tcount: {}", pi_idx);
  }

  // Gates: create instances + output nets (no inputs connected yet)
  logger->report("gates");
  ntk.foreach_gate([&](const Node& n) {
    const auto idx = node_index(n);

    CellMapping mapping = map_cell_from_standard_cell(ntk, n, logger);
    odb::dbMaster* master = findMasterOrDie(lib, mapping.master_name);

    std::string inst_name = "n_" + std::to_string(idx);
    odb::dbInst* inst = odb::dbInst::create(block, master, inst_name.c_str());
    if (!inst) {
      std::ostringstream oss;
      oss << "Failed to create dbInst " << inst_name << " for master "
          << mapping.master_name;
      throw std::runtime_error(oss.str());
    }

    auto out_mterms = getSignalOutputs(master);
    const uint32_t num_cell_outputs = static_cast<uint32_t>(out_mterms.size());
    const uint32_t num_node_outputs = ntk.num_outputs(n);

    if (num_node_outputs == 0) {
      // Shouldn't happen for mapped logic; skip
      return;
    }

    if (num_cell_outputs < num_node_outputs) {
      std::ostringstream oss;
      oss << "Cell " << mapping.master_name << " has only " << num_cell_outputs
          << " signal outputs but node " << idx << " has " << num_node_outputs
          << " outputs";
      throw std::runtime_error(oss.str());
    }

    node_out_nets[idx].resize(num_node_outputs);

    if (idx < N) {
      logger->report("\t{} {} {} {}",
                     inst_name,
                     mapping.master_name,
                     num_cell_outputs,
                     num_node_outputs);
    }

    for (uint32_t out_pin_idx = 0; out_pin_idx < num_node_outputs;
         ++out_pin_idx) {
      odb::dbMTerm* o_mterm = out_mterms[out_pin_idx];
      const std::string pin_name = o_mterm->getName();

      odb::dbITerm* o_iterm = inst->findITerm(pin_name.c_str());
      if (!o_iterm) {
        std::ostringstream oss;
        oss << "Instance " << inst_name << " has no OUTPUT ITerm \"" << pin_name
            << "\"";
        throw std::runtime_error(oss.str());
      }

      std::string net_name
          = "n_" + std::to_string(idx) + "_o" + std::to_string(out_pin_idx);

      odb::dbNet* net = odb::dbNet::create(block, net_name.c_str());
      if (!net) {
        std::ostringstream oss;
        oss << "Failed to create net " << net_name;
        throw std::runtime_error(oss.str());
      }

      o_iterm->connect(net);
      node_out_nets[idx][out_pin_idx] = net;

      if (idx < N) {
        logger->report("\t\t{} {}", pin_name, net_name);
      }
    }
  });

  ntk.foreach_gate([&](const Node& n) {
    const auto idx = node_index(n);

    CellMapping mapping = map_cell_from_standard_cell(ntk, n, logger);
    std::string inst_name = "n_" + std::to_string(idx);
    odb::dbInst* inst = block->findInst(inst_name.c_str());
    if (!inst) {
      std::ostringstream oss;
      oss << "Internal error: instance " << inst_name << " not found";
      throw std::runtime_error(oss.str());
    }

    if (idx < N) {
      logger->report("\t{}", inst_name);
    }

    uint32_t fanin_idx = 0;
    ntk.foreach_fanin(n, [&](const Signal& f) {
      if (fanin_idx >= mapping.input_pins.size()) {
        std::ostringstream oss;
        oss << "Not enough input pins in cell " << mapping.master_name
            << " (node " << idx << "), fanins=" << fanin_idx
            << " inputs=" << mapping.input_pins.size();
        throw std::runtime_error(oss.str());
      }

      odb::dbNet* src_net = nullptr;
      const Node src_node = ntk.get_node(f);

      if (ntk.is_constant(src_node)) {
        // constant node; complemented = 1
        const bool value = ntk.is_complemented(f);
        src_net = ensure_const_net(value);
      } else {
        const auto src_idx = node_index(src_node);
        uint32_t out_pin_idx = 0;
        if (ntk.is_multioutput(src_node)) {
          out_pin_idx = ntk.get_output_pin(f);
        }

        if (src_idx >= node_out_nets.size()
            || out_pin_idx >= node_out_nets[src_idx].size()
            || node_out_nets[src_idx][out_pin_idx] == nullptr) {
          std::ostringstream oss;
          oss << "Missing driver net for fanin of node " << idx;
          throw std::runtime_error(oss.str());
        }
        src_net = node_out_nets[src_idx][out_pin_idx];
        if (idx < N) {
          logger->report("\t\t{} -> {}", src_idx, out_pin_idx);
        }
      }

      const std::string& pin_name = mapping.input_pins[fanin_idx++];
      odb::dbITerm* it = inst->findITerm(pin_name.c_str());
      if (!it) {
        std::ostringstream oss;
        oss << "Master " << mapping.master_name << " has no input ITerm \""
            << pin_name << "\"";
        throw std::runtime_error(oss.str());
      }
      it->connect(src_net);

      if (idx < N) {
        logger->report("\t\t{} {}", pin_name, src_node);
      }
    });
  });

  // Primary Outputs: BTerms attached to driver nets
  logger->report("primary outputs");
  {
    uint32_t po_idx = 0;
    ntk.foreach_po([&](const Signal& f) {
      odb::dbNet* src_net = nullptr;
      const Node src_node = ntk.get_node(f);

      if (ntk.is_constant(src_node)) {
        const bool value = ntk.is_complemented(f);
        src_net = ensure_const_net(value);
      } else {
        const auto idx = node_index(src_node);
        uint32_t out_pin_idx = 0;
        if (ntk.is_multioutput(src_node)) {
          out_pin_idx = ntk.get_output_pin(f);
        }

        if (idx >= node_out_nets.size()
            || out_pin_idx >= node_out_nets[idx].size()
            || node_out_nets[idx][out_pin_idx] == nullptr) {
          std::ostringstream oss;
          oss << "Missing driver net for PO " << po_idx;
          throw std::runtime_error(oss.str());
        }
        src_net = node_out_nets[idx][out_pin_idx];
      }

      std::string name = "po_" + std::to_string(po_idx++);
      odb::dbBTerm* bt = odb::dbBTerm::create(src_net, name.c_str());
      bt->setSigType(odb::dbSigType::SIGNAL);
      bt->setIoType(odb::dbIoType::OUTPUT);

      // logger->report("\t{}", name);
    });
    logger->report("\tcount: {}", po_idx);
  }

  logger->report("import done");
}

void Restructure::emap(sta::Corner* corner, char* genlib_file_name, bool map_multioutput, bool verbose, char* workdir_name)
{
  mockturtle::emap_params ps;

  ps.map_multioutput = map_multioutput;
  ps.verbose = verbose;

  switch (opt_mode_) {
    case Mode::AREA_1: {
      ps.area_oriented_mapping = true;
      break;
    }
    case Mode::DELAY_1: {
      ps.area_oriented_mapping = false;
      ps.relax_required = 0.0;
      break;
    }
    default: {
      break;
    }
  }

  logger_->report("Creating AIG network");

  sta::dbNetwork* network = open_sta_->getDbNetwork();

  // Disable incremental timing.
  open_sta_->graphDelayCalc()->delaysInvalid();
  open_sta_->search()->arrivalsInvalid();
  open_sta_->search()->endpointsInvalid();

  cut::LogicExtractorFactory logic_extractor(open_sta_, logger_);
  for (sta::Vertex* endpoint : *open_sta_->endpoints()) {
    logic_extractor.AppendEndpoint(endpoint);
  }

  Aig_Man_t* aig;

  cut::AbcLibraryFactory factory(logger_);
  factory.AddDbSta(open_sta_);
  factory.AddResizer(resizer_);
  factory.SetCorner(corner);
  cut::AbcLibrary abc_library = factory.Build();

  cut::LogicCut cut = logic_extractor.BuildLogicCut(abc_library);

  utl::UniquePtrWithDeleter<abc::Abc_Ntk_t> mapped_abc_network
      = cut.BuildMappedAbcNetwork(abc_library, network, logger_);

  utl::UniquePtrWithDeleter<abc::Abc_Ntk_t> current_network(
      abc::Abc_NtkToLogic(
          const_cast<abc::Abc_Ntk_t*>(mapped_abc_network.get())),
      &abc::Abc_NtkDelete);

  {
    auto library
        = static_cast<abc::Mio_Library_t*>(mapped_abc_network->pManFunc);

    // Install library for NtkMap
    abc::Abc_FrameSetLibGen(library);

    logger_->report("Mapped ABC network has {} nodes and {} POs.",
                    abc::Abc_NtkNodeNum(current_network.get()),
                    abc::Abc_NtkPoNum(current_network.get()));

    current_network->pManFunc = library;

    {
      auto ntk = current_network.get();
      assert(!Abc_NtkIsStrash(ntk));
      // derive comb GIA
      auto strash = Abc_NtkStrash(ntk, false, true, false);
      aig = Abc_NtkToDar(strash, false, false);
      Abc_NtkDelete(strash);
    }
  }

  mockturtle::aig_network ntk = abc_to_mockturtle_aig(aig);

  logger_->report("Reading genlib file {}", genlib_file_name);
  std::vector<mockturtle::gate> gates;
  {
    auto code = lorina::read_genlib(std::string(genlib_file_name), mockturtle::genlib_reader(gates));

    if (code != lorina::return_code::success) {
      logger_->report("Error reading genlib file");
      return;
    }
  }

  static constexpr unsigned MaxInputs = 9u;
  mockturtle::tech_library<MaxInputs> tech_lib(gates);

  mockturtle::emap_stats st;

  logger_->report("Running emap");
  mockturtle::cell_view<mockturtle::block_network> mapped_ntk
      = mockturtle::emap(ntk, tech_lib, ps, &st);

  logger_->report(
      "Extended technology mapping stats:\n\tarea: {}\n\tdelay: {}\n\tpower: "
      "{}\n\tinverters: {}\n\tmultioutput gates: {}\n\ttime multioutput: "
      "{}\n\ttime total: {}",
      st.area,
      st.delay,
      st.power,
      st.inverters,
      st.multioutput_gates,
      st.time_multioutput,
      st.time_total);

  mapped_ntk.report_cells_usage();
  mapped_ntk.report_stats();

  mockturtle::write_verilog_with_cell(mapped_ntk, "mapped.v");

  odb::dbLib* lib = *ord::OpenRoad::openRoad()->getDb()->getLibs().begin();

  import_block_network_to_db(open_sta_, mapped_ntk, lib, "aes", logger_);

  // Notify OpenROAD
  ord::OpenRoad::openRoad()->designCreated();
}

}  // namespace rmp
