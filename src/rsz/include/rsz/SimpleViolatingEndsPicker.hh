#pragma once
#include "rsz/ViolatingEndPicker.hh"
#include "sta/Graph.hh"

namespace rsz {
class ViolatingEndsPicker : ViolatingEndsPickerAPI
{
public:
  void init(dbSta* sta, const float setup_slack_margin, int max_passes) override
  {
    this->max_passes = max_passes;
    sta_ = sta;
    
    const VertexSet* endpoints = sta_->endpoints();
    // logger_->setDebugLevel(RSZ, "repair_setup", 2);
    // Should check here whether we can figure out the clock domain for each
    // vertex. This may be the place where we can do some round robin fun to
    // individually control each clock domain instead of just fixating on fixing
    // one.
    for (Vertex* end : *endpoints) {
      const Slack end_slack = sta_->vertexSlack(end, MinMax::max());
      if (end_slack < setup_slack_margin) {
        violating_ends.emplace_back(end, end_slack);
      }
    }
    std::stable_sort(violating_ends.begin(),
                     violating_ends.end(),
                     [](const auto& end_slack1, const auto& end_slack2) {
                       return end_slack1.second < end_slack2.second;
                     });

    it = violating_ends.begin();
  }

  Vertex* getCurrent() override {
    if(it == violating_ends.end()) {
      return nullptr;
    }
    return it->first;
  }

  bool shouldGetNext() override {
    return pass > max_passes;
  }

  Vertex* getNext() override {
    resetEndpointStats();
    it++;
    return getCurrent();
  }

  void noteRepair(bool improved) override {
    pass++;
    if(!improved) {
      ++decreasing_slack_passes;
    }
  }

  bool shouldAllowSlackDecrease() override
  {
    return decreasing_slack_passes <= decreasing_slack_max_passes_;
  }

  size_t size() override {
    return violating_ends.size();
  }
    
  size_t empty() override {
    return violating_ends.empty();
  }

private:
  dbSta* sta_ = nullptr;
  std::vector<std::pair<Vertex*, Slack>> violating_ends;
  std::vector<std::pair<Vertex*, Slack>>::iterator it;
  int max_passes;

  static constexpr int decreasing_slack_max_passes_ = 50;

  void resetEndpointStats() {
    pass = 1;
    decreasing_slack_passes = 0;
  }

  // data specific to currently examined endpoint
  int pass = 1;
  int decreasing_slack_passes = 0;
};
}  // namespace rsz

