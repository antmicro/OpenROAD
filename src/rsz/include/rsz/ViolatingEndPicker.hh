#pragma once
#include "rsz/Resizer.hh"

namespace rsz {

class ViolatingEndsPicker;
// VIOLATING_ENDS_PICKER_HEADER should implement this class

class ViolatingEndsPickerAPI
{
// This abstract class is used only for documenting and enforcing the API of ViolatingEndsPicker
// (we don't use virtual calls)
public:
  virtual ~ViolatingEndsPickerAPI() {};
  virtual void init(dbSta* sta, float setup_slack_margin, int max_passes) = 0;

  virtual Vertex* getCurrent() = 0;
  virtual bool shouldGetNext() = 0;
  virtual Vertex* getNext() = 0;

  virtual void noteRepair(bool improved) = 0;

  virtual bool shouldAllowSlackDecrease() = 0;

  virtual size_t size() = 0;
  virtual size_t empty() = 0;
};
}  // namespace rsz


// VIOLATING_ENDS_PICKER_HEADER should contain definition of ViolatingEndsPicker that implements ViolatingEndsPickerAPI
#ifndef VIOLATING_ENDS_PICKER_HEADER
#define VIOLATING_ENDS_PICKER_HEADER "rsz/SimpleViolatingEndsPicker.hh"
#endif

#include VIOLATING_ENDS_PICKER_HEADER

