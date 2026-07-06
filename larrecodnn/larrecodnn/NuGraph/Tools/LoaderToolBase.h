#ifndef LOADERTOOLBASE_H
#define LOADERTOOLBASE_H

#include "art/Framework/Principal/Event.h"
#include "fhiclcpp/ParameterSet.h"

#include "lardataobj/RecoBase/Hit.h"

#include <string>
#include <utility>
#include <vector>

using std::string;
using std::vector;

struct NuGraphInput {
  NuGraphInput(string s, vector<int32_t> vi)
    : input_name(s), isInt(true), input_int32_vec(std::move(vi))
  {}
  NuGraphInput(string s, vector<float> vf)
    : input_name(s), isInt(false), input_float_vec(std::move(vf))
  {}
  string input_name;
  bool isInt;
  vector<int32_t> input_int32_vec;
  vector<float> input_float_vec;
};

class LoaderToolBase {

public:
  /**
   *  @brief  Virtual Destructor
   */
  virtual ~LoaderToolBase() noexcept = default;

  /**
   * @brief loadData virtual function
   *
   * @param art::Event event record, list of input, idsmap
   */
  virtual void loadData(art::Event& e,
                        vector<art::Ptr<recob::Hit>>& hitlist,
                        vector<NuGraphInput>& inputs,
                        vector<vector<size_t>>& idsmap) = 0;

  void setDebugAndPlanes(bool d, vector<std::string>& p)
  {
    debug = d;
    planes = p;
  }

protected:
  bool debug;
  vector<std::string> planes;
};

#endif
