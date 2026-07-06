#ifndef DECODERTOOLBASE_H
#define DECODERTOOLBASE_H

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "fhiclcpp/ParameterSet.h"

#include <string>
#include <utility>
#include <vector>

using std::string;
using std::vector;

struct NuGraphOutput {
  NuGraphOutput(string s, vector<float> vf) : output_name(s), output_vec(std::move(vf)) {}
  string output_name;
  vector<float> output_vec;
};

class DecoderToolBase {

public:
  /**
     *  @brief  Virtual Destructor
     */
  virtual ~DecoderToolBase() noexcept = default;

  /**
     *  @brief Construcutor
     *
     *  @param ParameterSet  The input set of parameters for configuration
     */
  DecoderToolBase(fhicl::ParameterSet const& p)
    : instancename{p.get<std::string>("instanceName", "filter")}
    , outputname{p.get<std::string>("outputName", "x_filter_")}
  {}

  /**
     * @brief declareProducts function
     *
     * @param art::ProducesCollector
     */
  virtual void declareProducts(art::ProducesCollector&) = 0;

  /**
     * @brief writeEmptyToEvent function
     *
     * @param art::Event event record
     */
  virtual void writeEmptyToEvent(art::Event& e, const vector<vector<size_t>>& idsmap) = 0;

  /**
     * @brief writeToEvent function
     *
     * @param art::Event event record
     * @param idsmap
     * @param infer_output
     */
  virtual void writeToEvent(art::Event& e,
                            const vector<vector<size_t>>& idsmap,
                            const vector<NuGraphOutput>& infer_output) = 0;

  // Function to print elements of a vector<float>
  void printVector(const std::vector<float>& vec)
  {
    for (size_t i = 0; i < vec.size(); ++i) {
      std::cout << vec[i];
      // Print space unless it's the last element
      if (i != vec.size() - 1) { std::cout << " "; }
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }

  template <typename T, size_t N>
  void softmax(std::array<T, N>& arr)
  {
    T m = -std::numeric_limits<T>::max();
    for (size_t i = 0; i < arr.size(); i++) {
      if (arr[i] > m) { m = arr[i]; }
    }
    T sum = 0.0;
    for (size_t i = 0; i < arr.size(); i++) {
      sum += expf(arr[i] - m);
    }
    T offset = m + logf(sum);
    for (size_t i = 0; i < arr.size(); i++) {
      arr[i] = expf(arr[i] - offset);
    }
    return;
  }

  void setDebugAndPlanes(bool d, vector<std::string>& p)
  {
    debug = d;
    planes = p;
  }

protected:
  bool debug;
  vector<std::string> planes;
  std::string instancename;
  std::string outputname;
};

#endif
