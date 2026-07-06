////////////////////////////////////////////////////////////////////////
// Class:       NuGraphInferenceSonicTriton
// Plugin Type: producer (Unknown Unknown)
// File:        NuGraphInferenceSonicTriton_module.cc
//
// Generated at Tue Nov 14 14:41:30 2023 by Giuseppe Cerati using cetskelgen
// from  version .
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Utilities/make_tool.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "fhiclcpp/ParameterSet.h"
#include "fhiclcpp/types/Table.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include <limits>
#include <memory>

#include "lardataobj/AnalysisBase/MVAOutput.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/Vertex.h"
#include "larrecodnn/ImagePatternAlgs/NuSonic/Triton/TritonClient.h"
#include "larrecodnn/ImagePatternAlgs/NuSonic/Triton/TritonData.h"

#include "larrecodnn/NuGraph/Tools/DecoderToolBase.h"
#include "larrecodnn/NuGraph/Tools/LoaderToolBase.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class NuGraphInferenceSonicTriton;

using anab::FeatureVector;
using anab::MVADescription;
using recob::Hit;
using std::vector;

class NuGraphInferenceSonicTriton : public art::EDProducer {
public:
  explicit NuGraphInferenceSonicTriton(fhicl::ParameterSet const& p);

  // Plugins should not be copied or assigned.
  NuGraphInferenceSonicTriton(NuGraphInferenceSonicTriton const&) = delete;
  NuGraphInferenceSonicTriton(NuGraphInferenceSonicTriton&&) = delete;
  NuGraphInferenceSonicTriton& operator=(NuGraphInferenceSonicTriton const&) = delete;
  NuGraphInferenceSonicTriton& operator=(NuGraphInferenceSonicTriton&&) = delete;

  // Required functions.
  void produce(art::Event& e) override;

private:
  size_t minHits;
  bool debug;
  vector<std::string> planes;
  fhicl::ParameterSet tritonPset;
  std::unique_ptr<lartriton::TritonClient> triton_client;

  // loader tool
  std::unique_ptr<LoaderToolBase> _loaderTool;
  // decoder tools
  std::vector<std::unique_ptr<DecoderToolBase>> _decoderToolsVec;

  template <class T>
  void setShapeAndToServer(lartriton::TritonData<triton::client::InferInput>& triton_input,
                           vector<T>& vec,
                           size_t batchSize)
  {
    triton_input.setShape({static_cast<long int>(vec.size())});
    triton_input.toServer(
      std::make_shared<lartriton::TritonInput<T>>(lartriton::TritonInput<T>(batchSize, vec)));
  }
};

NuGraphInferenceSonicTriton::NuGraphInferenceSonicTriton(fhicl::ParameterSet const& p)
  : EDProducer{p}
  , minHits(p.get<size_t>("minHits"))
  , debug(p.get<bool>("debug"))
  , planes(p.get<vector<std::string>>("planes"))
  , tritonPset(p.get<fhicl::ParameterSet>("TritonConfig"))
{

  // ... Create the Triton inference client
  if (debug) std::cout << "TritonConfig: " << tritonPset.to_string() << std::endl;
  triton_client = std::make_unique<lartriton::TritonClient>(tritonPset);

  // Loader Tool
  _loaderTool = art::make_tool<LoaderToolBase>(p.get<fhicl::ParameterSet>("LoaderTool"));
  _loaderTool->setDebugAndPlanes(debug, planes);

  // configure and construct Decoder Tools
  auto const tool_psets = p.get<fhicl::ParameterSet>("DecoderTools");
  for (auto const& tool_pset_labels : tool_psets.get_pset_names()) {
    std::cout << "decoder lablel: " << tool_pset_labels << std::endl;
    auto const tool_pset = tool_psets.get<fhicl::ParameterSet>(tool_pset_labels);
    _decoderToolsVec.push_back(art::make_tool<DecoderToolBase>(tool_pset));
    _decoderToolsVec.back()->setDebugAndPlanes(debug, planes);
    _decoderToolsVec.back()->declareProducts(producesCollector());
  }
}

void NuGraphInferenceSonicTriton::produce(art::Event& e)
{

  //
  // Load the data and fill the graph inputs
  //
  vector<art::Ptr<Hit>> hitlist;
  vector<vector<size_t>> idsmap;
  vector<NuGraphInput> graphinputs;
  _loaderTool->loadData(e, hitlist, graphinputs, idsmap);

  if (debug) std::cout << "Hits size=" << hitlist.size() << std::endl;
  if (hitlist.size() < minHits) {
    // Writing the empty outputs to the output root file
    for (size_t i = 0; i < _decoderToolsVec.size(); i++) {
      _decoderToolsVec[i]->writeEmptyToEvent(e, idsmap);
    }
    return;
  }

  //
  // NuSonic Triton Server section
  //
  auto start = std::chrono::high_resolution_clock::now();
  //
  //Here the input should be sent to Triton
  triton_client->reset();
  size_t batchSize = 1; //the code below assumes/has only been tested for batch size = 1
  triton_client->setBatchSize(batchSize); // set batch size
  //
  auto& inputs = triton_client->input();
  for (auto& input_pair : inputs) {
    const std::string& key = input_pair.first;
    auto& triton_input = input_pair.second;
    //
    for (auto& gi : graphinputs) {
      if (key != gi.input_name) continue;
      if (gi.isInt)
        setShapeAndToServer(triton_input, gi.input_int32_vec, batchSize);
      else
        setShapeAndToServer(triton_input, gi.input_float_vec, batchSize);
    }
  }
  // ~~~~ Send inference request
  triton_client->dispatch();
  // ~~~~ Retrieve inference results
  auto& infer_result = triton_client->output();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken for inference: " << elapsed.count() << " seconds" << std::endl;

  //
  // Get pointers to the result returned and write to the event
  //
  vector<NuGraphOutput> infer_output;
  for (const auto& [name, data] : infer_result) {
    const auto& prob = data.fromServer<float>();
    std::vector<float> out_data(prob[0].begin(), prob[0].end());
    infer_output.emplace_back(name, std::move(out_data));
  }

  // Write the outputs to the output root file
  for (size_t i = 0; i < _decoderToolsVec.size(); i++) {
    _decoderToolsVec[i]->writeToEvent(e, idsmap, infer_output);
  }
}

DEFINE_ART_MODULE(NuGraphInferenceSonicTriton)
