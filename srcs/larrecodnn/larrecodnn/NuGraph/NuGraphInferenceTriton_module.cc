////////////////////////////////////////////////////////////////////////
// Class:       NuGraphInferenceTriton
// Plugin Type: producer (Unknown Unknown)
// File:        NuGraphInferenceTriton_module.cc
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

#include <array>
#include <limits>
#include <memory>

#include "lardataobj/AnalysisBase/MVAOutput.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/Vertex.h"
#include "larrecodnn/NuGraph/Tools/DecoderToolBase.h"
#include "larrecodnn/NuGraph/Tools/LoaderToolBase.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "grpc_client.h"

class NuGraphInferenceTriton;

using anab::FeatureVector;
using anab::MVADescription;
using recob::Hit;
using std::array;
using std::vector;

#define FAIL_IF_ERR(X, MSG)                                                                        \
  {                                                                                                \
    tc::Error err = (X);                                                                           \
    if (!err.IsOk()) {                                                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl;                                 \
      exit(1);                                                                                     \
    }                                                                                              \
  }
namespace tc = triton::client;

class NuGraphInferenceTriton : public art::EDProducer {
public:
  explicit NuGraphInferenceTriton(fhicl::ParameterSet const& p);

  // Plugins should not be copied or assigned.
  NuGraphInferenceTriton(NuGraphInferenceTriton const&) = delete;
  NuGraphInferenceTriton(NuGraphInferenceTriton&&) = delete;
  NuGraphInferenceTriton& operator=(NuGraphInferenceTriton const&) = delete;
  NuGraphInferenceTriton& operator=(NuGraphInferenceTriton&&) = delete;

  // Required functions.
  void produce(art::Event& e) override;

private:
  size_t minHits;
  bool debug;
  vector<std::string> planes;
  std::string inference_url;
  std::string inference_model_name;
  std::string model_version;
  bool inference_ssl;
  std::string ssl_root_certificates;
  std::string ssl_private_key;
  std::string ssl_certificate_chain;
  bool verbose;
  uint32_t client_timeout;

  // loader tool
  std::unique_ptr<LoaderToolBase> _loaderTool;
  // decoder tools
  std::vector<std::unique_ptr<DecoderToolBase>> _decoderToolsVec;
};

NuGraphInferenceTriton::NuGraphInferenceTriton(fhicl::ParameterSet const& p)
  : EDProducer{p}
  , minHits(p.get<size_t>("minHits"))
  , debug(p.get<bool>("debug"))
  , planes(p.get<vector<std::string>>("planes"))
{

  fhicl::ParameterSet tritonPset = p.get<fhicl::ParameterSet>("TritonConfig");
  inference_url = tritonPset.get<std::string>("serverURL");
  inference_model_name = tritonPset.get<std::string>("modelName");
  inference_ssl = tritonPset.get<bool>("ssl");
  ssl_root_certificates = tritonPset.get<std::string>("sslRootCertificates", "");
  ssl_private_key = tritonPset.get<std::string>("sslPrivateKey", "");
  ssl_certificate_chain = tritonPset.get<std::string>("sslCertificateChain", "");
  verbose = tritonPset.get<bool>("verbose", "false");
  model_version = tritonPset.get<std::string>("modelVersion", "");
  client_timeout = tritonPset.get<unsigned>("timeout", 0);

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

void NuGraphInferenceTriton::produce(art::Event& e)
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
  // Triton-specific section
  //
  const vector<int32_t>* hit_table_hit_id_data = nullptr;
  const vector<int32_t>* hit_table_local_plane_data = nullptr;
  const vector<float>* hit_table_local_time_data = nullptr;
  const vector<int32_t>* hit_table_local_wire_data = nullptr;
  const vector<float>* hit_table_integral_data = nullptr;
  const vector<float>* hit_table_rms_data = nullptr;
  const vector<int32_t>* spacepoint_table_spacepoint_id_data = nullptr;
  const vector<int32_t>* spacepoint_table_hit_id_u_data = nullptr;
  const vector<int32_t>* spacepoint_table_hit_id_v_data = nullptr;
  const vector<int32_t>* spacepoint_table_hit_id_y_data = nullptr;
  for (const auto& gi : graphinputs) {
    if (gi.input_name == "hit_table_hit_id")
      hit_table_hit_id_data = &gi.input_int32_vec;
    else if (gi.input_name == "hit_table_local_plane")
      hit_table_local_plane_data = &gi.input_int32_vec;
    else if (gi.input_name == "hit_table_local_time")
      hit_table_local_time_data = &gi.input_float_vec;
    else if (gi.input_name == "hit_table_local_wire")
      hit_table_local_wire_data = &gi.input_int32_vec;
    else if (gi.input_name == "hit_table_integral")
      hit_table_integral_data = &gi.input_float_vec;
    else if (gi.input_name == "hit_table_rms")
      hit_table_rms_data = &gi.input_float_vec;
    else if (gi.input_name == "spacepoint_table_spacepoint_id")
      spacepoint_table_spacepoint_id_data = &gi.input_int32_vec;
    else if (gi.input_name == "spacepoint_table_hit_id_u")
      spacepoint_table_hit_id_u_data = &gi.input_int32_vec;
    else if (gi.input_name == "spacepoint_table_hit_id_v")
      spacepoint_table_hit_id_v_data = &gi.input_int32_vec;
    else if (gi.input_name == "spacepoint_table_hit_id_y")
      spacepoint_table_hit_id_y_data = &gi.input_int32_vec;
  }

  //Here the input should be sent to Triton
  tc::Headers http_headers;
  grpc_compression_algorithm compression_algorithm = grpc_compression_algorithm::GRPC_COMPRESS_NONE;
  bool test_use_cached_channel = false;
  bool use_cached_channel = true;

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  tc::SslOptions ssl_options = tc::SslOptions();
  std::string err;
  if (inference_ssl) {
    ssl_options.root_certificates = ssl_root_certificates;
    ssl_options.private_key = ssl_private_key;
    ssl_options.certificate_chain = ssl_certificate_chain;
    err = "unable to create secure grpc client";
  }
  else {
    err = "unable to create grpc client";
  }
  // Run with the same name to ensure cached channel is not used
  int numRuns = test_use_cached_channel ? 2 : 1;
  for (int i = 0; i < numRuns; ++i) {
    FAIL_IF_ERR(tc::InferenceServerGrpcClient::Create(&client,
                                                      inference_url,
                                                      verbose,
                                                      inference_ssl,
                                                      ssl_options,
                                                      tc::KeepAliveOptions(),
                                                      use_cached_channel),
                err);

    std::vector<int64_t> hit_table_shape{int64_t(hit_table_hit_id_data->size())};
    std::vector<int64_t> spacepoint_table_shape{
      int64_t(spacepoint_table_spacepoint_id_data->size())};

    // Initialize the inputs with the data.
    tc::InferInput* hit_table_hit_id;
    tc::InferInput* hit_table_local_plane;
    tc::InferInput* hit_table_local_time;
    tc::InferInput* hit_table_local_wire;
    tc::InferInput* hit_table_integral;
    tc::InferInput* hit_table_rms;

    tc::InferInput* spacepoint_table_spacepoint_id;
    tc::InferInput* spacepoint_table_hit_id_u;
    tc::InferInput* spacepoint_table_hit_id_v;
    tc::InferInput* spacepoint_table_hit_id_y;

    FAIL_IF_ERR(
      tc::InferInput::Create(&hit_table_hit_id, "hit_table_hit_id", hit_table_shape, "INT32"),
      "unable to get hit_table_hit_id");
    std::shared_ptr<tc::InferInput> hit_table_hit_id_ptr;
    hit_table_hit_id_ptr.reset(hit_table_hit_id);

    FAIL_IF_ERR(tc::InferInput::Create(
                  &hit_table_local_plane, "hit_table_local_plane", hit_table_shape, "INT32"),
                "unable to get hit_table_local_plane");
    std::shared_ptr<tc::InferInput> hit_table_local_plane_ptr;
    hit_table_local_plane_ptr.reset(hit_table_local_plane);

    FAIL_IF_ERR(tc::InferInput::Create(
                  &hit_table_local_time, "hit_table_local_time", hit_table_shape, "FP32"),
                "unable to get hit_table_local_time");
    std::shared_ptr<tc::InferInput> hit_table_local_time_ptr;
    hit_table_local_time_ptr.reset(hit_table_local_time);

    FAIL_IF_ERR(tc::InferInput::Create(
                  &hit_table_local_wire, "hit_table_local_wire", hit_table_shape, "INT32"),
                "unable to get hit_table_local_wire");
    std::shared_ptr<tc::InferInput> hit_table_local_wire_ptr;
    hit_table_local_wire_ptr.reset(hit_table_local_wire);

    FAIL_IF_ERR(
      tc::InferInput::Create(&hit_table_integral, "hit_table_integral", hit_table_shape, "FP32"),
      "unable to get hit_table_integral");
    std::shared_ptr<tc::InferInput> hit_table_integral_ptr;
    hit_table_integral_ptr.reset(hit_table_integral);

    FAIL_IF_ERR(tc::InferInput::Create(&hit_table_rms, "hit_table_rms", hit_table_shape, "FP32"),
                "unable to get hit_table_rms");
    std::shared_ptr<tc::InferInput> hit_table_rms_ptr;
    hit_table_rms_ptr.reset(hit_table_rms);

    FAIL_IF_ERR(tc::InferInput::Create(&spacepoint_table_spacepoint_id,
                                       "spacepoint_table_spacepoint_id",
                                       spacepoint_table_shape,
                                       "INT32"),
                "unable to get spacepoint_table_spacepoint_id");
    std::shared_ptr<tc::InferInput> spacepoint_table_spacepoint_id_ptr;
    spacepoint_table_spacepoint_id_ptr.reset(spacepoint_table_spacepoint_id);

    FAIL_IF_ERR(
      tc::InferInput::Create(
        &spacepoint_table_hit_id_u, "spacepoint_table_hit_id_u", spacepoint_table_shape, "INT32"),
      "unable to get spacepoint_table_spacepoint_hit_id_u");
    std::shared_ptr<tc::InferInput> spacepoint_table_hit_id_u_ptr;
    spacepoint_table_hit_id_u_ptr.reset(spacepoint_table_hit_id_u);

    FAIL_IF_ERR(
      tc::InferInput::Create(
        &spacepoint_table_hit_id_v, "spacepoint_table_hit_id_v", spacepoint_table_shape, "INT32"),
      "unable to get spacepoint_table_spacepoint_hit_id_v");
    std::shared_ptr<tc::InferInput> spacepoint_table_hit_id_v_ptr;
    spacepoint_table_hit_id_v_ptr.reset(spacepoint_table_hit_id_v);

    FAIL_IF_ERR(
      tc::InferInput::Create(
        &spacepoint_table_hit_id_y, "spacepoint_table_hit_id_y", spacepoint_table_shape, "INT32"),
      "unable to get spacepoint_table_spacepoint_hit_id_y");
    std::shared_ptr<tc::InferInput> spacepoint_table_hit_id_y_ptr;
    spacepoint_table_hit_id_y_ptr.reset(spacepoint_table_hit_id_y);

    FAIL_IF_ERR(hit_table_hit_id_ptr->AppendRaw(
                  reinterpret_cast<const uint8_t*>(hit_table_hit_id_data->data()),
                  hit_table_hit_id_data->size() * sizeof(int32_t)),
                "unable to set data for hit_table_hit_id");

    FAIL_IF_ERR(hit_table_local_plane_ptr->AppendRaw(
                  reinterpret_cast<const uint8_t*>(hit_table_local_plane_data->data()),
                  hit_table_local_plane_data->size() * sizeof(int32_t)),
                "unable to set data for hit_table_local_plane");

    FAIL_IF_ERR(hit_table_local_time_ptr->AppendRaw(
                  reinterpret_cast<const uint8_t*>(hit_table_local_time_data->data()),
                  hit_table_local_time_data->size() * sizeof(float)),
                "unable to set data for hit_table_local_time");

    FAIL_IF_ERR(hit_table_local_wire_ptr->AppendRaw(
                  reinterpret_cast<const uint8_t*>(hit_table_local_wire_data->data()),
                  hit_table_local_wire_data->size() * sizeof(int32_t)),
                "unable to set data for hit_table_local_wire");

    FAIL_IF_ERR(hit_table_integral_ptr->AppendRaw(
                  reinterpret_cast<const uint8_t*>(hit_table_integral_data->data()),
                  hit_table_integral_data->size() * sizeof(float)),
                "unable to set data for hit_table_integral");

    FAIL_IF_ERR(
      hit_table_rms_ptr->AppendRaw(reinterpret_cast<const uint8_t*>(hit_table_rms_data->data()),
                                   hit_table_rms_data->size() * sizeof(float)),
      "unable to set data for hit_table_rms");

    FAIL_IF_ERR(spacepoint_table_spacepoint_id_ptr->AppendRaw(
                  reinterpret_cast<const uint8_t*>(spacepoint_table_spacepoint_id_data->data()),
                  spacepoint_table_spacepoint_id_data->size() * sizeof(int32_t)),
                "unable to set data for spacepoint_table_spacepoint_id");

    FAIL_IF_ERR(spacepoint_table_hit_id_u_ptr->AppendRaw(
                  reinterpret_cast<const uint8_t*>(spacepoint_table_hit_id_u_data->data()),
                  spacepoint_table_hit_id_u_data->size() * sizeof(int32_t)),
                "unable to set data for spacepoint_table_hit_id_u");

    FAIL_IF_ERR(spacepoint_table_hit_id_v_ptr->AppendRaw(
                  reinterpret_cast<const uint8_t*>(spacepoint_table_hit_id_v_data->data()),
                  spacepoint_table_hit_id_v_data->size() * sizeof(int32_t)),
                "unable to set data for spacepoint_table_hit_id_v");

    FAIL_IF_ERR(spacepoint_table_hit_id_y_ptr->AppendRaw(
                  reinterpret_cast<const uint8_t*>(spacepoint_table_hit_id_y_data->data()),
                  spacepoint_table_hit_id_y_data->size() * sizeof(int32_t)),
                "unable to set data for spacepoint_table_hit_id_y");

    // Generate the outputs to be requested.
    tc::InferRequestedOutput* x_semantic_u;
    tc::InferRequestedOutput* x_semantic_v;
    tc::InferRequestedOutput* x_semantic_y;
    tc::InferRequestedOutput* x_filter_u;
    tc::InferRequestedOutput* x_filter_v;
    tc::InferRequestedOutput* x_filter_y;

    FAIL_IF_ERR(tc::InferRequestedOutput::Create(&x_semantic_u, "x_semantic_u"),
                "unable to get 'x_semantic_u'");
    std::shared_ptr<tc::InferRequestedOutput> x_semantic_u_ptr;
    x_semantic_u_ptr.reset(x_semantic_u);

    FAIL_IF_ERR(tc::InferRequestedOutput::Create(&x_semantic_v, "x_semantic_v"),
                "unable to get 'x_semantic_v'");
    std::shared_ptr<tc::InferRequestedOutput> x_semantic_v_ptr;
    x_semantic_v_ptr.reset(x_semantic_v);

    FAIL_IF_ERR(tc::InferRequestedOutput::Create(&x_semantic_y, "x_semantic_y"),
                "unable to get 'x_semantic_y'");
    std::shared_ptr<tc::InferRequestedOutput> x_semantic_y_ptr;
    x_semantic_y_ptr.reset(x_semantic_y);

    FAIL_IF_ERR(tc::InferRequestedOutput::Create(&x_filter_u, "x_filter_u"),
                "unable to get 'x_filter_u'");
    std::shared_ptr<tc::InferRequestedOutput> x_filter_u_ptr;
    x_filter_u_ptr.reset(x_filter_u);

    FAIL_IF_ERR(tc::InferRequestedOutput::Create(&x_filter_v, "x_filter_v"),
                "unable to get 'x_filter_v'");
    std::shared_ptr<tc::InferRequestedOutput> x_filter_v_ptr;
    x_filter_v_ptr.reset(x_filter_v);

    FAIL_IF_ERR(tc::InferRequestedOutput::Create(&x_filter_y, "x_filter_y"),
                "unable to get 'x_filter_y'");
    std::shared_ptr<tc::InferRequestedOutput> x_filter_y_ptr;
    x_filter_y_ptr.reset(x_filter_y);

    // The inference settings. Will be using default for now.
    tc::InferOptions options(inference_model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;

    std::vector<tc::InferInput*> inputs = {hit_table_hit_id_ptr.get(),
                                           hit_table_local_plane_ptr.get(),
                                           hit_table_local_time_ptr.get(),
                                           hit_table_local_wire_ptr.get(),
                                           hit_table_integral_ptr.get(),
                                           hit_table_rms_ptr.get(),
                                           spacepoint_table_spacepoint_id_ptr.get(),
                                           spacepoint_table_hit_id_u_ptr.get(),
                                           spacepoint_table_hit_id_v_ptr.get(),
                                           spacepoint_table_hit_id_y_ptr.get()};

    std::vector<const tc::InferRequestedOutput*> outputs = {x_semantic_u_ptr.get(),
                                                            x_semantic_v_ptr.get(),
                                                            x_semantic_y_ptr.get(),
                                                            x_filter_u_ptr.get(),
                                                            x_filter_v_ptr.get(),
                                                            x_filter_y_ptr.get()};

    tc::InferResult* results;
    auto start = std::chrono::high_resolution_clock::now();
    FAIL_IF_ERR(
      client->Infer(&results, options, inputs, outputs, http_headers, compression_algorithm),
      "unable to run model");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for inference: " << elapsed.count() << " seconds" << std::endl;
    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);

    //
    // Get pointers to the result returned and write to the event
    //
    vector<NuGraphOutput> infer_output;
    vector<string> outnames = {
      "x_semantic_u", "x_semantic_v", "x_semantic_y", "x_filter_u", "x_filter_v", "x_filter_y"};
    for (const auto& name : outnames) {
      const float* _data;
      size_t _byte_size;
      FAIL_IF_ERR(results_ptr->RawData(name, (const uint8_t**)&_data, &_byte_size),
                  "unable to get result data for " + name);
      size_t n_elements = _byte_size / sizeof(float);
      std::vector<float> out_data(_data, _data + n_elements);
      infer_output.push_back(NuGraphOutput(name, out_data));
    }

    // Write the outputs
    for (size_t i = 0; i < _decoderToolsVec.size(); i++) {
      _decoderToolsVec[i]->writeToEvent(e, idsmap, infer_output);
    }
  }
}
DEFINE_ART_MODULE(NuGraphInferenceTriton)
