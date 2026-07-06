////////////////////////////////////////////////////////////////////////
// Class:       NuGraphInference
// Plugin Type: producer (Unknown Unknown)
// File:        NuGraphInference_module.cc
//
// Generated at Tue Nov 14 14:41:30 2023 by Giuseppe Cerati using cetskelgen
// from  version .
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "art/Utilities/make_tool.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include <array>
#include <limits>
#include <memory>

#include "delaunator-header-only.hpp"
#include <torch/script.h>

#include "lardataobj/AnalysisBase/MVAOutput.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include "lardataobj/RecoBase/Vertex.h" //this creates a conflict with torch script if included before it...

#include "larrecodnn/NuGraph/Tools/DecoderToolBase.h"
#include "larrecodnn/NuGraph/Tools/LoaderToolBase.h"

class NuGraphInference;

using anab::FeatureVector;
using anab::MVADescription;
using recob::Hit;
using recob::SpacePoint;
using std::array;
using std::vector;

namespace {
  template <typename T, typename A>
  int arg_max(std::vector<T, A> const& vec)
  {
    return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
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
}

class NuGraphInference : public art::EDProducer {
public:
  explicit NuGraphInference(fhicl::ParameterSet const& p);

  // Plugins should not be copied or assigned.
  NuGraphInference(NuGraphInference const&) = delete;
  NuGraphInference(NuGraphInference&&) = delete;
  NuGraphInference& operator=(NuGraphInference const&) = delete;
  NuGraphInference& operator=(NuGraphInference&&) = delete;

  // Required functions.
  void produce(art::Event& e) override;

private:
  vector<std::string> planes;
  size_t minHits;
  bool debug;
  vector<vector<float>> avgs;
  vector<vector<float>> devs;
  vector<float> pos_norm;
  torch::jit::script::Module model;
  // loader tool
  std::unique_ptr<LoaderToolBase> _loaderTool;
  // decoder tools
  std::vector<std::unique_ptr<DecoderToolBase>> _decoderToolsVec;
};

NuGraphInference::NuGraphInference(fhicl::ParameterSet const& p)
  : EDProducer{p}
  , planes(p.get<vector<std::string>>("planes"))
  , minHits(p.get<size_t>("minHits"))
  , debug(p.get<bool>("debug"))
  , pos_norm(p.get<vector<float>>("pos_norm"))
{

  for (size_t ip = 0; ip < planes.size(); ++ip) {
    avgs.push_back(p.get<vector<float>>("avgs_" + planes[ip]));
    devs.push_back(p.get<vector<float>>("devs_" + planes[ip]));
  }

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

  cet::search_path sp("FW_SEARCH_PATH");
  model = torch::jit::load(sp.find_file(p.get<std::string>("modelFileName")));
}

void NuGraphInference::produce(art::Event& e)
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
  // libTorch-specific section: requires extracting inputs, create graph, run inference
  //
  const vector<int32_t>* spids = nullptr;
  const vector<int32_t>* hitids_u = nullptr;
  const vector<int32_t>* hitids_v = nullptr;
  const vector<int32_t>* hitids_y = nullptr;
  const vector<int32_t>* hit_plane = nullptr;
  const vector<float>* hit_time = nullptr;
  const vector<int32_t>* hit_wire = nullptr;
  const vector<float>* hit_integral = nullptr;
  const vector<float>* hit_rms = nullptr;
  for (const auto& gi : graphinputs) {
    if (gi.input_name == "spacepoint_table_spacepoint_id")
      spids = &gi.input_int32_vec;
    else if (gi.input_name == "spacepoint_table_hit_id_u")
      hitids_u = &gi.input_int32_vec;
    else if (gi.input_name == "spacepoint_table_hit_id_v")
      hitids_v = &gi.input_int32_vec;
    else if (gi.input_name == "spacepoint_table_hit_id_y")
      hitids_y = &gi.input_int32_vec;
    else if (gi.input_name == "hit_table_local_plane")
      hit_plane = &gi.input_int32_vec;
    else if (gi.input_name == "hit_table_local_time")
      hit_time = &gi.input_float_vec;
    else if (gi.input_name == "hit_table_local_wire")
      hit_wire = &gi.input_int32_vec;
    else if (gi.input_name == "hit_table_integral")
      hit_integral = &gi.input_float_vec;
    else if (gi.input_name == "hit_table_rms")
      hit_rms = &gi.input_float_vec;
  }

  // Reverse lookup from key to index in plane index
  vector<size_t> idsmapRev(hitlist.size(), hitlist.size());
  for (const auto& ipv : idsmap) {
    for (size_t ih = 0; ih < ipv.size(); ih++) {
      idsmapRev[ipv[ih]] = ih;
    }
  }

  struct Edge {
    size_t n1;
    size_t n2;
    bool operator==(const Edge& other) const
    {
      if (this->n1 == other.n1 && this->n2 == other.n2)
        return true;
      else
        return false;
    };
  };

  // Delauney graph construction
  auto start_preprocess1 = std::chrono::high_resolution_clock::now();
  vector<vector<Edge>> edge2d(planes.size(), vector<Edge>());
  for (size_t p = 0; p < planes.size(); p++) {
    vector<double> coords;
    for (size_t i = 0; i < hit_plane->size(); ++i) {
      if (size_t(hit_plane->at(i)) != p) continue;
      coords.push_back(hit_time->at(i) * pos_norm[1]);
      coords.push_back(hit_wire->at(i) * pos_norm[0]);
    }
    if (debug) std::cout << "Plane " << p << " has N hits=" << coords.size() / 2 << std::endl;
    if (coords.size() / 2 < 3) { continue; }
    try {
      delaunator::Delaunator d(coords);
      if (debug) std::cout << "Found N triangles=" << d.triangles.size() / 3 << std::endl;
      for (std::size_t i = 0; i < d.triangles.size(); i += 3) {
        //create edges in both directions
        Edge e;
        e.n1 = d.triangles[i];
        e.n2 = d.triangles[i + 1];
        edge2d[p].push_back(e);
        e.n1 = d.triangles[i + 1];
        e.n2 = d.triangles[i];
        edge2d[p].push_back(e);
        //
        e.n1 = d.triangles[i];
        e.n2 = d.triangles[i + 2];
        edge2d[p].push_back(e);
        e.n1 = d.triangles[i + 2];
        e.n2 = d.triangles[i];
        edge2d[p].push_back(e);
        //
        e.n1 = d.triangles[i + 1];
        e.n2 = d.triangles[i + 2];
        edge2d[p].push_back(e);
        e.n1 = d.triangles[i + 2];
        e.n2 = d.triangles[i + 1];
        edge2d[p].push_back(e);
        //
      }
    }
    catch (const std::exception& e) {
      mf::LogWarning("NuGraphInference") << "Failed Delauney triangulation: " << e.what();
      continue;
    }

    //sort and cleanup duplicate edges
    std::sort(edge2d[p].begin(), edge2d[p].end(), [](const auto& i, const auto& j) {
      return (i.n1 != j.n1 ? i.n1 < j.n1 : i.n2 < j.n2);
    });
    if (debug) {
      for (auto& e : edge2d[p]) {
        std::cout << "sorted plane=" << p << " e1=" << e.n1 << " e2=" << e.n2 << std::endl;
      }
    }
    edge2d[p].erase(std::unique(edge2d[p].begin(), edge2d[p].end()), edge2d[p].end());
  }

  if (debug) {
    for (size_t p = 0; p < planes.size(); p++) {
      for (auto& e : edge2d[p]) {
        std::cout << " plane=" << p << " e1=" << e.n1 << " e2=" << e.n2 << std::endl;
      }
    }
  }
  auto end_preprocess1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_preprocess1 = end_preprocess1 - start_preprocess1;

  // Nexus edges
  auto start_preprocess2 = std::chrono::high_resolution_clock::now();
  vector<vector<Edge>> edge3d(planes.size(), vector<Edge>());
  for (size_t i = 0; i < spids->size(); ++i) {
    if (hitids_u->at(i) >= 0) {
      Edge e;
      e.n1 = idsmapRev[hitids_u->at(i)];
      e.n2 = spids->at(i);
      edge3d[0].push_back(e);
    }
    if (hitids_v->at(i) >= 0) {
      Edge e;
      e.n1 = idsmapRev[hitids_v->at(i)];
      e.n2 = spids->at(i);
      edge3d[1].push_back(e);
    }
    if (hitids_y->at(i) >= 0) {
      Edge e;
      e.n1 = idsmapRev[hitids_y->at(i)];
      e.n2 = spids->at(i);
      edge3d[2].push_back(e);
    }
  }

  // Prepare inputs
  auto x = torch::Dict<std::string, torch::Tensor>();
  auto batch = torch::Dict<std::string, torch::Tensor>();
  for (size_t p = 0; p < planes.size(); p++) {
    vector<float> nodeft;
    for (size_t i = 0; i < hit_plane->size(); ++i) {
      if (size_t(hit_plane->at(i)) != p) continue;
      nodeft.push_back((hit_wire->at(i) * pos_norm[0] - avgs[hit_plane->at(i)][0]) /
                       devs[hit_plane->at(i)][0]);
      nodeft.push_back((hit_time->at(i) * pos_norm[1] - avgs[hit_plane->at(i)][1]) /
                       devs[hit_plane->at(i)][1]);
      nodeft.push_back((hit_integral->at(i) - avgs[hit_plane->at(i)][2]) /
                       devs[hit_plane->at(i)][2]);
      nodeft.push_back((hit_rms->at(i) - avgs[hit_plane->at(i)][3]) / devs[hit_plane->at(i)][3]);
    }
    long int dim = nodeft.size() / 4;
    torch::Tensor ix = torch::zeros({dim, 4}, torch::dtype(torch::kFloat32));
    if (debug) {
      std::cout << "plane=" << p << std::endl;
      std::cout << std::scientific;
      for (size_t n = 0; n < nodeft.size(); n = n + 4) {
        std::cout << nodeft[n] << " " << nodeft[n + 1] << " " << nodeft[n + 2] << " "
                  << nodeft[n + 3] << " " << std::endl;
      }
    }
    for (size_t n = 0; n < nodeft.size(); n = n + 4) {
      ix[n / 4][0] = nodeft[n];
      ix[n / 4][1] = nodeft[n + 1];
      ix[n / 4][2] = nodeft[n + 2];
      ix[n / 4][3] = nodeft[n + 3];
    }
    x.insert(planes[p], ix);
    torch::Tensor ib = torch::zeros({dim}, torch::dtype(torch::kInt64));
    batch.insert(planes[p], ib);
  }

  auto edge_index_plane = torch::Dict<std::string, torch::Tensor>();
  for (size_t p = 0; p < planes.size(); p++) {
    long int dim = edge2d[p].size();
    torch::Tensor ix = torch::zeros({2, dim}, torch::dtype(torch::kInt64));
    for (size_t n = 0; n < edge2d[p].size(); n++) {
      ix[0][n] = int(edge2d[p][n].n1);
      ix[1][n] = int(edge2d[p][n].n2);
    }
    edge_index_plane.insert(planes[p], ix);
    if (debug) {
      std::cout << "plane=" << p << std::endl;
      std::cout << "2d edge size=" << edge2d[p].size() << std::endl;
      for (size_t n = 0; n < edge2d[p].size(); n++) {
        std::cout << edge2d[p][n].n1 << " ";
      }
      std::cout << std::endl;
      for (size_t n = 0; n < edge2d[p].size(); n++) {
        std::cout << edge2d[p][n].n2 << " ";
      }
      std::cout << std::endl;
    }
  }

  auto edge_index_nexus = torch::Dict<std::string, torch::Tensor>();
  for (size_t p = 0; p < planes.size(); p++) {
    long int dim = edge3d[p].size();
    torch::Tensor ix = torch::zeros({2, dim}, torch::dtype(torch::kInt64));
    for (size_t n = 0; n < edge3d[p].size(); n++) {
      ix[0][n] = int(edge3d[p][n].n1);
      ix[1][n] = int(edge3d[p][n].n2);
    }
    edge_index_nexus.insert(planes[p], ix);
    if (debug) {
      std::cout << "plane=" << p << std::endl;
      std::cout << "3d edge size=" << edge3d[p].size() << std::endl;
      for (size_t n = 0; n < edge3d[p].size(); n++) {
        std::cout << edge3d[p][n].n1 << " ";
      }
      std::cout << std::endl;
      for (size_t n = 0; n < edge3d[p].size(); n++) {
        std::cout << edge3d[p][n].n2 << " ";
      }
      std::cout << std::endl;
    }
  }

  long int spdim = spids->size();
  auto nexus = torch::empty({spdim, 0}, torch::dtype(torch::kFloat32));

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(x);
  inputs.push_back(edge_index_plane);
  inputs.push_back(edge_index_nexus);
  inputs.push_back(nexus);
  inputs.push_back(batch);

  // Run inference
  auto end_preprocess2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_preprocess2 = end_preprocess2 - start_preprocess2;
  if (debug) std::cout << "FORWARD!" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  c10::IValue raw_outputs;

  {
    // Use NoGradGuard to disable gradient tracking.
    // Gradients are not needed for inference, so save some memory and computation
    // by disabling them.
    //
    // This is equivalent to `with torch.no_grad():` in Python.
    // Uses RAII, so is disabled once the guard goes out of scope.
    torch::NoGradGuard guard;
    raw_outputs = model.forward(inputs);
  }

  auto outputs = raw_outputs.toGenericDict();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  if (debug) {
    std::cout << "Time taken for inference: "
              << elapsed_preprocess1.count() + elapsed_preprocess2.count() + elapsed.count()
              << " seconds" << std::endl;
    std::cout << "output =" << outputs << std::endl;
  }

  //
  // Get pointers to the result returned and write to the event
  //
  vector<NuGraphOutput> infer_output;
  for (const auto& elem1 : outputs) {
    if (elem1.value().isTensor()) {
      torch::Tensor tensor = elem1.value().toTensor();
      std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
      infer_output.push_back(NuGraphOutput(elem1.key().to<std::string>(), vec));
    }
    else if (elem1.value().isGenericDict()) {
      for (const auto& elem2 : elem1.value().toGenericDict()) {
        torch::Tensor tensor = elem2.value().toTensor();
        std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
        infer_output.push_back(
          NuGraphOutput(elem1.key().to<std::string>() + "_" + elem2.key().to<std::string>(), vec));
      }
    }
  }

  // Write the outputs to the output root file
  for (size_t i = 0; i < _decoderToolsVec.size(); i++) {
    _decoderToolsVec[i]->writeToEvent(e, idsmap, infer_output);
  }
}

DEFINE_ART_MODULE(NuGraphInference)
