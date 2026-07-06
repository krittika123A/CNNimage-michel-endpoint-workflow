#include "DecoderToolBase.h"

#include "art/Utilities/ToolMacros.h"
#include "canvas/Persistency/Common/Assns.h"
#include "lardataobj/RecoBase/Hit.h"
#include <art/Persistency/Common/PtrMaker.h>

#include "lardataobj/AnalysisBase/MVAOutput.h"
#include <torch/torch.h>

using anab::FeatureVector;
using anab::MVADescription;

// fixme: this only works for 5 categories and should be extended to different sizes. This may require making the class templated.
class SemanticDecoder : public DecoderToolBase {

public:
  /**
   *  @brief  Constructor
   *
   *  @param  pset
   */
  SemanticDecoder(const fhicl::ParameterSet& pset);

  /**
   *  @brief  Virtual Destructor
   */
  virtual ~SemanticDecoder() noexcept = default;

  /**
   * @brief declareProducts function
   *
   * @param art::ProducesCollector
   */
  void declareProducts(art::ProducesCollector& collector) override
  {
    collector.produces<vector<FeatureVector<5>>>(instancename);
    collector.produces<MVADescription<5>>(instancename);
    collector.produces<art::Assns<FeatureVector<5>, recob::Hit>>(instancename);
  }

  /**
   * @brief writeEmptyToEvent function
   *
   * @param art::Event event record
   */
  void writeEmptyToEvent(art::Event& e, const vector<vector<size_t>>& idsmap) override;

  /**
   * @brief Decoder function
   *
   * @param art::Event event record for decoder
   */
  void writeToEvent(art::Event& e,
                    const vector<vector<size_t>>& idsmap,
                    const vector<NuGraphOutput>& infer_output) override;

private:
  std::vector<std::string> categories;
  art::InputTag hitInput;
};

SemanticDecoder::SemanticDecoder(const fhicl::ParameterSet& p)
  : DecoderToolBase(p)
  , categories{p.get<std::vector<std::string>>("categories")}
  , hitInput{p.get<art::InputTag>("hitInput")}
{}

void SemanticDecoder::writeEmptyToEvent(art::Event& e, const vector<vector<size_t>>& idsmap)
{
  //
  size_t size = 0;
  for (auto& v : idsmap)
    size += v.size();
  std::array<float, 5> arr;
  std::fill(arr.begin(), arr.end(), -1.);
  auto semtcol = std::make_unique<vector<FeatureVector<5>>>(size, FeatureVector<5>(arr));
  auto semtdes = std::make_unique<MVADescription<5>>(hitInput.label(), instancename, categories);
  auto outputFeatureHitAssns = std::make_unique<art::Assns<FeatureVector<5>, recob::Hit>>();
  // fill the assns as well
  art::ValidHandle<std::vector<recob::Hit>> hitsHandle =
    e.getValidHandle<std::vector<recob::Hit>>(hitInput);
  art::PtrMaker<FeatureVector<5>> fvPtrMaker{e, instancename};
  for (size_t p = 0; p < planes.size(); p++) {
    for (size_t id : idsmap[p]) {
      const art::Ptr<FeatureVector<5>> fvPtr = fvPtrMaker(outputFeatureHitAssns->size());
      const art::Ptr<recob::Hit> hitPtr(hitsHandle, id);
      outputFeatureHitAssns->addSingle(fvPtr, hitPtr);
    }
  }
  //
  e.put(std::move(semtcol), instancename);
  e.put(std::move(semtdes), instancename);
  e.put(std::move(outputFeatureHitAssns), instancename);
  //
}

void SemanticDecoder::writeToEvent(art::Event& e,
                                   const vector<vector<size_t>>& idsmap,
                                   const vector<NuGraphOutput>& infer_output)
{
  //
  auto semtdes = std::make_unique<MVADescription<5>>(hitInput.label(), instancename, categories);
  e.put(std::move(semtdes), instancename);
  auto outputFeatureHitAssns = std::make_unique<art::Assns<FeatureVector<5>, recob::Hit>>();
  art::PtrMaker<FeatureVector<5>> fvPtrMaker{e, instancename};
  art::ValidHandle<std::vector<recob::Hit>> hitsHandle =
    e.getValidHandle<std::vector<recob::Hit>>(hitInput);
  //
  std::vector<size_t> sorted_keys;
  size_t size = 0;
  for (auto& v : idsmap) {
    size += v.size();
    for (auto k : v)
      sorted_keys.push_back(k);
  }
  // preserve the order for backward compatibility
  std::sort(sorted_keys.begin(), sorted_keys.end());
  std::array<float, 5> arr;
  std::fill(arr.begin(), arr.end(), -1.);
  auto semtcol = std::make_unique<vector<FeatureVector<5>>>(size, FeatureVector<5>(arr));

  size_t n_cols = categories.size();
  for (size_t p = 0; p < planes.size(); p++) {
    //
    const std::vector<float>* x_semantic_data = 0;
    for (auto& io : infer_output) {
      if (io.output_name == outputname + planes[p]) x_semantic_data = &io.output_vec;
    }
    if (debug) {
      std::cout << outputname + planes[p] << std::endl;
      printVector(*x_semantic_data);
    }

    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    size_t n_rows = x_semantic_data->size() / n_cols;
    const torch::Tensor s =
      torch::from_blob(const_cast<float*>(x_semantic_data->data()),
                       {static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)},
                       options);

    if (debug) std::cout << "Sizes: " << s.sizes()[0] << " idsmap[p]:" << idsmap[p].size() << '\n';
    for (int i = 0; i < s.sizes()[0]; ++i) {
      size_t idx = idsmap[p][i];
      std::array<float, 5> input;
      for (size_t j = 0; j < n_cols; ++j)
        input[j] = s[i][j].item<float>();
      softmax(input);
      FeatureVector<5> semt = FeatureVector<5>(input);
      size_t filt_index =
        std::distance(sorted_keys.begin(), std::find(sorted_keys.begin(), sorted_keys.end(), idx));
      (*semtcol)[filt_index] = semt;
      const art::Ptr<FeatureVector<5>> fvPtr = fvPtrMaker(filt_index);
      const art::Ptr<recob::Hit> hitPtr(hitsHandle, idx);
      if (debug)
        std::cout << "Associating SemanticVector #" << fvPtr.key() << " with hit #" << hitPtr.key()
                  << '\n';
      outputFeatureHitAssns->addSingle(fvPtr, hitPtr);
    }
  }
  e.put(std::move(semtcol), instancename);
  e.put(std::move(outputFeatureHitAssns), instancename);
}

DEFINE_ART_CLASS_TOOL(SemanticDecoder)
