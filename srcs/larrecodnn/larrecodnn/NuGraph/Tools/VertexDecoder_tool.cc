#include "DecoderToolBase.h"

#include "art/Utilities/ToolMacros.h"

#include "lardataobj/RecoBase/Vertex.h"
#include <torch/torch.h>

class VertexDecoder : public DecoderToolBase {

public:
  /**
   *  @brief  Constructor
   *
   *  @param  pset
   */
  VertexDecoder(const fhicl::ParameterSet& pset);

  /**
   *  @brief  Virtual Destructor
   */
  virtual ~VertexDecoder() noexcept = default;

  /**
   * @brief declareProducts function
   *
   * @param art::ProducesCollector
   */
  void declareProducts(art::ProducesCollector& collector) override
  {
    collector.produces<vector<recob::Vertex>>(instancename);
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
  string outputDictElem;
};

VertexDecoder::VertexDecoder(const fhicl::ParameterSet& p)
  : DecoderToolBase(p), outputDictElem{p.get<string>("outputDictElem")}
{}

void VertexDecoder::writeEmptyToEvent(art::Event& e, const vector<vector<size_t>>& idsmap)
{
  //
  auto vertcol = std::make_unique<vector<recob::Vertex>>();
  e.put(std::move(vertcol), instancename);
  //
}

void VertexDecoder::writeToEvent(art::Event& e,
                                 const vector<vector<size_t>>& idsmap,
                                 const vector<NuGraphOutput>& infer_output)
{
  //
  auto vertcol = std::make_unique<vector<recob::Vertex>>();

  const std::vector<float>* x_vertex_data = nullptr;
  for (auto& io : infer_output) {
    if (io.output_name == outputDictElem) x_vertex_data = &io.output_vec;
  }
  if (x_vertex_data->size() == 3) {
    double vpos[3] = {(*x_vertex_data)[0], (*x_vertex_data)[1], (*x_vertex_data)[2]};
    vertcol->push_back(recob::Vertex(vpos));
    if (debug)
      std::cout << "NuGraph vertex pos=" << vpos[0] << ", " << vpos[1] << ", " << vpos[2]
                << std::endl;
  }
  else {
    std::cout << "ERROR -- Wrong size returned by NuGraph vertex decoder" << std::endl;
  }
  e.put(std::move(vertcol), instancename);
  //
}

DEFINE_ART_CLASS_TOOL(VertexDecoder)
