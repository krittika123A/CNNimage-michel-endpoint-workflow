#include "LoaderToolBase.h"

#include "art/Utilities/ToolMacros.h"

#include "canvas/Persistency/Common/FindManyP.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include <torch/torch.h>

class StandardLoader : public LoaderToolBase {

public:
  /**
   *  @brief  Constructor
   *
   *  @param  pset
   */
  StandardLoader(const fhicl::ParameterSet& pset);

  /**
   *  @brief  Virtual Destructor
   */
  virtual ~StandardLoader() noexcept = default;

  /**
   * @brief loadData function
   *
   * @param art::Event event record, list of input, idsmap
   */
  void loadData(art::Event& e,
                vector<art::Ptr<recob::Hit>>& hitlist,
                vector<NuGraphInput>& inputs,
                vector<vector<size_t>>& idsmap) override;

private:
  art::InputTag hitInput;
  art::InputTag spsInput;
};

StandardLoader::StandardLoader(const fhicl::ParameterSet& p)
  : hitInput{p.get<art::InputTag>("hitInput")}, spsInput{p.get<art::InputTag>("spsInput")}
{}

void StandardLoader::loadData(art::Event& e,
                              vector<art::Ptr<recob::Hit>>& hitlist,
                              vector<NuGraphInput>& inputs,
                              vector<vector<size_t>>& idsmap)
{
  //
  art::Handle<vector<recob::Hit>> hitListHandle;
  if (e.getByLabel(hitInput, hitListHandle)) { art::fill_ptr_vector(hitlist, hitListHandle); }
  //
  idsmap = std::vector<std::vector<size_t>>(planes.size(), std::vector<size_t>());
  for (auto h : hitlist) {
    idsmap[h->View()].push_back(h.key());
  }

  vector<int32_t> hit_table_hit_id_data;
  vector<int32_t> hit_table_local_plane_data;
  vector<float> hit_table_local_time_data;
  vector<int32_t> hit_table_local_wire_data;
  vector<float> hit_table_integral_data;
  vector<float> hit_table_rms_data;
  vector<int32_t> spacepoint_table_spacepoint_id_data;
  vector<int32_t> spacepoint_table_hit_id_u_data;
  vector<int32_t> spacepoint_table_hit_id_v_data;
  vector<int32_t> spacepoint_table_hit_id_y_data;

  // hit table
  for (auto h : hitlist) {
    hit_table_hit_id_data.push_back(h.key());
    hit_table_local_plane_data.push_back(h->View());
    hit_table_local_time_data.push_back(h->PeakTime());
    hit_table_local_wire_data.push_back(h->WireID().Wire);
    hit_table_integral_data.push_back(h->Integral());
    hit_table_rms_data.push_back(h->RMS());
  }

  // Get spacepoints from the event record
  art::Handle<vector<recob::SpacePoint>> spListHandle;
  vector<art::Ptr<recob::SpacePoint>> splist;
  if (e.getByLabel(spsInput, spListHandle)) { art::fill_ptr_vector(splist, spListHandle); }
  // Get assocations from spacepoints to hits
  vector<vector<art::Ptr<recob::Hit>>> sp2Hit(splist.size());
  if (splist.size() > 0) {
    art::FindManyP<recob::Hit> fmp(spListHandle, e, spsInput);
    for (size_t spIdx = 0; spIdx < sp2Hit.size(); ++spIdx) {
      sp2Hit[spIdx] = fmp.at(spIdx);
    }
  }

  // space point table
  for (size_t i = 0; i < splist.size(); ++i) {
    spacepoint_table_spacepoint_id_data.push_back(i);
    spacepoint_table_hit_id_u_data.push_back(-1);
    spacepoint_table_hit_id_v_data.push_back(-1);
    spacepoint_table_hit_id_y_data.push_back(-1);
    for (size_t j = 0; j < sp2Hit[i].size(); ++j) {
      if (sp2Hit[i][j]->View() == 0) spacepoint_table_hit_id_u_data.back() = sp2Hit[i][j].key();
      if (sp2Hit[i][j]->View() == 1) spacepoint_table_hit_id_v_data.back() = sp2Hit[i][j].key();
      if (sp2Hit[i][j]->View() == 2) spacepoint_table_hit_id_y_data.back() = sp2Hit[i][j].key();
    }
  }

  inputs.emplace_back("hit_table_hit_id", hit_table_hit_id_data);
  inputs.emplace_back("hit_table_local_plane", hit_table_local_plane_data);
  inputs.emplace_back("hit_table_local_time", hit_table_local_time_data);
  inputs.emplace_back("hit_table_local_wire", hit_table_local_wire_data);
  inputs.emplace_back("hit_table_integral", hit_table_integral_data);
  inputs.emplace_back("hit_table_rms", hit_table_rms_data);

  inputs.emplace_back("spacepoint_table_spacepoint_id", spacepoint_table_spacepoint_id_data);
  inputs.emplace_back("spacepoint_table_hit_id_u", spacepoint_table_hit_id_u_data);
  inputs.emplace_back("spacepoint_table_hit_id_v", spacepoint_table_hit_id_v_data);
  inputs.emplace_back("spacepoint_table_hit_id_y", spacepoint_table_hit_id_y_data);
}
DEFINE_ART_CLASS_TOOL(StandardLoader)
