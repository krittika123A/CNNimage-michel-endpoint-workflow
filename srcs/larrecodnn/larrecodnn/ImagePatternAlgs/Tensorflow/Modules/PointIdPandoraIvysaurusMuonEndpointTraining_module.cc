////////////////////////////////////////////////////////////////////////////////////////////////////
// Class:       PointIdPandoraIvysaurusMuonEndpointTraining
//
// Build one training-tree entry per selected Pandora PFP/track candidate.
// Each entry contains both StartGrid* and EndGrid* views. Reco hit charge fills
// the image grids directly. Truth labels come from TruthMatchUtils, SimChannel
// deposits, and MCParticle ancestry; no TrainingDataAlg bitmap Michel flag is used.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
#include "fhiclcpp/types/Comment.h"
#include "fhiclcpp/types/Name.h"
#include "fhiclcpp/types/Table.h"
#include "larcore/Geometry/WireReadout.h"
#include "larcorealg/Geometry/GeometryCore.h"
#include "lardata/DetectorInfoServices/DetectorClocksService.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesService.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/Simulation/SimChannel.h"
#include "larsim/Utils/TruthMatchUtils.h"
#include "messagefacility/MessageLogger/MessageLogger.h"
#include "nusimdata/SimulationBase/MCParticle.h"

#include "TH2F.h"
#include "TH2I.h"
#include "TTree.h"
#include "TVector3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

  constexpr unsigned int kNPlanes = 3U;

  enum TruthCode {
    kTruthEmpty = 0,
    kTruthMuon = 1,
    kTruthMichel = 2,
    kTruthEM = 3,
    kTruthOther = 4
  };

  struct DominantTPC {
    bool valid = false;
    int cryo = -1;
    int tpc = -1;
    std::size_t nhits = 0U;
  };

  struct EndpointProjection {
    bool valid = false;
    std::array<float, kNPlanes> wire{{0.F, 0.F, 0.F}};
    std::array<float, kNPlanes> tick{{0.F, 0.F, 0.F}};
    std::array<float, 3U> xyz{{0.F, 0.F, 0.F}};
  };

  struct HitTruth {
    int truthCode = kTruthOther;
    int dominantTrackId = 0;
    int dominantPdg = 0;
    int michelMuonTrackId = 0;
    int michelMuonPdg = 0;
  };

  struct EndpointTruth {
    int hasMichel = 0;
    int michelHits = 0;
    int muonHits = 0;
    int emHits = 0;
    int otherHits = 0;
    int dominantTrackId = 0;
    int dominantPdg = 0;
    int michelMuonTrackId = 0;
    int michelMuonPdg = 0;
  };

  struct EventDisplayBounds {
    int minWire = std::numeric_limits<int>::max();
    int maxWire = std::numeric_limits<int>::lowest();
    int minDrift = std::numeric_limits<int>::max();
    int maxDrift = std::numeric_limits<int>::lowest();

    void update(int wire, int drift)
    {
      minWire = std::min(minWire, wire);
      maxWire = std::max(maxWire, wire);
      minDrift = std::min(minDrift, drift);
      maxDrift = std::max(maxDrift, drift);
    }

    bool valid() const
    {
      return minWire <= maxWire && minDrift <= maxDrift;
    }
  };

} // namespace

namespace nnet {

  class PointIdPandoraIvysaurusMuonEndpointTraining : public art::EDAnalyzer {
  public:
    struct Config {
      using Name = fhicl::Name;
      using Comment = fhicl::Comment;

      fhicl::Atom<art::InputTag> PandoraPfoLabel{
        Name("PandoraPfoLabel"),
        Comment("Pandora PFParticle collection label")
      };
      fhicl::Atom<art::InputTag> PandoraTrackLabel{
        Name("PandoraTrackLabel"),
        Comment("Pandora track collection and PFParticle-to-track association label")
      };
      fhicl::Atom<art::InputTag> TrackHitAssocLabel{
        Name("TrackHitAssocLabel"),
        Comment("Pandora track-to-hit association label")
      };
      fhicl::Atom<art::InputTag> HitLabel{
        Name("HitLabel"),
        Comment("Full-event hit collection used for event-display histograms"),
        art::InputTag("hitfd")
      };

      fhicl::Atom<art::InputTag> ParticleLabel{Name("ParticleLabel"), Comment("MCParticle label")};
      fhicl::Atom<art::InputTag> SimChannelLabel{Name("SimChannelLabel"), Comment("SimChannel label")};

      fhicl::Atom<float> MinTrackLength{Name("MinTrackLength"), Comment("Minimum track length"), 10.F};
      fhicl::Atom<int> GridWires{Name("GridWires"), Comment("Wire bins in each grid"), 48};
      fhicl::Atom<int> GridDrifts{Name("GridDrifts"), Comment("Drift/time bins in each grid"), 48};
      fhicl::Atom<int> DriftBinWidth{Name("DriftBinWidth"), Comment("Hit PeakTime divisor"), 6};
      fhicl::Atom<float> MinHitIntegral{Name("MinHitIntegral"), Comment("Minimum hit integral"), 0.F};
      fhicl::Atom<int> MinMichelHits{Name("MinMichelHits"), Comment("Minimum Michel hits"), 1};
      fhicl::Atom<float> MinMichelFraction{Name("MinMichelFraction"), Comment("Minimum Michel fraction"), 0.05F};
      fhicl::Atom<bool> SaveEventDisplays{
        Name("SaveEventDisplays"),
        Comment("Write one full-event hit-charge TH2 per selected TPC/view"),
        false
      };
      fhicl::Atom<bool> SaveEventTruth{
        Name("SaveEventTruth"),
        Comment("Write one full-event truth-code TH2 per selected TPC/view"),
        false
      };
      fhicl::Atom<bool> SaveEndpointPatchHists{
        Name("SaveEndpointPatchHists"),
        Comment("Write start/end ROI patch TH2 histograms matching the CNN input grids"),
        true
      };
      fhicl::Atom<bool> SaveEndpointPatchTruth{
        Name("SaveEndpointPatchTruth"),
        Comment("Write matching start/end ROI truth-code TH2 histograms"),
        true
      };
      fhicl::Sequence<int> SelectedTPC{
        Name("SelectedTPC"),
        Comment("TPCs to write in endpoint patch displays; empty means all TPCs with hits"),
        std::vector<int>{}
      };
      fhicl::Sequence<int> SelectedView{
        Name("SelectedView"),
        Comment("Wire-plane views to write in endpoint patch displays; empty means all views with hits"),
        std::vector<int>{0, 1, 2}
      };
      fhicl::Atom<int> EventDisplayWireMargin{
        Name("EventDisplayWireMargin"),
        Comment("Wire-bin margin around occupied hits in event displays"),
        10
      };
      fhicl::Atom<int> EventDisplayDriftMargin{
        Name("EventDisplayDriftMargin"),
        Comment("Drift-bin margin around occupied hits in event displays"),
        10
      };
    };

    using Parameters = art::EDAnalyzer::Table<Config>;

    explicit PointIdPandoraIvysaurusMuonEndpointTraining(Parameters const& config);
    void beginJob() override;

  private:
    void analyze(art::Event const& event) override;

    DominantTPC dominantTPC(std::vector<art::Ptr<recob::Hit>> const& hits) const;

    EndpointProjection projectEndpoint(recob::Track const& track,
                                       bool useEnd,
                                       int cryo,
                                       int tpc,
                                       detinfo::DetectorPropertiesData const& detProp,
                                       geo::WireReadoutGeom const& wireReadoutGeom) const;

    HitTruth classifyHit(art::Ptr<recob::Hit> const& hit,
                         detinfo::DetectorClocksData const& clockData,
                         std::vector<sim::SimChannel> const& simChannels,
                         std::unordered_map<int, simb::MCParticle const*> const& particleMap) const;

    void fillEndpoint(EndpointProjection const& endpoint,
                      int cryo,
                      int tpc,
                      std::vector<art::Ptr<recob::Hit>> const& hits,
                      detinfo::DetectorClocksData const& clockData,
                      std::vector<sim::SimChannel> const& simChannels,
                      std::unordered_map<int, simb::MCParticle const*> const& particleMap,
                      std::array<std::vector<float>, kNPlanes>& grids,
                      std::array<std::vector<int>, kNPlanes>& truthGrids,
                      EndpointTruth& truth) const;

    std::vector<art::Ptr<recob::Hit>> collectEventHits(
      art::Event const& event,
      std::vector<recob::Track> const& tracks,
      art::FindManyP<recob::Hit> const& trackHitAssoc) const;

    void writeEventDisplays(
      art::Event const& event,
      detinfo::DetectorClocksData const& clockData,
      std::vector<art::Ptr<recob::Hit>> const& eventHits,
      std::vector<sim::SimChannel> const& simChannels,
      std::unordered_map<int, simb::MCParticle const*> const& particleMap) const;

    void writeEndpointPatchHists(
      art::Event const& event,
      int pfpKey,
      int trackKey,
      int cryo,
      int tpc,
      std::array<std::vector<float>, kNPlanes> const& startGrids,
      std::array<std::vector<float>, kNPlanes> const& endGrids,
      std::array<std::vector<int>, kNPlanes> const& startTruthGrids,
      std::array<std::vector<int>, kNPlanes> const& endTruthGrids) const;

    bool selected(std::vector<int> const& values, int value) const;

    bool isMuonDecayAncestor(
      simb::MCParticle const& particle,
      std::unordered_map<int, simb::MCParticle const*> const& particleMap) const;

    bool isMichelElectronFromAncestry(
      simb::MCParticle const& particle,
      std::unordered_map<int, simb::MCParticle const*> const& particleMap,
      int& muonTrackId,
      int& muonPdg) const;

    void resetBranches();
    void resetGridVectors();

    art::InputTag fPandoraPfoLabel;
    art::InputTag fPandoraTrackLabel;
    art::InputTag fTrackHitAssocLabel;
    art::InputTag fHitLabel;
    art::InputTag fParticleLabel;
    art::InputTag fSimChannelLabel;
    float fMinTrackLength;
    int fGridWires;
    int fGridDrifts;
    int fDriftBinWidth;
    float fMinHitIntegral;
    int fMinMichelHits;
    float fMinMichelFraction;
    bool fSaveEventDisplays;
    bool fSaveEventTruth;
    bool fSaveEndpointPatchHists;
    bool fSaveEndpointPatchTruth;
    std::vector<int> fSelectedTPC;
    std::vector<int> fSelectedView;
    int fEventDisplayWireMargin;
    int fEventDisplayDriftMargin;

    TTree* fTree = nullptr;

    int b_run = 0;
    int b_subrun = 0;
    int b_event = 0;
    int b_pfpKey = -1;
    int b_pfpSelf = 0;
    int b_pfpPdg = 0;
    int b_trackKey = -1;
    int b_cryo = -1;
    int b_tpc = -1;
    int b_nHits = 0;
    float b_trackLength = 0.F;

    float b_startXYZ[3] = {};
    float b_endXYZ[3] = {};
    float b_startWire[3] = {};
    float b_startTick[3] = {};
    float b_endWire[3] = {};
    float b_endTick[3] = {};

    int b_startHasMichel = 0;
    int b_endHasMichel = 0;
    int b_startMichelHits = 0;
    int b_endMichelHits = 0;
    int b_startMuonHits = 0;
    int b_endMuonHits = 0;
    int b_startEMHits = 0;
    int b_endEMHits = 0;
    int b_startOtherHits = 0;
    int b_endOtherHits = 0;
    int b_startDominantTrackId = 0;
    int b_endDominantTrackId = 0;
    int b_startDominantPdg = 0;
    int b_endDominantPdg = 0;
    int b_startMichelMuonTrackId = 0;
    int b_endMichelMuonTrackId = 0;
    int b_startMichelMuonPdg = 0;
    int b_endMichelMuonPdg = 0;

    std::vector<float> b_startGridU;
    std::vector<float> b_startGridV;
    std::vector<float> b_startGridW;
    std::vector<float> b_endGridU;
    std::vector<float> b_endGridV;
    std::vector<float> b_endGridW;
    std::vector<int> b_startTruthU;
    std::vector<int> b_startTruthV;
    std::vector<int> b_startTruthW;
    std::vector<int> b_endTruthU;
    std::vector<int> b_endTruthV;
    std::vector<int> b_endTruthW;
  };

  //-----------------------------------------------------------------------
  PointIdPandoraIvysaurusMuonEndpointTraining::PointIdPandoraIvysaurusMuonEndpointTraining(
    Parameters const& config)
    : art::EDAnalyzer(config)
    , fPandoraPfoLabel(config().PandoraPfoLabel())
    , fPandoraTrackLabel(config().PandoraTrackLabel())
    , fTrackHitAssocLabel(config().TrackHitAssocLabel())
    , fHitLabel(config().HitLabel())
    , fParticleLabel(config().ParticleLabel())
    , fSimChannelLabel(config().SimChannelLabel())
    , fMinTrackLength(config().MinTrackLength())
    , fGridWires(config().GridWires())
    , fGridDrifts(config().GridDrifts())
    , fDriftBinWidth(config().DriftBinWidth())
    , fMinHitIntegral(config().MinHitIntegral())
    , fMinMichelHits(config().MinMichelHits())
    , fMinMichelFraction(config().MinMichelFraction())
    , fSaveEventDisplays(config().SaveEventDisplays())
    , fSaveEventTruth(config().SaveEventTruth())
    , fSaveEndpointPatchHists(config().SaveEndpointPatchHists())
    , fSaveEndpointPatchTruth(config().SaveEndpointPatchTruth())
    , fSelectedTPC(config().SelectedTPC())
    , fSelectedView(config().SelectedView())
    , fEventDisplayWireMargin(config().EventDisplayWireMargin())
    , fEventDisplayDriftMargin(config().EventDisplayDriftMargin())
  {}

  //-----------------------------------------------------------------------
  void PointIdPandoraIvysaurusMuonEndpointTraining::beginJob()
  {
    art::ServiceHandle<art::TFileService> tfs;
    fTree = tfs->make<TTree>("ivysaurusMuonEndpointTraining", "Ivysaurus-style muon endpoint grids");

    resetGridVectors();

    fTree->Branch("run", &b_run, "run/I");
    fTree->Branch("subrun", &b_subrun, "subrun/I");
    fTree->Branch("event", &b_event, "event/I");
    fTree->Branch("pfp_key", &b_pfpKey, "pfp_key/I");
    fTree->Branch("pfp_self", &b_pfpSelf, "pfp_self/I");
    fTree->Branch("pfp_pdg", &b_pfpPdg, "pfp_pdg/I");
    fTree->Branch("track_key", &b_trackKey, "track_key/I");
    fTree->Branch("cryo", &b_cryo, "cryo/I");
    fTree->Branch("tpc", &b_tpc, "tpc/I");
    fTree->Branch("n_hits", &b_nHits, "n_hits/I");
    fTree->Branch("track_length", &b_trackLength, "track_length/F");

    fTree->Branch("StartXYZ", b_startXYZ, "StartXYZ[3]/F");
    fTree->Branch("EndXYZ", b_endXYZ, "EndXYZ[3]/F");
    fTree->Branch("StartWire", b_startWire, "StartWire[3]/F");
    fTree->Branch("StartTick", b_startTick, "StartTick[3]/F");
    fTree->Branch("EndWire", b_endWire, "EndWire[3]/F");
    fTree->Branch("EndTick", b_endTick, "EndTick[3]/F");

    fTree->Branch("StartHasMichel", &b_startHasMichel, "StartHasMichel/I");
    fTree->Branch("EndHasMichel", &b_endHasMichel, "EndHasMichel/I");
    fTree->Branch("StartMichelHits", &b_startMichelHits, "StartMichelHits/I");
    fTree->Branch("EndMichelHits", &b_endMichelHits, "EndMichelHits/I");
    fTree->Branch("StartMuonHits", &b_startMuonHits, "StartMuonHits/I");
    fTree->Branch("EndMuonHits", &b_endMuonHits, "EndMuonHits/I");
    fTree->Branch("StartEMHits", &b_startEMHits, "StartEMHits/I");
    fTree->Branch("EndEMHits", &b_endEMHits, "EndEMHits/I");
    fTree->Branch("StartOtherHits", &b_startOtherHits, "StartOtherHits/I");
    fTree->Branch("EndOtherHits", &b_endOtherHits, "EndOtherHits/I");
    fTree->Branch("StartDominantTrackId", &b_startDominantTrackId, "StartDominantTrackId/I");
    fTree->Branch("EndDominantTrackId", &b_endDominantTrackId, "EndDominantTrackId/I");
    fTree->Branch("StartDominantPdg", &b_startDominantPdg, "StartDominantPdg/I");
    fTree->Branch("EndDominantPdg", &b_endDominantPdg, "EndDominantPdg/I");
    fTree->Branch("StartMichelMuonTrackId", &b_startMichelMuonTrackId, "StartMichelMuonTrackId/I");
    fTree->Branch("EndMichelMuonTrackId", &b_endMichelMuonTrackId, "EndMichelMuonTrackId/I");
    fTree->Branch("StartMichelMuonPdg", &b_startMichelMuonPdg, "StartMichelMuonPdg/I");
    fTree->Branch("EndMichelMuonPdg", &b_endMichelMuonPdg, "EndMichelMuonPdg/I");

    fTree->Branch("StartGridU", &b_startGridU);
    fTree->Branch("StartGridV", &b_startGridV);
    fTree->Branch("StartGridW", &b_startGridW);
    fTree->Branch("EndGridU", &b_endGridU);
    fTree->Branch("EndGridV", &b_endGridV);
    fTree->Branch("EndGridW", &b_endGridW);
    fTree->Branch("StartTruthU", &b_startTruthU);
    fTree->Branch("StartTruthV", &b_startTruthV);
    fTree->Branch("StartTruthW", &b_startTruthW);
    fTree->Branch("EndTruthU", &b_endTruthU);
    fTree->Branch("EndTruthV", &b_endTruthV);
    fTree->Branch("EndTruthW", &b_endTruthW);
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraIvysaurusMuonEndpointTraining::analyze(art::Event const& event)
  {
    auto const clockData =
      art::ServiceHandle<detinfo::DetectorClocksService const>()->DataFor(event);
    auto const detProp =
      art::ServiceHandle<detinfo::DetectorPropertiesService const>()->DataFor(event, clockData);
    auto const& wireReadoutGeom = art::ServiceHandle<geo::WireReadout const>()->Get();

    auto const pfpHandle = event.getValidHandle<std::vector<recob::PFParticle>>(fPandoraPfoLabel);
    auto const trackHandle = event.getValidHandle<std::vector<recob::Track>>(fPandoraTrackLabel);
    art::FindManyP<recob::Track> pfpTrackAssoc(pfpHandle, event, fPandoraTrackLabel);
    art::FindManyP<recob::Hit> trackHitAssoc(trackHandle, event, fTrackHitAssocLabel);

    auto const particleHandle = event.getValidHandle<std::vector<simb::MCParticle>>(fParticleLabel);
    std::unordered_map<int, simb::MCParticle const*> particleMap;
    for (auto const& particle : *particleHandle) {
      particleMap.emplace(particle.TrackId(), &particle);
    }

    auto const simChannelHandle = event.getValidHandle<std::vector<sim::SimChannel>>(fSimChannelLabel);

    // Full-event displays were too broad for endpoint CNN training. Keep the
    // implementation available, but write endpoint-centred ROI patches below.
    // std::vector<art::Ptr<recob::Hit>> const eventHits =
    //   collectEventHits(event, *trackHandle, trackHitAssoc);
    // if (fSaveEventDisplays) {
    //   writeEventDisplays(event, clockData, eventHits, *simChannelHandle, particleMap);
    // }

    for (std::size_t iPfp = 0; iPfp < pfpHandle->size(); ++iPfp) {
      auto const tracks = pfpTrackAssoc.at(iPfp);
      if (tracks.empty()) continue;

      auto const hits = trackHitAssoc.at(tracks.front().key());
      if (hits.empty()) continue;

      recob::Track const& track = *(tracks.front());
      if (track.Length() < fMinTrackLength) continue;

      DominantTPC const domTPC = dominantTPC(hits);
      if (!domTPC.valid) continue;
      if (!selected(fSelectedTPC, domTPC.tpc)) continue;

      EndpointProjection const start =
        projectEndpoint(track, false, domTPC.cryo, domTPC.tpc, detProp, wireReadoutGeom);
      EndpointProjection const end =
        projectEndpoint(track, true, domTPC.cryo, domTPC.tpc, detProp, wireReadoutGeom);
      if (!start.valid || !end.valid) continue;

      resetBranches();

      recob::PFParticle const& pfp = pfpHandle->at(iPfp);
      b_run = event.run();
      b_subrun = event.subRun();
      b_event = event.id().event();
      b_pfpKey = static_cast<int>(iPfp);
      b_pfpSelf = pfp.Self();
      b_pfpPdg = pfp.PdgCode();
      b_trackKey = tracks.front().key();
      b_cryo = domTPC.cryo;
      b_tpc = domTPC.tpc;
      b_nHits = static_cast<int>(hits.size());
      b_trackLength = static_cast<float>(track.Length());

      for (unsigned int i = 0; i < 3U; ++i) {
        b_startXYZ[i] = start.xyz[i];
        b_endXYZ[i] = end.xyz[i];
      }
      for (unsigned int plane = 0; plane < kNPlanes; ++plane) {
        b_startWire[plane] = start.wire[plane];
        b_startTick[plane] = start.tick[plane];
        b_endWire[plane] = end.wire[plane];
        b_endTick[plane] = end.tick[plane];
      }

      std::array<std::vector<float>, kNPlanes> startGrids{b_startGridU, b_startGridV, b_startGridW};
      std::array<std::vector<float>, kNPlanes> endGrids{b_endGridU, b_endGridV, b_endGridW};
      std::array<std::vector<int>, kNPlanes> startTruthGrids{b_startTruthU, b_startTruthV, b_startTruthW};
      std::array<std::vector<int>, kNPlanes> endTruthGrids{b_endTruthU, b_endTruthV, b_endTruthW};

      EndpointTruth startTruth;
      EndpointTruth endTruth;
      fillEndpoint(start, domTPC.cryo, domTPC.tpc, hits, clockData, *simChannelHandle, particleMap, startGrids, startTruthGrids, startTruth);
      fillEndpoint(end, domTPC.cryo, domTPC.tpc, hits, clockData, *simChannelHandle, particleMap, endGrids, endTruthGrids, endTruth);

      if (fSaveEndpointPatchHists) {
        writeEndpointPatchHists(event,
                                static_cast<int>(iPfp),
                                static_cast<int>(tracks.front().key()),
                                domTPC.cryo,
                                domTPC.tpc,
                                startGrids,
                                endGrids,
                                startTruthGrids,
                                endTruthGrids);
      }

      b_startGridU.swap(startGrids[0]);
      b_startGridV.swap(startGrids[1]);
      b_startGridW.swap(startGrids[2]);
      b_endGridU.swap(endGrids[0]);
      b_endGridV.swap(endGrids[1]);
      b_endGridW.swap(endGrids[2]);
      b_startTruthU.swap(startTruthGrids[0]);
      b_startTruthV.swap(startTruthGrids[1]);
      b_startTruthW.swap(startTruthGrids[2]);
      b_endTruthU.swap(endTruthGrids[0]);
      b_endTruthV.swap(endTruthGrids[1]);
      b_endTruthW.swap(endTruthGrids[2]);

      b_startHasMichel = startTruth.hasMichel;
      b_endHasMichel = endTruth.hasMichel;
      b_startMichelHits = startTruth.michelHits;
      b_endMichelHits = endTruth.michelHits;
      b_startMuonHits = startTruth.muonHits;
      b_endMuonHits = endTruth.muonHits;
      b_startEMHits = startTruth.emHits;
      b_endEMHits = endTruth.emHits;
      b_startOtherHits = startTruth.otherHits;
      b_endOtherHits = endTruth.otherHits;
      b_startDominantTrackId = startTruth.dominantTrackId;
      b_endDominantTrackId = endTruth.dominantTrackId;
      b_startDominantPdg = startTruth.dominantPdg;
      b_endDominantPdg = endTruth.dominantPdg;
      b_startMichelMuonTrackId = startTruth.michelMuonTrackId;
      b_endMichelMuonTrackId = endTruth.michelMuonTrackId;
      b_startMichelMuonPdg = startTruth.michelMuonPdg;
      b_endMichelMuonPdg = endTruth.michelMuonPdg;

      fTree->Fill();
    }
  }

  //-----------------------------------------------------------------------
  DominantTPC PointIdPandoraIvysaurusMuonEndpointTraining::dominantTPC(
    std::vector<art::Ptr<recob::Hit>> const& hits) const
  {
    std::map<std::pair<int, int>, std::size_t> counts;
    for (auto const& hit : hits) {
      auto const key = std::make_pair(static_cast<int>(hit->WireID().Cryostat), static_cast<int>(hit->WireID().TPC));
      ++counts[key];
    }

    DominantTPC result;
    for (auto const& entry : counts) {
      if (entry.second <= result.nhits) continue;
      result.valid = true;
      result.cryo = entry.first.first;
      result.tpc = entry.first.second;
      result.nhits = entry.second;
    }
    return result;
  }

  //-----------------------------------------------------------------------
  EndpointProjection PointIdPandoraIvysaurusMuonEndpointTraining::projectEndpoint(
    recob::Track const& track,
    bool useEnd,
    int cryo,
    int tpc,
    detinfo::DetectorPropertiesData const& detProp,
    geo::WireReadoutGeom const& wireReadoutGeom) const
  {
    EndpointProjection endpoint;
    TVector3 const point = useEnd ? track.End<TVector3>() : track.Vertex<TVector3>();
    geo::Point_t const geoPoint{point.X(), point.Y(), point.Z()};
    endpoint.xyz[0] = point.X();
    endpoint.xyz[1] = point.Y();
    endpoint.xyz[2] = point.Z();

    for (unsigned int plane = 0; plane < kNPlanes; ++plane) {
      geo::PlaneID const planeID{static_cast<unsigned int>(cryo), static_cast<unsigned int>(tpc), plane};
      if (!wireReadoutGeom.HasPlane(planeID)) return endpoint;

      double const wire = wireReadoutGeom.Plane(planeID).WireCoordinate(geoPoint);
      double const tick = detProp.ConvertXToTicks(point.X(), plane, tpc, cryo) / static_cast<double>(fDriftBinWidth);
      if (!std::isfinite(wire) || !std::isfinite(tick)) return endpoint;

      endpoint.wire[plane] = static_cast<float>(wire);
      endpoint.tick[plane] = static_cast<float>(tick);
    }

    endpoint.valid = true;
    return endpoint;
  }

  //-----------------------------------------------------------------------
  HitTruth PointIdPandoraIvysaurusMuonEndpointTraining::classifyHit(
    art::Ptr<recob::Hit> const& hit,
    detinfo::DetectorClocksData const& clockData,
    std::vector<sim::SimChannel> const& simChannels,
    std::unordered_map<int, simb::MCParticle const*> const& particleMap) const
  {
    HitTruth truth;
    int const dominantId = TruthMatchUtils::TrueParticleID(clockData, hit, true);
    if (TruthMatchUtils::Valid(dominantId)) {
      auto const particleSearch = particleMap.find(std::abs(dominantId));
      if (particleSearch != particleMap.end()) {
        truth.dominantTrackId = particleSearch->second->TrackId();
        truth.dominantPdg = particleSearch->second->PdgCode();
      }
    }

    double bestCharge = 0.0;
    bool hasMichel = false;
    bool hasMuon = false;
    bool hasEM = false;
    bool hasOther = false;

    for (auto const& channel : simChannels) {
      if (channel.Channel() != hit->Channel()) continue;

      for (auto const& timeSlice : channel.TDCIDEMap()) {
        if (std::abs(hit->TimeDistanceAsRMS(timeSlice.first)) >= 1.0) continue;

        for (auto const& energyDeposit : timeSlice.second) {
          int const trackId = std::abs(energyDeposit.trackID);
          auto const particleSearch = particleMap.find(trackId);
          if (particleSearch == particleMap.end()) continue;

          simb::MCParticle const& particle = *(particleSearch->second);
          int const absPdg = std::abs(particle.PdgCode());
          double const charge = energyDeposit.numElectrons;

          if (charge > bestCharge) {
            bestCharge = charge;
            truth.dominantTrackId = particle.TrackId();
            truth.dominantPdg = particle.PdgCode();
          }

          int muonTrackId = 0;
          int muonPdg = 0;
          if (isMichelElectronFromAncestry(particle, particleMap, muonTrackId, muonPdg)) {
            hasMichel = true;
            truth.michelMuonTrackId = muonTrackId;
            truth.michelMuonPdg = muonPdg;
          }
          else if (absPdg == 13) {
            hasMuon = true;
          }
          else if (absPdg == 11 || absPdg == 22) {
            hasEM = true;
          }
          else {
            hasOther = true;
          }
        }
      }
    }

    if (hasMichel) truth.truthCode = kTruthMichel;
    else if (hasMuon) truth.truthCode = kTruthMuon;
    else if (hasEM) truth.truthCode = kTruthEM;
    else if (hasOther) truth.truthCode = kTruthOther;
    else if (std::abs(truth.dominantPdg) == 13) truth.truthCode = kTruthMuon;
    else if (std::abs(truth.dominantPdg) == 11 || std::abs(truth.dominantPdg) == 22) truth.truthCode = kTruthEM;

    return truth;
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraIvysaurusMuonEndpointTraining::fillEndpoint(
    EndpointProjection const& endpoint,
    int cryo,
    int tpc,
    std::vector<art::Ptr<recob::Hit>> const& hits,
    detinfo::DetectorClocksData const& clockData,
    std::vector<sim::SimChannel> const& simChannels,
    std::unordered_map<int, simb::MCParticle const*> const& particleMap,
    std::array<std::vector<float>, kNPlanes>& grids,
    std::array<std::vector<int>, kNPlanes>& truthGrids,
    EndpointTruth& truth) const
  {
    int endpointHits = 0;
    int const halfW = fGridWires / 2;
    int const halfD = fGridDrifts / 2;

    for (auto const& hit : hits) {
      unsigned int const plane = hit->WireID().Plane;
      if (plane >= kNPlanes) continue;
      if (static_cast<int>(hit->WireID().Cryostat) != cryo) continue;
      if (static_cast<int>(hit->WireID().TPC) != tpc) continue;
      if (hit->Integral() < fMinHitIntegral) continue;

      int const gridW = static_cast<int>(std::lround(static_cast<float>(hit->WireID().Wire) - endpoint.wire[plane])) + halfW;
      int const gridD = static_cast<int>(std::lround(hit->PeakTime() / static_cast<float>(fDriftBinWidth) - endpoint.tick[plane])) + halfD;
      if (gridW < 0 || gridW >= fGridWires) continue;
      if (gridD < 0 || gridD >= fGridDrifts) continue;

      std::size_t const index = static_cast<std::size_t>(gridD * fGridWires + gridW);
      grids[plane][index] += hit->Integral();
      ++endpointHits;

      HitTruth const hitTruth = classifyHit(hit, clockData, simChannels, particleMap);
      truthGrids[plane][index] = std::max(truthGrids[plane][index], hitTruth.truthCode);

      if (hitTruth.truthCode == kTruthMichel) {
        ++truth.michelHits;
        truth.michelMuonTrackId = hitTruth.michelMuonTrackId;
        truth.michelMuonPdg = hitTruth.michelMuonPdg;
      }
      else if (hitTruth.truthCode == kTruthMuon) ++truth.muonHits;
      else if (hitTruth.truthCode == kTruthEM) ++truth.emHits;
      else ++truth.otherHits;

      if (truth.dominantTrackId == 0 && hitTruth.dominantTrackId != 0) {
        truth.dominantTrackId = hitTruth.dominantTrackId;
        truth.dominantPdg = hitTruth.dominantPdg;
      }
    }

    float const michelFraction =
      (endpointHits > 0) ? static_cast<float>(truth.michelHits) / static_cast<float>(endpointHits) : 0.F;
    truth.hasMichel =
      (truth.michelHits >= fMinMichelHits && michelFraction >= fMinMichelFraction) ? 1 : 0;
  }

  //-----------------------------------------------------------------------
  std::vector<art::Ptr<recob::Hit>> PointIdPandoraIvysaurusMuonEndpointTraining::collectEventHits(
    art::Event const& event,
    std::vector<recob::Track> const& tracks,
    art::FindManyP<recob::Hit> const& trackHitAssoc) const
  {
    std::vector<art::Ptr<recob::Hit>> eventHits;
    art::Handle<std::vector<recob::Hit>> hitHandle;
    if (event.getByLabel(fHitLabel, hitHandle)) {
      art::fill_ptr_vector(eventHits, hitHandle);
      return eventHits;
    }

    std::map<std::size_t, art::Ptr<recob::Hit>> uniqueAssociatedHits;
    for (std::size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
      for (auto const& hit : trackHitAssoc.at(iTrack)) {
        uniqueAssociatedHits.emplace(hit.key(), hit);
      }
    }
    for (auto const& entry : uniqueAssociatedHits) eventHits.push_back(entry.second);

    mf::LogWarning("PointIdPandoraIvysaurusMuonEndpointTraining")
      << "HitLabel " << fHitLabel.encode()
      << " was not found; using union of Pandora track-associated hits for event displays";
    return eventHits;
  }

  //-----------------------------------------------------------------------
  bool PointIdPandoraIvysaurusMuonEndpointTraining::selected(
    std::vector<int> const& values, int value) const
  {
    return values.empty() || std::find(values.begin(), values.end(), value) != values.end();
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraIvysaurusMuonEndpointTraining::writeEndpointPatchHists(
    art::Event const& event,
    int pfpKey,
    int trackKey,
    int cryo,
    int tpc,
    std::array<std::vector<float>, kNPlanes> const& startGrids,
    std::array<std::vector<float>, kNPlanes> const& endGrids,
    std::array<std::vector<int>, kNPlanes> const& startTruthGrids,
    std::array<std::vector<int>, kNPlanes> const& endTruthGrids) const
  {
    art::ServiceHandle<art::TFileService> tfs;

    auto writeOnePatch = [&](char const* endpointName,
                             std::array<std::vector<float>, kNPlanes> const& grids,
                             std::array<std::vector<int>, kNPlanes> const& truthGrids) {
      for (unsigned int plane = 0; plane < kNPlanes; ++plane) {
        if (!selected(fSelectedView, static_cast<int>(plane))) continue;
        if (grids[plane].size() != static_cast<std::size_t>(fGridWires * fGridDrifts)) continue;
        if (truthGrids[plane].size() != static_cast<std::size_t>(fGridWires * fGridDrifts)) continue;

        bool hasCharge = false;
        for (float const value : grids[plane]) {
          if (value > 0.0F) {
            hasCharge = true;
            break;
          }
        }
        if (!hasCharge) continue;

        std::ostringstream baseName;
        baseName << endpointName
                 << "_patch_event_" << event.id().event()
                 << "_run_" << event.run()
                 << "_subrun_" << event.subRun()
                 << "_pfp_" << pfpKey
                 << "_track_" << trackKey
                 << "_cryo_" << cryo
                 << "_tpc_" << tpc
                 << "_view_" << plane;

        TH2F* chargeHist = tfs->make<TH2F>((baseName.str() + "_charge").c_str(),
                                           "Endpoint CNN input hit charge;local wire bin;local drift bin",
                                           fGridWires,
                                           0,
                                           fGridWires,
                                           fGridDrifts,
                                           0,
                                           fGridDrifts);

        TH2I* truthHist = nullptr;
        if (fSaveEndpointPatchTruth) {
          truthHist = tfs->make<TH2I>((baseName.str() + "_truthcode").c_str(),
                                      "Endpoint CNN truth code;local wire bin;local drift bin",
                                      fGridWires,
                                      0,
                                      fGridWires,
                                      fGridDrifts,
                                      0,
                                      fGridDrifts);
        }

        for (int drift = 0; drift < fGridDrifts; ++drift) {
          for (int wire = 0; wire < fGridWires; ++wire) {
            std::size_t const index = static_cast<std::size_t>(drift * fGridWires + wire);
            chargeHist->SetBinContent(wire + 1, drift + 1, grids[plane][index]);
            if (truthHist) truthHist->SetBinContent(wire + 1, drift + 1, truthGrids[plane][index]);
          }
        }
      }
    };

    writeOnePatch("start", startGrids, startTruthGrids);
    writeOnePatch("end", endGrids, endTruthGrids);
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraIvysaurusMuonEndpointTraining::writeEventDisplays(
    art::Event const& event,
    detinfo::DetectorClocksData const& clockData,
    std::vector<art::Ptr<recob::Hit>> const& eventHits,
    std::vector<sim::SimChannel> const& simChannels,
    std::unordered_map<int, simb::MCParticle const*> const& particleMap) const
  {
    using DisplayKey = std::tuple<int, int, int>;
    std::map<DisplayKey, EventDisplayBounds> boundsByDisplay;

    for (auto const& hit : eventHits) {
      auto const& wid = hit->WireID();
      int const cryo = static_cast<int>(wid.Cryostat);
      int const tpc = static_cast<int>(wid.TPC);
      int const plane = static_cast<int>(wid.Plane);
      if (!selected(fSelectedTPC, tpc) || !selected(fSelectedView, plane)) continue;

      int const wire = static_cast<int>(wid.Wire);
      int const drift = static_cast<int>(std::lround(hit->PeakTime() / static_cast<float>(fDriftBinWidth)));
      boundsByDisplay[DisplayKey{cryo, tpc, plane}].update(wire, drift);
    }

    if (boundsByDisplay.empty()) return;

    std::ostringstream eventName;
    eventName << "event_" << event.id().event()
              << "_run_" << event.run()
              << "_subrun_" << event.subRun();

    art::ServiceHandle<art::TFileService> tfs;

    for (auto const& entry : boundsByDisplay) {
      int cryo = 0;
      int tpc = 0;
      int plane = 0;
      std::tie(cryo, tpc, plane) = entry.first;
      EventDisplayBounds const& bounds = entry.second;
      if (!bounds.valid()) continue;

      int const w0 = std::max(0, bounds.minWire - fEventDisplayWireMargin);
      int const w1 = std::max(w0 + 1, bounds.maxWire + fEventDisplayWireMargin + 1);
      int const d0 = std::max(0, bounds.minDrift - fEventDisplayDriftMargin);
      int const d1 = std::max(d0 + 1, bounds.maxDrift + fEventDisplayDriftMargin + 1);

      std::ostringstream baseName;
      baseName << "event_display_" << eventName.str()
               << "_cryo_" << cryo
               << "_tpc_" << tpc
               << "_view_" << plane;

      TH2F* chargeHist = tfs->make<TH2F>(
        (baseName.str() + "_charge").c_str(),
        "Event hit charge;wire;drift",
        w1 - w0,
        w0,
        w1,
        d1 - d0,
        d0,
        d1);

      TH2I* truthHist = nullptr;
      if (fSaveEventTruth) {
        truthHist = tfs->make<TH2I>(
          (baseName.str() + "_truthcode").c_str(),
          "Event truth code;wire;drift",
          w1 - w0,
          w0,
          w1,
          d1 - d0,
          d0,
          d1);
      }

      for (auto const& hit : eventHits) {
        auto const& wid = hit->WireID();
        if (static_cast<int>(wid.Cryostat) != cryo) continue;
        if (static_cast<int>(wid.TPC) != tpc) continue;
        if (static_cast<int>(wid.Plane) != plane) continue;
        if (hit->Integral() < fMinHitIntegral) continue;

        int const wire = static_cast<int>(wid.Wire);
        int const drift = static_cast<int>(std::lround(hit->PeakTime() / static_cast<float>(fDriftBinWidth)));
        if (wire < w0 || wire >= w1 || drift < d0 || drift >= d1) continue;

        float charge = static_cast<float>(hit->Integral());
        if (!std::isfinite(charge) || charge <= 0.0F) {
          charge = static_cast<float>(hit->PeakAmplitude());
        }
        if (!std::isfinite(charge) || charge <= 0.0F) continue;

        chargeHist->Fill(wire, drift, charge);
        if (truthHist) {
          HitTruth const hitTruth = classifyHit(hit, clockData, simChannels, particleMap);
          int const bin = truthHist->FindBin(wire, drift);
          int const currentTruth = static_cast<int>(truthHist->GetBinContent(bin));
          truthHist->SetBinContent(bin, std::max(currentTruth, hitTruth.truthCode));
        }
      }
    }
  }

  //-----------------------------------------------------------------------
  bool PointIdPandoraIvysaurusMuonEndpointTraining::isMuonDecayAncestor(
    simb::MCParticle const& particle,
    std::unordered_map<int, simb::MCParticle const*> const& particleMap) const
  {
    if (std::abs(particle.PdgCode()) != 13) return false;

    bool hasElectron = false;
    bool hasNuMu = false;
    bool hasNuE = false;

    for (int i = 0; i < particle.NumberDaughters(); ++i) {
      auto const daughterSearch = particleMap.find(particle.Daughter(i));
      if (daughterSearch == particleMap.end()) continue;

      int const daughterPdg = std::abs(daughterSearch->second->PdgCode());
      if (daughterPdg == 11) hasElectron = true;
      else if (daughterPdg == 14) hasNuMu = true;
      else if (daughterPdg == 12) hasNuE = true;
    }

    return hasElectron && hasNuMu && hasNuE;
  }

  //-----------------------------------------------------------------------
  bool PointIdPandoraIvysaurusMuonEndpointTraining::isMichelElectronFromAncestry(
    simb::MCParticle const& particle,
    std::unordered_map<int, simb::MCParticle const*> const& particleMap,
    int& muonTrackId,
    int& muonPdg) const
  {
    muonTrackId = 0;
    muonPdg = 0;

    if (std::abs(particle.PdgCode()) != 11) return false;

    int motherTrackId = particle.Mother();
    for (int depth = 0; depth < 32 && motherTrackId != 0; ++depth) {
      auto const motherSearch = particleMap.find(motherTrackId);
      if (motherSearch == particleMap.end()) break;

      simb::MCParticle const& mother = *(motherSearch->second);
      if (isMuonDecayAncestor(mother, particleMap)) {
        muonTrackId = mother.TrackId();
        muonPdg = mother.PdgCode();
        return true;
      }

      motherTrackId = mother.Mother();
    }

    return false;
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraIvysaurusMuonEndpointTraining::resetBranches()
  {
    b_pfpKey = -1;
    b_pfpSelf = 0;
    b_pfpPdg = 0;
    b_trackKey = -1;
    b_cryo = -1;
    b_tpc = -1;
    b_nHits = 0;
    b_trackLength = 0.F;

    std::fill(std::begin(b_startXYZ), std::end(b_startXYZ), 0.F);
    std::fill(std::begin(b_endXYZ), std::end(b_endXYZ), 0.F);
    std::fill(std::begin(b_startWire), std::end(b_startWire), 0.F);
    std::fill(std::begin(b_startTick), std::end(b_startTick), 0.F);
    std::fill(std::begin(b_endWire), std::end(b_endWire), 0.F);
    std::fill(std::begin(b_endTick), std::end(b_endTick), 0.F);

    b_startHasMichel = 0;
    b_endHasMichel = 0;
    b_startMichelHits = 0;
    b_endMichelHits = 0;
    b_startMuonHits = 0;
    b_endMuonHits = 0;
    b_startEMHits = 0;
    b_endEMHits = 0;
    b_startOtherHits = 0;
    b_endOtherHits = 0;
    b_startDominantTrackId = 0;
    b_endDominantTrackId = 0;
    b_startDominantPdg = 0;
    b_endDominantPdg = 0;
    b_startMichelMuonTrackId = 0;
    b_endMichelMuonTrackId = 0;
    b_startMichelMuonPdg = 0;
    b_endMichelMuonPdg = 0;

    resetGridVectors();
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraIvysaurusMuonEndpointTraining::resetGridVectors()
  {
    std::size_t const gridSize = static_cast<std::size_t>(fGridWires) * static_cast<std::size_t>(fGridDrifts);

    b_startGridU.assign(gridSize, 0.F);
    b_startGridV.assign(gridSize, 0.F);
    b_startGridW.assign(gridSize, 0.F);
    b_endGridU.assign(gridSize, 0.F);
    b_endGridV.assign(gridSize, 0.F);
    b_endGridW.assign(gridSize, 0.F);
    b_startTruthU.assign(gridSize, kTruthEmpty);
    b_startTruthV.assign(gridSize, kTruthEmpty);
    b_startTruthW.assign(gridSize, kTruthEmpty);
    b_endTruthU.assign(gridSize, kTruthEmpty);
    b_endTruthV.assign(gridSize, kTruthEmpty);
    b_endTruthW.assign(gridSize, kTruthEmpty);
  }

  DEFINE_ART_MODULE(PointIdPandoraIvysaurusMuonEndpointTraining)

} // namespace nnet
