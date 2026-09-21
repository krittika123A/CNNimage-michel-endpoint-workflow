////////////////////////////////////////////////////////////////////////////////////////////////////
// Class: PointIdSimpleGeantProcessTruth
//
// Build one W-plane charge image at each reconstructed Pandora track extremum.  Pandora is used
// only to provide reconstructed tracks and their extrema.  Patch truth comes directly from
// SimChannel IDE track IDs and the largeant MCParticle records produced from GEANT4.
//
// A patch is Michel-positive when at least one hit in the corresponding U, V, or W endpoint
// window contains charge from the direct electron daughter in a primary-muon three-body decay
// (electron + electron-flavour neutrino + muon-flavour neutrino), and ParticleInventory maps that
// muon to the outgoing lepton of a generator-level CC nu_mu/anti-nu_mu interaction.  No descendant
// propagation, majority vote, minimum Michel-hit count, or Michel fraction is applied.  The
// optional W truth bitmap uses the explicit priority Michel > muon > EM > other.
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
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Comment.h"
#include "fhiclcpp/types/Name.h"
#include "fhiclcpp/types/Sequence.h"
#include "fhiclcpp/types/Table.h"
#include "larcore/Geometry/Geometry.h"
#include "larcore/Geometry/WireReadout.h"
#include "larcorealg/Geometry/GeometryCore.h"
#include "lardata/DetectorInfoServices/DetectorClocksService.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesService.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/Simulation/SimChannel.h"
#include "larsim/MCCheater/ParticleInventoryService.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "nusimdata/SimulationBase/MCTruth.h"

#include "TH2F.h"
#include "TH2I.h"
#include "TTree.h"
#include "TVector3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

  constexpr unsigned int kNPlanes = 3U;
  constexpr unsigned int kWPlane = 2U;

  enum TruthCode {
    kTruthEmpty = 0,
    kTruthMuon = 1,
    kTruthMichel = 2,
    kTruthEM = 3,
    kTruthOther = 4
  };

  int truthPriority(int code)
  {
    if (code == kTruthMichel) return 4;
    if (code == kTruthMuon) return 3;
    if (code == kTruthEM) return 2;
    if (code == kTruthOther) return 1;
    return 0;
  }

  int mergeTruthCodes(int current, int incoming)
  {
    return truthPriority(incoming) > truthPriority(current) ? incoming : current;
  }

  struct DominantTPC {
    bool valid = false;
    int cryo = -1;
    int tpc = -1;
    std::size_t nHits = 0U;
  };

  struct EndpointProjection {
    bool valid = false;
    std::array<float, kNPlanes> wire{{0.F, 0.F, 0.F}};
    std::array<float, kNPlanes> tick{{0.F, 0.F, 0.F}};
    std::array<float, 3U> xyz{{0.F, 0.F, 0.F}};
  };

  struct NeutrinoTruth {
    int associated = 0;
    int neutrinoSet = 0;
    int validatedCCNuMu = 0;
    int truthKey = -1;
    int origin = -1;
    int neutrinoPdg = 0;
    int leptonPdg = 0;
    int ccnc = -1;
    int mode = -1;
    int interactionType = -1;
    std::array<float, 3U> vertexXYZ{{0.F, 0.F, 0.F}};
    float muonVertexDistance = -1.F;
  };

  struct HitTruth {
    int truthCode = kTruthOther;
    int dominantTrackId = 0;
    int dominantPdg = 0;
    double dominantCharge = 0.0;
    double michelCharge = 0.0;
    double michelDepositEnergyMeV = 0.0;
    int michelElectronTrackId = 0;
    int michelElectronPdg = 0;
    int michelMuonTrackId = 0;
    int michelMuonPdg = 0;
    std::string michelElectronProcess;
    std::string michelElectronEndProcess;
    std::string michelMuonProcess;
    std::string michelMuonEndProcess;
    int geantMichelCandidate = 0;
    NeutrinoTruth neutrinoTruth;
  };

  struct EndpointTruth {
    int hasMichel = 0;
    int hasMichelW = 0;
    int patchTruthCode = kTruthEmpty;
    int totalHits = 0;
    int michelHits = 0;
    int muonHits = 0;
    int emHits = 0;
    int otherHits = 0;
    int wImageValid = 0;
    std::array<int, kNPlanes> planeHits{{0, 0, 0}};
    std::array<int, kNPlanes> planeMichelHits{{0, 0, 0}};
    std::array<double, kNPlanes> planeMichelCharge{{0.0, 0.0, 0.0}};
    double michelCharge = 0.0;
    double michelDepositEnergyMeV = 0.0;
    double bestMichelCharge = 0.0;
    int dominantTrackId = 0;
    int dominantPdg = 0;
    double dominantCharge = 0.0;
    int michelElectronTrackId = 0;
    int michelElectronPdg = 0;
    int michelMuonTrackId = 0;
    int michelMuonPdg = 0;
    int geantMichelHits = 0;
    int mcTruthRejectedMichelHits = 0;
    NeutrinoTruth neutrinoTruth;
    std::string michelElectronProcess;
    std::string michelElectronEndProcess;
    std::string michelMuonProcess;
    std::string michelMuonEndProcess;
  };

} // namespace

namespace nnet {

  class PointIdSimpleGeantProcessTruth : public art::EDAnalyzer {
  public:
    struct Config {
      using Name = fhicl::Name;
      using Comment = fhicl::Comment;

      fhicl::Atom<art::InputTag> PandoraPfoLabel{
        Name("PandoraPfoLabel"), Comment("Pandora PFParticle collection label")};
      fhicl::Atom<art::InputTag> PandoraTrackLabel{
        Name("PandoraTrackLabel"),
        Comment("Pandora track collection and PFParticle-to-track association label")};
      fhicl::Atom<art::InputTag> TrackHitAssocLabel{
        Name("TrackHitAssocLabel"), Comment("Pandora track-to-hit association label")};
      fhicl::Atom<art::InputTag> HitLabel{
        Name("HitLabel"), Comment("Full-event reconstructed hit collection used to fill patches")};
      fhicl::Atom<art::InputTag> ParticleLabel{
        Name("ParticleLabel"), Comment("Direct GEANT MCParticle collection, normally largeant")};
      fhicl::Atom<art::InputTag> SimChannelLabel{
        Name("SimChannelLabel"), Comment("SimChannel collection containing GEANT IDE track IDs")};

      fhicl::Sequence<std::string> MichelElectronProcesses{
        Name("MichelElectronProcesses"),
        Comment("GEANT Process() values accepted for a muon-decay electron"),
        std::vector<std::string>{"Decay", "DecayWithSpin"}};
      fhicl::Sequence<std::string> MichelMuonEndProcesses{
        Name("MichelMuonEndProcesses"),
        Comment("GEANT EndProcess() values accepted for the parent muon"),
        std::vector<std::string>{"Decay", "DecayWithSpin"}};
      fhicl::Sequence<std::string> PrimaryMuonProcesses{
        Name("PrimaryMuonProcesses"),
        Comment("GEANT Process() values accepted for a neutrino-interaction primary muon"),
        std::vector<std::string>{"primary"}};
      fhicl::Atom<float> HitTimeWindowRMS{
        Name("HitTimeWindowRMS"),
        Comment("Maximum |hit.TimeDistanceAsRMS(simulated tick)| used for IDE matching"),
        1.F};
      fhicl::Atom<float> MCTruthVertexTolerance{
        Name("MCTruthVertexTolerance"),
        Comment("Maximum distance in cm between the GEANT primary-muon start and GENIE neutrino vertex"),
        1.F};

      fhicl::Atom<float> MinTrackLength{
        Name("MinTrackLength"), Comment("Minimum reconstructed track length in cm"), 10.F};
      fhicl::Atom<bool> RequireContainedTrack{
        Name("RequireContainedTrack"),
        Comment("Require both reconstructed track extrema inside the active TPC"),
        true};
      fhicl::Atom<float> ContainmentMargin{
        Name("ContainmentMargin"),
        Comment("Inward margin from every X, Y, and Z active-volume face in cm"),
        20.F};
      fhicl::Sequence<int> SelectedTPC{
        Name("SelectedTPC"),
        Comment("TPC numbers to keep; empty keeps every TPC"),
        std::vector<int>{}};

      fhicl::Atom<int> GridWires{Name("GridWires"), Comment("Wire bins in each W image"), 48};
      fhicl::Atom<int> GridDrifts{Name("GridDrifts"), Comment("Drift/time bins in each W image"), 48};
      fhicl::Atom<int> DriftBinWidth{
        Name("DriftBinWidth"), Comment("Number of reconstructed ticks per image drift bin"), 6};
      fhicl::Atom<float> MinHitIntegral{
        Name("MinHitIntegral"), Comment("Minimum reconstructed hit integral used in a patch"), 0.F};
      fhicl::Atom<float> ChargeScale{
        Name("ChargeScale"),
        Comment("Single global multiplier applied to every hit integral; 1 keeps raw charge"),
        1.F};
      fhicl::Atom<bool> SaveEndpointPatchHists{
        Name("SaveEndpointPatchHists"), Comment("Write W-plane start/end charge TH2 histograms"), true};
      fhicl::Atom<bool> SaveEndpointPatchTruth{
        Name("SaveEndpointPatchTruth"), Comment("Write W-plane per-pixel truth-code TH2 histograms"), true};
    };

    using Parameters = art::EDAnalyzer::Table<Config>;

    explicit PointIdSimpleGeantProcessTruth(Parameters const& config);
    void beginJob() override;

  private:
    using ParticleMap = std::unordered_map<int, simb::MCParticle const*>;
    using SimChannelMap = std::unordered_map<unsigned int, sim::SimChannel const*>;

    void analyze(art::Event const& event) override;

    DominantTPC dominantTPC(std::vector<art::Ptr<recob::Hit>> const& hits) const;
    EndpointProjection projectEndpoint(recob::Track const& track,
                                       bool useEnd,
                                       int cryo,
                                       int tpc,
                                       detinfo::DetectorPropertiesData const& detProp,
                                       geo::WireReadoutGeom const& wireReadoutGeom) const;
    bool endpointContained(TVector3 const& point, int cryo, int tpc) const;
    bool selectedTPC(int tpc) const;
    bool acceptsProcess(std::vector<std::string> const& accepted,
                        std::string const& process) const;
    bool isDirectMichelElectron(simb::MCParticle const& contributor,
                                ParticleMap const& particleMap,
                                simb::MCParticle const*& parentMuon) const;
    bool validateMuonNeutrinoTruth(simb::MCParticle const& parentMuon,
                                   cheat::ParticleInventoryService const& particleInventory,
                                   NeutrinoTruth& truth) const;
    HitTruth classifyHit(art::Ptr<recob::Hit> const& hit,
                         detinfo::DetectorClocksData const& clockData,
                         SimChannelMap const& simChannelMap,
                         ParticleMap const& particleMap,
                         cheat::ParticleInventoryService const& particleInventory) const;
    void fillEndpoint(EndpointProjection const& endpoint,
                      int cryo,
                      int tpc,
                      std::vector<art::Ptr<recob::Hit>> const& eventHits,
                      detinfo::DetectorClocksData const& clockData,
                      SimChannelMap const& simChannelMap,
                      ParticleMap const& particleMap,
                      cheat::ParticleInventoryService const& particleInventory,
                      std::vector<float>& wGrid,
                      std::vector<int>& wTruthGrid,
                      EndpointTruth& truth) const;
    void writeEndpointPatchHists(art::Event const& event,
                                 int pfpKey,
                                 int trackKey,
                                 int cryo,
                                 int tpc,
                                 std::vector<float> const& startGrid,
                                 std::vector<float> const& endGrid,
                                 std::vector<int> const& startTruthGrid,
                                 std::vector<int> const& endTruthGrid) const;
    void copyTruthToBranches(EndpointTruth const& startTruth, EndpointTruth const& endTruth);
    void resetBranches();

    art::InputTag fPandoraPfoLabel;
    art::InputTag fPandoraTrackLabel;
    art::InputTag fTrackHitAssocLabel;
    art::InputTag fHitLabel;
    art::InputTag fParticleLabel;
    art::InputTag fSimChannelLabel;
    std::vector<std::string> fMichelElectronProcesses;
    std::vector<std::string> fMichelMuonEndProcesses;
    std::vector<std::string> fPrimaryMuonProcesses;
    float fHitTimeWindowRMS;
    float fMCTruthVertexTolerance;
    float fMinTrackLength;
    bool fRequireContainedTrack;
    float fContainmentMargin;
    std::vector<int> fSelectedTPC;
    int fGridWires;
    int fGridDrifts;
    int fDriftBinWidth;
    float fMinHitIntegral;
    float fChargeScale;
    bool fSaveEndpointPatchHists;
    bool fSaveEndpointPatchTruth;

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
    int b_nTrackHits = 0;
    float b_trackLength = 0.F;
    float b_startXYZ[3] = {};
    float b_endXYZ[3] = {};
    float b_startWireW = 0.F;
    float b_startTickW = 0.F;
    float b_endWireW = 0.F;
    float b_endTickW = 0.F;

    int b_startLabel = 0;
    int b_endLabel = 0;
    int b_startHasMichelW = 0;
    int b_endHasMichelW = 0;
    int b_startTruthCode = kTruthEmpty;
    int b_endTruthCode = kTruthEmpty;
    int b_startWImageValid = 0;
    int b_endWImageValid = 0;
    int b_startTotalHits = 0;
    int b_endTotalHits = 0;
    int b_startMichelHits = 0;
    int b_endMichelHits = 0;
    int b_startMuonHits = 0;
    int b_endMuonHits = 0;
    int b_startEMHits = 0;
    int b_endEMHits = 0;
    int b_startOtherHits = 0;
    int b_endOtherHits = 0;
    int b_startPlaneHits[3] = {};
    int b_endPlaneHits[3] = {};
    int b_startPlaneMichelHits[3] = {};
    int b_endPlaneMichelHits[3] = {};
    float b_startMichelCharge = 0.F;
    float b_endMichelCharge = 0.F;
    float b_startMichelDepositEnergyMeV = 0.F;
    float b_endMichelDepositEnergyMeV = 0.F;
    int b_startDominantTrackId = 0;
    int b_endDominantTrackId = 0;
    int b_startDominantPdg = 0;
    int b_endDominantPdg = 0;
    int b_startMichelElectronTrackId = 0;
    int b_endMichelElectronTrackId = 0;
    int b_startMichelElectronPdg = 0;
    int b_endMichelElectronPdg = 0;
    int b_startMichelMuonTrackId = 0;
    int b_endMichelMuonTrackId = 0;
    int b_startMichelMuonPdg = 0;
    int b_endMichelMuonPdg = 0;
    int b_startGeantMichelHits = 0;
    int b_endGeantMichelHits = 0;
    int b_startMCTruthRejectedMichelHits = 0;
    int b_endMCTruthRejectedMichelHits = 0;
    int b_startMichelMCTruthAssociated = 0;
    int b_endMichelMCTruthAssociated = 0;
    int b_startMichelMCTruthNeutrinoSet = 0;
    int b_endMichelMCTruthNeutrinoSet = 0;
    int b_startMichelMCTruthValidatedCCNuMu = 0;
    int b_endMichelMCTruthValidatedCCNuMu = 0;
    int b_startMichelMCTruthKey = -1;
    int b_endMichelMCTruthKey = -1;
    int b_startMichelMCTruthOrigin = -1;
    int b_endMichelMCTruthOrigin = -1;
    int b_startMichelNuPdg = 0;
    int b_endMichelNuPdg = 0;
    int b_startMichelLeptonPdg = 0;
    int b_endMichelLeptonPdg = 0;
    int b_startMichelCCNC = -1;
    int b_endMichelCCNC = -1;
    int b_startMichelMode = -1;
    int b_endMichelMode = -1;
    int b_startMichelInteractionType = -1;
    int b_endMichelInteractionType = -1;
    float b_startMichelNuVertexXYZ[3] = {};
    float b_endMichelNuVertexXYZ[3] = {};
    float b_startMichelMuonVertexDistance = -1.F;
    float b_endMichelMuonVertexDistance = -1.F;
    std::string b_startMichelElectronProcess;
    std::string b_endMichelElectronProcess;
    std::string b_startMichelElectronEndProcess;
    std::string b_endMichelElectronEndProcess;
    std::string b_startMichelMuonProcess;
    std::string b_endMichelMuonProcess;
    std::string b_startMichelMuonEndProcess;
    std::string b_endMichelMuonEndProcess;
    std::vector<float> b_startGridW;
    std::vector<float> b_endGridW;
    std::vector<int> b_startTruthW;
    std::vector<int> b_endTruthW;
  };

  PointIdSimpleGeantProcessTruth::PointIdSimpleGeantProcessTruth(Parameters const& config)
    : art::EDAnalyzer(config)
    , fPandoraPfoLabel(config().PandoraPfoLabel())
    , fPandoraTrackLabel(config().PandoraTrackLabel())
    , fTrackHitAssocLabel(config().TrackHitAssocLabel())
    , fHitLabel(config().HitLabel())
    , fParticleLabel(config().ParticleLabel())
    , fSimChannelLabel(config().SimChannelLabel())
    , fMichelElectronProcesses(config().MichelElectronProcesses())
    , fMichelMuonEndProcesses(config().MichelMuonEndProcesses())
    , fPrimaryMuonProcesses(config().PrimaryMuonProcesses())
    , fHitTimeWindowRMS(config().HitTimeWindowRMS())
    , fMCTruthVertexTolerance(config().MCTruthVertexTolerance())
    , fMinTrackLength(config().MinTrackLength())
    , fRequireContainedTrack(config().RequireContainedTrack())
    , fContainmentMargin(config().ContainmentMargin())
    , fSelectedTPC(config().SelectedTPC())
    , fGridWires(config().GridWires())
    , fGridDrifts(config().GridDrifts())
    , fDriftBinWidth(config().DriftBinWidth())
    , fMinHitIntegral(config().MinHitIntegral())
    , fChargeScale(config().ChargeScale())
    , fSaveEndpointPatchHists(config().SaveEndpointPatchHists())
    , fSaveEndpointPatchTruth(config().SaveEndpointPatchTruth())
  {}

  void PointIdSimpleGeantProcessTruth::beginJob()
  {
    art::ServiceHandle<art::TFileService> tfs;
    fTree = tfs->make<TTree>("simpleGeantProcessTruth", "W-plane endpoint patches with direct GEANT process truth");

    fTree->Branch("run", &b_run, "run/I");
    fTree->Branch("subrun", &b_subrun, "subrun/I");
    fTree->Branch("event", &b_event, "event/I");
    fTree->Branch("pfp_key", &b_pfpKey, "pfp_key/I");
    fTree->Branch("pfp_self", &b_pfpSelf, "pfp_self/I");
    fTree->Branch("pfp_pdg", &b_pfpPdg, "pfp_pdg/I");
    fTree->Branch("track_key", &b_trackKey, "track_key/I");
    fTree->Branch("cryo", &b_cryo, "cryo/I");
    fTree->Branch("tpc", &b_tpc, "tpc/I");
    fTree->Branch("n_track_hits", &b_nTrackHits, "n_track_hits/I");
    fTree->Branch("track_length", &b_trackLength, "track_length/F");
    fTree->Branch("StartXYZ", b_startXYZ, "StartXYZ[3]/F");
    fTree->Branch("EndXYZ", b_endXYZ, "EndXYZ[3]/F");
    fTree->Branch("StartWireW", &b_startWireW, "StartWireW/F");
    fTree->Branch("StartTickW", &b_startTickW, "StartTickW/F");
    fTree->Branch("EndWireW", &b_endWireW, "EndWireW/F");
    fTree->Branch("EndTickW", &b_endTickW, "EndTickW/F");

    fTree->Branch("StartLabel", &b_startLabel, "StartLabel/I");
    fTree->Branch("EndLabel", &b_endLabel, "EndLabel/I");
    fTree->Branch("StartHasMichelW", &b_startHasMichelW, "StartHasMichelW/I");
    fTree->Branch("EndHasMichelW", &b_endHasMichelW, "EndHasMichelW/I");
    fTree->Branch("StartTruthCode", &b_startTruthCode, "StartTruthCode/I");
    fTree->Branch("EndTruthCode", &b_endTruthCode, "EndTruthCode/I");
    fTree->Branch("StartWImageValid", &b_startWImageValid, "StartWImageValid/I");
    fTree->Branch("EndWImageValid", &b_endWImageValid, "EndWImageValid/I");
    fTree->Branch("StartTotalHits", &b_startTotalHits, "StartTotalHits/I");
    fTree->Branch("EndTotalHits", &b_endTotalHits, "EndTotalHits/I");
    fTree->Branch("StartMichelHits", &b_startMichelHits, "StartMichelHits/I");
    fTree->Branch("EndMichelHits", &b_endMichelHits, "EndMichelHits/I");
    fTree->Branch("StartMuonHits", &b_startMuonHits, "StartMuonHits/I");
    fTree->Branch("EndMuonHits", &b_endMuonHits, "EndMuonHits/I");
    fTree->Branch("StartEMHits", &b_startEMHits, "StartEMHits/I");
    fTree->Branch("EndEMHits", &b_endEMHits, "EndEMHits/I");
    fTree->Branch("StartOtherHits", &b_startOtherHits, "StartOtherHits/I");
    fTree->Branch("EndOtherHits", &b_endOtherHits, "EndOtherHits/I");
    fTree->Branch("StartPlaneHits", b_startPlaneHits, "StartPlaneHits[3]/I");
    fTree->Branch("EndPlaneHits", b_endPlaneHits, "EndPlaneHits[3]/I");
    fTree->Branch("StartPlaneMichelHits", b_startPlaneMichelHits, "StartPlaneMichelHits[3]/I");
    fTree->Branch("EndPlaneMichelHits", b_endPlaneMichelHits, "EndPlaneMichelHits[3]/I");
    fTree->Branch("StartMichelCharge", &b_startMichelCharge, "StartMichelCharge/F");
    fTree->Branch("EndMichelCharge", &b_endMichelCharge, "EndMichelCharge/F");
    fTree->Branch("StartMichelDepositEnergyMeV", &b_startMichelDepositEnergyMeV, "StartMichelDepositEnergyMeV/F");
    fTree->Branch("EndMichelDepositEnergyMeV", &b_endMichelDepositEnergyMeV, "EndMichelDepositEnergyMeV/F");
    fTree->Branch("StartDominantTrackId", &b_startDominantTrackId, "StartDominantTrackId/I");
    fTree->Branch("EndDominantTrackId", &b_endDominantTrackId, "EndDominantTrackId/I");
    fTree->Branch("StartDominantPdg", &b_startDominantPdg, "StartDominantPdg/I");
    fTree->Branch("EndDominantPdg", &b_endDominantPdg, "EndDominantPdg/I");
    fTree->Branch("StartMichelElectronTrackId", &b_startMichelElectronTrackId, "StartMichelElectronTrackId/I");
    fTree->Branch("EndMichelElectronTrackId", &b_endMichelElectronTrackId, "EndMichelElectronTrackId/I");
    fTree->Branch("StartMichelElectronPdg", &b_startMichelElectronPdg, "StartMichelElectronPdg/I");
    fTree->Branch("EndMichelElectronPdg", &b_endMichelElectronPdg, "EndMichelElectronPdg/I");
    fTree->Branch("StartMichelMuonTrackId", &b_startMichelMuonTrackId, "StartMichelMuonTrackId/I");
    fTree->Branch("EndMichelMuonTrackId", &b_endMichelMuonTrackId, "EndMichelMuonTrackId/I");
    fTree->Branch("StartMichelMuonPdg", &b_startMichelMuonPdg, "StartMichelMuonPdg/I");
    fTree->Branch("EndMichelMuonPdg", &b_endMichelMuonPdg, "EndMichelMuonPdg/I");
    fTree->Branch("StartGeantMichelHits", &b_startGeantMichelHits, "StartGeantMichelHits/I");
    fTree->Branch("EndGeantMichelHits", &b_endGeantMichelHits, "EndGeantMichelHits/I");
    fTree->Branch("StartMCTruthRejectedMichelHits",
                  &b_startMCTruthRejectedMichelHits,
                  "StartMCTruthRejectedMichelHits/I");
    fTree->Branch("EndMCTruthRejectedMichelHits",
                  &b_endMCTruthRejectedMichelHits,
                  "EndMCTruthRejectedMichelHits/I");
    fTree->Branch("StartMichelMCTruthAssociated",
                  &b_startMichelMCTruthAssociated,
                  "StartMichelMCTruthAssociated/I");
    fTree->Branch("EndMichelMCTruthAssociated",
                  &b_endMichelMCTruthAssociated,
                  "EndMichelMCTruthAssociated/I");
    fTree->Branch("StartMichelMCTruthNeutrinoSet",
                  &b_startMichelMCTruthNeutrinoSet,
                  "StartMichelMCTruthNeutrinoSet/I");
    fTree->Branch("EndMichelMCTruthNeutrinoSet",
                  &b_endMichelMCTruthNeutrinoSet,
                  "EndMichelMCTruthNeutrinoSet/I");
    fTree->Branch("StartMichelMCTruthValidatedCCNuMu",
                  &b_startMichelMCTruthValidatedCCNuMu,
                  "StartMichelMCTruthValidatedCCNuMu/I");
    fTree->Branch("EndMichelMCTruthValidatedCCNuMu",
                  &b_endMichelMCTruthValidatedCCNuMu,
                  "EndMichelMCTruthValidatedCCNuMu/I");
    fTree->Branch("StartMichelMCTruthKey", &b_startMichelMCTruthKey, "StartMichelMCTruthKey/I");
    fTree->Branch("EndMichelMCTruthKey", &b_endMichelMCTruthKey, "EndMichelMCTruthKey/I");
    fTree->Branch("StartMichelMCTruthOrigin",
                  &b_startMichelMCTruthOrigin,
                  "StartMichelMCTruthOrigin/I");
    fTree->Branch("EndMichelMCTruthOrigin",
                  &b_endMichelMCTruthOrigin,
                  "EndMichelMCTruthOrigin/I");
    fTree->Branch("StartMichelNuPdg", &b_startMichelNuPdg, "StartMichelNuPdg/I");
    fTree->Branch("EndMichelNuPdg", &b_endMichelNuPdg, "EndMichelNuPdg/I");
    fTree->Branch("StartMichelLeptonPdg", &b_startMichelLeptonPdg, "StartMichelLeptonPdg/I");
    fTree->Branch("EndMichelLeptonPdg", &b_endMichelLeptonPdg, "EndMichelLeptonPdg/I");
    fTree->Branch("StartMichelCCNC", &b_startMichelCCNC, "StartMichelCCNC/I");
    fTree->Branch("EndMichelCCNC", &b_endMichelCCNC, "EndMichelCCNC/I");
    fTree->Branch("StartMichelMode", &b_startMichelMode, "StartMichelMode/I");
    fTree->Branch("EndMichelMode", &b_endMichelMode, "EndMichelMode/I");
    fTree->Branch("StartMichelInteractionType",
                  &b_startMichelInteractionType,
                  "StartMichelInteractionType/I");
    fTree->Branch("EndMichelInteractionType",
                  &b_endMichelInteractionType,
                  "EndMichelInteractionType/I");
    fTree->Branch("StartMichelNuVertexXYZ",
                  b_startMichelNuVertexXYZ,
                  "StartMichelNuVertexXYZ[3]/F");
    fTree->Branch("EndMichelNuVertexXYZ",
                  b_endMichelNuVertexXYZ,
                  "EndMichelNuVertexXYZ[3]/F");
    fTree->Branch("StartMichelMuonVertexDistance",
                  &b_startMichelMuonVertexDistance,
                  "StartMichelMuonVertexDistance/F");
    fTree->Branch("EndMichelMuonVertexDistance",
                  &b_endMichelMuonVertexDistance,
                  "EndMichelMuonVertexDistance/F");
    fTree->Branch("StartMichelElectronProcess", &b_startMichelElectronProcess);
    fTree->Branch("EndMichelElectronProcess", &b_endMichelElectronProcess);
    fTree->Branch("StartMichelElectronEndProcess", &b_startMichelElectronEndProcess);
    fTree->Branch("EndMichelElectronEndProcess", &b_endMichelElectronEndProcess);
    fTree->Branch("StartMichelMuonProcess", &b_startMichelMuonProcess);
    fTree->Branch("EndMichelMuonProcess", &b_endMichelMuonProcess);
    fTree->Branch("StartMichelMuonEndProcess", &b_startMichelMuonEndProcess);
    fTree->Branch("EndMichelMuonEndProcess", &b_endMichelMuonEndProcess);
    fTree->Branch("StartGridW", &b_startGridW);
    fTree->Branch("EndGridW", &b_endGridW);
    fTree->Branch("StartTruthW", &b_startTruthW);
    fTree->Branch("EndTruthW", &b_endTruthW);
  }

  void PointIdSimpleGeantProcessTruth::analyze(art::Event const& event)
  {
    auto const clockData =
      art::ServiceHandle<detinfo::DetectorClocksService const>()->DataFor(event);
    auto const detProp =
      art::ServiceHandle<detinfo::DetectorPropertiesService const>()->DataFor(event, clockData);
    auto const& wireReadoutGeom = art::ServiceHandle<geo::WireReadout const>()->Get();
    auto const& particleInventory =
      *art::ServiceHandle<cheat::ParticleInventoryService>();

    auto const pfpHandle = event.getValidHandle<std::vector<recob::PFParticle>>(fPandoraPfoLabel);
    auto const trackHandle = event.getValidHandle<std::vector<recob::Track>>(fPandoraTrackLabel);
    auto const hitHandle = event.getValidHandle<std::vector<recob::Hit>>(fHitLabel);
    auto const particleHandle = event.getValidHandle<std::vector<simb::MCParticle>>(fParticleLabel);
    auto const simChannelHandle = event.getValidHandle<std::vector<sim::SimChannel>>(fSimChannelLabel);

    art::FindManyP<recob::Track> pfpTrackAssoc(pfpHandle, event, fPandoraTrackLabel);
    art::FindManyP<recob::Hit> trackHitAssoc(trackHandle, event, fTrackHitAssocLabel);

    std::vector<art::Ptr<recob::Hit>> eventHits;
    art::fill_ptr_vector(eventHits, hitHandle);

    ParticleMap particleMap;
    particleMap.reserve(particleHandle->size());
    for (auto const& particle : *particleHandle) {
      particleMap.emplace(std::abs(particle.TrackId()), &particle);
    }

    SimChannelMap simChannelMap;
    simChannelMap.reserve(simChannelHandle->size());
    for (auto const& channel : *simChannelHandle) {
      simChannelMap.emplace(static_cast<unsigned int>(channel.Channel()), &channel);
    }

    for (std::size_t iPfp = 0; iPfp < pfpHandle->size(); ++iPfp) {
      recob::PFParticle const& pfp = pfpHandle->at(iPfp);
      auto const tracks = pfpTrackAssoc.at(iPfp);

      for (auto const& trackPtr : tracks) {
        recob::Track const& track = *trackPtr;
        if (track.Length() < fMinTrackLength) continue;

        auto const trackHits = trackHitAssoc.at(trackPtr.key());
        if (trackHits.empty()) continue;

        DominantTPC const domTPC = dominantTPC(trackHits);
        if (!domTPC.valid || !selectedTPC(domTPC.tpc)) continue;

        EndpointProjection const start =
          projectEndpoint(track, false, domTPC.cryo, domTPC.tpc, detProp, wireReadoutGeom);
        EndpointProjection const end =
          projectEndpoint(track, true, domTPC.cryo, domTPC.tpc, detProp, wireReadoutGeom);
        if (!start.valid || !end.valid) continue;

        if (fRequireContainedTrack &&
            (!endpointContained(TVector3(start.xyz[0], start.xyz[1], start.xyz[2]), domTPC.cryo, domTPC.tpc) ||
             !endpointContained(TVector3(end.xyz[0], end.xyz[1], end.xyz[2]), domTPC.cryo, domTPC.tpc))) {
          continue;
        }

        resetBranches();
        b_run = event.run();
        b_subrun = event.subRun();
        b_event = event.id().event();
        b_pfpKey = static_cast<int>(iPfp);
        b_pfpSelf = pfp.Self();
        b_pfpPdg = pfp.PdgCode();
        b_trackKey = static_cast<int>(trackPtr.key());
        b_cryo = domTPC.cryo;
        b_tpc = domTPC.tpc;
        b_nTrackHits = static_cast<int>(trackHits.size());
        b_trackLength = static_cast<float>(track.Length());
        for (unsigned int i = 0; i < 3U; ++i) {
          b_startXYZ[i] = start.xyz[i];
          b_endXYZ[i] = end.xyz[i];
        }
        b_startWireW = start.wire[kWPlane];
        b_startTickW = start.tick[kWPlane];
        b_endWireW = end.wire[kWPlane];
        b_endTickW = end.tick[kWPlane];

        EndpointTruth startTruth;
        EndpointTruth endTruth;
        fillEndpoint(start,
                     domTPC.cryo,
                     domTPC.tpc,
                     eventHits,
                     clockData,
                     simChannelMap,
                     particleMap,
                     particleInventory,
                     b_startGridW,
                     b_startTruthW,
                     startTruth);
        fillEndpoint(end,
                     domTPC.cryo,
                     domTPC.tpc,
                     eventHits,
                     clockData,
                     simChannelMap,
                     particleMap,
                     particleInventory,
                     b_endGridW,
                     b_endTruthW,
                     endTruth);

        copyTruthToBranches(startTruth, endTruth);

        if (fSaveEndpointPatchHists) {
          writeEndpointPatchHists(event,
                                  b_pfpKey,
                                  b_trackKey,
                                  b_cryo,
                                  b_tpc,
                                  b_startGridW,
                                  b_endGridW,
                                  b_startTruthW,
                                  b_endTruthW);
        }

        fTree->Fill();
      }
    }
  }

  DominantTPC PointIdSimpleGeantProcessTruth::dominantTPC(
    std::vector<art::Ptr<recob::Hit>> const& hits) const
  {
    std::map<std::pair<int, int>, std::size_t> counts;
    for (auto const& hit : hits) {
      ++counts[{static_cast<int>(hit->WireID().Cryostat), static_cast<int>(hit->WireID().TPC)}];
    }

    DominantTPC result;
    for (auto const& entry : counts) {
      if (entry.second <= result.nHits) continue;
      result.valid = true;
      result.cryo = entry.first.first;
      result.tpc = entry.first.second;
      result.nHits = entry.second;
    }
    return result;
  }

  EndpointProjection PointIdSimpleGeantProcessTruth::projectEndpoint(
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
    endpoint.xyz = {{static_cast<float>(point.X()),
                     static_cast<float>(point.Y()),
                     static_cast<float>(point.Z())}};

    for (unsigned int plane = 0; plane < kNPlanes; ++plane) {
      geo::PlaneID const planeID{static_cast<unsigned int>(cryo),
                                 static_cast<unsigned int>(tpc),
                                 plane};
      if (!wireReadoutGeom.HasPlane(planeID)) return endpoint;

      double const wire = wireReadoutGeom.Plane(planeID).WireCoordinate(geoPoint);
      double const tick =
        detProp.ConvertXToTicks(point.X(), plane, tpc, cryo) / static_cast<double>(fDriftBinWidth);
      if (!std::isfinite(wire) || !std::isfinite(tick)) return endpoint;
      endpoint.wire[plane] = static_cast<float>(wire);
      endpoint.tick[plane] = static_cast<float>(tick);
    }

    endpoint.valid = true;
    return endpoint;
  }

  bool PointIdSimpleGeantProcessTruth::endpointContained(
    TVector3 const& point, int cryo, int tpc) const
  {
    art::ServiceHandle<geo::Geometry const> geom;
    auto const& tpcGeo =
      geom->TPC(geo::TPCID{static_cast<unsigned int>(cryo), static_cast<unsigned int>(tpc)});
    auto const& active = tpcGeo.ActiveBoundingBox();
    return active.InFiducialX(point.X(), fContainmentMargin) &&
           active.InFiducialY(point.Y(), fContainmentMargin) &&
           active.InFiducialZ(point.Z(), fContainmentMargin);
  }

  bool PointIdSimpleGeantProcessTruth::selectedTPC(int tpc) const
  {
    return fSelectedTPC.empty() ||
           std::find(fSelectedTPC.begin(), fSelectedTPC.end(), tpc) != fSelectedTPC.end();
  }

  bool PointIdSimpleGeantProcessTruth::acceptsProcess(
    std::vector<std::string> const& accepted, std::string const& process) const
  {
    return std::find(accepted.begin(), accepted.end(), process) != accepted.end();
  }

  bool PointIdSimpleGeantProcessTruth::isDirectMichelElectron(
    simb::MCParticle const& contributor,
    ParticleMap const& particleMap,
    simb::MCParticle const*& parentMuon) const
  {
    parentMuon = nullptr;
    if (std::abs(contributor.PdgCode()) != 11 ||
        !acceptsProcess(fMichelElectronProcesses, contributor.Process())) {
      return false;
    }

    auto const motherSearch = particleMap.find(std::abs(contributor.Mother()));
    if (motherSearch == particleMap.end()) return false;

    simb::MCParticle const* mother = motherSearch->second;
    if (std::abs(mother->PdgCode()) != 13 ||
        !acceptsProcess(fPrimaryMuonProcesses, mother->Process()) ||
        mother->Mother() != 0 ||
        !acceptsProcess(fMichelMuonEndProcesses, mother->EndProcess())) {
      return false;
    }

    int const expectedElectronPdg = mother->PdgCode() > 0 ? 11 : -11;
    int const expectedElectronNeutrinoPdg = mother->PdgCode() > 0 ? -12 : 12;
    int const expectedMuonNeutrinoPdg = mother->PdgCode() > 0 ? 14 : -14;
    if (contributor.PdgCode() != expectedElectronPdg) return false;

    bool hasContributorElectron = false;
    bool hasElectronNeutrino = false;
    bool hasMuonNeutrino = false;
    for (int iDaughter = 0; iDaughter < mother->NumberDaughters(); ++iDaughter) {
      auto const daughterSearch =
        particleMap.find(std::abs(mother->Daughter(iDaughter)));
      if (daughterSearch == particleMap.end()) continue;

      simb::MCParticle const& daughter = *(daughterSearch->second);
      if (std::abs(daughter.TrackId()) == std::abs(contributor.TrackId()) &&
          daughter.PdgCode() == expectedElectronPdg) {
        hasContributorElectron = true;
      }
      else if (daughter.PdgCode() == expectedElectronNeutrinoPdg) {
        hasElectronNeutrino = true;
      }
      else if (daughter.PdgCode() == expectedMuonNeutrinoPdg) {
        hasMuonNeutrino = true;
      }
    }
    if (!hasContributorElectron || !hasElectronNeutrino || !hasMuonNeutrino) return false;

    parentMuon = mother;
    return true;
  }

  bool PointIdSimpleGeantProcessTruth::validateMuonNeutrinoTruth(
    simb::MCParticle const& parentMuon,
    cheat::ParticleInventoryService const& particleInventory,
    NeutrinoTruth& truth) const
  {
    art::Ptr<simb::MCTruth> truthPtr;
    try {
      truthPtr = particleInventory.TrackIdToMCTruth_P(parentMuon.TrackId());
    }
    catch (cet::exception const&) {
      return false;
    }
    if (!truthPtr) return false;

    truth.associated = 1;
    truth.truthKey = static_cast<int>(truthPtr.key());
    truth.origin = static_cast<int>(truthPtr->Origin());
    truth.neutrinoSet = truthPtr->NeutrinoSet() ? 1 : 0;
    if (!truth.neutrinoSet) return false;

    simb::MCNeutrino const& neutrino = truthPtr->GetNeutrino();
    truth.neutrinoPdg = neutrino.Nu().PdgCode();
    truth.leptonPdg = neutrino.Lepton().PdgCode();
    truth.ccnc = neutrino.CCNC();
    truth.mode = neutrino.Mode();
    truth.interactionType = neutrino.InteractionType();
    truth.vertexXYZ = {{static_cast<float>(neutrino.Nu().EndX()),
                        static_cast<float>(neutrino.Nu().EndY()),
                        static_cast<float>(neutrino.Nu().EndZ())}};

    double const dx = parentMuon.Vx() - truth.vertexXYZ[0];
    double const dy = parentMuon.Vy() - truth.vertexXYZ[1];
    double const dz = parentMuon.Vz() - truth.vertexXYZ[2];
    truth.muonVertexDistance = static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz));

    bool const isCCNuMu = neutrino.CCNC() == simb::kCC &&
                          std::abs(neutrino.Nu().PdgCode()) == 14 &&
                          neutrino.Lepton().PdgCode() == parentMuon.PdgCode();
    bool const atInteractionVertex =
      truth.muonVertexDistance <= fMCTruthVertexTolerance;
    truth.validatedCCNuMu = isCCNuMu && atInteractionVertex ? 1 : 0;
    return truth.validatedCCNuMu != 0;
  }

  HitTruth PointIdSimpleGeantProcessTruth::classifyHit(
    art::Ptr<recob::Hit> const& hit,
    detinfo::DetectorClocksData const& clockData,
    SimChannelMap const& simChannelMap,
    ParticleMap const& particleMap,
    cheat::ParticleInventoryService const& particleInventory) const
  {
    HitTruth truth;
    bool hasMichel = false;
    bool hasMuon = false;
    bool hasEM = false;
    bool hasOther = false;
    double bestMichelCharge = 0.0;

    auto const channelSearch = simChannelMap.find(static_cast<unsigned int>(hit->Channel()));
    if (channelSearch == simChannelMap.end()) return truth;

    for (auto const& timeSlice : channelSearch->second->TDCIDEMap()) {
      double const simTick = clockData.TPCTDC2Tick(static_cast<double>(timeSlice.first));
      if (std::abs(hit->TimeDistanceAsRMS(simTick)) >= fHitTimeWindowRMS) continue;

      for (auto const& ide : timeSlice.second) {
        auto const particleSearch = particleMap.find(std::abs(ide.trackID));
        if (particleSearch == particleMap.end()) {
          hasOther = true;
          continue;
        }

        simb::MCParticle const& particle = *(particleSearch->second);
        double const charge = std::max(0.0, static_cast<double>(ide.numElectrons));
        if (charge > truth.dominantCharge) {
          truth.dominantCharge = charge;
          truth.dominantTrackId = particle.TrackId();
          truth.dominantPdg = particle.PdgCode();
        }

        simb::MCParticle const* parentMuon = nullptr;
        if (isDirectMichelElectron(particle, particleMap, parentMuon)) {
          truth.geantMichelCandidate = 1;
          NeutrinoTruth neutrinoTruth;
          if (validateMuonNeutrinoTruth(*parentMuon, particleInventory, neutrinoTruth)) {
            hasMichel = true;
            truth.michelCharge += charge;
            truth.michelDepositEnergyMeV += std::max(0.0, static_cast<double>(ide.energy));
            if (charge > bestMichelCharge) {
              bestMichelCharge = charge;
              truth.neutrinoTruth = neutrinoTruth;
              truth.michelElectronTrackId = particle.TrackId();
              truth.michelElectronPdg = particle.PdgCode();
              truth.michelMuonTrackId = parentMuon->TrackId();
              truth.michelMuonPdg = parentMuon->PdgCode();
              truth.michelElectronProcess = particle.Process();
              truth.michelElectronEndProcess = particle.EndProcess();
              truth.michelMuonProcess = parentMuon->Process();
              truth.michelMuonEndProcess = parentMuon->EndProcess();
            }
          }
          else {
            if (!truth.neutrinoTruth.validatedCCNuMu) truth.neutrinoTruth = neutrinoTruth;
            hasEM = true;
          }
        }
        else if (std::abs(particle.PdgCode()) == 13) {
          hasMuon = true;
        }
        else if (std::abs(particle.PdgCode()) == 11 || std::abs(particle.PdgCode()) == 22) {
          hasEM = true;
        }
        else {
          hasOther = true;
        }
      }
    }

    if (hasMichel) truth.truthCode = kTruthMichel;
    else if (hasMuon) truth.truthCode = kTruthMuon;
    else if (hasEM) truth.truthCode = kTruthEM;
    else if (hasOther) truth.truthCode = kTruthOther;
    return truth;
  }

  void PointIdSimpleGeantProcessTruth::fillEndpoint(
    EndpointProjection const& endpoint,
    int cryo,
    int tpc,
    std::vector<art::Ptr<recob::Hit>> const& eventHits,
    detinfo::DetectorClocksData const& clockData,
    SimChannelMap const& simChannelMap,
    ParticleMap const& particleMap,
    cheat::ParticleInventoryService const& particleInventory,
    std::vector<float>& wGrid,
    std::vector<int>& wTruthGrid,
    EndpointTruth& truth) const
  {
    int const halfW = fGridWires / 2;
    int const halfD = fGridDrifts / 2;

    for (auto const& hit : eventHits) {
      unsigned int const plane = hit->WireID().Plane;
      if (plane >= kNPlanes) continue;
      if (static_cast<int>(hit->WireID().Cryostat) != cryo) continue;
      if (static_cast<int>(hit->WireID().TPC) != tpc) continue;

      float const integral = static_cast<float>(hit->Integral());
      if (!std::isfinite(integral) || integral <= fMinHitIntegral) continue;

      int const gridW =
        static_cast<int>(std::lround(static_cast<float>(hit->WireID().Wire) - endpoint.wire[plane])) + halfW;
      int const gridD =
        static_cast<int>(std::lround(hit->PeakTime() / static_cast<float>(fDriftBinWidth) - endpoint.tick[plane])) + halfD;
      if (gridW < 0 || gridW >= fGridWires || gridD < 0 || gridD >= fGridDrifts) continue;

      HitTruth const hitTruth =
        classifyHit(hit, clockData, simChannelMap, particleMap, particleInventory);
      ++truth.totalHits;
      ++truth.planeHits[plane];
      truth.patchTruthCode = mergeTruthCodes(truth.patchTruthCode, hitTruth.truthCode);

      if (hitTruth.geantMichelCandidate) {
        ++truth.geantMichelHits;
        if (!hitTruth.neutrinoTruth.validatedCCNuMu) {
          ++truth.mcTruthRejectedMichelHits;
          if (!truth.neutrinoTruth.validatedCCNuMu) {
            truth.neutrinoTruth = hitTruth.neutrinoTruth;
          }
        }
      }

      if (hitTruth.dominantCharge > truth.dominantCharge) {
        truth.dominantCharge = hitTruth.dominantCharge;
        truth.dominantTrackId = hitTruth.dominantTrackId;
        truth.dominantPdg = hitTruth.dominantPdg;
      }

      if (hitTruth.truthCode == kTruthMichel) {
        truth.hasMichel = 1;
        ++truth.michelHits;
        ++truth.planeMichelHits[plane];
        truth.michelCharge += hitTruth.michelCharge;
        truth.planeMichelCharge[plane] += hitTruth.michelCharge;
        truth.michelDepositEnergyMeV += hitTruth.michelDepositEnergyMeV;
        if (plane == kWPlane) truth.hasMichelW = 1;

        if (hitTruth.michelCharge > truth.bestMichelCharge) {
          truth.bestMichelCharge = hitTruth.michelCharge;
          truth.michelElectronTrackId = hitTruth.michelElectronTrackId;
          truth.michelElectronPdg = hitTruth.michelElectronPdg;
          truth.michelMuonTrackId = hitTruth.michelMuonTrackId;
          truth.michelMuonPdg = hitTruth.michelMuonPdg;
          truth.michelElectronProcess = hitTruth.michelElectronProcess;
          truth.michelElectronEndProcess = hitTruth.michelElectronEndProcess;
          truth.michelMuonProcess = hitTruth.michelMuonProcess;
          truth.michelMuonEndProcess = hitTruth.michelMuonEndProcess;
          truth.neutrinoTruth = hitTruth.neutrinoTruth;
        }
      }
      else if (hitTruth.truthCode == kTruthMuon) {
        ++truth.muonHits;
      }
      else if (hitTruth.truthCode == kTruthEM) {
        ++truth.emHits;
      }
      else {
        ++truth.otherHits;
      }

      if (plane == kWPlane) {
        std::size_t const index = static_cast<std::size_t>(gridD * fGridWires + gridW);
        wGrid[index] += integral * fChargeScale;
        wTruthGrid[index] = mergeTruthCodes(wTruthGrid[index], hitTruth.truthCode);
        truth.wImageValid = 1;
      }
    }
  }

  void PointIdSimpleGeantProcessTruth::writeEndpointPatchHists(
    art::Event const& event,
    int pfpKey,
    int trackKey,
    int cryo,
    int tpc,
    std::vector<float> const& startGrid,
    std::vector<float> const& endGrid,
    std::vector<int> const& startTruthGrid,
    std::vector<int> const& endTruthGrid) const
  {
    art::ServiceHandle<art::TFileService> tfs;

    auto writeOne = [&](char const* endpoint,
                        std::vector<float> const& grid,
                        std::vector<int> const& truthGrid) {
      if (std::none_of(grid.begin(), grid.end(), [](float value) { return value > 0.F; })) return;

      std::ostringstream name;
      name << endpoint << "_w_patch_event_" << event.id().event()
           << "_run_" << event.run()
           << "_subrun_" << event.subRun()
           << "_pfp_" << pfpKey
           << "_track_" << trackKey
           << "_cryo_" << cryo
           << "_tpc_" << tpc;

      TH2F* chargeHist = tfs->make<TH2F>((name.str() + "_charge").c_str(),
                                         "W endpoint hit integral;local wire bin;local drift bin",
                                         fGridWires,
                                         0,
                                         fGridWires,
                                         fGridDrifts,
                                         0,
                                         fGridDrifts);
      TH2I* truthHist = nullptr;
      if (fSaveEndpointPatchTruth) {
        truthHist = tfs->make<TH2I>((name.str() + "_truthcode").c_str(),
                                    "W endpoint GEANT truth code;local wire bin;local drift bin",
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
          chargeHist->SetBinContent(wire + 1, drift + 1, grid[index]);
          if (truthHist) truthHist->SetBinContent(wire + 1, drift + 1, truthGrid[index]);
        }
      }
    };

    writeOne("start", startGrid, startTruthGrid);
    writeOne("end", endGrid, endTruthGrid);
  }

  void PointIdSimpleGeantProcessTruth::copyTruthToBranches(
    EndpointTruth const& startTruth, EndpointTruth const& endTruth)
  {
    b_startLabel = startTruth.hasMichel;
    b_endLabel = endTruth.hasMichel;
    b_startHasMichelW = startTruth.hasMichelW;
    b_endHasMichelW = endTruth.hasMichelW;
    b_startTruthCode = startTruth.patchTruthCode;
    b_endTruthCode = endTruth.patchTruthCode;
    b_startWImageValid = startTruth.wImageValid;
    b_endWImageValid = endTruth.wImageValid;
    b_startTotalHits = startTruth.totalHits;
    b_endTotalHits = endTruth.totalHits;
    b_startMichelHits = startTruth.michelHits;
    b_endMichelHits = endTruth.michelHits;
    b_startMuonHits = startTruth.muonHits;
    b_endMuonHits = endTruth.muonHits;
    b_startEMHits = startTruth.emHits;
    b_endEMHits = endTruth.emHits;
    b_startOtherHits = startTruth.otherHits;
    b_endOtherHits = endTruth.otherHits;
    for (unsigned int plane = 0; plane < kNPlanes; ++plane) {
      b_startPlaneHits[plane] = startTruth.planeHits[plane];
      b_endPlaneHits[plane] = endTruth.planeHits[plane];
      b_startPlaneMichelHits[plane] = startTruth.planeMichelHits[plane];
      b_endPlaneMichelHits[plane] = endTruth.planeMichelHits[plane];
    }
    b_startMichelCharge = static_cast<float>(startTruth.michelCharge);
    b_endMichelCharge = static_cast<float>(endTruth.michelCharge);
    b_startMichelDepositEnergyMeV = static_cast<float>(startTruth.michelDepositEnergyMeV);
    b_endMichelDepositEnergyMeV = static_cast<float>(endTruth.michelDepositEnergyMeV);
    b_startDominantTrackId = startTruth.dominantTrackId;
    b_endDominantTrackId = endTruth.dominantTrackId;
    b_startDominantPdg = startTruth.dominantPdg;
    b_endDominantPdg = endTruth.dominantPdg;
    b_startMichelElectronTrackId = startTruth.michelElectronTrackId;
    b_endMichelElectronTrackId = endTruth.michelElectronTrackId;
    b_startMichelElectronPdg = startTruth.michelElectronPdg;
    b_endMichelElectronPdg = endTruth.michelElectronPdg;
    b_startMichelMuonTrackId = startTruth.michelMuonTrackId;
    b_endMichelMuonTrackId = endTruth.michelMuonTrackId;
    b_startMichelMuonPdg = startTruth.michelMuonPdg;
    b_endMichelMuonPdg = endTruth.michelMuonPdg;
    b_startGeantMichelHits = startTruth.geantMichelHits;
    b_endGeantMichelHits = endTruth.geantMichelHits;
    b_startMCTruthRejectedMichelHits = startTruth.mcTruthRejectedMichelHits;
    b_endMCTruthRejectedMichelHits = endTruth.mcTruthRejectedMichelHits;
    b_startMichelMCTruthAssociated = startTruth.neutrinoTruth.associated;
    b_endMichelMCTruthAssociated = endTruth.neutrinoTruth.associated;
    b_startMichelMCTruthNeutrinoSet = startTruth.neutrinoTruth.neutrinoSet;
    b_endMichelMCTruthNeutrinoSet = endTruth.neutrinoTruth.neutrinoSet;
    b_startMichelMCTruthValidatedCCNuMu = startTruth.neutrinoTruth.validatedCCNuMu;
    b_endMichelMCTruthValidatedCCNuMu = endTruth.neutrinoTruth.validatedCCNuMu;
    b_startMichelMCTruthKey = startTruth.neutrinoTruth.truthKey;
    b_endMichelMCTruthKey = endTruth.neutrinoTruth.truthKey;
    b_startMichelMCTruthOrigin = startTruth.neutrinoTruth.origin;
    b_endMichelMCTruthOrigin = endTruth.neutrinoTruth.origin;
    b_startMichelNuPdg = startTruth.neutrinoTruth.neutrinoPdg;
    b_endMichelNuPdg = endTruth.neutrinoTruth.neutrinoPdg;
    b_startMichelLeptonPdg = startTruth.neutrinoTruth.leptonPdg;
    b_endMichelLeptonPdg = endTruth.neutrinoTruth.leptonPdg;
    b_startMichelCCNC = startTruth.neutrinoTruth.ccnc;
    b_endMichelCCNC = endTruth.neutrinoTruth.ccnc;
    b_startMichelMode = startTruth.neutrinoTruth.mode;
    b_endMichelMode = endTruth.neutrinoTruth.mode;
    b_startMichelInteractionType = startTruth.neutrinoTruth.interactionType;
    b_endMichelInteractionType = endTruth.neutrinoTruth.interactionType;
    for (unsigned int coordinate = 0; coordinate < 3U; ++coordinate) {
      b_startMichelNuVertexXYZ[coordinate] = startTruth.neutrinoTruth.vertexXYZ[coordinate];
      b_endMichelNuVertexXYZ[coordinate] = endTruth.neutrinoTruth.vertexXYZ[coordinate];
    }
    b_startMichelMuonVertexDistance = startTruth.neutrinoTruth.muonVertexDistance;
    b_endMichelMuonVertexDistance = endTruth.neutrinoTruth.muonVertexDistance;
    b_startMichelElectronProcess = startTruth.michelElectronProcess;
    b_endMichelElectronProcess = endTruth.michelElectronProcess;
    b_startMichelElectronEndProcess = startTruth.michelElectronEndProcess;
    b_endMichelElectronEndProcess = endTruth.michelElectronEndProcess;
    b_startMichelMuonProcess = startTruth.michelMuonProcess;
    b_endMichelMuonProcess = endTruth.michelMuonProcess;
    b_startMichelMuonEndProcess = startTruth.michelMuonEndProcess;
    b_endMichelMuonEndProcess = endTruth.michelMuonEndProcess;
  }

  void PointIdSimpleGeantProcessTruth::resetBranches()
  {
    b_pfpKey = -1;
    b_pfpSelf = 0;
    b_pfpPdg = 0;
    b_trackKey = -1;
    b_cryo = -1;
    b_tpc = -1;
    b_nTrackHits = 0;
    b_trackLength = 0.F;
    std::fill(std::begin(b_startXYZ), std::end(b_startXYZ), 0.F);
    std::fill(std::begin(b_endXYZ), std::end(b_endXYZ), 0.F);
    b_startWireW = b_startTickW = b_endWireW = b_endTickW = 0.F;
    b_startLabel = b_endLabel = 0;
    b_startHasMichelW = b_endHasMichelW = 0;
    b_startTruthCode = b_endTruthCode = kTruthEmpty;
    b_startWImageValid = b_endWImageValid = 0;
    b_startTotalHits = b_endTotalHits = 0;
    b_startMichelHits = b_endMichelHits = 0;
    b_startMuonHits = b_endMuonHits = 0;
    b_startEMHits = b_endEMHits = 0;
    b_startOtherHits = b_endOtherHits = 0;
    std::fill(std::begin(b_startPlaneHits), std::end(b_startPlaneHits), 0);
    std::fill(std::begin(b_endPlaneHits), std::end(b_endPlaneHits), 0);
    std::fill(std::begin(b_startPlaneMichelHits), std::end(b_startPlaneMichelHits), 0);
    std::fill(std::begin(b_endPlaneMichelHits), std::end(b_endPlaneMichelHits), 0);
    b_startMichelCharge = b_endMichelCharge = 0.F;
    b_startMichelDepositEnergyMeV = b_endMichelDepositEnergyMeV = 0.F;
    b_startDominantTrackId = b_endDominantTrackId = 0;
    b_startDominantPdg = b_endDominantPdg = 0;
    b_startMichelElectronTrackId = b_endMichelElectronTrackId = 0;
    b_startMichelElectronPdg = b_endMichelElectronPdg = 0;
    b_startMichelMuonTrackId = b_endMichelMuonTrackId = 0;
    b_startMichelMuonPdg = b_endMichelMuonPdg = 0;
    b_startGeantMichelHits = b_endGeantMichelHits = 0;
    b_startMCTruthRejectedMichelHits = b_endMCTruthRejectedMichelHits = 0;
    b_startMichelMCTruthAssociated = b_endMichelMCTruthAssociated = 0;
    b_startMichelMCTruthNeutrinoSet = b_endMichelMCTruthNeutrinoSet = 0;
    b_startMichelMCTruthValidatedCCNuMu = b_endMichelMCTruthValidatedCCNuMu = 0;
    b_startMichelMCTruthKey = b_endMichelMCTruthKey = -1;
    b_startMichelMCTruthOrigin = b_endMichelMCTruthOrigin = -1;
    b_startMichelNuPdg = b_endMichelNuPdg = 0;
    b_startMichelLeptonPdg = b_endMichelLeptonPdg = 0;
    b_startMichelCCNC = b_endMichelCCNC = -1;
    b_startMichelMode = b_endMichelMode = -1;
    b_startMichelInteractionType = b_endMichelInteractionType = -1;
    std::fill(std::begin(b_startMichelNuVertexXYZ), std::end(b_startMichelNuVertexXYZ), 0.F);
    std::fill(std::begin(b_endMichelNuVertexXYZ), std::end(b_endMichelNuVertexXYZ), 0.F);
    b_startMichelMuonVertexDistance = b_endMichelMuonVertexDistance = -1.F;
    b_startMichelElectronProcess.clear();
    b_endMichelElectronProcess.clear();
    b_startMichelElectronEndProcess.clear();
    b_endMichelElectronEndProcess.clear();
    b_startMichelMuonProcess.clear();
    b_endMichelMuonProcess.clear();
    b_startMichelMuonEndProcess.clear();
    b_endMichelMuonEndProcess.clear();

    std::size_t const gridSize =
      static_cast<std::size_t>(fGridWires) * static_cast<std::size_t>(fGridDrifts);
    b_startGridW.assign(gridSize, 0.F);
    b_endGridW.assign(gridSize, 0.F);
    b_startTruthW.assign(gridSize, kTruthEmpty);
    b_endTruthW.assign(gridSize, kTruthEmpty);
  }

  DEFINE_ART_MODULE(PointIdSimpleGeantProcessTruth)

} // namespace nnet
