////////////////////////////////////////////////////////////////////////////////////////////////////
// Class:       PointIdPandoraEndpointTrainingData
// Author:      adapted from PointIdTrainingData
//
// Additive extension of the original PointIdTrainingData_module.cc.
// Keeps the original full-image ROOT / text / NumPy outputs,
// and adds a Pandora endpoint TTree for endpoint-centred patch selection.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "larcore/Geometry/Geometry.h"
#include "larcore/Geometry/WireReadout.h"
#include "larcorealg/Geometry/GeometryCore.h"
#include "lardata/DetectorInfoServices/DetectorClocksService.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesService.h"
#include "larrecodnn/ImagePatternAlgs/Modules/c2numpy.h"
#include "larrecodnn/ImagePatternAlgs/Tensorflow/PointIdAlg/PointIdAlg.h"

// art extensions
#include "nurandom/RandomUtils/NuRandomService.h"

// Framework includes
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "canvas/Utilities/Exception.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Comment.h"
#include "fhiclcpp/types/Name.h"
#include "fhiclcpp/types/Sequence.h"
#include "fhiclcpp/types/Table.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/Track.h"

#include "CLHEP/Random/RandFlat.h"

#include "TH2F.h" // ADC and deposit maps
#include "TH2I.h" // PDG+vertex info map
#include "TTree.h"
#include "TVector3.h"

// C++ Includes
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {
  template <typename Hist>
  void writeAndDelete(Hist*& hist)
  {
    if (!hist) return;
    hist->Write();
    delete hist;
    hist = nullptr;
  } // writeAndDelete()

  struct DominantTPCInfo {
    bool valid = false;
    int cryo = -1;
    int tpc = -1;
    std::size_t nhits = 0U;
  };
} // local namespace

namespace nnet {

  class PointIdPandoraEndpointTrainingData : public art::EDAnalyzer {
  public:
    struct Config {
      using Name = fhicl::Name;
      using Comment = fhicl::Comment;

      fhicl::Table<nnet::TrainingDataAlg::Config> TrainingDataAlg{
        Name("TrainingDataAlg")
      };

      fhicl::Sequence<int> SelectedTPC{
        Name("SelectedTPC"),
        Comment("use selected TPCs only, or all TPCs if empty list")
      };

      fhicl::Sequence<int> SelectedView{
        Name("SelectedView"),
        Comment("Views to dump; empty means all")
      };

      fhicl::Atom<bool> Crop{
        Name("Crop"),
        Comment("Crop the projection to the event region plus margin")
      };

      fhicl::Atom<int> Patch_size_w{
        Name("Patch_size_w"),
        Comment("Patch size in wire dimension")
      };

      fhicl::Atom<int> Patch_size_d{
        Name("Patch_size_d"),
        Comment("Patch size in drift dimension")
      };

      fhicl::Atom<double> Em{
        Name("Em"),
        Comment("Fraction of Em patches to keep")
      };

      fhicl::Atom<double> Trk{
        Name("Trk"),
        Comment("Fraction of Trk patches to keep")
      };

      fhicl::Atom<double> Michel{
        Name("Michel"),
        Comment("Fraction of Michel patches to keep")
      };

      fhicl::Atom<double> None{
        Name("None"),
        Comment("Fraction of None patches to keep")
      };

      fhicl::Atom<double> StopTrk{
        Name("StopTrk"),
        Comment("Fraction of stopping Trk patches to keep")
      };

      fhicl::Atom<double> CleanTrk{
        Name("CleanTrk"),
        Comment("Fraction of clean track patches to keep")
      };

      fhicl::Atom<std::string> OutTextFilePath{
        Name("OutTextFilePath"),
        Comment("Text files with all needed data dumped.")
      };

      fhicl::Atom<std::string> OutNumpyFileName{
        Name("OutNumpyFileName"),
        Comment("Numpy files with patches.")
      };

      fhicl::Atom<bool> DumpToRoot{
        Name("DumpToRoot"),
        Comment("Dump to ROOT histogram file (replaces the text files)")
      };

      fhicl::Atom<bool> DumpToNumpy{
        Name("DumpToNumpy"),
        Comment("Dump to Numpy file (replaces the text files)")
      };

      fhicl::Atom<art::InputTag> PandoraTrackLabel{
        Name("PandoraTrackLabel"),
        Comment("Track label carrying recob::Track and Track<->Hit associations")
      };

      fhicl::Atom<double> MinTrackLength{
        Name("MinTrackLength"),
        Comment("Skip very short tracks when filling the endpoint tree"),
        10.0
      };

      fhicl::Atom<bool> SaveEndpointTree{
        Name("SaveEndpointTree"),
        Comment("Write a TTree with per-track endpoint coordinates and truth flags"),
        true
      };

      fhicl::Atom<bool> UseBothEndpoints{
        Name("UseBothEndpoints"),
        Comment("If false, write only the track end point; if true, write both start and end points"),
        true
      };

      fhicl::Atom<double> TruthDepositThreshold{
        Name("TruthDepositThreshold"),
        Comment("Deposit threshold used when tagging endpoint truth"),
        2.0e-5
      };

      fhicl::Atom<int> TruthCentreRadiusW{
        Name("TruthCentreRadiusW"),
        Comment("Half-width of the endpoint truth window in wires"),
        6
      };

      fhicl::Atom<int> TruthCentreRadiusD{
        Name("TruthCentreRadiusD"),
        Comment("Half-width of the endpoint truth window in drifts/ticks"),
        6
      };

      fhicl::Atom<int> TruthMichelMinPixels{
        Name("TruthMichelMinPixels"),
        Comment("Minimum Michel-labelled pixels required to flag the endpoint as Michel-positive"),
        3
      };

      fhicl::Atom<int> TruthMuonMinPixels{
        Name("TruthMuonMinPixels"),
        Comment("Minimum muon-labelled pixels required near endpoint to consider it a muon end"),
        5
      };
    };
    using Parameters = art::EDAnalyzer::Table<Config>;

    explicit PointIdPandoraEndpointTrainingData(Parameters const& config);

    void beginJob() override;
    void endJob() override;

  private:
    void analyze(const art::Event& event) override;

    DominantTPCInfo dominantTPC(std::vector<art::Ptr<recob::Hit>> const& hits) const;

    bool projectEndpoint(recob::Track const& trk,
                         int endpoint,
                         int cryo,
                         int tpc,
                         int plane,
                         detinfo::DetectorPropertiesData const& detProp,
                         geo::WireReadoutGeom const& wireReadoutGeom,
                         float& wireAbs,
                         float& tickAbs,
                         float xyz[3]) const;

    void evaluateEndpointTruth(float wireAbs,
                               float tickAbs,
                               int& nMichel,
                               int& nMuon,
                               int& nTrack,
                               int& nShower) const;

    int WeightedFit(int n,
                    std::vector<double> const& x,
                    std::vector<double> const& y,
                    std::vector<double> const& w,
                    double* params) const;

    nnet::TrainingDataAlg fTrainingDataAlg;

    art::InputTag fPandoraTrackLabel;

    std::string fOutTextFilePath;
    std::string fOutNumpyFileName;
    bool fDumpToRoot;
    bool fDumpToNumpy;

    std::vector<int> fSelectedTPC;
    std::vector<int> fSelectedPlane;

    bool fCrop;

    int fPatch_size_w;
    int fPatch_size_d;

    double fEm, fTrk, fMichel, fNone;
    double fStopTrk, fCleanTrk;

    double fMinTrackLength;
    bool fSaveEndpointTree;
    bool fUseBothEndpoints;
    double fTruthDepositThreshold;
    int fTruthCentreRadiusW;
    int fTruthCentreRadiusD;
    int fTruthMichelMinPixels;
    int fTruthMuonMinPixels;

    int nEm, nTrk, nMichel, nNone;
    int nEm_sel, nTrk_sel, nMichel_sel, nNone_sel;
    int nStopTrk_sel, nCleanTrk_sel;

    c2numpy_writer npywriter;

    CLHEP::HepRandomEngine& fEngine; ///< art-managed random-number engine

    TTree* fEndpointTree{nullptr};

    int b_run{};
    int b_subrun{};
    int b_event{};
    int b_trackKey{};
    int b_endpoint{}; // 0=start, 1=end
    int b_cryo{};
    int b_tpc{};
    int b_plane{};
    float b_trackLength{};
    float b_centerWireAbs{};
    float b_centerTickAbs{};
    float b_centerWireLocal{};
    float b_centerTickLocal{};
    int b_histWireOffset{};
    int b_histTickOffset{};
    int b_histNWires{};
    int b_histNDrifts{};
    int b_patchFits{};
    int b_truthHasMichel{};
    int b_truthMichelPixels{};
    int b_truthMuonPixels{};
    int b_truthTrackPixels{};
    int b_truthShowerPixels{};
    int b_nHits{};
    float b_startXYZ[3]{};
    float b_endXYZ[3]{};
  };

  //-----------------------------------------------------------------------
  PointIdPandoraEndpointTrainingData::PointIdPandoraEndpointTrainingData(
    PointIdPandoraEndpointTrainingData::Parameters const& config)
    : art::EDAnalyzer(config)
    , fTrainingDataAlg(config().TrainingDataAlg())
    , fPandoraTrackLabel(config().PandoraTrackLabel())
    , fOutTextFilePath(config().OutTextFilePath())
    , fOutNumpyFileName(config().OutNumpyFileName())
    , fDumpToRoot(config().DumpToRoot())
    , fDumpToNumpy(config().DumpToNumpy())
    , fSelectedTPC(config().SelectedTPC())
    , fSelectedPlane(config().SelectedView())
    , fCrop(config().Crop())
    , fPatch_size_w(config().Patch_size_w())
    , fPatch_size_d(config().Patch_size_d())
    , fEm(config().Em())
    , fTrk(config().Trk())
    , fMichel(config().Michel())
    , fNone(config().None())
    , fStopTrk(config().StopTrk())
    , fCleanTrk(config().CleanTrk())
    , fMinTrackLength(config().MinTrackLength())
    , fSaveEndpointTree(config().SaveEndpointTree())
    , fUseBothEndpoints(config().UseBothEndpoints())
    , fTruthDepositThreshold(config().TruthDepositThreshold())
    , fTruthCentreRadiusW(config().TruthCentreRadiusW())
    , fTruthCentreRadiusD(config().TruthCentreRadiusD())
    , fTruthMichelMinPixels(config().TruthMichelMinPixels())
    , fTruthMuonMinPixels(config().TruthMuonMinPixels())
    , fEngine(art::ServiceHandle<rndm::NuRandomService>()->registerAndSeedEngine(createEngine(0)))
  {
    art::ServiceHandle<geo::Geometry const> geom;
    auto const& wireReadoutGeom = art::ServiceHandle<geo::WireReadout>()->Get();

    if (fSelectedTPC.empty()) {
      for (size_t tpc = 0; tpc < geom->NTPC(); ++tpc)
        fSelectedTPC.push_back(static_cast<int>(tpc));
    }

    if (fSelectedPlane.empty()) {
      for (size_t p = 0; p < wireReadoutGeom.MaxPlanes(); ++p)
        fSelectedPlane.push_back(static_cast<int>(p));
    }
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraEndpointTrainingData::beginJob()
  {
    if (fDumpToNumpy) {
      c2numpy_init(&npywriter, fOutNumpyFileName, 50000);
      c2numpy_addcolumn(&npywriter, "run", C2NUMPY_UINT32);
      c2numpy_addcolumn(&npywriter, "subrun", C2NUMPY_UINT32);
      c2numpy_addcolumn(&npywriter, "evt", C2NUMPY_UINT32);
      c2numpy_addcolumn(&npywriter, "tpc", C2NUMPY_UINT8);
      c2numpy_addcolumn(&npywriter, "plane", C2NUMPY_UINT8);
      c2numpy_addcolumn(&npywriter, "wire", C2NUMPY_UINT16);
      c2numpy_addcolumn(&npywriter, "tck", C2NUMPY_UINT16);
      c2numpy_addcolumn(&npywriter, "y0", C2NUMPY_UINT8);
      c2numpy_addcolumn(&npywriter, "y1", C2NUMPY_UINT8);
      c2numpy_addcolumn(&npywriter, "y2", C2NUMPY_UINT8);
      c2numpy_addcolumn(&npywriter, "y3", C2NUMPY_UINT8);
      for (int i = 0; i < fPatch_size_w * fPatch_size_d; ++i) {
        c2numpy_addcolumn(&npywriter, Form("x%d", i), C2NUMPY_FLOAT32);
      }
    }

    nEm = 0;
    nTrk = 0;
    nMichel = 0;
    nNone = 0;
    nEm_sel = 0;
    nTrk_sel = 0;
    nMichel_sel = 0;
    nNone_sel = 0;
    nStopTrk_sel = 0;
    nCleanTrk_sel = 0;

    if (fSaveEndpointTree) {
      art::ServiceHandle<art::TFileService const> tfs;
      fEndpointTree =
        tfs->make<TTree>("pandoraEndpoints", "Pandora endpoint metadata for patch selection");

      fEndpointTree->Branch("run", &b_run, "run/I");
      fEndpointTree->Branch("subrun", &b_subrun, "subrun/I");
      fEndpointTree->Branch("event", &b_event, "event/I");
      fEndpointTree->Branch("track_key", &b_trackKey, "track_key/I");
      fEndpointTree->Branch("endpoint", &b_endpoint, "endpoint/I");
      fEndpointTree->Branch("cryo", &b_cryo, "cryo/I");
      fEndpointTree->Branch("tpc", &b_tpc, "tpc/I");
      fEndpointTree->Branch("plane", &b_plane, "plane/I");
      fEndpointTree->Branch("track_length", &b_trackLength, "track_length/F");

      fEndpointTree->Branch("center_wire_abs", &b_centerWireAbs, "center_wire_abs/F");
      fEndpointTree->Branch("center_tick_abs", &b_centerTickAbs, "center_tick_abs/F");
      fEndpointTree->Branch("center_wire_local", &b_centerWireLocal, "center_wire_local/F");
      fEndpointTree->Branch("center_tick_local", &b_centerTickLocal, "center_tick_local/F");

      fEndpointTree->Branch("hist_wire_offset", &b_histWireOffset, "hist_wire_offset/I");
      fEndpointTree->Branch("hist_tick_offset", &b_histTickOffset, "hist_tick_offset/I");
      fEndpointTree->Branch("hist_nwires", &b_histNWires, "hist_nwires/I");
      fEndpointTree->Branch("hist_ndrifts", &b_histNDrifts, "hist_ndrifts/I");
      fEndpointTree->Branch("patch_fits", &b_patchFits, "patch_fits/I");

      fEndpointTree->Branch("truth_has_michel", &b_truthHasMichel, "truth_has_michel/I");
      fEndpointTree->Branch("truth_michel_pixels", &b_truthMichelPixels, "truth_michel_pixels/I");
      fEndpointTree->Branch("truth_muon_pixels", &b_truthMuonPixels, "truth_muon_pixels/I");
      fEndpointTree->Branch("truth_track_pixels", &b_truthTrackPixels, "truth_track_pixels/I");
      fEndpointTree->Branch("truth_shower_pixels", &b_truthShowerPixels, "truth_shower_pixels/I");

      fEndpointTree->Branch("n_associated_hits", &b_nHits, "n_associated_hits/I");
      fEndpointTree->Branch("start_xyz", b_startXYZ, "start_xyz[3]/F");
      fEndpointTree->Branch("end_xyz", b_endXYZ, "end_xyz[3]/F");
    }
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraEndpointTrainingData::endJob()
  {
    std::cout << "nEm = " << nEm << std::endl;
    std::cout << "nTrk = " << nTrk << std::endl;
    std::cout << "nMichel = " << nMichel << std::endl;
    std::cout << "nNone = " << nNone << std::endl;
    std::cout << std::endl;
    std::cout << "nEm_sel = " << nEm_sel << std::endl;
    std::cout << "nTrk_sel = " << nTrk_sel << std::endl;
    std::cout << "nMichel_sel = " << nMichel_sel << std::endl;
    std::cout << "nNone_sel = " << nNone_sel << std::endl;
    std::cout << "nStopTrk_sel = " << nStopTrk_sel << std::endl;
    std::cout << "nCleanTrk_sel = " << nCleanTrk_sel << std::endl;

    if (fDumpToNumpy) c2numpy_close(&npywriter);
  }

  //-----------------------------------------------------------------------
  DominantTPCInfo PointIdPandoraEndpointTrainingData::dominantTPC(
    std::vector<art::Ptr<recob::Hit>> const& hits) const
  {
    DominantTPCInfo info;
    std::map<std::pair<int, int>, std::size_t> counts;

    for (auto const& hit : hits) {
      auto const& wid = hit->WireID();
      ++counts[std::make_pair(static_cast<int>(wid.Cryostat), static_cast<int>(wid.TPC))];
    }

    for (auto const& entry : counts) {
      if (!info.valid || entry.second > info.nhits) {
        info.valid = true;
        info.cryo = entry.first.first;
        info.tpc = entry.first.second;
        info.nhits = entry.second;
      }
    }

    return info;
  }

  //-----------------------------------------------------------------------
  bool PointIdPandoraEndpointTrainingData::projectEndpoint(
    recob::Track const& trk,
    int endpoint,
    int cryo,
    int tpc,
    int plane,
    detinfo::DetectorPropertiesData const& detProp,
    geo::WireReadoutGeom const& wireReadoutGeom,
    float& wireAbs,
    float& tickAbs,
    float xyz[3]) const
  {
    TVector3 point = (endpoint == 0) ? trk.Vertex<TVector3>() : trk.End<TVector3>();

    xyz[0] = point.X();
    xyz[1] = point.Y();
    xyz[2] = point.Z();

    geo::PlaneID const planeID{static_cast<unsigned int>(cryo),
                               static_cast<unsigned int>(tpc),
                               static_cast<unsigned int>(plane)};

    geo::Point_t pos(point.X(), point.Y(), point.Z());
    auto const& planeGeo = wireReadoutGeom.Plane(planeID);

    // Keep this in the same plane-local wire coordinate system used by
    // fTrainingDataAlg.wireData(w), the dumped histograms, and the patch script.
    wireAbs = static_cast<float>(planeGeo.WireCoordinate(pos));
    tickAbs = static_cast<float>(detProp.ConvertXToTicks(point.X(), plane, tpc, cryo));

    return std::isfinite(wireAbs) && std::isfinite(tickAbs);
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraEndpointTrainingData::evaluateEndpointTruth(
    float wireAbs,
    float tickAbs,
    int& nMichel,
    int& nMuon,
    int& nTrack,
    int& nShower) const
  {
    nMichel = 0;
    nMuon = 0;
    nTrack = 0;
    nShower = 0;

    int const wireCentre = static_cast<int>(std::lround(wireAbs));
    int const tickCentre = static_cast<int>(std::lround(tickAbs));

    int const wMin = std::max(0, wireCentre - fTruthCentreRadiusW);
    int const wMax =
      std::min(static_cast<int>(fTrainingDataAlg.NWires()) - 1, wireCentre + fTruthCentreRadiusW);

    int const dMin = std::max(0, tickCentre - fTruthCentreRadiusD);
    int const dMax = std::min(static_cast<int>(fTrainingDataAlg.NScaledDrifts()) - 1,
                              tickCentre + fTruthCentreRadiusD);

    for (int w = wMin; w <= wMax; ++w) {
      auto const& raw = fTrainingDataAlg.wireData(w);
      auto const& edep = fTrainingDataAlg.wireEdep(w);
      auto const& pdg = fTrainingDataAlg.wirePdg(w);

      for (int d = dMin; d <= dMax; ++d) {
        if (edep[d] < fTruthDepositThreshold || raw[d] < 0.05F) continue;

        int const pdgValue = pdg[d];
        int const absPdg = std::abs(pdgValue & 0x0FFF);
        bool const isMichel = ((pdgValue & 0xF000) == 0x2000) && (absPdg == 11);

        if (isMichel) ++nMichel;
        if (absPdg == 13) ++nMuon;
        if (absPdg == 11) ++nShower;
        else ++nTrack;
      }
    }
  }

  //-----------------------------------------------------------------------
  void PointIdPandoraEndpointTrainingData::analyze(const art::Event& event)
  {
    int const eventNumber = event.id().event();
    int const runNumber = event.run();
    int const subRunNumber = event.subRun();

    bool const saveSim = fTrainingDataAlg.saveSimInfo() && !event.isRealData();

    std::ostringstream os;
    os << "event_" << eventNumber << "_run_" << runNumber << "_subrun_" << subRunNumber;

    std::cout << "analyze " << os.str() << std::endl;

    auto const clockData =
      art::ServiceHandle<detinfo::DetectorClocksService const>()->DataFor(event);
    auto const detProp =
      art::ServiceHandle<detinfo::DetectorPropertiesService const>()->DataFor(event, clockData);
    auto const& wireReadoutGeom = art::ServiceHandle<geo::WireReadout>()->Get();

    auto const trackHandle = event.getValidHandle<std::vector<recob::Track>>(fPandoraTrackLabel);
    art::FindManyP<recob::Hit> trackHitAssoc(trackHandle, event, fPandoraTrackLabel);

    CLHEP::RandFlat flat(fEngine);

    // //for (int currentCryo = 0; currentCryo < static_cast<int>(geom.Ncryostats()); ++currentCryo) {
    //   for (size_t i = 0; i < fSelectedTPC.size(); ++i) {
    //     for (size_t v = 0; v < fSelectedPlane.size(); ++v) {

    //       int const currentTPC = fSelectedTPC[i];
    //       int const currentPlane = fSelectedPlane[v];
    //       int const currentCryo = 0; // matches original PointIdTrainingData usage
    
    for (unsigned int currentCryo = 0; 
        currentCryo < geom.Ncryostats(); 
        ++currentCryo) 
      {
        for (unsigned int currentTPC = 0; 
          currentTPC < geom.NTPC(currentCryo); 
          ++currentTPC) 
          {
            for (unsigned int currentPlane = 0; 
                 currentPlane < wireReadoutGeom.Nplanes(); 
                 ++currentPlane) 
                {
              
                  // Debug: print current cryostat, TPC, plane
                  std::cout << "CRYO=" << currentCryo
                            << " TPC=" << currentTPC
                            << " PLANE=" << currentPlane
                            << std::endl;
                
                            fTrainingDataAlg.setEventData(
                              event, clockData, detProp, currentPlane, currentTPC, currentCryo);

                              unsigned int w0, w1, d0, d1;
                                        if (fCrop && saveSim) {
                                          if (fTrainingDataAlg.findCrop(0.004F, w0, w1, d0, d1)) {
                                            std::cout << "   crop: " << w0 << " " << w1 << " " << d0 << " " << d1 << std::endl;
                                          }
            else {
              std::cout << "   skip empty tpc:" << currentTPC << " / view:" << currentPlane
                        << std::endl;
              continue;
            }
          }
          else {
            w0 = 0;
            w1 = fTrainingDataAlg.NWires();
            d0 = 0;
            d1 = fTrainingDataAlg.NScaledDrifts();
          }

          // ------------------------------------------------------------------
          // ORIGINAL ROOT DUMP PATH: KEPT INTACT
          // ------------------------------------------------------------------
          if (fDumpToRoot) {
            std::ostringstream ss1;
            ss1 << "raw_" << os.str() << "_tpc_" << currentTPC << "_view_" << currentPlane;

            art::ServiceHandle<art::TFileService const> tfs;
            TH2F* rawHist =
              tfs->make<TH2F>((ss1.str() + "_raw").c_str(), "ADC", w1 - w0, w0, w1, d1 - d0, d0, d1);
            TH2F* depHist = nullptr;
            TH2I* pdgHist = nullptr;

            if (saveSim) {
              depHist = tfs->make<TH2F>((ss1.str() + "_deposit").c_str(),
                                      "Deposit",
                                      w1 - w0,
                                      w0,
                                      w1,
                                      d1 - d0,
                                      d0,
                                      d1);
              pdgHist = tfs->make<TH2I>(
                (ss1.str() + "_pdg").c_str(), "PDG", w1 - w0, w0, w1, d1 - d0, d0, d1);
            }

            for (size_t w = w0; w < w1; ++w) {
              auto const& raw = fTrainingDataAlg.wireData(w);
              for (size_t d = d0; d < d1; ++d) {
                rawHist->Fill(w, d, raw[d]);
              }

              if (saveSim) {
                auto const& edep = fTrainingDataAlg.wireEdep(w);
                for (size_t d = d0; d < d1; ++d) {
                  depHist->Fill(w, d, edep[d]);
                }

                auto const& pdg = fTrainingDataAlg.wirePdg(w);
                for (size_t d = d0; d < d1; ++d) {
                  pdgHist->Fill(w, d, pdg[d]);
                }
              }
            }

            writeAndDelete(rawHist);
            writeAndDelete(depHist);
            writeAndDelete(pdgHist);
          }

          // ------------------------------------------------------------------
          // ORIGINAL NUMPY DUMP PATH: KEPT INTACT
          // ------------------------------------------------------------------
          else if (fDumpToNumpy) {
            for (size_t w = w0; w < w1; ++w) {
              int w_start = static_cast<int>(w) - fPatch_size_w / 2;
              int w_stop = w_start + fPatch_size_w;
              if (w_start < static_cast<int>(w0) || w_start > static_cast<int>(w1)) continue;
              if (w_stop < static_cast<int>(w0) || w_stop > static_cast<int>(w1)) continue;

              auto const& pdg = fTrainingDataAlg.wirePdg(w);
              auto const& deposit = fTrainingDataAlg.wireEdep(w);
              auto const& raw = fTrainingDataAlg.wireData(w);

              for (size_t d = d0; d < d1; ++d) {
                int d_start = static_cast<int>(d) - fPatch_size_d / 2;
                int d_stop = d_start + fPatch_size_d;
                if (d_start < static_cast<int>(d0) || d_start > static_cast<int>(d1)) continue;
                if (d_stop < static_cast<int>(d0) || d_stop > static_cast<int>(d1)) continue;

                int y0 = 0, y1 = 0, y2 = 0, y3 = 0;

                if (deposit[d] < 2e-5 || raw[d] < 0.05) { // empty pixel
                  y3 = 1;
                  ++nNone;
                  if (flat.fire() > fNone) continue;
                  ++nNone_sel;
                }
                else if ((pdg[d] & 0x0FFF) == 11) { // shower
                  y1 = 1;
                  ++nEm;
                  if ((pdg[d] & 0xF000) == 0x2000) { // Michel
                    y2 = 1;
                    ++nMichel;
                    if (flat.fire() > fMichel) continue;
                    ++nMichel_sel;
                  }
                  else {
                    if (flat.fire() > fEm) continue;
                    ++nEm_sel;
                  }
                }
                else { // track
                  y0 = 1;
                  ++nTrk;

                  int nPxlw0 = 0, nPxlw1 = 0, nPxld0 = 0, nPxld1 = 0;
                  std::vector<double> wfit, dfit, chgfit;
                  double total_trkchg = 0.0;

                  for (int ww = w_start; ww < w_stop; ++ww) {
                    auto const& pdg1 = fTrainingDataAlg.wirePdg(ww);
                    auto const& deposit1 = fTrainingDataAlg.wireEdep(ww);
                    auto const& raw1 = fTrainingDataAlg.wireData(ww);

                    for (int dd = d_start; dd < d_stop; ++dd) {
                      if (deposit1[dd] < 2e-5 || raw1[dd] < 0.05) continue;

                      if ((pdg1[dd] & 0x0FFF) != 11) {
                        wfit.push_back(ww);
                        dfit.push_back(dd);
                        chgfit.push_back(raw1[dd]);
                        total_trkchg += raw1[dd];

                        if (ww - w_start < 3) ++nPxlw0;
                        if (w_stop - ww - 1 < 3) ++nPxlw1;
                        if (dd - d_start < 3) ++nPxld0;
                        if (d_stop - dd - 1 < 3) ++nPxld1;
                      }
                    }
                  }

                  double fit_trkchg = 0.0;
                  double parm[2] = {};

                  if (!wfit.empty()) {
                    if (!WeightedFit(static_cast<int>(wfit.size()), wfit, dfit, chgfit, &parm[0])) {
                      for (size_t j = 0; j < wfit.size(); ++j) {
                        if (std::abs((dfit[j] - (parm[0] + wfit[j] * parm[1])) * cos(atan(parm[1]))) <
                            3) {
                          fit_trkchg += chgfit[j];
                        }
                      }
                    }
                    else if (!WeightedFit(
                               static_cast<int>(dfit.size()), dfit, wfit, chgfit, &parm[0])) {
                      for (size_t j = 0; j < dfit.size(); ++j) {
                        if (std::abs((wfit[j] - (parm[0] + dfit[j] * parm[1])) * cos(atan(parm[1]))) <
                            3) {
                          fit_trkchg += chgfit[j];
                          }
                        }
                      }
                    }
                  }

                  if (((nPxlw0) && (!nPxlw1) && (!nPxld0) && (!nPxld1)) ||
                      ((!nPxlw0) && (nPxlw1) && (!nPxld0) && (!nPxld1)) ||
                      ((!nPxlw0) && (!nPxlw1) && (nPxld0) && (!nPxld1)) ||
                      ((!nPxlw0) && (!nPxlw1) && (!nPxld0) && (nPxld1))) {
                    if (flat.fire() > fStopTrk) continue;
                    ++nStopTrk_sel;
                  }
                  else if (total_trkchg && fit_trkchg / total_trkchg > 0.9) {
                    if (flat.fire() > fCleanTrk) continue;
                    ++nCleanTrk_sel;
                  }
                  else {
                    if (flat.fire() > fTrk) continue;
                    ++nTrk_sel;
                  }
                }

                c2numpy_uint32(&npywriter, event.id().run());
                c2numpy_uint32(&npywriter, event.id().subRun());
                c2numpy_uint32(&npywriter, event.id().event());
                c2numpy_uint8(&npywriter, currentTPC);
                c2numpy_uint8(&npywriter, currentPlane);
                c2numpy_uint16(&npywriter, w);
                c2numpy_uint16(&npywriter, d);
                c2numpy_uint8(&npywriter, y0);
                c2numpy_uint8(&npywriter, y1);
                c2numpy_uint8(&npywriter, y2);
                c2numpy_uint8(&npywriter, y3);

                for (int ww = w_start; ww < w_stop; ++ww) {
                  auto const& raw1 = fTrainingDataAlg.wireData(ww);
                  for (int dd = d_start; dd < d_stop; ++dd) {
                    c2numpy_float32(&npywriter, raw1[dd]);
                  }
                }
              }
            }
          }

          // ------------------------------------------------------------------
          // ORIGINAL TEXT DUMP PATH: KEPT INTACT
          // ------------------------------------------------------------------
          else {
            std::ostringstream ss1;
            ss1 << fOutTextFilePath << "/raw_" << os.str() << "_tpc_" << currentTPC << "_view_"
                << currentPlane;

            std::ofstream fout_raw, fout_deposit, fout_pdg;

            fout_raw.open(ss1.str() + ".raw");
            if (saveSim) {
              fout_deposit.open(ss1.str() + ".deposit");
              fout_pdg.open(ss1.str() + ".pdg");
            }

            for (size_t w = w0; w < w1; ++w) {
              auto const& raw = fTrainingDataAlg.wireData(w);
              for (size_t d = d0; d < d1; ++d) {
                fout_raw << raw[d] << " ";
              }
              fout_raw << std::endl;

              if (saveSim) {
                auto const& edep = fTrainingDataAlg.wireEdep(w);
                for (size_t d = d0; d < d1; ++d) {
                  fout_deposit << edep[d] << " ";
                }
                fout_deposit << std::endl;

                auto const& pdg = fTrainingDataAlg.wirePdg(w);
                for (size_t d = d0; d < d1; ++d) {
                  fout_pdg << pdg[d] << " ";
                }
                fout_pdg << std::endl;
              }
            }

            fout_raw.close();
            if (saveSim) {
              fout_deposit.close();
              fout_pdg.close();
            }
          }

          // ------------------------------------------------------------------
          // NEW ADDITIVE PANDORA ENDPOINT TREE
          // ------------------------------------------------------------------
          if (!fSaveEndpointTree) continue;

          for (size_t iTrack = 0; iTrack < trackHandle->size(); ++iTrack) {
            auto const& trk = trackHandle->at(iTrack);
            if (trk.Length() < fMinTrackLength) continue;

            auto const hits = trackHitAssoc.at(iTrack);
            DominantTPCInfo const domTPC = dominantTPC(hits);

            if (!domTPC.valid) continue;
            if (domTPC.cryo != currentCryo) continue;
            if (domTPC.tpc != currentTPC) continue;

            TVector3 const start = trk.Vertex<TVector3>();
            TVector3 const end = trk.End<TVector3>();

            b_startXYZ[0] = start.X();
            b_startXYZ[1] = start.Y();
            b_startXYZ[2] = start.Z();

            b_endXYZ[0] = end.X();
            b_endXYZ[1] = end.Y();
            b_endXYZ[2] = end.Z();

            int const endpointFirst = fUseBothEndpoints ? 0 : 1;
            int const endpointLast = 1; // always include the end point; start is optional

            for (int endpoint = endpointFirst; endpoint <= endpointLast; ++endpoint) {
              float xyz[3] = {0.F, 0.F, 0.F};
              float wireAbs = 0.F;
              float tickAbs = 0.F;

              if (!projectEndpoint(
                    trk, endpoint, currentCryo, currentTPC, currentPlane, detProp, wireReadoutGeom, wireAbs, tickAbs, xyz)) {
                continue;
              }

              int nMichelPix = 0;
              int nMuonPix = 0;
              int nTrackPix = 0;
              int nShowerPix = 0;

              if (saveSim) {
                evaluateEndpointTruth(wireAbs, tickAbs, nMichelPix, nMuonPix, nTrackPix, nShowerPix);
              }

              int const localWire = static_cast<int>(std::lround(wireAbs)) - static_cast<int>(w0);
              int const localTick = static_cast<int>(std::lround(tickAbs)) - static_cast<int>(d0);
              int const halfW = fPatch_size_w / 2;
              int const halfD = fPatch_size_d / 2;

              bool const patchFits = (localWire - halfW >= 0) &&
                                   (localWire + (fPatch_size_w - halfW) <= static_cast<int>(w1 - w0)) &&
                                   (localTick - halfD >= 0) &&
                                   (localTick + (fPatch_size_d - halfD) <= static_cast<int>(d1 - d0));

              b_run = runNumber;
              b_subrun = subRunNumber;
              b_event = eventNumber;
              b_trackKey = static_cast<int>(iTrack);
              b_endpoint = endpoint;
              b_cryo = currentCryo;
              b_tpc = currentTPC;
              b_plane = currentPlane;
              b_trackLength = trk.Length();

              b_centerWireAbs = wireAbs;
              b_centerTickAbs = tickAbs;

              b_centerWireLocal = wireAbs - static_cast<float>(w0);
              b_centerTickLocal = tickAbs - static_cast<float>(d0);

              b_histWireOffset = static_cast<int>(w0);
              b_histTickOffset = static_cast<int>(d0);
              b_histNWires = static_cast<int>(w1 - w0);
              b_histNDrifts = static_cast<int>(d1 - d0);

              b_patchFits = patchFits ? 1 : 0;

              b_truthMichelPixels = nMichelPix;
              b_truthMuonPixels = nMuonPix;
              b_truthTrackPixels = nTrackPix;
              b_truthShowerPixels = nShowerPix;

              b_truthHasMichel =
                (saveSim && nMichelPix >= fTruthMichelMinPixels && nMuonPix >= fTruthMuonMinPixels)
                  ? 1
                  : 0;

              b_nHits = static_cast<int>(hits.size());

              fEndpointTree->Fill();
            }
          }

        } // plane loop
      }   // tpc loop
    }   // cryo loop
  }     // analyze()

  //-----------------------------------------------------------------------
  int PointIdPandoraEndpointTrainingData::WeightedFit(int n,
                                                      std::vector<double> const& x,
                                                      std::vector<double> const& y,
                                                      std::vector<double> const& w,
                                                      double* params) const
  {
    Double_t sumx = 0.;
    Double_t sumx2 = 0.;
    Double_t sumy = 0.;
    Double_t sumy2 = 0.;
    Double_t sumxy = 0.;
    Double_t sumw = 0.;
    Double_t eparams[2] = {};

    for (Int_t i = 0; i < n; i++) {
      sumx += x[i] * w[i];
      sumx2 += x[i] * x[i] * w[i];
      sumy += y[i] * w[i];
      sumy2 += y[i] * y[i] * w[i];
      sumxy += x[i] * y[i] * w[i];
      sumw += w[i];
    }

    if (sumx2 * sumw - sumx * sumx == 0.) return 1;
    if (sumx2 - sumx * sumx / sumw == 0.) return 1;

    params[0] = (sumy * sumx2 - sumx * sumxy) / (sumx2 * sumw - sumx * sumx);
    params[1] = (sumxy - sumx * sumy / sumw) / (sumx2 - sumx * sumx / sumw);

    eparams[0] = sumx2 * (sumx2 * sumw - sumx * sumx);
    eparams[1] = (sumx2 - sumx * sumx / sumw);

    if (eparams[0] < 0. || eparams[1] < 0.) return 1;

    eparams[0] = sqrt(eparams[0]) / (sumx2 * sumw - sumx * sumx);
    eparams[1] = sqrt(eparams[1]) / (sumx2 - sumx * sumx / sumw);

    return 0;
  }

  DEFINE_ART_MODULE(PointIdPandoraEndpointTrainingData)

} // namespace nnet
