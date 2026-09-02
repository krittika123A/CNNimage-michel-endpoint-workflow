cet_build_plugin(EmTrackClusterId2out art::EDProducer
  LIBRARIES PRIVATE
  larrecodnn::ImagePatternAlgs_Tensorflow_PointIdAlg
  lardata::ArtDataHelper
  lardata::DetectorClocksService
  lardata::DetectorPropertiesService
  lardata::AssociationUtil
  lardataobj::RecoBase
  larcoreobj::SimpleTypesAndConstants
  art::Framework_Services_System_TriggerNamesService_service
  art::Framework_Services_Registry
  canvas::canvas
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
  cetlib_except::cetlib_except
)

cet_build_plugin(EmTrackClusterId3out art::EDProducer
  LIBRARIES PRIVATE
  larrecodnn::ImagePatternAlgs_Tensorflow_PointIdAlg
  lardata::ArtDataHelper
  lardata::DetectorClocksService
  lardata::DetectorPropertiesService
  lardata::AssociationUtil
  lardataobj::RecoBase
  larcoreobj::SimpleTypesAndConstants
  art::Framework_Services_System_TriggerNamesService_service
  art::Framework_Services_Registry
  canvas::canvas
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
  cetlib_except::cetlib_except
)

cet_build_plugin(EmTrackMichelId art::EDProducer
  LIBRARIES PRIVATE
  larrecodnn::ImagePatternAlgs_Tensorflow_PointIdAlg
  lardata::ArtDataHelper
  lardata::DetectorClocksService
  lardata::DetectorPropertiesService
  lardata::AssociationUtil
  lardataobj::RecoBase
  larcoreobj::SimpleTypesAndConstants
  art::Framework_Services_System_TriggerNamesService_service
  art::Framework_Services_Registry
  canvas::canvas
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
  cetlib_except::cetlib_except
)

cet_build_plugin(ParticleDecayId art::EDProducer
  LIBRARIES PRIVATE
  larrecodnn::ImagePatternAlgs_Tensorflow_PointIdAlg
  lardata::DetectorClocksService
  lardata::DetectorPropertiesService
  lardata::AssociationUtil
  lardataobj::RecoBase
  art::Framework_Services_Registry
  canvas::canvas
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
)

cet_build_plugin(PointIdTrainingData art::EDAnalyzer
  LIBRARIES PRIVATE
  larrecodnn::ImagePatternAlgs_Tensorflow_PointIdAlg
  lardata::DetectorClocksService
  lardata::DetectorPropertiesService
  larcore::Geometry_Geometry_service
  larcorealg::Geometry
  nurandom::RandomUtils_NuRandomService_service
  art_root_io::TFileService_service
  art::Framework_Services_Registry
  canvas::canvas
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
  CLHEP::Random
  ROOT::Hist
)

cet_build_plugin(PointIdPandoraEndpointTrainingData art::EDAnalyzer
  LIBRARIES PRIVATE
  larrecodnn::ImagePatternAlgs_Tensorflow_PointIdAlg
  lardata::DetectorClocksService
  lardata::DetectorPropertiesService
  larcore::Geometry_Geometry_service
  larcorealg::Geometry
  nurandom::RandomUtils_NuRandomService_service
  art_root_io::TFileService_service
  art::Framework_Services_Registry
  canvas::canvas
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
  CLHEP::Random
  ROOT::Hist
  ROOT::Graf
  ROOT::Tree
)

cet_build_plugin(PointIdPandoraEndpointTrainingDataTruthOverlap art::EDAnalyzer
  LIBRARIES PRIVATE
  larrecodnn::ImagePatternAlgs_Tensorflow_PointIdAlg
  lardata::DetectorClocksService
  lardata::DetectorPropertiesService
  larcore::Geometry_Geometry_service
  larcorealg::Geometry
  nurandom::RandomUtils_NuRandomService_service
  art_root_io::TFileService_service
  art::Framework_Services_Registry
  canvas::canvas
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
  CLHEP::Random
  ROOT::Hist
  ROOT::Graf
  ROOT::Tree
)

cet_build_plugin(PointIdPandoraEndpointPatchTree art::EDAnalyzer
  LIBRARIES PRIVATE
  larrecodnn::ImagePatternAlgs_Tensorflow_PointIdAlg
  lardata::DetectorClocksService
  lardata::DetectorPropertiesService
  larcore::Geometry_Geometry_service
  larcorealg::Geometry
  nurandom::RandomUtils_NuRandomService_service
  art_root_io::TFileService_service
  art::Framework_Services_Registry
  canvas::canvas
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
  CLHEP::Random
  ROOT::Hist
  ROOT::Tree
)

cet_build_plugin(PointIdPandoraEndpointHitPatchTree art::EDAnalyzer
  LIBRARIES PRIVATE
  larrecodnn::ImagePatternAlgs_Tensorflow_PointIdAlg
  lardata::DetectorClocksService
  lardata::DetectorPropertiesService
  larcore::Geometry_Geometry_service
  larcorealg::Geometry
  nurandom::RandomUtils_NuRandomService_service
  art_root_io::TFileService_service
  art::Framework_Services_Registry
  canvas::canvas
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
  CLHEP::Random
  ROOT::Hist
  ROOT::Tree
)

install_headers()
install_fhicl()
install_source()
