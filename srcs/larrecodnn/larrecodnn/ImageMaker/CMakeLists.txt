# at this time, hep_hpc does not define target libraries
include_directories($ENV{HEP_HPC_INC})

cet_build_plugin(SaveImageH5 art::EDAnalyzer
  LIBRARIES PRIVATE
  ${HEP_HPC_HDF5}
  HDF5::HDF5
)

cet_build_plugin(SavePiMu art::tool
  LIBRARIES PRIVATE
  art::Framework_Core
  larcore::Geometry_Geometry_service
  larcorealg::Geometry
  lardataobj::RecoBase
  nusimdata::SimulationBase
  ${HEP_HPC_HDF5}
  HDF5::HDF5
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12.5")
  # GCC 12.5 (and above) aggressively/spuriously warns about array-bounds issues.
  target_compile_options(larrecodnn_ImageMaker_SavePiMu_tool
                         PRIVATE "-Wno-array-bounds;-Wno-stringop-overread")
endif()

install_headers()
install_fhicl()
install_source()
