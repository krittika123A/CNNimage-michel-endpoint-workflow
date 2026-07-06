add_subdirectory(Tools)

cet_set_compiler_flags(DIAGS VIGILANT DWARF_VER 4
  WERROR
  NO_UNDEFINED
  EXTRA_FLAGS -pedantic)

cet_build_plugin(NuGraphInference art::EDProducer
    LIBRARIES PRIVATE
    lardataobj::RecoBase
    TorchScatter::TorchScatter
    IMPL_TARGET_VAR NuGraphInference_module
)

find_path(DELAUNATOR_INC delaunator-header-only.hpp HINTS ENV DELAUNATOR_INC REQUIRED DOC "Header location for delaunator_cpp")
target_include_directories(${NuGraphInference_module} PRIVATE ${DELAUNATOR_INC})

cet_build_plugin(NuSliceHitsProducer art::EDProducer
    LIBRARIES PRIVATE
    nusimdata::SimulationBase
    lardataobj::RecoBase
)

cet_build_plugin(NuGraphAnalyzer art::EDAnalyzer
    LIBRARIES PRIVATE
    art_root_io::TFileService_service
    ROOT::Tree
    lardata::RecoBaseProxy
)

cet_build_plugin(NuGraphInferenceSonicTriton art::EDProducer
    LIBRARIES PRIVATE
    lardataobj::RecoBase
    larrecodnn_ImagePatternAlgs_NuSonic_Triton
    TorchScatter::TorchScatter
    IMPL_TARGET_VAR NuGraphInferenceSonicTriton_module
)

cet_build_plugin(NuGraphInferenceTriton art::EDProducer
    LIBRARIES PRIVATE
    lardataobj::RecoBase
    TorchScatter::TorchScatter
    TritonClient::grpcclient
    gRPC::grpc # Should be a transitive (INTERFACE) dependency of TritonClient::grpcclient
    IMPL_TARGET_VAR NuGraphInferenceTriton_module
)

install_headers()
install_fhicl()
install_source()
