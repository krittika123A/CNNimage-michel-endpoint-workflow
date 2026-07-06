cet_enable_asserts()

cet_make_library(LIBRARY_NAME NuGraphBaseTools
                 INTERFACE SOURCE LoaderToolBase.h DecoderToolBase.h
                 LIBRARIES INTERFACE
                   lardataobj::RecoBase
                   art::Framework_Principal
                   art::Framework_Core
                   fhiclcpp::fhiclcpp
                   art_plugin_types::tool
                )

set( nugraph_tool_lib_list
       lardataobj::RecoBase
       lardataobj::AnalysisBase
       larrecodnn_ImagePatternAlgs_NuSonic_Triton
       TorchScatter::TorchScatter
       art::Framework_Core
       hep_concurrency::hep_concurrency
       art_plugin_support::toolMaker
)

cet_build_plugin(FilterDecoder art::tool
  LIBRARIES PRIVATE
  ${nugraph_tool_lib_list}
)

cet_build_plugin(SemanticDecoder art::tool
  LIBRARIES PRIVATE
  ${nugraph_tool_lib_list}
)

cet_build_plugin(StandardLoader art::tool
  LIBRARIES PRIVATE
  ${nugraph_tool_lib_list}
)

cet_build_plugin(VertexDecoder art::tool
  LIBRARIES PRIVATE
  ${nugraph_tool_lib_list}
)

install_headers()
install_fhicl()
install_source()
