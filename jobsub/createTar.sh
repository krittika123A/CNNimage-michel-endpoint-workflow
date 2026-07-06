#!/bin/bash

echo "Making tar... bear with... bear with... bear with..."

#tar czf larsoft_shower2.tar.gz -C /exp/dune/app/users/imawby/dunesw_validationRelease/ localProducts_larsoft_v10_20_05_e26_prof -C /exp/dune/app/users/imawby/dunesw_validationRelease/models matching_net_inter_cluster_sim.pt  matching_net_inter_cluster_attn.pt matching_net_intra_cluster_encoder.pt PandoraNet_ShowerGrowing_DUNEFD_HD_Encoder_v05_00_00.pt PandoraNet_ShowerGrowing_DUNEFD_HD_Attn_v05_00_00.pt PandoraNet_ShowerGrowing_DUNEFD_HD_Sim_v05_00_00.pt -C /exp/dune/app/users/imawby/dunesw_validationRelease/POMS/ setup-grid PandoraSettings_Master_DUNEFD_Shower.xml PandoraSettings_Neutrino_DUNEFD_Shower.xml pandora_shower.fcl -C /exp/dune/app/users/imawby/dunesw_pandoraBugFix/models PandoraNet_Vertex_DUNEFD_HD_Accel_1_U_v04_06_00.pt PandoraNet_Vertex_DUNEFD_HD_Accel_1_V_v04_06_00.pt PandoraNet_Vertex_DUNEFD_HD_Accel_1_W_v04_06_00.pt PandoraNet_Vertex_DUNEFD_HD_Accel_2_U_v04_06_00.pt PandoraNet_Vertex_DUNEFD_HD_Accel_2_V_v04_06_00.pt PandoraNet_Vertex_DUNEFD_HD_Accel_2_W_v04_06_00.pt PandoraBdt_PfoCharacterisation_DUNEFD_HD_v04_06_00.xml PandoraNet_Hierarchy_DUNEFD_HD_T_Edge_v014_15_00.pt PandoraNet_Hierarchy_DUNEFD_HD_T_Class_v014_15_00.pt PandoraNet_Hierarchy_DUNEFD_HD_TT_Edge_v014_15_00.pt PandoraNet_Hierarchy_DUNEFD_HD_TT_Class_v014_15_00.pt PandoraNet_Hierarchy_DUNEFD_HD_TS_Edge_v014_15_00.pt PandoraNet_Hierarchy_DUNEFD_HD_TS_Class_v014_15_00.pt PandoraNet_Hierarchy_DUNEFD_HD_S_Class_v014_15_00.pt

tar czf larsoft.tar.gz \
  -C /exp/dune/app/users/kadhikar/CNNimage localProducts_larsoft_v10_20_03_01_prof_e26 PointIdTrackEndpointDump.fcl \
  -C /exp/dune/app/users/kadhikar/CNNimage/jobsub setup-grid

#mv larsoft.tar.gz /exp/dune/app/users/kadhikar/CNNimage/jobsub/

echo "BACK!"
