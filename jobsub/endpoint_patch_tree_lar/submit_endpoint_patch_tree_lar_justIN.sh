#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Dedicated submitter for CNNimage endpoint-patch-tree JustIN workflow.
# It selects atmospheric reco2.root files and runs PointIdTrackEndpointPatchTree.fcl.
# This produces direct endpoint ROI patch TTrees, avoiding the Python crop stage.

source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup python v3_9_15
setup rucio
setup justin

export RUCIO_ACCOUNT=justinreadonly
setup metacat
export METACAT_SERVER_URL=https://metacat.fnal.gov:9443/dune_meta_prod/app
export METACAT_AUTH_SERVER_URL=https://metacat.fnal.gov:8143/auth/dune

justin get-token || exit 1

TARBALL=${TARBALL:-larsoft_endpoint_patch_tree_lar.tar.gz}
INPUT_TAR_DIR_LOCAL=`justin-cvmfs-upload "$TARBALL"` || exit 1
USERF=$USER
DIR=${OUTPUT_SUBDIR:-CNNimage/jobsub/endpoint_patch_tree_lar}
FNALURL='https://fndcadoor.fnal.gov:2880/dune/scratch/users'
DATASET=${DATASET:-"fardet-hd:fardet-hd__full-reconstructed__v09_85_00d00__reco2_atmos_dune10kt_1x2x6_geov5__prodgenie_atmnu_max_weighted_randompolicy_dune10kt_1x2x6__out1__v1_official"}

NUM_FILES=${NUM_FILES:-8000}
SKIP_FILES=${SKIP_FILES:-0}
WORKFLOW_NAME=${WORKFLOW_NAME:-cnnimage-endpoint-patch-tree-lar-${NUM_FILES}}

if [ "$SKIP_FILES" = "0" ] ; then
  MQL_QUERY="files from $DATASET limit $NUM_FILES ordered"
else
  MQL_QUERY="files from $DATASET skip $SKIP_FILES limit $NUM_FILES ordered"
fi

echo "[submit_endpoint_patch_tree_lar] Input tar directory: $INPUT_TAR_DIR_LOCAL"
echo "[submit_endpoint_patch_tree_lar] Workflow name: $WORKFLOW_NAME"
echo "[submit_endpoint_patch_tree_lar] Input stage: atmospheric reco2.root files selected by MetaCat/JustIN"
echo "[submit_endpoint_patch_tree_lar] Analyzer/FCL: PointIdPandoraEndpointPatchTree via PointIdTrackEndpointPatchTree.fcl"
echo "[submit_endpoint_patch_tree_lar] Expected output per job: endpoint_patch_tree_<input_reco2_basename>.root"
echo "[submit_endpoint_patch_tree_lar] Scratch output destination: $FNALURL/$USERF/$DIR"
echo "[submit_endpoint_patch_tree_lar] MQL query: $MQL_QUERY"

justin simple-workflow \
  --name "$WORKFLOW_NAME" \
  --mql "$MQL_QUERY" \
  --jobscript jobscript_endpoint_patch_tree_lar.jobscript \
  --env INPUT_TAR_DIR_LOCAL="$INPUT_TAR_DIR_LOCAL" \
  --rss-mb 2000 \
  --wall-seconds 7200 \
  --max-distance 30 \
  --output-pattern "*endpoint_patch_tree*.root:$FNALURL/$USERF/$DIR" \
  --output-pattern "*.logs.tgz:$FNALURL/$USERF/$DIR"
