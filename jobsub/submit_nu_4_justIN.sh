#!/bin/bash

# Setup DUNE
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup python v3_9_15 

setup rucio

# Setup justin
setup justin

export RUCIO_ACCOUNT=justinreadonly
setup metacat
export METACAT_SERVER_URL=https://metacat.fnal.gov:9443/dune_meta_prod/app
export METACAT_AUTH_SERVER_URL=https://metacat.fnal.gov:8143/auth/dune

# Get a token
justin get-token

# Job config
INPUT_TAR_DIR_LOCAL=`justin-cvmfs-upload larsoft.tar.gz`
USERF=$USER
DIR='CNNimage/jobsub'
FNALURL='https://fndcadoor.fnal.gov:2880/dune/scratch/users'  # output location for files
DATASET="fardet-hd:fardet-hd__full-reconstructed__v09_85_00d00__reco2_atmos_dune10kt_1x2x6_geov5__prodgenie_atmnu_max_weighted_randompolicy_dune10kt_1x2x6__out1__v1_official"  # Rucio/MetaCat dataset
MQL_QUERY="files from $DATASET limit 20 ordered" 
#MQL_QUERY="files from $DATASET skip 4000 limit 1000 ordered"  # specific criteria

echo 'Input tar directory: '
echo $INPUT_TAR_DIR_LOCAL

# Now do justin submit
# Codex change 2026-07-03: long-statistics run over the selected 20 files.
# NUM_EVENTS is intentionally not passed below, so jobscript.jobscript does not add lar -n.
# This makes lar process all available events in each selected atmospheric file.
# Previous short-test option, intentionally disabled now:
#   --env NUM_EVENTS=100 \
#justin simple-workflow \
#   --mql "$MQL_QUERY" \
#   --jobscript jobscript.jobscript \
#--env INPUT_TAR_DIR_LOCAL="$INPUT_TAR_DIR_LOCAL" \
#   --env NUM_EVENTS=100 \
#   --rss-mb 2000 \
#   --wall-seconds 7200 \
#   --max-distance 30 \
#--output-pattern "*ShowerMatchingTraining*.root:$FNALURL/$USERF/$DIR" \
#    --output-pattern "*Validation*.root:$FNALURL/$USERF/$DIR"

justin simple-workflow \
    --mql "$MQL_QUERY" \
    --jobscript jobscript.jobscript \
    --env INPUT_TAR_DIR_LOCAL="$INPUT_TAR_DIR_LOCAL" \
    --rss-mb 2000 \
    --wall-seconds 7200 \
    --max-distance 30 \
    --output-pattern "*reco_hist*.root:$FNALURL/$USERF/$DIR"

