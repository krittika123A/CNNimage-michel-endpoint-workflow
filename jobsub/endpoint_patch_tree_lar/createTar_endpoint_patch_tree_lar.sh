#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Build the payload for CNNimage endpoint-patch-tree JustIN workflow.
# The tarball carries the built localProducts area, the patch-tree FCL, and setup-grid.

set -e

echo "[createTar_endpoint_patch_tree_lar] Making tarball for direct endpoint patch-tree analyzer workflow"
echo "[createTar_endpoint_patch_tree_lar] Payload includes localProducts, PointIdTrackEndpointPatchTree.fcl, and setup-grid"

tar czf larsoft_endpoint_patch_tree_lar.tar.gz \
  -C /exp/dune/app/users/kadhikar/CNNimage localProducts_larsoft_v10_20_03_01_prof_e26 PointIdTrackEndpointPatchTree.fcl \
  -C /exp/dune/app/users/kadhikar/CNNimage/jobsub setup-grid

rc=$?
echo "[createTar_endpoint_patch_tree_lar] tar exit code: $rc"
exit $rc
