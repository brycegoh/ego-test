#!/usr/bin/env bash
# Fetch the SO-101 URDF + meshes from TheRobotStudio/SO-ARM100 into assets/urdf/.
#
# The render stage (src/egodex_robot/stages/render.py) loads this URDF into rerun.
# Requires network access to github.com (may be blocked in sandboxed environments).
#
# Usage: bash scripts/fetch_urdf.sh
set -euo pipefail

DEST="assets/urdf"
REPO="https://github.com/TheRobotStudio/SO-ARM100.git"
mkdir -p "${DEST}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# Sparse-checkout just the SO101 simulation assets to avoid cloning the whole repo.
git clone --depth 1 --filter=blob:none --sparse "${REPO}" "${TMP}/SO-ARM100"
git -C "${TMP}/SO-ARM100" sparse-checkout set Simulation/SO101

cp -r "${TMP}/SO-ARM100/Simulation/SO101/." "${DEST}/"

echo "SO-101 URDF assets copied to ${DEST}/ (expecting so101_new_calib.urdf)."
