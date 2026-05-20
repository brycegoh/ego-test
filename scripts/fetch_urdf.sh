#!/usr/bin/env bash
# Fetch the Mobile ALOHA URDF + meshes from AgileX's simulation repo into assets/urdf/.
#
#   https://github.com/agilexrobotics/mobile_aloha_sim
#
# Mobile ALOHA is bimanual (two ViperX-class 6-DOF arms with parallel-jaw grippers) on an
# AgileX Tracer mobile base. This repo ships a *flat* URDF (no xacro/ROS expansion needed):
#
#   aloha_description/aloha/urdf/aloha.urdf      # robot model
#   aloha_description/aloha/meshes/*.STL         # meshes (referenced as package://aloha/...)
#
# The render stage loads `assets/urdf/aloha/urdf/aloha.urdf` via rerun's built-in URDF
# loader (rerun-sdk >= 0.29). NOTE: the URDF references meshes with `package://aloha/...`;
# we lay the files out so the package root `aloha/` sits next to the urdf. If rerun cannot
# resolve `package://`, rewrite those refs to relative paths (sed) or point the loader at
# this package root.
#
# Requires network access to github.com (may be blocked in sandboxed environments).
# Usage: bash scripts/fetch_urdf.sh
set -euo pipefail

DEST="assets/urdf"
REPO="https://github.com/agilexrobotics/mobile_aloha_sim.git"
mkdir -p "${DEST}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# Sparse-checkout only the ALOHA description (arms + grippers). The repo also has
# tracer2_description (mobile base) and realsense2_description (camera) if needed later.
git clone --depth 1 --filter=blob:none --sparse "${REPO}" "${TMP}/mobile_aloha_sim"
git -C "${TMP}/mobile_aloha_sim" sparse-checkout set aloha_description/aloha

# Copy the `aloha` ROS package so `package://aloha/...` resolves to assets/urdf/aloha/...
cp -r "${TMP}/mobile_aloha_sim/aloha_description/aloha" "${DEST}/"

URDF="${DEST}/aloha/urdf/aloha.urdf"
if [[ -f "${URDF}" ]]; then
  echo "Mobile ALOHA URDF ready at ${URDF}"
else
  echo "Expected ${URDF} not found; inspect ${DEST}/aloha and update render.DEFAULT_URDF." >&2
  exit 1
fi
