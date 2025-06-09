#!/usr/bin/env bash
set -euo pipefail

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────

# Local source directory (trailing slash means "contents of")
SOURCE="."

# Remote destination. Format: user@host:/path/to/remote/folder/
DEST="n.fahrni@slurmlogin.cs.technik.fhnw.ch:/home2/n.fahrni/sequoia/"

# Exclude patterns: add paths relative to SOURCE here.
# For example: "cache/", "tmp/", "node_modules/"
EXCLUDES=(
  "data/raw/"
  "data/processed/meshes/"
  "Hunyuan3D-2/"
  "InstantMesh/"
  "notebooks/"
  "Pixel_Nerf/"
  "Pixel2Mesh/"
  ".git/"
  ".venv/"
)

# ─── BUILD RSYNC ARGS ────────────────────────────────────────────────────────────

# Basic options:
#   -a : archive mode (recurses, preserves perms/times/etc.)
#   -v : verbose
#   -z : compress during transfer
#   -h : human-readable numbers
#   -P : show progress & keep partials
RSYNC_OPTS=(-avvzhP)

# Add each exclude pattern
for pat in "${EXCLUDES[@]}"; do
  RSYNC_OPTS+=(--exclude="${pat}")
done

# ─── RUN RSYNC ──────────────────────────────────────────────────────────────────

echo "Starting rsync from ${SOURCE} to ${DEST}…"
rsync "${RSYNC_OPTS[@]}" "${SOURCE}" "${DEST}"

echo "Done!"