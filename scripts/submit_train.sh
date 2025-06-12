#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
  if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [time] [-- hydra overrides…]"
    exit 1
  fi

  EXP_NAME="$1"; shift
  TIME="${1:-04:00:00}"; shift

  sbatch \
    --job-name="${EXP_NAME}" \
    --time="${TIME}" \
    --output="logs/train_%j.out" \
    --error="logs/train_%j.err" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=16G \
    --partition=performance \
    "$0" "$EXP_NAME" "$@" \
  && echo "Submitted as job \"$EXP_NAME\" with time=${TIME}"
  exit
fi

SIF_PATH="./singularity/pix2vox.sif"

BIND_OPTS="--bind ${SLURM_SUBMIT_DIR}:/workspace"

EXP_NAME="$1"; shift
OVERRIDES="$@"

cd "${SLURM_SUBMIT_DIR}"
singularity exec --nv \
  ${BIND_OPTS} \
  ${SIF_PATH} \
  uv run python /workspace/core/train.py \
    experiment="${EXP_NAME}" \
    ${OVERRIDES}
