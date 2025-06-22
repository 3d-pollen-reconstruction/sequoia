#!/usr/bin/env bash

if [ -z "$SLURM_JOB_ID" ]; then
  sbatch --job-name="optuna_ctrl" \
         --partition=performance \
         --time=00:15:00 \
         --cpus-per-task=2  --mem=4G \
         --gres=gpu:0 \
         --output="logs/optuna_ctrl_%j.out" \
         "$0" "$@"
  echo "Controller job submitted."
  exit
fi

module load singularity
SIF_PATH=./singularity/pix2vox.sif

singularity exec --nv "$SIF_PATH" \
  python train.py -m +sweep=pix2vox_optuna "$@"
