#!/usr/bin/env bash
if [ -z "$SLURM_JOB_ID" ]; then
  sbatch --job-name="pix2vox_optuna_ctrl" \
         --partition=performance \
         --time=00:15:00         \
         --cpus-per-task=2  --mem=4G \
         --gres=gpu:0            \
         --output="logs/optuna_ctrl_%j.out" \
         "$0" "$@"
  echo "Controller job submitted."
  exit
fi

IMG="./singularity/pix2vox.sif"
BIND="--bind ${SLURM_SUBMIT_DIR}:/workspace"

/usr/bin/singularity exec --nv $BIND "$IMG" \
  python3 train.py -m +sweep=pix2vox_optuna "$@"
