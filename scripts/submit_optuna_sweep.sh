#!/usr/bin/env bash
###############################################################################
# Usage (login node):                                                         #
#   bash submit_optuna_sweep.sh [-- hydra overrides …]                        #
#                                                                             #
# Anything after -- is forwarded unchanged to Hydra.                          #
###############################################################################

# ───── 1. If we're on the login node, re-submit ourselves with sbatch ────── #
if [ -z "$SLURM_JOB_ID" ]; then
  sbatch --job-name="pix2vox_optuna_ctrl" \
         --partition=performance \
         --time=00:15:00         \  # enough for Optuna to dispatch trials
         --cpus-per-task=2  --mem=4G \
         --gres=gpu:0            \  # the controller holds *no* GPU
         --output="logs/optuna_ctrl_%j.out" \
         "$0" "$@"               # re-invoke inside the allocation
  echo "Controller job submitted."
  exit
fi

# ───── 2. Inside the controller allocation (compute node) ───────────────── #
IMG="./singularity/pix2vox.sif"
BIND="--bind ${SLURM_SUBMIT_DIR}:/workspace"   # mirror your working dir

#   ▸ No `module load` here – avoids the missing /usr/share/lmod error
#   ▸ Singularity binary is usually in /usr/bin; adjust if different
/usr/bin/singularity exec --nv $BIND "$IMG" \
  python3 train.py -m +sweep=pix2vox_optuna "$@"
