#!/usr/bin/env bash
#
#SBATCH --job-name=recon_${1}           # job name (includes experiment)
#SBATCH --output=logs/%x_%j.out         # %x = job-name, %j = jobid
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4               # adjust as needed
#SBATCH --gres=gpu:1                    # request 1 GPU
#SBATCH --time=24:00:00
#SBATCH --partition=standard            # default; override via sbatch --partition=...
#SBATCH --export=ALL

# --- sanity check ---
if [[ -z "$1" ]]; then
  echo "Usage: $0 <experiment_name> [HYDRA_OVERRIDES...]"
  echo "  e.g.: sbatch run_train.slurm my_experiment data.batch_size=64 trainer.max_epochs=50"
  exit 1
fi

EXP_NAME="$1"
shift  # pop off experiment name

# --- environment setup on host ---
module purge
module load cuda/11.7            # ensure GPU drivers
# (No conda environment needed—everything is inside the container)

# go to your project root
cd $SLURM_SUBMIT_DIR

# --- launch inside Singularity container ---
singularity exec --nv train.sif \
    --config-name "$EXP_NAME" \
    "$@"
