#!/bin/bash
#SBATCH --job-name=instantmesh_pollen_perf
#SBATCH --partition=p4500
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/instantmesh_%j.out
#SBATCH --error=logs/instantmesh_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Optional: Load environment modules if needed
# module load singularity

# Run InstantMesh inside the Singularity container
singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/InstantMesh:/code \
  --pwd /code \
  instantmesh_latest.sif \
  bash -c "source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate instantmesh && python train.py --base configs/instant-meshpollen-base_train.yaml --gpus 0 --num_nodes 1"