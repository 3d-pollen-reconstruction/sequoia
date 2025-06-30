#!/bin/bash
#SBATCH --job-name=sparsefusion_pollen_perf
#SBATCH --partition=performance
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=6-24:00:00
#SBATCH --output=logs/sparsefusion_%j.out
#SBATCH --error=logs/sparsefusion_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Optional: Load environment modules if needed
# module load singularity

# Run SparseFusion inside the Singularity container
singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/SparseFusion:/workspace \
  --bind /home2/etienne.roulet/sequoia/SparseFusion/checkpoints:/workspace/checkpoints \
  --pwd /workspace \
  sparsefusion_latest.sif \
  bash -c "source activate sparsefusion && python demo.py -d co3d_toy -c apple --eft ./checkpoints/sf/apple/ckpt_latest_eft.pt"