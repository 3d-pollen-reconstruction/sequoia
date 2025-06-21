#!/bin/bash
#SBATCH --job-name=pixel2mesh_shapenet_performance
#SBATCH --partition=performance
#SBATCH --mem=25G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/pixel2mesh_shapenet_%j.out
#SBATCH --error=logs/pixel2mesh_shapenet_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

export WANDB_API_KEY=<YOUAPIKEY>


singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/Pixel2MeshPlusPlus/:/code \
  --pwd /code \
  pixel2meshplusplus_latest.sif \
  tensorboard --logdir /code/results/coarse_mvp2m/summaries --port 6006