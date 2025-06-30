#!/bin/bash
#SBATCH --job-name=splatterimage
#SBATCH --partition=performance
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxA4500:1
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/splatterimage_pollen_%j.out
#SBATCH --error=logs/splatterimage_pollen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#docker build -t pixelnerf-latest --build-arg CACHE_DATE=$(Get-Date -UFormat %s) .


# Load any required modules (optional)
# module load singularity
export WANDB_API_KEY=<YOUAPIKEY>
# Run the training inside Singularity
singularity exec --nv \
  --bind /home2/etienne.roulet/checkpoints:/container/checkpoints \
  --bind /home2/etienne.roulet/sequoia/SplatterImage/:/code \
  --pwd /code \
  splatter-image_0.1.sif \
  python train_network.py \
    +dataset=pollen \
    cam_embd=pose_pos \
    data.input_images=2 \
    opt.imgs_per_obj=5 \
