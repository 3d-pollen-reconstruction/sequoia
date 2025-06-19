#!/bin/bash
#SBATCH --job-name=128_pixelnerf_pollen_performance
#SBATCH --partition=performance
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=6-24:00:00
#SBATCH --output=logs/pixelnerf_pollen_%j.out
#SBATCH --error=logs/pixelnerf_pollen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#docker build -t pixelnerf-latest --build-arg CACHE_DATE=$(Get-Date -UFormat %s) .


# Load any required modules (optional)
# module load singularity
export WANDB_API_KEY=<YOUAPIKEY>
# Run the training inside Singularity
singularity exec --nv \
  --bind /home2/etienne.roulet/checkpoints:/container/checkpoints \
  --bind /home2/etienne.roulet/sequoia/Pixel_Nerf/:/code \
  --pwd /code \
  --env HF_TOKEN=something \
  pixelnerf_new.sif \
  python3 train/org_train.py \
    -n pollen_128 \
    -c conf/exp/pollen128.conf \
    -D /code/pollen \
    --checkpoints_path /container/checkpoints \
    --visual_path /container/checkpoints/visuals \
    --logs_path /container/checkpoints/logs \
    --gpu_id='0' \
    --resume \
    --lr 0.00001 \
    --epochs 10000000000000 \
    --gamma 0.995 \
    --batch_size 16 \
    --nviews "2" \
    --resume 
    
