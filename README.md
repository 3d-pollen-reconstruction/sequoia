# Recommended Training Method

**Local training is strongly NOT recommended.**  
Instead, use the official Docker containers provided at [https://hub.docker.com/repositories/etiir](https://hub.docker.com/repositories/etiir) for all training tasks.

## Why use Docker?

- Ensures a consistent, reproducible environment
- Avoids dependency and hardware issues on local machines
- Simplifies deployment on HPC clusters

## Running on SLURM

To run training jobs on a SLURM-managed cluster, use the provided Singularity containers and submit your jobs with the following example scripts.

---

### PixelNeRF Training (`train_pixelnerf.sbatch`)

```bash
#!/bin/bash
#SBATCH --job-name=4_4pixelnerf_pollen_performance
#SBATCH --partition=performance
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=6-24:00:00
#SBATCH --output=logs/pixelnerf_pollen_%j.out
#SBATCH --error=logs/pixelnerf_pollen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Optional: Load modules
# module load singularity

export WANDB_API_KEY=XXXX

singularity exec --nv \
  --bind /home2/etienne.roulet/checkpoints:/container/checkpoints \
  --bind /home2/etienne.roulet/sequoia/Pixel_Nerf/:/code \
  --pwd /code \
  --env HF_TOKEN=something \
  pixelnerf_new.sif \
  python3 train/org_train.py \
    -n pollen_256_4_4 \
    -c conf/exp/pollen.conf \
    -D /code/pollen \
    --checkpoints_path /container/checkpoints \
    --visual_path /container/checkpoints/visuals \
    --logs_path /container/checkpoints/logs \
    --gpu_id='0' \
    --resume \
    --lr 0.00001 \
    --epochs 10000000000000 \
    --gamma 0.99999 \
    --batch_size 2 \
    --nviews "4" \
    --resume
```

---

### SparseFusion Training (`sparsefusion.sbatch`)

```bash
#!/bin/bash
#SBATCH --job-name=sparsefusion_pollen_perf
#SBATCH --partition=performance
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=6-24:00:00
#SBATCH --output=logs/sparsefusion_%j.out
#SBATCH --error=logs/sparsefusion_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Optional: Load modules
# module load singularity

singularity exec --nv \
  --bind /home2/etienne.roulet/sequoia/SparseFusion:/workspace \
  --bind /home2/etienne.roulet/sequoia/SparseFusion/checkpoints:/workspace/checkpoints \
  --pwd /workspace \
  sparsefusion_latest.sif \
  bash -c "source activate sparsefusion && python demo.py -d co3d_toy -c apple --eft ./checkpoints/sf/apple/ckpt_latest_eft.pt"
```

---

## Notes

- Replace paths and environment variables as needed for your setup.
- Always use the provided containers for best results and support.
- For more containers and updates, see [https://hub.docker.com/repositories/etiir](https://hub.docker.com/repositories/etiir).