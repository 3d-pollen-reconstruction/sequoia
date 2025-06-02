#!/usr/bin/env python3
"""
examples/pollen_trainer.py

Full fine-tuning of the Hunyuan3D-2mv DiT UNet on pollen renders.
Each subfolder under --data_dir must contain:
    imgs/train_0.png … train_{num_views-1}.png
"""

import os
import argparse
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Hunyuan3D-2mv UNet on pollen views"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root folder with subfolders each containing imgs/train_*.png"
    )
    parser.add_argument("--num_views",  type=int,   default=8)
    parser.add_argument("--batch_size", type=int,   default=1)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--max_steps",  type=int,   default=1000)
    parser.add_argument("--output_dir",              default="pollen_full_ckpts")
    parser.add_argument("--device",                  default="cuda")
    return parser.parse_args()

class PollenViews(Dataset):
    def __init__(self, root_dir, num_views, transform):
        self.num_views = num_views
        self.folders   = sorted(
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        if not self.folders:
            raise RuntimeError(f"No subfolders found in {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        views  = []
        for i in range(self.num_views):
            img_path = os.path.join(folder, "imgs", f"train_{i}.png")
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Missing view: {img_path}")
            img = Image.open(img_path).convert("RGB")
            views.append(self.transform(img))
        # returns tensor [V, C, H, W]
        return torch.stack(views)

def collate_fn(batch):
    # batch: list of [V,C,H,W] → [B, V, C, H, W]
    return torch.stack(batch, dim=0)

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Resize → ToTensor → Normalize to [-1,1]
    transform = transforms.Compose([
        transforms.Resize((512, 512), antialias=True),
        transforms.ToTensor(),                           
        transforms.Normalize([0.5, 0.5, 0.5],             
                             [0.5, 0.5, 0.5])             
    ])

    # DataLoader
    dataset = PollenViews(args.data_dir, args.num_views, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Load pipeline (FP16) directly onto device
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv",
        use_safetensors=True,
        torch_dtype=torch.float16,
        device=device
    )

    # Extract components
    unet      = pipeline.model       # DiT UNet
    scheduler = pipeline.scheduler   # Diffusers scheduler
    vae       = pipeline.vae         # Shape-VAE

    # Freeze VAE, train UNet
    for p in vae.parameters():  
        p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # VAE scaling factor (if defined)
    scaling_factor = getattr(vae, "scaling_factor", 1.0)

    step = 0
    while step < args.max_steps:
        for batch in dataloader:
            # [B, V, C, H, W] → [B*V, C, H, W]
            B, V, C, H, W = batch.shape
            images = batch.to(device, dtype=torch.float16).view(B*V, C, H, W)

            # Encode → latent (freeze grads)
            with torch.no_grad():
                latent, _aux = vae(images)
                latent = latent * scaling_factor

            # Noise & timesteps
            noise = torch.randn_like(latent)
            t     = torch.randint(
                        low=0,
                        high=scheduler.config.num_train_timesteps,
                        size=(latent.size(0),),
                        device=device
                    )

            noisy = scheduler.add_noise(latent, noise, t)
            pred  = unet(noisy, t).sample
            loss  = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"[{step:04d}/{args.max_steps}] loss = {loss.item():.4f}")
            if step and step % 200 == 0:
                ckpt_path = os.path.join(args.output_dir, f"ckpt-{step}")
                pipeline.save_pretrained(ckpt_path)

            step += 1
            if step >= args.max_steps:
                break

    # Final save
    pipeline.save_pretrained(os.path.join(args.output_dir, "final"))
    print("✅ Training complete — saved to", args.output_dir)

if __name__ == "__main__":
    main()
