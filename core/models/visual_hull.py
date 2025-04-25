# models/visual_hull.py
import torch
import numpy as np
import lightning.pytorch as pl

from core.metrics import MetricsMixin

class VisualHull(MetricsMixin, pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        front_tensor: torch.Tensor,  # (B, 1, H, W)
        side_tensor:  torch.Tensor,  # (B, 1, H, D)
        step:         int = 1
    ) -> torch.Tensor:
        B, _, H, W = front_tensor.shape
        D = side_tensor.shape[-1]

        # binary volumes as bool tensors, squeeze channel
        front_bin = (front_tensor > 0).squeeze(1)  # (B, H, W)
        side_bin  = (side_tensor  > 0).squeeze(1)  # (B, H, D)

        # broadcast to full 3D volumes
        #    front_vol[b,i,j,k] = front_bin[b,i,j]
        #    side_vol [b,i,j,k] = side_bin [b,i,k]
        front_vol = front_bin.unsqueeze(3).expand(-1, -1, -1, D)  # (B, H, W, D)
        side_vol  = side_bin.unsqueeze(2).expand(-1, -1, W, -1)   # (B, H, W, D)

        # intersection: bool tensor
        visual_hull = front_vol & side_vol  # (B, H, W, D)

        # down‑sample if step > 1
        if step > 1:
            visual_hull = visual_hull[:, ::step, ::step, ::step]

        return visual_hull

    def training_step(self, batch, batch_idx):
        (L, R), _, _, voxels = batch
        y_pred = self(L, R)
        self.log_train_metrics(y_pred, voxels)

    def validation_step(self, batch, batch_idx):
        (L, R), _, _, voxels = batch
        y_pred = self(L, R)
        self.log_val_metrics(y_pred, voxels)

    def configure_optimizers(self):
        # return None to indicate no optimization is needed for this baseline
        return None