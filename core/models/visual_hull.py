# models/visual_hull.py
import torch
import numpy as np
import lightning.pytorch as pl

class VisualHull(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, front_tensor: torch.Tensor, side_tensor: torch.Tensor, step: int = 1) -> list:
        batch_voxels = []
        
        for ft, st in zip(front_tensor, side_tensor):
            front_np = ft.squeeze().detach().cpu().numpy()  # This should be (1024, 1024)
            side_np = st.squeeze().detach().cpu().numpy()     # This should be (1024, 1024)
            
            front_bin = (front_np > 0).astype(np.uint8)
            side_bin = (side_np > 0).astype(np.uint8)
            
            H, W = front_bin.shape
            D = W  # Assuming depth equals width, adjust accordingly

            front_vol = np.repeat(front_bin[:, :, np.newaxis], D, axis=2)
            side_vol = np.repeat(side_bin[:, np.newaxis, :], W, axis=1)
            visual_hull = np.logical_and(front_vol, side_vol)
            voxel_data = visual_hull[::step, ::step, ::step]
            
            batch_voxels.append(voxel_data)
        
        return batch_voxels

    def training_step(self, batch, batch_idx):
        (left_image, right_image), _, rotations, _ = batch
        _ = self.forward(left_image, right_image)
        loss = self.dummy_param * 0.0
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (left_image, right_image), _, rotations, _ = batch
        voxel_data_list = self.forward(left_image, right_image)
        # Compute the total voxel count over the batch
        voxel_count = float(sum([v.sum() for v in voxel_data_list]))
        self.log("val_voxel_count", voxel_count, prog_bar=True)
        loss = self.dummy_param * 0.0
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
