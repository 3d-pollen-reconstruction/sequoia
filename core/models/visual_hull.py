import torch
import numpy as np
import lightning as pl

class VisualHull(pl.LightningModule):
    def __init__(self):
        super(VisualHull, self).__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, front_tensor: torch.Tensor, side_tensor: torch.Tensor, step: int = 1) -> np.ndarray:
        front_np = front_tensor.squeeze(0).detach().cpu().numpy()
        side_np = side_tensor.squeeze(0).detach().cpu().numpy()

        front_bin = (front_np > 0).astype(np.uint8)
        side_bin = (side_np > 0).astype(np.uint8)

        H, W = front_bin.shape
        D = W

        front_vol = np.repeat(front_bin[:, :, np.newaxis], D, axis=2)
        side_vol = np.repeat(side_bin[:, np.newaxis, :], W, axis=1)

        visual_hull = np.logical_and(front_vol, side_vol)
        voxel_data = visual_hull[::step, ::step, ::step]
        
        return voxel_data

    def training_step(self, batch, batch_idx):
        (left_image, right_image), (vertices, faces), rotations = batch
        _ = self.forward(left_image, right_image)
       
        loss = self.dummy_param * 0.0
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
