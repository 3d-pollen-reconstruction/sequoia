# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified by ChatGPT – added configurable dropout

import torch


class Refiner(torch.nn.Module):
    """Voxel grid refiner with configurable dropout probability.

    Parameters
    ----------
    cfg : Any
        Experiment/config object (kept for backward‑compatibility).
    dropout_p : float, optional (default=0.3)
        Dropout probability applied after most activations. Set to ``0.0`` to
        disable dropout entirely.
    """

    def __init__(self, cfg, refiner_dropout: float = 0.3):
        super().__init__()
        self.cfg = cfg
        self.dropout_p = refiner_dropout

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=self.dropout_p),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=self.dropout_p),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=self.dropout_p),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        # ------------------------------------------------------------------
        # Bottleneck (fully‑connected)
        # ------------------------------------------------------------------
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(8192, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=self.dropout_p)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=self.dropout_p)
        )

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout3d(p=self.dropout_p)
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout3d(p=self.dropout_p)
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.Sigmoid()
        )

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(self, coarse_volumes: torch.Tensor) -> torch.Tensor:
        """Refine a coarse 32³ occupancy grid.

        Parameters
        ----------
        coarse_volumes : torch.Tensor
            Flattened tensor of shape ``(B, 32*32*32)`` or volumetric tensor of
            shape ``(B, 1, 32, 32, 32)``.
        Returns
        -------
        torch.Tensor
            Refined voxel grid of shape ``(B, 32, 32, 32)``.
        """
        # Ensure correct shape (B, 1, 32, 32, 32)
        volumes_32_l = coarse_volumes.view((-1, 1, 32, 32, 32))

        # Encoder
        volumes_16_l = self.layer1(volumes_32_l)
        volumes_8_l  = self.layer2(volumes_16_l)
        volumes_4_l  = self.layer3(volumes_8_l)

        # Bottleneck
        flatten_features = self.layer4(volumes_4_l.view(-1, 8192))
        flatten_features = self.layer5(flatten_features)
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)

        # Decoder with skip connections
        volumes_8_r  = volumes_8_l  + self.layer6(volumes_4_r)
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5

        return volumes_32_r.view((-1, 32, 32, 32))
