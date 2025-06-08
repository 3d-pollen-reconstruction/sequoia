from pathlib import Path
from typing import Dict, Any, Tuple

import logging
import torch
import torch.nn as nn
import lightning.pytorch as pl

from .encoder import Encoder
from .decoder import Decoder
from .merger  import Merger
from .refiner import Refiner
from metrics import MetricsMixin

logger = logging.getLogger(__name__)


class Pix2Vox(MetricsMixin, pl.LightningModule):
    """Lightning wrapper for the Pix2Vox model (supports an arbitrary
    number of input views)."""

    # ─────────────────── initialisation ────────────────────────────────── #

    def __init__(
        self,
        cfg: Dict[str, Any] | None = None,
        lr: float = 1e-4,
        pretrained: str | None = None,
        merger_kickin: int = 50,
        refiner_kickin: int = 100,
    ):
        super().__init__()
        MetricsMixin.__init__(self)

        self.lr             = lr
        self.merger_kickin  = merger_kickin
        self.refiner_kickin = refiner_kickin

        self.encoder  = Encoder(cfg)
        self.decoder  = Decoder(cfg)
        self.merger   = Merger(cfg)
        self.refiner  = Refiner(cfg)

        self.criterion = nn.BCELoss()

        if pretrained is not None:
            if Path(pretrained).is_file():
                ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
                parts = {
                    "encoder_state_dict": self.encoder,
                    "decoder_state_dict": self.decoder,
                    "merger_state_dict":  self.merger,
                    "refiner_state_dict": self.refiner,
                }
                for key, module in parts.items():
                    if key in ckpt:
                        module.load_state_dict(ckpt[key], strict=False)
                logger.info(f"Loaded Pix2Vox weights from {pretrained}")
            else:
                logger.warning(
                    f"Pretrained weights file {pretrained} does not exist – skipping load."
                )

    # ─────────────────── utility ───────────────────────────────────────── #

    @staticmethod
    def _build_img_batch(views: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Convert a tuple of B-sized tensors (one per view) into a stacked
        5-D tensor [B,V,C,H,W], replicating channels as needed.

        Each element of *views* arrives from the default PyTorch
        DataLoader collate_fn as shape either [B,H,W] or [B,1,H,W].
        """
        processed = []
        for v in views:                      # v: [B,H,W] or [B,1,H,W] or [B,3,H,W]
            if v.dim() == 3:                 # [B,H,W]  → add channel dim
                v = v.unsqueeze(1)           # [B,1,H,W]
            if v.shape[1] == 1:              # grayscale → RGB
                v = v.repeat(1, 3, 1, 1)     # [B,3,H,W]
            processed.append(v)
        return torch.stack(processed, dim=1)  # [B,V,3,H,W]

    # ─────────────────── forward & generation ──────────────────────────── #

    def _generate(
        self,
        imgs: torch.Tensor,
        apply_merger: bool,
        apply_refiner: bool,
    ) -> torch.Tensor:
        """Create a voxel grid given *multi-view* images (shape [B,V,C,H,W])."""
        feats       = self.encoder(imgs)
        raw, gen    = self.decoder(feats)          # raw [B,V,D,H,W] (weights)
        gen         = self.merger(raw, gen) if apply_merger else gen.mean(1)
        gen         = self.refiner(gen) if apply_refiner else gen
        return gen                                 # [B,D,H,W]

    def forward(self, imgs: torch.Tensor):
        """Inference with merger and refiner enabled."""
        return self._generate(imgs, True, True)

    # ─────────────────── training & evaluation steps ───────────────────── #

    def training_step(self, batch: Tuple, batch_idx: int):
        views, _, vox = batch                    # views is a tuple of tensors
        imgs   = self._build_img_batch(views)    # [B,V,3,H,W]
        vox_gt = vox.to(torch.float32)           # [B,D,H,W]

        use_merger  = self.current_epoch >= self.merger_kickin
        use_refiner = self.current_epoch >= self.refiner_kickin

        preds = self._generate(imgs, use_merger, use_refiner)
        loss  = self.criterion(preds, vox_gt) * 10.0

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_stage_metrics("train", preds, vox_gt)
        return loss

    # ─────────────────── shared eval logic ─────────────────────────────── #

    def _shared_eval_step(self, batch: Tuple, stage: str):
        views, _, vox = batch
        imgs   = self._build_img_batch(views)
        vox_gt = vox.to(torch.float32)

        preds = self._generate(imgs, True, True)
        loss  = self.criterion(preds, vox_gt) * 10.0

        self.log(f"{stage}/loss", loss, prog_bar=False, batch_size=imgs.size(0))
        self.log_stage_metrics(stage, preds, vox_gt)

    # Lightning entry-points
    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "test")

    # ─────────────────── optimiser ─────────────────────────────────────── #

    def configure_optimizers(self):
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.merger.parameters())
            + list(self.refiner.parameters())
        )
        return torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
