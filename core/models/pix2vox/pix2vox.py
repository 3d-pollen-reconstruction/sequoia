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


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a leading 'module.' (from DataParallel) if present."""
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


class Pix2Vox(MetricsMixin, pl.LightningModule):
    """Lightning wrapper for the Pix2Vox model."""
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

        self.lr            = lr
        self.merger_kickin = merger_kickin
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
                    "encoder_state_dict":  self.encoder,
                    "decoder_state_dict":  self.decoder,
                    "merger_state_dict":   self.merger,
                    "refiner_state_dict":  self.refiner,
                }
                for key, module in parts.items():
                    if key in ckpt:
                        module.load_state_dict(ckpt[key], strict=False)
                logger.info(f"Loaded Pix2Vox weights from {pretrained}")
            else:
                logger.warning(f"Pretrained weights file {pretrained} does not exist – skipping load.")

    def _generate(self, imgs: torch.Tensor, apply_merger: bool, apply_refiner: bool) -> torch.Tensor:
        """Create a voxel grid given two‑view images."""
        feats = self.encoder(imgs)
        raw, gen = self.decoder(feats) # raw: weighting vols, gen: [B,V,D,H,W]
        gen = self.merger(raw, gen) if apply_merger else gen.mean(1)
        gen = self.refiner(gen) if apply_refiner else gen
        return gen # [B,D,H,W]

    def forward(self, imgs: torch.Tensor):
        return self._generate(imgs, True, True)

    def training_step(self, batch: Tuple, batch_idx: int):
        (left, right), _, _, vox = batch
        if left.shape[1] == 1:
            left  = left.repeat(1, 3, 1, 1)
            right = right.repeat(1, 3, 1, 1)

        imgs   = torch.stack([left, right], dim=1) # [B,2,C,H,W]
        vox_gt = vox.to(torch.float32) # [B,D,H,W]

        use_merger  = self.current_epoch >= self.merger_kickin
        use_refiner = self.current_epoch >= self.refiner_kickin

        preds = self._generate(imgs, use_merger, use_refiner)
        loss  = self.criterion(preds, vox_gt) * 10.0

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_stage_metrics("train", preds, vox_gt)
        return loss

    def _shared_eval_step(self, batch: Tuple, stage: str):
        (left, right), _, _, vox = batch
        if left.shape[1] == 1:
            left  = left.repeat(1, 3, 1, 1)
            right = right.repeat(1, 3, 1, 1)
        imgs   = torch.stack([left, right], dim=1)
        vox_gt = vox.to(torch.float32)

        preds = self._generate(imgs, True, True)
        loss  = self.criterion(preds, vox_gt) * 10.0

        self.log(f"{stage}/loss", loss, prog_bar=False, batch_size=imgs.size(0))
        self.log_stage_metrics(stage, preds, vox_gt)

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "test")

    def configure_optimizers(self):
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.merger.parameters())
            + list(self.refiner.parameters())
        )
        return torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
