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
from metrics import batch_iou

logger = logging.getLogger(__name__)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a leading 'module.' (from DataParallel) if present."""
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


class Pix2Vox(pl.LightningModule):
    """Lightning wrapper that **faithfully reproduces** the original imper‑
    ative training loop with `MERGER_KICKIN` / `REFINER_KICKIN` logic.
    """

    def __init__(
        self,
        cfg: Dict[str, Any] | None = None,
        lr: float = 1e-4,
        pretrained: str | None = None,
        merger_kickin: int = 50,
        refiner_kickin: int = 100,
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=True)

        self.encoder  = Encoder(cfg)
        self.decoder  = Decoder(cfg)
        self.merger   = Merger(cfg)
        self.refiner  = Refiner(cfg)

        self.criterion = nn.BCELoss()

        if pretrained is not None and Path(pretrained).is_file():
            ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
            parts = {
                "encoder_state_dict":  self.encoder,
                "decoder_state_dict":  self.decoder,
                "merger_state_dict":   self.merger,
                "refiner_state_dict":  self.refiner,
            }
            for key, module in parts.items():
                if key in ckpt:
                    module.load_state_dict(_strip_module_prefix(ckpt[key]), strict=False)
            logger.info(f"Loaded Pix2Vox weights from {pretrained}")

    def _generate(
        self,
        imgs: torch.Tensor,
        apply_merger: bool,
        apply_refiner: bool,
    ) -> torch.Tensor:
        """Create a voxel grid given two‑view images."""
        feats       = self.encoder(imgs)
        raw, gen    = self.decoder(feats)      # raw: weighting vols, gen: [B,V,32,32,32]
        gen         = self.merger(raw, gen) if apply_merger else gen.mean(1)
        gen         = self.refiner(gen)        if apply_refiner else gen
        return gen

    def forward(self, imgs: torch.Tensor):
        return self._generate(imgs, True, True)

    def training_step(self, batch: Tuple, batch_idx: int):
        (left, right), _, _, vox = batch

        if left.shape[1] == 1:
            left  = left.repeat(1, 3, 1, 1)
            right = right.repeat(1, 3, 1, 1)

        imgs   = torch.stack([left, right], dim=1)
        vox_gt = vox.to(torch.float32)

        epoch          = self.current_epoch
        use_merger     = epoch >= self.hparams.merger_kickin
        use_refiner    = epoch >= self.hparams.refiner_kickin

        preds = self._generate(imgs, use_merger, use_refiner)
        loss  = self.criterion(preds, vox_gt) * 10.0

        iou = batch_iou(
            preds.detach().unsqueeze(1),
            vox_gt.unsqueeze(1),
            reduction="mean",
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/iou",  iou,  on_step=False, on_epoch=True, prog_bar=True)
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
        iou   = batch_iou(preds.detach().unsqueeze(1), vox_gt.unsqueeze(1), reduction="mean")

        self.log(f"{stage}/loss", loss, prog_bar=False, batch_size=imgs.size(0))
        self.log(f"{stage}/iou",  iou,  prog_bar=True,  batch_size=imgs.size(0))

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
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr, betas=(0.9, 0.999))
        return optimizer
