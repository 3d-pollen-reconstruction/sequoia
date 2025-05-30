import logging
from typing import Sequence
import torch
from torch import nn
from torchmetrics import MetricCollection

from torchmetrics import Metric
from lightning.pytorch.loggers import NeptuneLogger, WandbLogger

logger = logging.getLogger(__name__)

class IoU3D(Metric):
    """
    Computes the Intersection over Union (IoU) metric for 3D voxel grids.

    Args:
        threshold (float): Threshold for converting continuous predictions to binary masks. Default is 0.5.
        dist_sync_on_step (bool): Synchronize metric state across processes at each `forward()` before returning the value at `compute()`. Default is False.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold

        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric states with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predictions of shape (N, D, H, W) or (N, 1, D, H, W),
                                  with values in [0, 1].
            target (torch.Tensor): Ground truth of same shape, binary values {0,1}.
        """
        if preds.shape != target.shape:
            raise ValueError(
                f"Prediction and target must have the same shape, got {preds.shape} and {target.shape}."
            )

        if preds.ndim == 5 and preds.size(1) == 1:
            preds = preds.squeeze(1)
            target = target.squeeze(1)

        preds_bin = preds > self.threshold
        target_bin = target.bool()

        batch_size = preds_bin.size(0)
        preds_flat = preds_bin.view(batch_size, -1)
        target_flat = target_bin.view(batch_size, -1)

        intersection = (preds_flat & target_flat).sum(dim=1).float()
        union = (preds_flat | target_flat).sum(dim=1).float()

        self.intersection += intersection.sum()
        self.union += union.sum()

    def compute(self) -> torch.Tensor:
        """
        Compute the final IoU value.

        Returns:
            torch.Tensor: The overall 3D IoU.
        """
        if self.union == 0:
            return torch.tensor(1.0, device=self.intersection.device)
        return self.intersection / self.union

def batch_iou(pred_vol, gt_vol, thresh=0.5, eps=1e-6, reduction="mean"):
    """
    pred_vol : Tensor  [B, 1, D, H, W]  – probabilities in [0,1]
    gt_vol   : Tensor  [B, 1, D, H, W]  – binary (0/1) occupancy
    thresh   : float or (list/tuple)     – threshold(s) used to binarise pred_vol
    eps      : float                     – small value to avoid division-by-zero
    reduction: "none" | "mean" | "sum"   – behaviour identical to PyTorch losses

    Returns
    -------
    iou      : Tensor
        * shape [B]   if reduction=="none" and single threshold
        * shape [B,T] if reduction=="none" and multiple thresholds
        * scalar      otherwise
    """
    if isinstance(thresh, (list, tuple)):
        # ––– multiple thresholds at once –––
        ious = []
        for t in thresh:
            ious.append(batch_iou(pred_vol, gt_vol, t, eps, reduction="none"))
        ious = torch.stack(ious, dim=-1)  # [B, T]
        if reduction == "none":
            return ious
        elif reduction == "mean":
            return ious.mean()
        elif reduction == "sum":
            return ious.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    # ––– single threshold –––
    pred_bin = (pred_vol >= thresh)          # bool
    gt_bin   = (gt_vol   >= 0.5)             # ensure bool even if 0/1 floats

    inter = (pred_bin & gt_bin).float().sum(dim=(1,2,3,4))   # [B]
    union = (pred_bin | gt_bin).float().sum(dim=(1,2,3,4))   # [B]

    iou = inter / (union + eps)                              # [B]

    if reduction == "none":
        return iou
    elif reduction == "mean":
        return iou.mean()            # scalar
    elif reduction == "sum":
        return iou.sum()             # scalar
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def init_metrics(
    stage: str,
    metrics: Sequence[Metric] = None
) -> MetricCollection:
    
    logger.info(f"Initializing metrics for stage: {stage}")
    
    default = [IoU3D(threshold=0.5)]
    
    all_metrics = default + (metrics if isinstance(metrics, list) else [])
    
    return MetricCollection({m.__class__.__name__.lower(): m for m in all_metrics},
                             prefix=f"{stage}/")


class MetricsMixin:
    def __init__(self, *args, metric_list: Sequence[Metric]=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.metric_collections = nn.ModuleDict({
            "train_metrics": init_metrics("train", metric_list),
            "val_metrics":   init_metrics("val",   metric_list),
            "test_metrics":  init_metrics("test",  metric_list),
        })
 
    def log_train_metrics(self, preds, target):
        logs = self.train_metrics(preds, target)
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)

    def log_val_metrics(self, preds, target):
        logs = self.val_metrics(preds, target)
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)
        
    def log_test_metrics(self, preds, target):
        logs = self.test_metrics(preds, target)
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)
        
    def log_stage_metrics(self, stage: str, preds, target):
        mc = self.metric_collections[f"{stage}_metrics"]
        logs = mc(preds, target)
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)