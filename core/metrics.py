import logging
from typing import Optional, Literal, Dict
import torch
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np

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

        # States to accumulate intersections and unions
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
        # Ensure shapes match
        if preds.shape != target.shape:
            raise ValueError(
                f"Prediction and target must have the same shape, got {preds.shape} and {target.shape}."
            )

        # Squeeze channel dimension if present
        if preds.ndim == 5 and preds.size(1) == 1:
            preds = preds.squeeze(1)
            target = target.squeeze(1)

        # Binarize predictions
        preds_bin = preds > self.threshold
        target_bin = target.bool()

        # Flatten spatial dimensions
        batch_size = preds_bin.size(0)
        preds_flat = preds_bin.view(batch_size, -1)
        target_flat = target_bin.view(batch_size, -1)

        # Compute per-sample intersection and union
        intersection = (preds_flat & target_flat).sum(dim=1).float()
        union = (preds_flat | target_flat).sum(dim=1).float()

        # Update states
        self.intersection += intersection.sum()
        self.union += union.sum()

    def compute(self) -> torch.Tensor:
        """
        Compute the final IoU value.

        Returns:
            torch.Tensor: The overall 3D IoU.
        """
        if self.union == 0: # if both intersection and union are 0
            return torch.tensor(1.0, device=self.intersection.device)
        return self.intersection / self.union

def init_metrics(
    module: torch.nn.Module,
    stage: str,
) -> None:
    
    logger.info(f"Initializing metrics for stage: {stage} and module: {module.__class__.__name__}")
    
    metrics = {}
    
    metrics["iou"] = IoU3D(
        threshold=0.5,
        dist_sync_on_step=False,
    )
    
    setattr(module, f"metrics_{stage}", torch.nn.ModuleDict(metrics))

class MetricsMixin:
    def compute_metrics(self, y_pred, y_true, stage: str):
        """Compute and update all metrics for a given stage."""
        metrics: Dict[str, Metric] = getattr(self, f"metrics_{stage}")
        
        for name, metric in metrics.items():
            metric.update(y_pred, y_true)
        logger.info(f"Updated metrics for stage: {stage} with {metrics}")
    
    def finalize_metrics(self, stage: str) -> None:
        """Compute and log metrics with label-aware names."""
        metrics: Dict[str, Metric] = getattr(self, f"metrics_{stage}")
        logger.info(f"Finalizing metrics for stage: {stage} with {metrics}")
        log_dict = {}
        for key, metric in metrics.items():
            name = key.split(f"_{stage}")[0]
            log_key = f"fold_{self.fold}/{stage}/{name}"
            try:
                metric_value = metric.compute()
                logger.info(f"Computed metric {name} with value: {metric_value}")
                log_dict[log_key] = (
                    metric_value.item() if isinstance(metric_value, torch.Tensor)
                    else metric_value
                )

                metric.reset()
            except Exception as e:
                print(f"Error computing metric {name}: {str(e)}")
        logger.info(f"Setting self.log_dict to {log_dict}")
        self.log_dict(log_dict, prog_bar=True)