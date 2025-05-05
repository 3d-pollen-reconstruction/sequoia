import gc
import sys
import os
import logging
import rootutils
from collections import defaultdict
from typing import Dict
import torch
import lightning.pytorch as pl
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
import hydra
import numpy as np

from metrics import init_metrics

sys.path.insert(0, os.getcwd())

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)

CONFIG_ROOT = PROJECT_ROOT / "configs"

logger = logging.getLogger(__name__)

def train_fold(
    cfg: DictConfig,
    fold: int,
    wandb_logger: WandbLogger
) -> Dict[str, float]:
    """
    Run one fold: instantiate data, model, train, and return final metrics for that fold.
    """
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    cfg.data.fold_idx = fold
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    model = instantiate(cfg.model)
    model.fold = fold
    init_metrics(model, "train")
    init_metrics(model, "val")

    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=wandb_logger,
        callbacks=instantiate(cfg.get("callbacks")),
    )
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    train_results = trainer.callback_metrics

    val_results = trainer.validate(model, datamodule=datamodule)[0]

    torch.cuda.empty_cache()
    gc.collect()
    
    return val_results 

def run_cv(cfg: DictConfig) -> None:
    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandbLogger(
        project="reconstruction",
        name=cfg.experiment.name,
        config=flat_cfg,
        reinit=False,
    )
    run = wandb_logger.experiment
    run.define_metric("fold_*", step_metric="epoch")

    metrics_across_folds: Dict[str, list[float]] = defaultdict(list)

    for fold in range(cfg.data.n_splits):
        logger.info(f"Starting fold {fold}")
        
        fold_val_logs = train_fold(cfg, fold, wandb_logger)
        
        for key, value in fold_val_logs.items():
            short = "/".join(key.split("/")[1:])
            metrics_across_folds[short].append(value)

        logger.info(f"Fold {fold} metrics: {fold_val_logs}")

    summary: Dict[str, float] = {}
    for metric_name, values in metrics_across_folds.items():
        arr = np.array(values, dtype=float)
        mean, std = arr.mean(), arr.std(ddof=1)
        summary[f"{metric_name}_mean"] = mean
        summary[f"{metric_name}_std"]  = std
        
        run.summary[f"{metric_name}_mean"] = mean
        run.summary[f"{metric_name}_std"]  = std

    logger.info(f"CV Summary: {summary}")
    run.finish()

@hydra.main(config_path=str(CONFIG_ROOT), config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_cv(cfg)

if __name__ == "__main__":  
    main()
