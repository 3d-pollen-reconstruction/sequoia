import gc
import sys
import os
import re
import logging
import rootutils
from collections import defaultdict
from typing import Dict, List
import torch
import lightning.pytorch as pl
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
import hydra
import statistics

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

    # prepare data for this fold
    cfg.data.fold_idx = fold
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    # instantiate model and metrics
    model = instantiate(cfg.model)
    model.fold = fold
    init_metrics(model, "train")
    init_metrics(model, "val")

    # trainer reuses shared W&B logger
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=wandb_logger,
        callbacks=instantiate(cfg.get("callbacks")),
    )

    # training
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    val_results_list = trainer.validate(model, datamodule=datamodule)

    if not val_results_list:          # safety check, shouldn’t happen
        val_results = {}
    elif len(val_results_list) == 1:  # the common case: one val dataloader
        val_results = val_results_list[0]
    else:                             # >1 dataloader – merge the dicts
        val_results = {}
        for d, dl_dict in enumerate(val_results_list):
            for k, v in dl_dict.items():
                val_results[f"dl{d}/{k}"] = v
    # cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_results

def run_cv(cfg: DictConfig) -> Dict[str, float]:
    """
    Orchestrate the entire CV experiment in one W&B run.
    """
    # flatten config once
    flat_cfg = OmegaConf.to_container(cfg, resolve=True)

    # single W&B run for all folds
    wandb_logger = WandbLogger(
        project="reconstruction",
        name=cfg.experiment.name,
        config=flat_cfg,
        reinit=False,
    )
    run = wandb_logger.experiment
    # plot all fold_* metrics vs epoch
    run.define_metric("fold_*", step_metric="epoch")

    all_fold_metrics: List[Dict[str, float]] = []
    for fold in range(cfg.data.n_splits):
        logger.info(f"Starting fold {fold}")
        fold_metrics = train_fold(cfg, fold, wandb_logger)
        all_fold_metrics.append(fold_metrics)

        # log fold metrics, formatting only numeric values
        entries: List[str] = []
        for k, v in fold_metrics.items():
            if isinstance(v, (int, float)):
                entries.append(f"{k}={v:.4f}")
            else:
                entries.append(f"{k}={v}")
        logger.info(f"Fold {fold} metrics: " + ", ".join(entries))

    
    metric_buckets: defaultdict[str, list] = defaultdict(list)
    rx = re.compile(r"(?:dl\d+/)?fold_\d+/(.*)")   # capture metric suffix

    for fold_dict in all_fold_metrics:
        for k, v in fold_dict.items():
            m = rx.match(k)
            metric_name = m.group(1) if m else k       # fallback: keep key
            metric_buckets[metric_name].append(v)

    summary_metrics: Dict[str, float] = {}
    for metric_name, values in metric_buckets.items():
        nums = [v for v in values if isinstance(v, (int, float))]
        if not nums:
            continue

        mean_val = statistics.mean(nums)
        std_val  = statistics.stdev(nums) if len(nums) > 1 else 0.0

        safe = metric_name.replace('/', '_')
        summary_metrics[f"cv/mean_{safe}"] = mean_val
        summary_metrics[f"cv/std_{safe}"]  = std_val

    # log summary to W&B run
    for k, v in summary_metrics.items():
        run.summary[k] = v

    run.finish()
    return summary_metrics

@hydra.main(config_path=str(CONFIG_ROOT), config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    summary = run_cv(cfg)
    logger.info(f"CV Summary: {summary}")
    return summary

if __name__ == "__main__":  
    main()
