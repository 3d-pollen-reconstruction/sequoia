import gc
import sys
import os
import logging
import rootutils
from typing import Dict
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from hydra.utils import instantiate
import hydra

from metrics import init_metrics

sys.path.insert(0, os.getcwd())

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_ROOT = PROJECT_ROOT / "configs"

logger = logging.getLogger(__name__)


def train_and_evaluate(cfg: DictConfig) -> Dict[str, float]:
    """Run a single train–val–test cycle and return validation & test metrics."""
    
    logger.info("Starting training with configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    # ---------------------------------------------------------------------
    # Seed everything for reproducibility
    # ---------------------------------------------------------------------
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # ---------------------------------------------------------------------
    # DataModule ─ expects that `setup(stage)` creates the corresponding splits
    #   stage="fit"  -> train + val
    #   stage="test" -> test
    # ---------------------------------------------------------------------
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    # ---------------------------------------------------------------------
    # Model & metrics
    # ---------------------------------------------------------------------
    print(cfg.model, type(cfg.model))
    model = instantiate(cfg.model)
    init_metrics("train", model)
    init_metrics("val", model)
    init_metrics("test", model)

    # ---------------------------------------------------------------------
    # WandB logger – flatten cfg for nicer UI
    # ---------------------------------------------------------------------
    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandbLogger(
        project="reconstruction",
        name=cfg.name,
        config=flat_cfg,
        reinit=False,
    )

    # ---------------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------------
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=wandb_logger,
        callbacks=instantiate(cfg.get("callbacks")),
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    val_metrics = trainer.callback_metrics  # from the last validation epoch

    # ------------------------------------------------------------------
    # Testing (uses the best checkpoint by default if trainer.fit saved one)
    # ------------------------------------------------------------------
    datamodule.setup("test")
    test_metrics = trainer.test(model, datamodule=datamodule)[0]

    # Merge val & test metrics for convenience
    results = {**val_metrics, **{f"test/{k}": v for k, v in test_metrics.items()}}

    # ------------------------------------------------------------------
    # Clean‑up
    # ------------------------------------------------------------------
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # Log a nice summary to WandB (this will appear in the run summary pane)
    # ------------------------------------------------------------------
    for key, val in results.items():
        wandb_logger.experiment.summary[key] = val

    wandb_logger.experiment.finish()

    return results


@hydra.main(config_path=str(CONFIG_ROOT), config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    """Entry point launched by Hydra."""
    metrics = train_and_evaluate(cfg)
    logger.info("Final metrics (val + test): %s", metrics)


if __name__ == "__main__":
    main()
