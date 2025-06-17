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
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    model: torch.nn.Module = instantiate(cfg.model)

    if cfg.model.get("pretrained"):
        ckpt_path = cfg.model.pretrained
        state = torch.load(ckpt_path, map_location="cpu")
        
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained weights from {ckpt_path}")

    frozen = cfg.model.get("frozen", None)
    if frozen:
        if not isinstance(frozen, (list, tuple)):
            raise ValueError(f"cfg.model.frozen must be a list, got {type(frozen)}")
        for submod_name in frozen:
            submod = getattr(model, submod_name, None)
            if submod is None:
                logger.warning(f"Cannot freeze '{submod_name}': not found on model")
                continue
            for p in submod.parameters():
                p.requires_grad = False
            logger.info(f"Froze parameters in model.{submod_name}")
    
    init_metrics("train", model)
    init_metrics("val", model)
    init_metrics("test", model)

    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandbLogger(
        project="reconstruction",
        name=cfg.name,
        config=flat_cfg,
        reinit=False,
    )

    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=wandb_logger,
        callbacks=instantiate_callbacks(cfg.get("callbacks")),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    val_metrics = trainer.callback_metrics

    datamodule.setup("test")
    test_metrics = trainer.test(model, datamodule=datamodule)[0]

    results = {**val_metrics, **{f"test/{k}": v for k, v in test_metrics.items()}}

    torch.cuda.empty_cache()
    gc.collect()

    for key, val in results.items():
        wandb_logger.experiment.summary[key] = val

    wandb_logger.experiment.finish()

    return results


def instantiate_callbacks(callbacks_cfg: DictConfig):
    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return []

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    callbacks = []
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(instantiate(cb_conf))

    return callbacks


@hydra.main(config_path=str(CONFIG_ROOT), config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    """Entry point launched by Hydra."""
    metrics = train_and_evaluate(cfg)
    logger.info("Final metrics (val + test): %s", metrics)


if __name__ == "__main__":
    main()
