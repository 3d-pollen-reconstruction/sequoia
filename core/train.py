import sys
import os
import gc
import logging
from typing import Optional

import hydra
import lightning as L
import torch
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig

sys.path.insert(0, os.getcwd())

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)

CONFIG_ROOT = PROJECT_ROOT / "configs"

logger = logging.getLogger(__name__)

def train(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple of metrics dictionaries.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks = instantiate(cfg.get("callbacks"))

    logger.info("Instantiating loggers...")
    log_instances = instantiate(cfg.get("logger"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate(cfg.trainer, logger=log_instances, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
        "callbacks": callbacks,
    }

    if cfg.get("train"):
        logger.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        logger.info("Training completed.")
    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        logger.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        logger.info(f"Loading best model from {ckpt_path} for testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        logger.info("Testing completed.")

    test_metrics = trainer.callback_metrics

    logger.info("Cleaning up...")
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Goodbye!")

    return {**train_metrics, **test_metrics}, object_dict


@hydra.main(config_path=str(CONFIG_ROOT), config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    fold_metrics = {}

    for fold in range(cfg.data.n_splits):
        logger.info(f"Starting fold {fold}")
        cfg.data.fold_idx = fold

        current_metrics, _ = train(cfg)

        for key, value in current_metrics.items():
            fold_metrics.setdefault(key, []).append(value)
        logger.info(f"Fold {fold} Metrics: {current_metrics}")

    averaged_metrics = {key: sum(values)/len(values) for key, values in fold_metrics.items()}
    logger.info(f"Averaged Metrics across {cfg.data.n_splits} folds: {averaged_metrics}")

    return averaged_metrics

if __name__ == "__main__":
    main()