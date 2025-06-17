import os
import logging
import inspect
from typing import Tuple

import numpy as np
import torch
import lightning.pytorch as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra
import rootutils
from skimage.measure import marching_cubes
from scipy.ndimage import binary_closing, binary_fill_holes
import trimesh

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root",
                                    pythonpath=True)
CONFIG_ROOT = PROJECT_ROOT / "configs"
logger = logging.getLogger(__name__)


def voxelgrid_to_mesh(vox: np.ndarray) -> trimesh.Trimesh:
    """Binary 32³ → (centred, unit-sphere) mesh via marching-cubes."""
    if vox.sum() == 0:
        raise ValueError("Empty voxel grid – cannot create mesh.")

    verts, faces, *_ = marching_cubes(vox.astype(np.float32), level=0.5)
    verts -= verts.mean(axis=0)
    verts /= np.linalg.norm(verts, axis=1).max()
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _stack_views_to_imgs(views: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Generic fallback: convert tuple([B,H,W] | [B,1,H,W]) → [B,V,3,H,W].
    Mirrors Pix2Vox's _build_img_batch logic for other models.
    """
    processed = []
    for v in views:
        if v.dim() == 3:
            v = v.unsqueeze(1)
        if v.shape[1] == 1:
            v = v.repeat(1, 3, 1, 1)
        processed.append(v)
    return torch.stack(processed, dim=1)


def _run_model(model, views, rotations):
    """Adaptive call that works for 1- or 2-arg forward()."""
    sig_len = len(inspect.signature(model.forward).parameters)

    if sig_len == 1:
        imgs = (model._build_img_batch(tuple(views))
                if hasattr(model, "_build_img_batch")
                else _stack_views_to_imgs(tuple(views))
               )
        return model(imgs.to(views[0].device))

    elif sig_len == 2:
        return model(views, rotations)

    else:
        raise RuntimeError(
            f"Unsupported forward() signature with {sig_len} positional args."
        )


def predict_and_export(cfg: DictConfig) -> None:
    """Run inference on the test split and write STL meshes to disk."""
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    exp_name = cfg.name
    ckpt_path = cfg.get("ckpt_path") or os.path.join(
        "checkpoints", exp_name, "last.ckpt"
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    datamodule = instantiate(cfg.data)
    datamodule.batch_size = 1
    datamodule.setup("test")
    dataset = datamodule.test_dataset
    loader = datamodule.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.model)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    logger.info("Loaded model from %s", ckpt_path)
    model.to(device).eval()

    out_dir = os.path.join("predictions", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            views, rotations, _ = batch
            views = tuple(v.to(device) for v in views)
            rotations = rotations.to(device)

            logits = _run_model(model, views, rotations)
            
            vox = (logits.squeeze().cpu().numpy() >= 0.5)
            vox = binary_closing(vox, iterations=1)
            vox = binary_fill_holes(vox) 
            
            stem = dataset.stems[idx]
            try:
                mesh = voxelgrid_to_mesh(vox)
            except ValueError:
                logger.warning("Sample '%s' is empty – skipped.", stem)
                continue
            mesh.export(os.path.join(out_dir, f"{stem}.stl"))   
            logger.info("%s.stl written", stem)

    logger.info("All meshes saved to %s", out_dir)
    

@hydra.main(config_path=str(CONFIG_ROOT), config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    """Re-use *train.yaml* as the primary Hydra config – no extra files needed."""
    predict_and_export(cfg)


if __name__ == "__main__":
    main()
