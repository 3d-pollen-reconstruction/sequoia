import os
import re
import logging
import inspect
from typing import Tuple

import numpy as np
import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import hydra
import rootutils
from skimage.measure import marching_cubes
from scipy.ndimage import binary_closing, binary_fill_holes
import trimesh

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_ROOT = PROJECT_ROOT / "configs"
logger = logging.getLogger(__name__)


def voxelgrid_to_mesh(vox: np.ndarray) -> trimesh.Trimesh:
    """Binary 32³ -> (centred, unit-sphere) mesh via marching-cubes."""
    if vox.sum() == 0:
        raise ValueError("Empty voxel grid – cannot create mesh.")

    verts, faces, *_ = marching_cubes(vox.astype(np.float32), level=0.5)
    verts -= verts.mean(axis=0)
    verts /= np.linalg.norm(verts, axis=1).max()
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _stack_views_to_imgs(views: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Generic fallback: convert tuple([B,H,W] | [B,1,H,W]) -> [B,V,3,H,W].
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


def _run_model(model, views, rotations=None):
    """
    Call `model.forward` no matter whether it expects
    (imgs), (views), (views, rotations), or (views_list).

    Parameters
    ----------
    model : torch.nn.Module
    views : tuple[Tensor] | Tensor
    rotations : Tensor | None
    """
    params = list(inspect.signature(model.forward).parameters.keys())
    sig_len = len(params)

    if isinstance(views, torch.Tensor):
        views = (views,)
    elif isinstance(views, list):
        views = tuple(views)

    if sig_len == 1:
        imgs = (model._build_img_batch(tuple(views))
                if hasattr(model, "_build_img_batch")
                else _stack_views_to_imgs(tuple(views)))
        return model(imgs.to(views[0].device))

    elif sig_len == 2:
        if params[1] == "rotations":
            # model wants rotations *too* … only pass them if we have them
            return model(views, rotations) if rotations is not None else model(views)
        else:
            # e.g. forward(self, views_list)
            return model(list(views))

    else:
        raise RuntimeError(f"Unsupported forward() signature with {sig_len} positional args.")



def _select_latest_checkpoint(exp_name: str, user_ckpt_path: str | None) -> str:
    """Return explicit ckpt_path if given, else latest epoch checkpoint in dir."""
    if user_ckpt_path:
        return user_ckpt_path

    ckpt_dir = os.path.join("checkpoints", exp_name)
    if not os.path.isdir(ckpt_dir):
        logger.info("Checkpoint directory '%s' does not exist. Assuming no checkpoints.")
        return ""

    pattern = rf"^{re.escape(exp_name)}_epochepoch=(\d+)\.ckpt$"
    candidates: list[tuple[int, str]] = []
    for fname in os.listdir(ckpt_dir):
        m = re.match(pattern, fname)
        if m:
            candidates.append((int(m.group(1)), fname))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching '{pattern}' in {ckpt_dir}")

    _, latest_file = max(candidates, key=lambda x: x[0])
    resolved_dir = os.path.join(ckpt_dir, latest_file)
    logger.info("Using checkpoint: %s", resolved_dir)
    return resolved_dir


def predict_and_export(cfg: DictConfig) -> None:
    """Run inference on the test split and write STL meshes to disk."""

    # Seed
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    exp_name = cfg.name
    ckpt_path = _select_latest_checkpoint(exp_name, cfg.get("ckpt_path"))

    # Data
    datamodule = instantiate(cfg.data)
    datamodule.batch_size = 1
    datamodule.setup("test")
    dataset = datamodule.test_dataset
    loader = datamodule.test_dataloader()

    # --- Model instantiation mirroring training script ----------------------
    tmp_model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    tmp_model_cfg.pop("frozen", None)  # remove unsupported key
    tmp_model_cfg.pop("pretrained", None)  # remove unsupported key
    model: torch.nn.Module = instantiate(tmp_model_cfg)

    # Freeze sub‑modules (harmless at inference, but keeps parity with training)
    for submod_name in cfg.model.get("frozen", []):
        submod = getattr(model, submod_name, None)
        if submod is None:
            logger.warning("Cannot freeze '%s': not found on model", submod_name)
            continue
        for p in submod.parameters():
            p.requires_grad = False
        logger.info("Froze parameters in model.%s", submod_name)
    # ------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path != "":
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("state_dict", state), strict=False)
    
    model.to(device).eval()

    out_dir = os.path.join("predictions", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if len(batch) == 3:
                # default dataset: (views, rotations, stems)
                views, rotations, stems = batch
            elif len(batch) == 2:
                # holo dataset: ((img0,img1), label)
                views, label = batch
                rotations = None
                stems = label  # save whatever identifier we got
            else:
                raise ValueError(f"Unexpected batch structure of length {len(batch)}.")

            # make everything a tuple of tensors and move to device
            if isinstance(views, torch.Tensor):
                views = (views,)  # single-view model
            elif isinstance(views, list):
                views = tuple(views)

            views = tuple(v.to(device) for v in views)
            rotations = rotations.to(device) if isinstance(rotations, torch.Tensor) else None

            logits = _run_model(model, views, rotations)
            vox = (logits.squeeze().cpu().numpy() >= 0.5)
            vox = binary_closing(vox, iterations=1)
            vox = binary_fill_holes(vox)

            if isinstance(stems, (list, tuple)):
                stem = stems[0]
            else:
                stem = stems

            if rotations is None:
                p0_path = dataset.pairs[idx][0]
                stem   = os.path.splitext(os.path.basename(p0_path))[0]

            stub = stem
            suffix = 1
            while os.path.exists(os.path.join(out_dir, f"{stub}.stl")):
                stub = f"{stem}_{suffix}"
                suffix += 1
            stem = stub

            try:
                mesh = voxelgrid_to_mesh(vox)
            except ValueError:
                logger.warning("Sample '%s' produced an empty mesh – skipped.", stem)
                continue

            mesh.export(os.path.join(out_dir, f"{stem}.stl"))
            logger.info("%s.stl written", stem)

    logger.info("All meshes saved to %s", out_dir)


@hydra.main(config_path=str(CONFIG_ROOT), config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    """Entry point - re-use *train.yaml* Hydra config."""
    predict_and_export(cfg)


if __name__ == "__main__":
    main()
