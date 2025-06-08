import os
import json
from pathlib import Path
from typing import Iterable, Sequence, Union, List

from .pollen_dataset import PollenDataset

from dotenv import load_dotenv
import torch

load_dotenv()

def list_files(path: str) -> Iterable[str]:
    """Return names of regular files (no sub‑dirs) in *path*."""
    return [entry.name for entry in os.scandir(path) if entry.is_file()]

def list_files_of_type(
    directory: Union[str, Path],
    extensions: Sequence[str],
    *,
    recursive: bool = False,
) -> List[Path]:
    """
    Return a list of Path objects whose suffix matches *extensions*.

    Parameters
    ----------
    directory   : str | Path
        Folder to search.
    extensions  : iterable of str
        File‑type suffixes ('.txt', '.py', 'jpg', ...).  The leading
        dot is optional.  Matching is case‑insensitive.
    recursive   : bool, default False
        If True, descend into sub‑directories (uses rglob).

    Examples
    --------
    >>> list_files_of_type('/logs', ['.log'])
    [PosixPath('/logs/today.log'), PosixPath('/logs/yesterday.log')]

    >>> list_files_of_type('.', ('py', 'ipynb'), recursive=True)
    [PosixPath('app/main.py'), PosixPath('notebooks/demo.ipynb')]
    """
    # normalise extensions once (make sure they start with a dot, lower‑case)
    exts = {
        ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
        for ext in extensions
    }

    directory = Path(directory)
    iterator: Iterable[Path] = (
        directory.rglob('*') if recursive else directory.iterdir()
    )
    return [p for p in iterator if p.is_file() and p.suffix.lower() in exts]


def make_splits_from_json(
    split_json_path: str,
    image_transforms=None,
    n_images: int = 2,
    device: torch.device = torch.device("cpu"),
    include_augmentations: bool = True,
) -> tuple[PollenDataset, PollenDataset, PollenDataset]:
    """
    Args:
      split_json_path: path to a JSON containing {"train": [...], "val": [...], "test": [...]}
        where each list contains mesh‐filenames like
          "18227_Ragweed_Ambrosia_sp_pollen_grain_showing_pore_distensions.stl",
          "17885_Peat_moss_Sphagnum_sp_spore.stl", etc.

      image_transforms: torchvision transforms to apply to images
      device: torch.device for storing the rotation tensors
      include_augmentations: 
          - If True: train set = everything in processed/images/ whose first 5 chars match a train ID.
          - If False: train set = only the exact stems from the JSON (no "_aug" suffixes).

    Returns:
      train_dataset, val_dataset, test_dataset (each a PollenDataset restricted to certain stems)
    """
    with open(split_json_path, 'r') as f:
        splits = json.load(f)

    train_list = splits["train"]
    val_list   = splits["val"]
    test_list  = splits["test"]

    train_origins = {fname.split(".")[0] for fname in train_list}
    val_origins   = {fname.split(".")[0] for fname in val_list}
    test_origins  = {fname.split(".")[0] for fname in test_list}

    train_ids_5char = {origin[:5] for origin in train_origins}

    images_dir = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "images")
    all_pngs = sorted([
        fname for fname in os.listdir(images_dir)
        if fname.lower().endswith(".png")
    ])
    all_stems = [png_fname.rsplit(".", 1)[0] for png_fname in all_pngs]

    if include_augmentations:
        train_stems = [stem for stem in all_stems if stem[:5] in train_ids_5char]
    else:
        train_stems = [stem for stem in all_stems if stem in train_origins]

    val_stems = [stem for stem in all_stems if stem in val_origins]

    test_stems = [stem for stem in all_stems if stem in test_origins]

    train_ds = PollenDataset(
        image_transforms=image_transforms,
        n_images=n_images,
        device=device,
        file_list=train_stems
    )
    val_ds = PollenDataset(
        image_transforms=image_transforms,
        n_images=n_images,
        device=device,
        file_list=val_stems
    )
    test_ds = PollenDataset(
        image_transforms=image_transforms,
        n_images=n_images,
        device=device,
        file_list=test_stems
    )

    return train_ds, val_ds, test_ds
