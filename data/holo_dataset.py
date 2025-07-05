import os
import json

import cv2
import numpy as np
from PIL import Image
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

class HolographicPollenDataset(Dataset):
    def __init__(self, transform=None, extensions=None):
        self.root_dir = os.path.join(os.getenv("DATA_DIR_PATH"), "subset_poleno")
        self.transform = transform
        self.extensions = extensions or [".png"]

        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        raw_samples = []
        for taxa in self.classes:
            cls_dir = os.path.join(self.root_dir, taxa)
            for fname in os.listdir(cls_dir):
                if any(fname.lower().endswith(ext) for ext in self.extensions):
                    path = os.path.join(cls_dir, fname)
                    raw_samples.append((path, taxa))

        groups = {}
        for path, taxa in raw_samples:

            fname = os.path.basename(path)
            if 'image_pairs' not in fname:
                continue
            base = fname.split('image_pairs')[0]
            groups.setdefault((base, taxa), []).append(path)

        # Build pairs list
        self.pairs = []  # (path0, path1, taxa_name)
        for (base, taxa), paths in groups.items():
            p0 = next((p for p in paths if '.0.' in os.path.basename(p)), None)
            p1 = next((p for p in paths if '.1.' in os.path.basename(p)), None)
            
            if p0 and p1:
                self.pairs.append((p0, p1, taxa))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path0, path1, taxa = self.pairs[idx]

        def load_and_normalize(p):
            img = Image.open(p)
            arr = np.array(img).astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255.0
            else:
                arr = np.zeros_like(arr)
            return Image.fromarray(arr.astype(np.uint8), mode='L')

        img0 = load_and_normalize(path0)
        img1 = load_and_normalize(path1)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (img0, img1), taxa


class RemoveRipples(object):
    def __init__(self,
                 method: str = 'otsu',
                 blur_ksize: int = 5,
                 adaptive_blocksize: int = 51,
                 adaptive_C: int = 2,
                 morph_ksize: int = 5,
                 max_scale: float = 1.7):
        self.method = method.lower()
        self.blur_ksize = blur_ksize
        self.adaptive_blocksize = adaptive_blocksize
        self.adaptive_C = adaptive_C
        self.morph_ksize = morph_ksize
        self.max_scale = max_scale

        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_ksize, self.morph_ksize)
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        orig = np.array(img.convert('L'), dtype=np.uint8)
        h, w = orig.shape

        # blur & threshold as before
        blur = cv2.GaussianBlur(orig, (self.blur_ksize,)*2, 0)
        if self.method == 'otsu':
            _, mask = cv2.threshold(
                blur, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
        else:
            mask = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.adaptive_blocksize,
                self.adaptive_C
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        # apply mask
        result = np.where(mask == 255, orig, 255).astype(np.uint8)

        # find bbox of object
        ys, xs = np.where(result < 255)
        if len(xs) == 0 or len(ys) == 0:
            return Image.new('L', (w, h), color=255)

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        crop = result[y1:y2+1, x1:x2+1]

        crop_h, crop_w = crop.shape
        max_by_frame = min(w / crop_w, h / crop_h)
        scale = min(self.max_scale, max_by_frame)

        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)

        # resize & paste
        crop_pil = Image.fromarray(crop, mode='L')
        resized = crop_pil.resize((new_w, new_h), resample=Image.LANCZOS)
        canvas = Image.new('L', (w, h), color=255)
        x_off = (w - new_w) // 2
        y_off = (h - new_h) // 2
        canvas.paste(resized, (x_off, y_off))

        return canvas


class HolographicPolenoDataModule(pl.LightningDataModule):
    """
    A Lightning DataModule that exposes the holographic Poleno subset
    **only for testing / inference** – the dataset has no ground-truth
    labels for training.

    Parameters
    ----------
    batch_size : int
        How many paired holograms to return per step.
    selection_json_path : str
        Path to the `poleno_selection.json` file that lists the images
        to keep.  Defaults to  `<DATA_DIR_PATH>/poleno_selection.json`.
    num_workers : int, default 4
        Workers passed to the underlying DataLoader.
    image_transforms : callable | None
        Any torchvision / custom transform that should be applied to
        both images of the pair.
    extensions : list[str] | None
        Image filename extensions to consider (defaults to [".png"]).
    """

    def __init__(
        self,
        batch_size: int = 1,
        selection_json_path: str | None = None,
        num_workers: int = 4,
        image_transforms=None,
        extensions: list[str] | None = None,
    ):
        super().__init__()
        self.data_dir = os.getenv("DATA_DIR_PATH")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_transforms = image_transforms
        self.extensions = extensions
        self.selection_json_path = (
            selection_json_path
            or os.path.join(self.data_dir, "poleno_selection.json")
        )

        # will be set inside `setup`
        self.test_dataset = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        """Build the dataset and filter it to the requested subset."""
        if stage not in (None, "test", "predict", "fit"):
            # no other stages expected
            return

        # base dataset (contains *all* pairs)
        ds = HolographicPollenDataset(
            transform=self.image_transforms,
            extensions=self.extensions,
        )

        # ------------------------------------------------------------------
        # Filter pairs so that only those whose *.0.* file is listed in
        # poleno_selection.json (one-per-taxon) remain in the dataset.
        # ------------------------------------------------------------------
        with open(self.selection_json_path, "r", encoding="utf-8") as f:
            selection = json.load(f)

        # create a {(taxa, basename_without_ext)} set for O(1) lookup
        allowed = {
            (taxa, name) for taxa, names in selection.items() for name in names
        }

        # keep a pair if the basename of its ".0." file is allowed
        filtered_pairs = [
            (p0, p1, taxa)
            for p0, p1, taxa in ds.pairs
            if (taxa, os.path.splitext(os.path.basename(p0))[0]) in allowed
        ]

        # overwrite the dataset's internal pair list
        ds.pairs = filtered_pairs

        self.test_dataset = ds

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    # The holographic dataset comes *without* labels – disable training.
    def train_dataloader(self):
        raise RuntimeError(
            "HolographicPolenoDataModule does not provide a training split."
        )

    def val_dataloader(self):
        raise RuntimeError(
            "HolographicPolenoDataModule does not provide a validation split."
        )
