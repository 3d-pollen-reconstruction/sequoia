import os
from typing import List, Tuple

from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision.transforms as transforms

load_dotenv()

MAX_IMAGES_PER_STRIP = 8

class PollenDataset(Dataset):
    def __init__(
        self,
        image_transforms: transforms.Compose | None = None,
        n_images: int = 2,
        device: torch.device = torch.device("cpu"),
        file_list: List[str] | None = None,
    ):
        if not 1 <= n_images <= MAX_IMAGES_PER_STRIP:
            raise ValueError(
                f"n_images must be in [1, {MAX_IMAGES_PER_STRIP}], got {n_images}"
            )

        self.images_path = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "images")
        self.voxels_path = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "voxels")
        self.rotations_csv = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "rotations.csv")

        self.image_transform = image_transforms
        self.device = device
        self.n_images = n_images

        if file_list is None:
            all_pngs = sorted(
                fname for fname in os.listdir(self.images_path) if fname.lower().endswith(".png")
            )
            self.stems = [png_fname.rsplit(".", 1)[0] for png_fname in all_pngs]
        else:
            self.stems = sorted(file_list)

        self.rotations = pd.read_csv(self.rotations_csv)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]

        strip_path = os.path.join(self.images_path, f"{stem}.png")
        strip_img = Image.open(strip_path).convert("L")
        strip_w, strip_h = strip_img.size

        if strip_w % strip_h != 0:
            raise RuntimeError(
                f"{strip_path} is not an exact grid of squares "
                f"(width {strip_w} px is not a multiple of height {strip_h} px)."
            )

        n_available = strip_w // strip_h
        if self.n_images > n_available:
            raise RuntimeError(
                f"Requested {self.n_images} images but only {n_available} available in strip {stem}."
            )

        to_tensor = transforms.ToTensor()
        patches: list[torch.Tensor] = []

        for i in range(self.n_images):
            left, upper = i * strip_h, 0
            right, lower = left + strip_h, strip_h
            patch_img = strip_img.crop((left, upper, right, lower))

            tensor = (
                self.image_transform(patch_img)
                if self.image_transform
                else to_tensor(patch_img).squeeze(0)
            ).to(torch.float32)

            patches.append(tensor)

        images_tuple: Tuple[torch.Tensor, ...] = tuple(patches)

        df_row = self.rotations.loc[self.rotations["sample"] == stem]
        if df_row.empty:
            raise KeyError(f"No rotation entry found for sample '{stem}' in {self.rotations_csv}")

        rot_x, rot_y, rot_z = map(float, df_row.iloc[0][["rot_x", "rot_y", "rot_z"]])
        rotations = torch.tensor([rot_x, rot_y, rot_z], dtype=torch.float32, device=self.device)

        voxels_path = os.path.join(self.voxels_path, f"{stem}.pt")
        voxels = torch.load(voxels_path)

        return images_tuple, rotations, voxels
