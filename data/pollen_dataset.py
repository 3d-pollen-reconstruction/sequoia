import os

from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision.transforms as transforms

load_dotenv()

class PollenDataset(Dataset):
    def __init__(
        self,
        image_transforms=None,
        device: torch.device = torch.device('cpu'),
        file_list: list[str] | None = None,
    ):
        self.images_path   = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "images")
        self.voxels_path   = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "voxels")
        self.rotations_csv = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "rotations.csv")

        self.image_transform = image_transforms
        self.device          = device

        if file_list is None:
            all_pngs = sorted([
                fname for fname in os.listdir(self.images_path)
                if fname.lower().endswith(".png")
            ])
            self.stems = [png_fname.rsplit(".", 1)[0] for png_fname in all_pngs]
        else:
            self.stems = sorted(file_list)

        self.rotations = pd.read_csv(self.rotations_csv)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        img_path = os.path.join(self.images_path, f"{stem}.png")
        image    = Image.open(img_path).convert("L")
        w, h     = image.size
        left_img  = image.crop((0, 0, w//2, h))
        right_img = image.crop((w//2, 0, w, h))

        if self.image_transform:
            left_tensor  = self.image_transform(left_img).to(torch.float32)
            right_tensor = self.image_transform(right_img).to(torch.float32)
        else:
            t            = transforms.ToTensor()
            left_tensor  = t(left_img).squeeze(0).to(torch.float32)
            right_tensor = t(right_img).squeeze(0).to(torch.float32)

        sample_id = stem[:5]
        row       = self.rotations.loc[self.rotations['sample'] == sample_id].iloc[0]
        rotations = torch.tensor(row[1:].astype(float).values, dtype=torch.float32, device=self.device)

        voxels_path = os.path.join(self.voxels_path, f"{stem}.pt")
        voxels      = torch.load(voxels_path)

        return (left_tensor, right_tensor), rotations, voxels
