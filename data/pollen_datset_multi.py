import os
import json
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

load_dotenv()

class PollenNeRFDataset(Dataset):
    def __init__(self,
                 transform=None,
                 device: torch.device = torch.device('cpu')
                 ):
        # base paths
        data_dir        = Path(os.getenv("DATA_DIR_PATH")) / "processed" / "nerf_dataset"
        self.img_dir    = data_dir
        self.meta_path  = data_dir / "transforms.json"

        # load metadata
        meta = json.loads(self.meta_path.read_text())
        self.frames       = meta["frames"]              # list of {file_path, transform_matrix}
        self.camera_angle = meta["camera_angle_x"]      # horizontal FOV in radians

        # optional image transform
        self.transform = transform or T.ToTensor()
        self.device    = device

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        # load image
        img_path = self.img_dir / frame["file_path"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img).to(self.device)  # [C,H,W], floats in [0,1]

        # load camera-to-world matrix
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=self.device)

        return {
            "image": img,
            "c2w":   c2w,
            "fov":   self.camera_angle
        }