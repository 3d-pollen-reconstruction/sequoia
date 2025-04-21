import os
import json
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .utils import list_files

load_dotenv()

class MultiViewPollenDataset(Dataset):
    """
    Dataset that loads N-view concatenated grayscale images, splits into individual views,
    applies transforms, and returns:
      - views_tensor: Tensor of shape (N, H, W)
      - pointcloud: Tensor of shape (P, 3)
      - rotations: Tensor of shape (N, 3)
      - voxels: loaded voxel tensor
    """
    def __init__(
        self,
        num_views: int = 4,
        image_transforms=None,
        device: torch.device = torch.device('cpu')
    ):
        self.num_views = num_views
        root = os.getenv("DATA_DIR_PATH")
        self.images_path      = os.path.join(root, "processed", f"images_{num_views}")
        self.meta_path        = os.path.join(self.images_path, "metadata")
        self.pointclouds_path = os.path.join(root, "processed", "pointclouds")
        self.voxels_path      = os.path.join(root, "processed", "voxels")
        self.rotations_csv    = os.path.join(root, "processed", f"rotations_{num_views}.csv")

        self.image_transform  = image_transforms or transforms.ToTensor()
        self.device           = device

        # list of image filenames: sample_Nviews.png
        self.image_files = sorted([f for f in list_files(self.images_path) if f.endswith(f"_{num_views}views.png")])
        # rotations dataframe: sample,view_index,rot_x,rot_y,rot_z
        self.rotations_df = pd.read_csv(self.rotations_csv)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # load concatenated image
        fname      = self.image_files[idx]
        sample_id  = Path(fname).stem.replace(f"_{self.num_views}views", "")
        img_path   = os.path.join(self.images_path, fname)
        image      = Image.open(img_path).convert("L")
        w, h       = image.size
        w_each     = w // self.num_views

        # split into N views, apply transforms, stack
        views = []
        for i in range(self.num_views):
            left = i * w_each
            box = (left, 0, left + w_each, h)
            view_img = image.crop(box)
            tensor = self.image_transform(view_img).squeeze(0).to(torch.float32)
            views.append(tensor)
        views_tensor = torch.stack(views, dim=0).to(self.device)

        # load point cloud
        pc_file = f"{sample_id}.npz"
        pc_data = np.load(os.path.join(self.pointclouds_path, pc_file))
        points  = torch.from_numpy(pc_data["points"]).to(torch.float32).to(self.device)

        # load rotations for this sample
        subset = self.rotations_df[self.rotations_df["sample"] == sample_id]
        # ensure sorted by view_index
        subset = subset.sort_values("view_index")
        rots = subset[["rot_x","rot_y","rot_z"]].values.astype(np.float32)
        rotations = torch.from_numpy(rots).to(torch.float32).to(self.device)  # shape (N,3)

        # load voxels
        vox_file = f"{sample_id}.pt"
        voxels   = torch.load(os.path.join(self.voxels_path, vox_file)).to(self.device)

        return views_tensor, points, rotations, voxels


def get_train_test_split(test_ratio=0.2, seed=42, **kwargs):
    dataset = MultiViewPollenDataset(**kwargs)
    N = len(dataset)
    indices = list(range(N))
    # reproducible split
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = int(np.floor(test_ratio * N))
    test_idx, train_idx = indices[:split], indices[split:]
    return dataset, train_idx, test_idx
