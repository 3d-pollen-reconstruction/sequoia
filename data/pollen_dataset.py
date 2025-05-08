import os
import json
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
import numpy as np
import pandas as pd
import trimesh
from trimesh.transformations import euler_matrix, translation_matrix, scale_matrix
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .utils import list_files

load_dotenv()

class PollenDataset(Dataset):
    def __init__(self,
                 image_transforms=None,
                 mesh_transforms=None,
                 device: torch.device = torch.device('cpu')
                 ):
        self.images_path      = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "images")
        self.meshes_path      = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "meshes")
        self.voxels_path      = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "voxels")
        self.pointclouds_path = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "pointclouds")
        self.rotations_csv    = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "rotations.csv")

        self.image_transform  = image_transforms
        self.mesh_transform   = mesh_transforms  # if you want per‑mesh augmentations
        self.device           = device

        self.image_files = sorted(list_files(self.images_path))
        self.rotations   = pd.read_csv(self.rotations_csv)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- load & split stereo image ---
        fname       = self.image_files[idx]
        image      = Image.open(os.path.join(self.images_path, fname)).convert("L")
        w, h       = image.size
        left_img   = image.crop((0, 0, w//2, h))
        right_img  = image.crop((w//2, 0, w, h))

        # apply transforms or default ToTensor
        if self.image_transform:
            left_tensor  = self.image_transform(left_img).to(torch.float32)
            right_tensor = self.image_transform(right_img).to(torch.float32)
        else:
            t           = transforms.ToTensor()
            left_tensor  = t(left_img).squeeze(0).to(torch.float32)
            right_tensor = t(right_img).squeeze(0).to(torch.float32)

        # --- load point cloud ---
        pc_name = fname.replace(".png", ".pt")
        pc_data = torch.load(os.path.join(self.pointclouds_path, pc_name))
        points  = pc_data["points"].to(torch.float32).to(self.device)   # (N,3)
        normals = pc_data["normals"].to(torch.float32).to(self.device)  # (N,3)

        # center & scale to unit radius
        center = points.mean(dim=0, keepdim=True)       # (1,3)
        points = points - center                        # zero-center
        max_dist = points.norm(dim=1).max()             # largest radius
        if max_dist > 0:
            points = points / max_dist                  # now all ≤1


        # --- load rotations ---
        sample_id = fname.split(".")[0]
        row       = self.rotations.loc[self.rotations['sample'] == sample_id].iloc[0]
        rotations = torch.tensor(row[1:].astype(float).values, dtype=torch.float32, device=self.device)

        # --- load & voxelize mesh →
        mesh_name = fname.replace(".png", ".stl")
        # mesh_path = os.path.join(self.meshes_path, mesh_name)
        # meta_path = os.path.join(self.images_path, "metadata", mesh_name.replace(".stl", "_cam.json"))
        voxels_path      = os.path.join(self.voxels_path, mesh_name.replace(".stl", ".pt"))
        voxels           = torch.load(voxels_path)
        

        return (left_tensor, right_tensor), (points, normals), rotations, voxels

def get_train_test_split(test_ratio=0.2, seed=42, **kwargs):
    dataset = PollenDataset(**kwargs)
    train_ids, test_ids = train_test_split(
        list(range(len(dataset))), test_size=test_ratio, random_state=seed
    )
    return dataset, train_ids, test_ids
