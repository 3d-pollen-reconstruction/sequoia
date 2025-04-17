import os

from dotenv import load_dotenv
from PIL import Image
import numpy as np
import pandas as pd
import trimesh
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

load_dotenv()

class PollenDataset(Dataset):
    def __init__(self, image_transforms=None, mesh_transforms=None):
        self.images_path = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "images")
        self.meshes_path = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "meshes")
        self.pointclouds_path = os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "pointclouds")
        self.image_transform = image_transforms
        self.mesh_transform = mesh_transforms
        self.image_files = os.listdir(self.images_path)
        self.mesh_files = os.listdir(self.meshes_path)
        self.rotations = pd.read_csv(os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "rotations.csv"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.image_files[idx])
        mesh_path = self.image_files[idx].replace(".png", ".stl")
        pointcloud_path = self.image_files[idx].replace(".png", ".npz")

        image = Image.open(image_path).convert("L")
        width, height = image.size
        
        left_image = image.crop((0, 0, width // 2, height))
        right_image = image.crop((width // 2, 0, width, height))
        
        if self.image_transform:
            left_image = self.image_transform(left_image).to(torch.float32)
            right_image = self.image_transform(right_image).to(torch.float32)
        else:
            left_image = transforms.ToTensor()(left_image).squeeze(0).to(torch.float32)
            right_image = transforms.ToTensor()(right_image).squeeze(0).to(torch.float32)
        
        mesh_path = os.path.join(self.meshes_path, mesh_path)
        #mesh = trimesh.load(mesh_path)
        #vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        #faces = torch.tensor(mesh.faces, dtype=torch.long)
        
        #if self.mesh_transform:
        #    vertices, faces = self.mesh_transform(vertices, faces)
        
        x_rotation, y_rotation, z_rotation = [
            float(val) for val in self.rotations.loc[
                self.rotations['sample'] == self.image_files[idx].split(".")[0]
            ].values[0][1:]
        ]

        rotations = self.rotations.loc[
            self.rotations['sample'] == self.image_files[idx].split(".")[0]
        ].values[0][1:]
        # Convert each rotation value to a Python float before creating the tensor
        rotations = [float(r) for r in rotations]
        rotations = torch.tensor(rotations, dtype=torch.float32)
        x_rotation, y_rotation, z_rotation = rotations

        pointcloud_path = os.path.join(self.pointclouds_path, pointcloud_path)
        points = torch.from_numpy(np.load(pointcloud_path)['points']).to(torch.float32)

        return (left_image, right_image), points, (x_rotation, y_rotation, z_rotation), mesh_path

def get_train_test_split(test_ratio=0.2, seed=42, **kwargs):
    dataset = PollenDataset(**kwargs)
    train_ids, test_ids = train_test_split(
        list(range(len(dataset))), test_size=test_ratio, random_state=seed
    )
    return dataset, train_ids, test_ids
