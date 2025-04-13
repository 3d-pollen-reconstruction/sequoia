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
        self.image_transform = image_transforms
        self.mesh_transform = mesh_transforms
        self.image_files = os.listdir(self.images_path)
        self.mesh_files = os.listdir(self.meshes_path)
        self.rotations = pd.read_csv(os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "rotations.csv"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.image_files[idx])
        image = Image.open(image_path).convert("L")
        width, height = image.size
        
        left_image = image.crop((0, 0, width // 2, height))
        right_image = image.crop((width // 2, 0, width, height))
        
        if self.image_transform:
            left_image = self.image_transform(left_image)
            right_image = self.image_transform(right_image)
        else:
            left_image = transforms.ToTensor()(left_image).squeeze(0)
            right_image = transforms.ToTensor()(right_image).squeeze(0)
        
        mesh_path = os.path.join(self.meshes_path, self.mesh_files[idx])
        mesh = trimesh.load(mesh_path)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        
        if self.mesh_transform:
            vertices, faces = self.mesh_transform(vertices, faces)
        
        # take the rotation values from the csv file where the 'sample' column is the filename without the extension
        x_rotation, y_rotation, z_rotation = self.rotations.loc[self.rotations['sample'] == self.image_files[idx].split(".")[0]].values[0][1:]
        
        return (left_image, right_image), (vertices, faces), (x_rotation, y_rotation, z_rotation)
        return (left_image, right_image), (vertices, faces)
    
# At the bottom of your PollenDataset file
from sklearn.model_selection import train_test_split

def get_train_test_split(test_ratio=0.2, seed=42, **kwargs):
    dataset = PollenDataset(**kwargs)
    train_ids, test_ids = train_test_split(
        list(range(len(dataset))), test_size=test_ratio, random_state=seed
    )
    return dataset, train_ids, test_ids
