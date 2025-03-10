import os
import random
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class PollenDataset(Dataset):
    def __init__(self, data_dir, processed_dir="data/processed", transform=None, fold=None):
        """
        Args:
            data_dir (str): Path of 3D STL models (cleaned folder).
            processed_dir (str): Path to the processed images directory.
            transform (callable, optional): Transformation(s) that get applied to the images.
            fold (int, optional): If specified, load images only from this fold.
        """
        self.data_dir = os.path.normpath(data_dir)
        self.processed_dir = os.path.normpath(processed_dir)
        self.transform = transform
        self.fold = fold
        
        # Create a map to store image paths for each STL file
        self.image_paths = {}

        self.exclude_list = [
            # Add any files to exclude here
        ]
        
        self.files = [
            f for f in os.listdir(self.data_dir)
            if f.endswith(".stl") and f not in self.exclude_list
        ]
        if len(self.files) == 0:
            raise RuntimeError("No STL-files after filtering found.")

        # Verify that the processed directory exists
        if not os.path.exists(self.processed_dir):
            raise RuntimeError(f"Processed directory not found: {self.processed_dir}")
            
        # If fold is specified, verify that the fold directory exists
        if self.fold is not None:
            fold_dir = os.path.join(self.processed_dir, f"fold_{self.fold}")
            if not os.path.exists(fold_dir):
                raise RuntimeError(f"Fold directory not found: {fold_dir}")
        
        # Pre-locate all image files and map them to the corresponding STL files
        self._find_image_paths()
        
        # Validate that we found at least one image
        if not self.image_paths:
            raise RuntimeError("No matching processed images found for STL files")

    def _find_image_paths(self):
        """Find and store paths to all corresponding images."""
        search_dir = os.path.join(self.processed_dir, f"fold_{self.fold}") if self.fold is not None else self.processed_dir
        
        # Get all combined.png files in the directory
        all_images = glob.glob(os.path.join(search_dir, "*_combined.png"))
        
        # Create a mapping from base name to image path
        image_map = {}
        for img_path in all_images:
            base_name = os.path.basename(img_path).replace("_combined.png", "")
            image_map[base_name] = img_path
        
        # Map STL files to their corresponding image paths
        for file_name in self.files:
            base_name = os.path.splitext(file_name)[0]
            if base_name in image_map:
                self.image_paths[file_name] = image_map[base_name]
        
        # Filter files to only those with matching images
        self.files = [f for f in self.files if f in self.image_paths]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        model_3d_path = os.path.join(self.data_dir, file_name)
        
        # Get the pre-located image path
        img_path = self.image_paths[file_name]
        
        # Load the combined image
        combined_img = Image.open(img_path).convert('L')  # Load as grayscale
        
        # Split the combined image into left and right views
        width, height = combined_img.size
        left_img = combined_img.crop((0, 0, width//2, height))
        right_img = combined_img.crop((width//2, 0, width, height))
        
        # Random rotation for consistency with the original code
        rotation = (
            random.uniform(0, 360),
            random.uniform(0, 360),
            random.uniform(0, 360),
        )
        
        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        # Always include the 3D model path as we need it for reconstruction loss
        sample = {
            'left_view': left_img,
            'right_view': right_img,
            'rotation': rotation,
            'file_name': file_name,
            '3d_model_path': model_3d_path
        }

        return sample