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
import trimesh
from scipy.ndimage import zoom, binary_dilation, binary_closing
import traceback


class PollenDataset(Dataset):
    def __init__(
        self, data_dir, processed_dir="data/processed", transform=None, folds=None
    ):
        """
        Args:
            data_dir (str): Path of 3D STL models (cleaned folder).
            processed_dir (str): Path to the processed images directory.
            transform (callable, optional): Transformation(s) that get applied to the images.
            folds (list, optional): If specified, load images from these folds. If None, load from all folds.
        """
        self.data_dir = os.path.normpath(data_dir)
        self.processed_dir = os.path.normpath(processed_dir)
        self.transform = transform
        self.folds = folds

        # Create a map to store image paths for each STL file
        self.image_paths = {}

        self.exclude_list = [
            # Add any files to exclude here
        ]

        self.files = [
            f
            for f in os.listdir(self.data_dir)
            if f.endswith(".stl") and f not in self.exclude_list
        ]
        if len(self.files) == 0:
            raise RuntimeError("No STL-files after filtering found.")

        # Verify that the processed directory exists
        if not os.path.exists(self.processed_dir):
            raise RuntimeError(f"Processed directory not found: {self.processed_dir}")

        # If folds are specified, verify that all fold directories exist
        if self.folds is not None:
            for fold in self.folds:
                fold_dir = os.path.join(self.processed_dir, f"fold_{fold}")
                if not os.path.exists(fold_dir):
                    raise RuntimeError(f"Fold directory not found: {fold_dir}")

        # Pre-locate all image files and map them to the corresponding STL files
        self._find_image_paths()

        # Validate that we found at least one image
        if not self.image_paths:
            raise RuntimeError("No matching processed images found for STL files")

    def _find_image_paths(self):
        """Find and store paths to all corresponding images from all specified folds."""
        # If no specific folds are provided, try to find all fold directories
        if self.folds is None:
            # Find all directories named "fold_X" in the processed directory
            self.folds = []
            for item in os.listdir(self.processed_dir):
                if os.path.isdir(
                    os.path.join(self.processed_dir, item)
                ) and item.startswith("fold_"):
                    try:
                        fold_num = int(item.split("_")[1])
                        self.folds.append(fold_num)
                    except (IndexError, ValueError):
                        continue

            if not self.folds:
                # If no fold directories found, just search in the processed directory directly
                self.folds = None

        # Map STL files to their corresponding image paths from all folds
        image_map = {}

        # Option 1: Search in specified folds
        if self.folds is not None:
            for fold in self.folds:
                fold_dir = os.path.join(self.processed_dir, f"fold_{fold}")
                all_images = glob.glob(os.path.join(fold_dir, "*_combined.png"))

                for img_path in all_images:
                    base_name = os.path.basename(img_path).replace("_combined.png", "")
                    # If we already have this base_name from another fold, we might want to keep track of fold info
                    image_map[base_name] = img_path
        # Option 2: No folds specified, search directly in processed directory
        else:
            all_images = glob.glob(os.path.join(self.processed_dir, "*_combined.png"))
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
        combined_img = Image.open(img_path).convert("L")  # Load as grayscale

        # Split the combined image into left and right views
        width, height = combined_img.size
        left_img = combined_img.crop((0, 0, width // 2, height))
        right_img = combined_img.crop((width // 2, 0, width, height))

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
            "left_view": left_img,
            "right_view": right_img,
            "rotation": rotation,
            "file_name": file_name,
            "3d_model_path": model_3d_path,
        }

        return sample
    
    def get_3d_model(self, idx, output_type='voxels', resolution=64, normalize=True):
        """
        Returns a 3D model representation from an STL file.
        
        Args:
            idx (int): Index of the dataset item
            output_type (str): Type of output model ('voxels' or 'mesh')
            resolution (int): Resolution for voxel grid (if output_type='voxels')
            normalize (bool): Whether to normalize and center the model
            
        Returns:
            numpy.ndarray or trimesh.Trimesh: 3D model as voxel grid or mesh
        """
        file_name = self.files[idx]
        model_3d_path = os.path.join(self.data_dir, file_name)
        
        if output_type.lower() == 'mesh':
            return self.stl_to_mesh(model_3d_path, normalize)
        else:  # Default to voxels
            return self.stl_to_voxels(model_3d_path, resolution, normalize)

    def stl_to_mesh(self, stl_path, normalize=True):
        """
        Loads an STL file as a mesh.
        
        Args:
            stl_path (str): Path to the STL file
            normalize (bool): Whether to normalize and center the mesh
            
        Returns:
            trimesh.Trimesh: Loaded mesh
        """
        try:
            # Load the mesh
            mesh = trimesh.load(stl_path)
            
            if normalize:
                # Center the mesh
                mesh.vertices -= mesh.bounding_box.centroid
                
                # Normalize size
                max_dim = np.max(mesh.bounding_box.extents)
                if max_dim > 0:
                    mesh.vertices /= max_dim
            
            return mesh
            
        except Exception as e:
            print(f"Error loading mesh from {stl_path}: {e}")
            return None

    def stl_to_voxels(self, stl_path, resolution=64, normalize=True):
        """
        Converts an STL file to a voxel grid.
        
        Args:
            stl_path (str): Path to the STL file
            resolution (int): Size of the voxel grid (resolution³)
            normalize (bool): Whether to normalize and center the mesh
            
        Returns:
            numpy.ndarray: Boolean voxel grid representation
        """
        try:
            # Load the mesh
            mesh = trimesh.load(stl_path)
            
            if normalize:
                # Center the mesh
                mesh.vertices -= mesh.bounding_box.centroid
                
                # Normalize size
                max_dim = np.max(mesh.bounding_box.extents)
                if max_dim > 0:
                    mesh.vertices /= max_dim
                    mesh.vertices *= 0.95  # Scale slightly to ensure it fits in grid
            
            # Try multiple voxelization methods until one works
            voxel_grid = None
            
            # Method 1: Use trimesh's ray-based voxelization
            try:
                temp_grid = mesh.voxelized(pitch=2.0/resolution, method='ray')
                temp_grid = temp_grid.fill()
                voxel_matrix = temp_grid.matrix
                
                if np.sum(voxel_matrix) > 0:
                    voxel_grid = voxel_matrix
            except Exception as e1:
                print(f"Ray-based voxelization failed: {e1}")
            
            # Method 2: Use trimesh's subdivision voxelization if method 1 failed
            if voxel_grid is None:
                try:
                    temp_grid = mesh.voxelized(pitch=2.0/resolution, method='subdivide')
                    temp_grid = temp_grid.fill()
                    voxel_matrix = temp_grid.matrix
                    
                    if np.sum(voxel_matrix) > 0:
                        voxel_grid = voxel_matrix
                except Exception as e2:
                    print(f"Subdivision voxelization failed: {e2}")
            
            # Method 3: Manual point-in-mesh test if both previous methods failed
            if voxel_grid is None:
                try:
                    x = np.linspace(-1, 1, resolution)
                    y = np.linspace(-1, 1, resolution)
                    z = np.linspace(-1, 1, resolution)
                    xx, yy, zz = np.meshgrid(x, y, z)
                    points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
                    
                    inside = mesh.contains(points)
                    voxel_grid = inside.reshape((resolution, resolution, resolution))
                except Exception as e3:
                    print(f"Manual voxelization failed: {e3}")
            
            # If all methods failed, create a simple sphere as fallback
            if voxel_grid is None or np.sum(voxel_grid) == 0:
                print(f"All voxelization methods failed for {stl_path}. Creating fallback sphere.")
                x = np.linspace(-1, 1, resolution)
                y = np.linspace(-1, 1, resolution)
                z = np.linspace(-1, 1, resolution)
                xx, yy, zz = np.meshgrid(x, y, z)
                
                # Create a sphere
                dist = np.sqrt(xx**2 + yy**2 + zz**2)
                voxel_grid = dist <= 0.5
            
            # Ensure correct dimensions
            if voxel_grid.shape != (resolution, resolution, resolution):
                try:
                    scale_factors = (
                        resolution / voxel_grid.shape[0],
                        resolution / voxel_grid.shape[1],
                        resolution / voxel_grid.shape[2]
                    )
                    voxel_grid = zoom(voxel_grid, scale_factors, order=0, mode='nearest')
                except Exception as e4:
                    print(f"Error resizing voxel grid: {e4}")
                    
                    # Create a new grid and copy what fits
                    new_grid = np.zeros((resolution, resolution, resolution), dtype=bool)
                    min_x = min(voxel_grid.shape[0], resolution)
                    min_y = min(voxel_grid.shape[1], resolution)
                    min_z = min(voxel_grid.shape[2], resolution)
                    new_grid[:min_x, :min_y, :min_z] = voxel_grid[:min_x, :min_y, :min_z]
                    voxel_grid = new_grid
            
            # Apply post-processing for better quality if needed
            if np.sum(voxel_grid) < resolution:
                voxel_grid = binary_closing(voxel_grid)
                
                if np.sum(voxel_grid) < resolution:
                    voxel_grid = binary_dilation(voxel_grid)
            
            return voxel_grid
            
        except Exception as e:
            print(f"Error in stl_to_voxels for {stl_path}: {e}")
            traceback.print_exc()
            return np.zeros((resolution, resolution, resolution), dtype=bool)


    def get_all_folds(processed_dir="data/processed"):
        """
        returns a list of all fold numbers found in the processed directory

        Args:
            processed_dir (str): Path to the processed images directory.

        Returns:
            list: List of fold numbers found in the processed directory.
        """
        processed_dir = os.path.normpath(processed_dir)
        folds = []

        if os.path.exists(processed_dir):
            for item in os.listdir(processed_dir):
                if os.path.isdir(os.path.join(processed_dir, item)) and item.startswith(
                    "fold_"
                ):
                    try:
                        fold_num = int(item.split("_")[1])
                        folds.append(fold_num)
                    except (IndexError, ValueError):
                        continue

        return sorted(folds)
