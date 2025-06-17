import torch
import numpy as np
from skimage import measure
import trimesh
import os

# === CONFIGURATION ===
scene_path = r"C:\Users\super\Downloads\apple_000_c2.pt"
save_path = r"./apple_000_c2_mesh.ply"
iso_level = 0.0  # TSDF iso-surface level
data = torch.load(scene_path, map_location="cpu")

print("Top-level keys in the .pt file:")
for key in data:
    print("-", key)
    
# === LOAD SCENE ===
data = torch.load(scene_path, map_location="cpu")
tsdf = data['tsdf']  # (D, H, W) torch tensor
voxel_size = data['voxel_size']
origin = data['origin'].numpy()

tsdf_np = tsdf.numpy()

# === MARCHING CUBES ===
verts, faces, normals, _ = measure.marching_cubes(tsdf_np, level=iso_level)

# Convert from voxel index to world coordinates
verts = verts * voxel_size + origin

# === SAVE MESH ===
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
mesh.export(save_path)
print(f"[✓] Saved mesh to: {save_path}")
