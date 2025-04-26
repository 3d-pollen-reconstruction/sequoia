import logging
import os
import json
from pathlib import Path

from tqdm import tqdm
import pyvista as pv
import trimesh
import torch
from trimesh.transformations import euler_matrix, translation_matrix, scale_matrix
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoxelGenerator:
    def __init__(self, raw_mesh_dir='raw', output_dir='processed'):
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir

    def _get_missing_files(self, files: list = None):
        folder_files = os.listdir(os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "voxels"))
        missing_meshes = set(files) - set(folder_files)
        return list(missing_meshes)
    
    def _mesh_to_voxel_tensor(self,
                         mesh: trimesh.Trimesh,
                         meta,
                         res: int = 128,
                         fill: bool = True,
                         device: torch.device = torch.device('cpu')) -> torch.BoolTensor:
        """
        Load a mesh + its metadata, voxelize at resolution `res`,
        and return a BoolTensor of shape (res, res, res) on `device`.
        """
        # converting from world → voxel units
        scale_cam = meta["camera_front"]["parallel_scale"]
        S = scale_matrix((res / 2) / scale_cam)
        O = translation_matrix([res/2]*3)
        mesh.apply_transform(O @ S)

        # voxelize and fill interior
        vox = mesh.voxelized(pitch=1, method="subdivide")
        if fill:
            vox = vox.fill()

        # build dense occupancy grid
        idx = vox.sparse_indices  # (n,3) in X,Y,Z order
        # reorder to Y,X,Z → index as occ[y,x,z]
        idx = idx[:, (1,0,2)]
        occ = np.zeros((res, res, res), dtype=bool)
        mask = np.all((idx >= 0) & (idx < res), axis=1)
        occ[tuple(idx[mask].T)] = True

        return torch.from_numpy(occ).to(torch.bool).to(device)

    def process(self, files: list = None):
        voxels_dir = os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "voxels")
        os.makedirs(voxels_dir, exist_ok=True)
        
        missing_files = self._get_missing_files(files)
        
        if len(missing_files) != 0:
            logger.info(f"Found {len(missing_files)} of {len(files)} meshes to turn into voxels.")
            for file in tqdm(missing_files, desc="Voxelizing Meshes"):
                mesh = trimesh.load(os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "meshes", file))
                metadata = json.loads(Path(os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "images", "metadata", file.replace(".stl", "_cam.json"))).read_text())
                
                voxel = self._mesh_to_voxel_tensor(mesh, metadata, res=128, fill=True, device=torch.device('cpu'))
                torch.save(voxel, os.path.join(voxels_dir, file.replace(".stl", ".pt")))
        else:
            logger.info("Meshes have already been turned into Voxels.")