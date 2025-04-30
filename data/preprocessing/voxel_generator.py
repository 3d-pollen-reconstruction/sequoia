import logging
import os
import json
from pathlib import Path

from tqdm import tqdm
import trimesh
import torch
import numpy as np
from trimesh.transformations import euler_matrix, translation_matrix, scale_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoxelGenerator:
    def __init__(self, raw_mesh_dir='raw', output_dir='processed'):
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir

    def _get_missing_files(self, files: list = None):
        voxels_folder = os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "voxels")
        if not os.path.exists(voxels_folder):
            return files or []
        existing = {os.path.splitext(f)[0] for f in os.listdir(voxels_folder)}
        return [f for f in (files or []) if os.path.splitext(f)[0] not in existing]

    def _mesh_to_voxel_tensor(
        self,
        mesh: trimesh.Trimesh,
        res: int = 128,
        fill: bool = True,
        device: torch.device = torch.device('cpu')
    ) -> torch.BoolTensor:
        """
        Voxelize a mesh into a (res x res x res) boolean occupancy grid.
        Uses mesh bounding-box to compute a consistent scale and centering transform,
        independent of camera parallel_scale.
        """
        # Compute bounding box and center
        bbox_min, bbox_max = mesh.bounds
        center = (bbox_min + bbox_max) * 0.5
        max_dim = np.max(bbox_max - bbox_min)
        if max_dim <= 0:
            raise RuntimeError("Mesh has zero size bounding-box")

        # Build transform: translate to origin, scale to fit [0, res-1], then center in voxel grid
        T_center_to_origin = translation_matrix(-center)
        scale_factor = (res - 1) / max_dim
        S = scale_matrix(scale_factor)
        T_to_grid_center = translation_matrix([res/2.0, res/2.0, res/2.0])
        transform = T_to_grid_center @ S @ T_center_to_origin
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(transform)

        # Voxelize at unit pitch
        vox = mesh_copy.voxelized(pitch=1.0, method='subdivide')
        if fill:
            vox = vox.fill()

        # Build occupancy grid
        indices = vox.sparse_indices  # (n,3) int
        # trimesh gives (x,y,z) indices: reorder to (y,x,z) for standard indexing
        idx = indices[:, (1, 0, 2)]
        occ = np.zeros((res, res, res), dtype=bool)
        mask = np.all((idx >= 0) & (idx < res), axis=1)
        occ[tuple(idx[mask].T)] = True

        return torch.from_numpy(occ).to(torch.bool).to(device)

    def process(self, files: list = None):
        """
        For each mesh in files, voxelize and save a .pt boolean tensor at resolution 128.
        """
        voxels_dir = os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "voxels")
        os.makedirs(voxels_dir, exist_ok=True)

        if files is None:
            raw_dir = os.path.join(os.getenv("DATA_DIR_PATH"), self.raw_mesh_dir)
            files = os.listdir(raw_dir)

        missing_files = self._get_missing_files(files)
        if missing_files:
            logger.info(f"Voxelizing {len(missing_files)}/{len(files)} meshes...")
            for fname in tqdm(missing_files, desc="Voxelizing Meshes"):
                try:
                    mesh_path = os.path.join(
                        os.getenv("DATA_DIR_PATH"), self.output_dir, "meshes", fname
                    )
                    mesh = trimesh.load(mesh_path, force='mesh')
                    if not isinstance(mesh, trimesh.Trimesh):
                        logger.error(f"{fname} is not a valid mesh. Skipping.")
                        continue

                    voxel_tensor = self._mesh_to_voxel_tensor(mesh, res=128, fill=True)
                    out_path = os.path.join(voxels_dir, fname.replace('.stl', '.pt'))
                    torch.save(voxel_tensor, out_path)
                except Exception as e:
                    logger.error(f"Failed to voxelize {fname}: {e}")
        else:
            logger.info("All meshes have been voxelized already.")
