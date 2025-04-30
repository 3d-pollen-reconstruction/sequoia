import logging
import os
import numpy as np
import torch
from tqdm import tqdm
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointCloudGenerator:
    def __init__(self, input_mesh_dir='processed/meshes', output_dir='processed'):
        self.input_mesh_dir = input_mesh_dir
        self.output_dir = output_dir

    def _get_missing_files(self, files: list = None):
        """
        Determine which mesh files have not yet been converted to point clouds.
        """
        processed_folder = os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "pointclouds")
        if not os.path.exists(processed_folder):
            return files
        folder_files = os.listdir(processed_folder)
        processed_bases = {os.path.splitext(f)[0] for f in folder_files}
        return [f for f in files if os.path.splitext(f)[0] not in processed_bases]

    def process(self, files: list = None, num_points: int = 4000):
        """
        Process mesh files into point clouds with normals.
        Saves both a NumPy .npz and a PyTorch .pt file containing:
          - points:   (num_points, 3)
          - normals:  (num_points, 3)
        """
        output_folder = os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "pointclouds")
        os.makedirs(output_folder, exist_ok=True)

        # If no file list provided, use all meshes in input directory
        if files is None:
            raw_dir = os.path.join(os.getenv("DATA_DIR_PATH"), self.input_mesh_dir)
            files = os.listdir(raw_dir)

        missing_files = self._get_missing_files(files)

        if missing_files:
            logger.info(f"Found {len(missing_files)} of {len(files)} meshes to turn into point clouds.")
            for file in tqdm(missing_files, desc="Generating Pointclouds"):
                mesh_path = os.path.join(os.getenv("DATA_DIR_PATH"), self.input_mesh_dir, file)
                try:
                    # Load mesh
                    mesh = trimesh.load(mesh_path, force='mesh')
                    if not isinstance(mesh, trimesh.Trimesh):
                        logger.error(f"File {file} is not a valid mesh. Skipping.")
                        continue

                    # Sample points and compute normals
                    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
                    normals = mesh.face_normals[face_indices]

                    base_name = os.path.splitext(file)[0]

                    # Save as NumPy .npz
                    output_npz = os.path.join(output_folder, base_name + ".npz")
                    np.savez_compressed(output_npz, points=points, normals=normals)

                    # Save as PyTorch tensor .pt
                    tensor_dict = {
                        'points': torch.from_numpy(points).float(),
                        'normals': torch.from_numpy(normals).float()
                    }
                    output_pt = os.path.join(output_folder, base_name + ".pt")
                    torch.save(tensor_dict, output_pt)

                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")
        else:
            logger.info("Meshes have already been turned into point clouds.")
