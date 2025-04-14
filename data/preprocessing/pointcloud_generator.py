import logging
import os
import numpy as np
from tqdm import tqdm
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointCloudGenerator:
    def __init__(self, input_mesh_dir='processed/meshes', output_dir='processed'):
        self.input_mesh_dir = input_mesh_dir
        self.output_dir = output_dir

    def _get_missing_files(self, files: list = None):
        # List all files already processed in the output pointclouds folder.
        processed_folder = os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "pointclouds")
        if not os.path.exists(processed_folder):
            return files
        folder_files = os.listdir(processed_folder)
        # Assume output files use the same base name but with a .npz extension.
        processed_bases = {os.path.splitext(f)[0] for f in folder_files}
        missing_meshes = [f for f in files if os.path.splitext(f)[0] not in processed_bases]
        print(f'missing meshes: {missing_meshes}')
        return missing_meshes

    def process(self, files: list = None, num_points: int = 4000):
        """
        Process a list of mesh files: each mesh is loaded with trimesh and uniformly sampled
        to produce a point cloud with normals. The result is saved to the output folder
        as a compressed .npz file containing the points and normals.
        """
        # Set up output directory for point clouds.
        output_folder = os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "pointclouds")
        os.makedirs(output_folder, exist_ok=True)
        
        if files is None:
            # If no specific file list is provided, use all files in the raw directory.
            raw_dir = os.path.join(os.getenv("DATA_DIR_PATH"), self.input_mesh_dir)
            files = os.listdir(raw_dir)
        
        missing_files = self._get_missing_files(files)
        
        if len(missing_files) != 0:
            logger.info(f"Found {len(missing_files)} of {len(files)} meshes to turn into point clouds.")
            for file in tqdm(missing_files, desc="Generating Pointclouds"):
                mesh_path = os.path.join(os.getenv("DATA_DIR_PATH"), self.input_mesh_dir, file)
                try:
                    # Load the mesh using trimesh (forcing the load as a mesh)
                    mesh = trimesh.load(mesh_path, force='mesh')
                    if not isinstance(mesh, trimesh.Trimesh):
                        logger.error(f"File {file} is not a valid mesh. Skipping.")
                        continue

                    # Uniformly sample the surface to get points and the corresponding face indices.
                    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
                    # Compute normals per sampled point from the face normals.
                    normals = mesh.face_normals[face_indices]
                    
                    # Create the output filename using a .npz extension.
                    base_name = os.path.splitext(file)[0]
                    output_file = os.path.join(output_folder, base_name + ".npz")
                    
                    # Save the point cloud as a compressed file with points and normals.
                    np.savez_compressed(output_file, points=points, normals=normals)
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")
        else:
            logger.info("Meshes have already been turned into point clouds.")
