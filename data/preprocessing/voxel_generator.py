import logging
import os

from tqdm import tqdm
import pyvista as pv
import trimesh

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

    def process(self, files: list = None):
        voxels_dir = os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "voxels")
        os.makedirs(voxels_dir, exist_ok=True)
        
        missing_files = self._get_missing_files(files)
        
        if len(missing_files) != 0:
            logger.info(f"Found {len(missing_files)} of {len(files)} meshes to turn into voxels.")
            for file in tqdm(missing_files, desc="Voxelizing Meshes"):
                mesh = trimesh.load(os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "meshes", file))
                voxel = mesh.voxelize(pitch=1.0)
                voxel.save(os.path.join(voxels_dir, file.replace(".stl", ".npz")))
        else:
            logger.info("Meshes have already been turned into Voxels.")