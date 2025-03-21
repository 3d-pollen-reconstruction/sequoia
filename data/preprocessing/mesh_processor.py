import logging
import os

from tqdm import tqdm
import fast_simplification
import pyvista as pv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeshProcessor:
    def __init__(self, raw_mesh_dir='raw', output_dir='processed'):
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir

    def _simplify_mesh(self, mesh, n_target_faces=2000):
        out = fast_simplification.simplify_mesh(mesh, target_count=n_target_faces)
        return out

    def _get_missing_files(self, files: list = None):
        folder_files = os.listdir(os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "meshes"))
        missing_meshes = set(files) - set(folder_files)
        return list(missing_meshes)

    def process(self, files: list = None):
        meshes_dir = os.path.join(os.getenv("DATA_DIR_PATH"), self.output_dir, "meshes")
        os.makedirs(meshes_dir, exist_ok=True)
        
        missing_files = self._get_missing_files(files)
        
        if len(missing_files) != 0:
            logger.info(f"Found {len(missing_files)} of {len(files)} files to simplify.")
            for file in tqdm(missing_files, desc="Simplifying meshes"):
                mesh = pv.read(os.path.join(os.getenv("DATA_DIR_PATH"), self.raw_mesh_dir, file))
                simplified_mesh = self._simplify_mesh(mesh)
                
                mesh_path = os.path.join(meshes_dir, file)
                
                pv.save_meshio(mesh=simplified_mesh, filename=mesh_path)
        else:
            logger.info("Meshes have already been simplified.")