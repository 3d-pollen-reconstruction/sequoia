import logging
import os

from tqdm import tqdm
import fast_simplification
import pyvista as pv
import trimesh
import numpy as np
import pymeshlab
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeshProcessor:
    def __init__(self, raw_mesh_dir='raw', output_dir='processed'):
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir

    def _simplify_mesh(self, mesh, n_target_faces=2000):
        out = fast_simplification.simplify_mesh(mesh, target_count=n_target_faces)
        return out
    
    def _make_watertight(self, mesh):
        center = mesh.centroid

        face_centers = mesh.triangles_center
        face_normals = mesh.face_normals

        # calculate vectors from face centers to object center
        vectors_to_center = center - face_centers

        vectors_to_center /= np.linalg.norm(vectors_to_center, axis=1)[:, np.newaxis]

        dot_products = np.einsum("ij,ij->i", face_normals, vectors_to_center)

        # keep faces with normals NOT pointing towards the center (dot product < 0 means facing away)
        keep_faces = dot_products < 0.1

        filtered_mesh = mesh.submesh([keep_faces], append=True)

        filled_mesh = filtered_mesh.copy()
        filled_mesh.fill_holes()

        m = pymeshlab.Mesh(filled_mesh.vertices, filled_mesh.faces)

        ms = pymeshlab.MeshSet()
        ms.add_mesh(m)

        ms.meshing_repair_non_manifold_edges(method='Remove Faces')
        ms.generate_surface_reconstruction_screened_poisson(preclean=True)

        mesh = ms.current_mesh()
        mesh = trimesh.Trimesh(vertices=mesh.vertex_matrix(), faces=mesh.face_matrix())
        
        ms.clear()
        
        return mesh

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
                mesh = trimesh.load_mesh(os.path.join(os.getenv("DATA_DIR_PATH"), self.raw_mesh_dir, file))
                if not mesh.is_watertight:
                    mesh = self._make_watertight(mesh)

                # turning the mesh into a pyvista mesh for simplification
                mesh = pv.wrap(mesh)                
                simplified_mesh = self._simplify_mesh(mesh)
                
                mesh_path = os.path.join(meshes_dir, file)
                
                pv.save_meshio(mesh=simplified_mesh, filename=mesh_path)
        else:
            logger.info("Meshes have already been simplified.")