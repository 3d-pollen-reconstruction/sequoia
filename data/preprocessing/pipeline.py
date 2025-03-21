import os
import logging

from image_generator import ImageGenerator
from mesh_processor import MeshProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, raw_mesh_dir='raw', output_dir='processed'):
        self.mesh_processor = MeshProcessor(raw_mesh_dir=raw_mesh_dir, output_dir=output_dir)
        self.image_generator = ImageGenerator(raw_mesh_dir=raw_mesh_dir, output_dir=output_dir)
        self.raw_mesh_dir = raw_mesh_dir
        self.raw_meshes = None

    def _load_meshes(self):
        files = os.path.join(os.getenv("DATA_DIR_PATH"), self.raw_mesh_dir)
        self.raw_meshes = [f for f in os.listdir(files) if f.lower().endswith('.stl')]
        if self.raw_meshes is None:
            logger.error(f"No .stl files found in {files}")
        logger.info(f"Found {len(self.raw_meshes)} meshes in {files}")

    def run(self):
        self._load_meshes()
        self.image_generator.process(self.raw_meshes)
        self.mesh_processor.process(self.raw_meshes)