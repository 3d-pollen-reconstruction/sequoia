import logging
import os
from typing import Tuple, List

from tqdm import tqdm
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import vtk_to_numpy

logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self, raw_mesh_dir: str = 'raw', output_dir: str = 'processed'):
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir
        self.data_dir = os.getenv("DATA_DIR_PATH")
        if not self.data_dir:
            raise EnvironmentError("DATA_DIR_PATH environment variable not set.")

    def _smooth_mesh(self, reader: vtk.vtkSTLReader, smoothing_iterations: int = 30, relaxation_factor: float = 0.1) -> vtk.vtkSmoothPolyDataFilter:
        """
        Smooths a mesh using VTK's vtkSmoothPolyDataFilter.
        """
        smooth_filter = vtk.vtkSmoothPolyDataFilter()
        smooth_filter.SetInputConnection(reader.GetOutputPort())
        smooth_filter.SetNumberOfIterations(smoothing_iterations)
        smooth_filter.SetRelaxationFactor(relaxation_factor)
        smooth_filter.FeatureEdgeSmoothingOff()
        smooth_filter.BoundarySmoothingOff()
        smooth_filter.Update()
        return smooth_filter

    def _setup_renderers(self, actor1: vtk.vtkActor, actor2: vtk.vtkActor, center: Tuple[float, float, float], distance: float) -> Tuple[vtk.vtkRenderer, vtk.vtkRenderer]:
        """
        Sets up two renderers for orthogonal views with cameras and shared lighting.
        """
        renderer1 = vtk.vtkRenderer()
        renderer2 = vtk.vtkRenderer()
        renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
        renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)
        renderer1.SetBackground(0, 0, 0)
        renderer2.SetBackground(0, 0, 0)
        renderer1.AddActor(actor1)
        renderer2.AddActor(actor2)

        # Setup cameras
        camera1 = vtk.vtkCamera()
        camera1.SetFocalPoint(center)
        camera1.SetPosition(center[0], center[1], center[2] + distance)
        camera1.SetViewUp(0, 1, 0)
        renderer1.SetActiveCamera(camera1)

        camera2 = vtk.vtkCamera()
        camera2.SetFocalPoint(center)
        camera2.SetPosition(center[0] + distance, center[1], center[2])
        camera2.SetViewUp(0, 1, 0)
        renderer2.SetActiveCamera(camera2)

        # Add shared lighting
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(1, 1, 1)
        light.SetFocalPoint(0, 0, 0)
        light.SetColor(1, 1, 1)
        light.SetIntensity(1.0)
        renderer1.AddLight(light)
        renderer2.AddLight(light)

        return renderer1, renderer2

    def _render_orthogonal_views(self, mesh_file_path: str, rotation: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Renders two orthogonal views of a mesh file.
        Returns two numpy arrays for the left and right views.
        """
        if not os.path.exists(mesh_file_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file_path}")

        reader = vtk.vtkSTLReader()
        reader.SetFileName(mesh_file_path)
        reader.Update()

        smooth_filter = self._smooth_mesh(reader)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smooth_filter.GetOutputPort())

        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper)
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper)
        for actor in (actor1, actor2):
            actor.GetProperty().SetColor(1, 1, 1)
            actor.GetProperty().SetAmbient(0.3)
            actor.GetProperty().SetDiffuse(0.7)
            actor.GetProperty().SetSpecular(0.5)
            actor.GetProperty().SetSpecularPower(50)

        # Center and compute view distance
        polydata = smooth_filter.GetOutput()
        bounds = polydata.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
        center = polydata.GetCenter()
        max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        distance = max_dim * 2.5 if max_dim > 0 else 100

        # Apply rotation and set actor origins
        for actor in (actor1, actor2):
            actor.SetOrigin(center)
            actor.SetOrientation(rotation)

        renderer1, renderer2 = self._setup_renderers(actor1, actor2, center, distance)

        # Setup offscreen render window
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(2048, 1024)
        render_window.SetOffScreenRendering(1)
        render_window.AddRenderer(renderer1)
        render_window.AddRenderer(renderer2)
        renderer1.ResetCameraClippingRange()
        renderer2.ResetCameraClippingRange()
        render_window.Render()

        # Capture the image
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(render_window)
        w2if.Update()
        vtk_image = w2if.GetOutput()
        dims = vtk_image.GetDimensions()
        num_comp = vtk_image.GetPointData().GetScalars().GetNumberOfComponents()
        vtk_array = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        height, width = dims[1], dims[0]
        arr = vtk_array.reshape(height, width, num_comp)
        arr = np.flipud(arr)

        gray_image = Image.fromarray(arr.astype(np.uint8)).convert("L")
        arr = np.array(gray_image)

        half_width = width // 2
        left_view = arr[:, :half_width]
        right_view = arr[:, half_width:]
        return left_view, right_view

    def _concatenate_images(self, left_view: np.ndarray, right_view: np.ndarray) -> np.ndarray:
        """
        Concatenates left and right views horizontally.
        """
        return np.concatenate((left_view, right_view), axis=1)

    def _get_missing_files(self, files: List[str]) -> List[str]:
        """
        Determines which meshes (by base name) do not have corresponding images.
        Returns a list of missing mesh filenames with a .stl extension.
        """
        images_dir = os.path.join(self.data_dir, self.output_dir, "images")
        if os.path.exists(images_dir):
            folder_files = os.listdir(images_dir)
        else:
            folder_files = []
        folder_base_names = {os.path.splitext(f)[0] for f in folder_files}
        base_names = {os.path.splitext(f)[0] for f in files}
        missing_base_names = base_names - folder_base_names
        return [f"{base}.stl" for base in missing_base_names]

    def process(self, files: List[str]) -> None:
        """
        For each missing mesh file, generate a concatenated image of two orthogonal views.
        """
        images_dir = os.path.join(self.data_dir, self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        missing_files = self._get_missing_files(files)
        if missing_files:
            logger.info(f"Found {len(missing_files)} out of {len(files)} files to generate images for.")
            for file in tqdm(missing_files, desc="Generating orthogonal image pairs"):
                mesh_path = os.path.join(self.data_dir, self.raw_mesh_dir, file)
                rotation = tuple(np.random.uniform(0, 360, 3))
                try:
                    left_view, right_view = self._render_orthogonal_views(mesh_path, rotation)
                except Exception as e:
                    logger.error(f"Failed to render images for {file}: {e}")
                    continue

                concatenated = self._concatenate_images(left_view, right_view)
                sample_name = os.path.splitext(file)[0]
                image_filename = f"{sample_name}.png"
                image_path = os.path.join(images_dir, image_filename)
                try:
                    Image.fromarray(np.uint8(concatenated)).save(image_path)
                except Exception as e:
                    logger.error(f"Failed to save image for {file}: {e}")
        else:
            logger.info("Images have already been generated.")
