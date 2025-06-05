import logging
import os
import csv
import json
from typing import Tuple, List

from tqdm import tqdm
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import vtk_to_numpy

logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self, raw_mesh_dir: str = 'raw', output_dir: str = 'processed', random_seed: int = 1337):
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.data_dir = os.getenv("DATA_DIR_PATH")
        if not self.data_dir:
            raise EnvironmentError("DATA_DIR_PATH environment variable not set.")
        
    def _vtk4x4_to_numpy(self, mat: vtk.vtkMatrix4x4) -> np.ndarray:
        return np.array([[mat.GetElement(r, c) for c in range(4)] for r in range(4)])

    def _camera_dict(self, cam: vtk.vtkCamera, renderer: vtk.vtkRenderer) -> dict:
        # --- model‑view --------------------------------------------------------
        mv_vtk = cam.GetModelViewTransformMatrix()              # ★ no arg
        mv_np  = self._vtk4x4_to_numpy(mv_vtk)                       # ★

        # --- projection -------------------------------------------------------
        near_, far_ = cam.GetClippingRange()
        aspect      = renderer.GetAspect()[0]                   # renderer’s X/Y
        proj_vtk = cam.GetProjectionTransformMatrix(aspect, near_, far_)  # ★ no out‑mat
        proj_np  = self._vtk4x4_to_numpy(proj_vtk)                   # ★

        return {
            "position"        : cam.GetPosition(),
            "focal_point"     : cam.GetFocalPoint(),
            "view_up"         : cam.GetViewUp(),
            "parallel_scale"  : cam.GetParallelScale(),
            "clipping_range"  : [near_, far_],
            "modelview"       : mv_np.tolist(),
            "projection"      : proj_np.tolist(),
            "proj_modelview"  : (proj_np @ mv_np).tolist()
        }

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

    def _setup_renderers(
            self, actor1: vtk.vtkActor, actor2: vtk.vtkActor,
            center: Tuple[float, float, float], distance: float, max_dim: float
        ) -> Tuple[Tuple[vtk.vtkRenderer, dict], Tuple[vtk.vtkRenderer, dict]]:

        renderer1, renderer2 = vtk.vtkRenderer(), vtk.vtkRenderer()
        renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
        renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)
        for r in (renderer1, renderer2):
            r.SetBackground(0, 0, 0)

        renderer1.AddActor(actor1)
        renderer2.AddActor(actor2)

        scale = max_dim * 0.6

        # -------- camera 1 (front) ------------------------------------------
        cam1 = vtk.vtkCamera()
        cam1.SetFocalPoint(center)
        cam1.SetPosition(center[0], center[1], center[2] + distance)
        cam1.SetViewUp(0, 1, 0)
        cam1.ParallelProjectionOn()
        cam1.SetParallelScale(scale)

        renderer1.SetActiveCamera(cam1)    # no ResetCamera()

        # -------- camera 2 (side) -------------------------------------------
        cam2 = vtk.vtkCamera()
        cam2.SetFocalPoint(center)
        cam2.SetPosition(center[0] + distance, center[1], center[2])
        cam2.SetViewUp(0, 1, 0)
        cam2.ParallelProjectionOn()
        cam2.SetParallelScale(scale)

        renderer2.SetActiveCamera(cam2)


        # --------------------------- shared light ------------------------------
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(1, 1, 1)
        light.SetFocalPoint(0, 0, 0)
        light.SetColor(1, 1, 1)
        light.SetIntensity(1.0)
        renderer1.AddLight(light)
        renderer2.AddLight(light)

        # return renderers and camera dictionaries
        return (
            (renderer1, self._camera_dict(cam1, renderer1)),
            (renderer2, self._camera_dict(cam2, renderer2))
        )

    def _render_orthogonal_views(
        self, mesh_file_path: str, rotation: Tuple[float, float, float] = (0, 0, 0)
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
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

        (renderer1, cam1_dict), (renderer2, cam2_dict) = self._setup_renderers(actor1, actor2, center, distance, max_dim)

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
        
        meta = {
            "rotation_deg"  : list(rotation),
            "center"        : list(center),
            "distance"      : distance,
            "image_height"  : height,          # == half of render window's height
            "image_width"   : half_width,      # whole window / 2
            "camera_front"  : cam1_dict,
            "camera_side"   : cam2_dict
        }
        return left_view, right_view, meta


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

    def process(self, files: List[str], mesh_path: str = None) -> None:
        """
        For each missing mesh file, generate a concatenated image of two orthogonal views.
        Also, update the rotation records in a CSV file so that only new images get a new rotation
        and any outdated rotations for missing images are replaced.
        """
        np.random.seed(self.random_seed)
        
        images_dir = os.path.join(self.data_dir, self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        csv_path = os.path.join(self.data_dir, self.output_dir, "rotations.csv")
        rotations_dict = {}
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r", newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        sample = row["sample"]
                        rotations_dict[sample] = (float(row["rot_x"]), float(row["rot_y"]), float(row["rot_z"]))
            except Exception as e:
                logger.error(f"Failed to load existing rotation CSV file: {e}")

        missing_files = self._get_missing_files(files)
        if missing_files:
            logger.info(f"Found {len(missing_files)} out of {len(files)} files to generate images for.")
            for file in tqdm(missing_files, desc="Generating orthogonal image pairs"):
                curr_mesh_path = os.path.join(mesh_path, file)
                rotation = tuple(np.random.uniform(0, 360, 3))
                try:
                    left_view, right_view, meta = self._render_orthogonal_views(curr_mesh_path, rotation)
                except Exception as e:
                    logger.error(f"Failed to render images for {file}: {e}")
                    break

                concatenated = self._concatenate_images(left_view, right_view)
                sample_name = os.path.splitext(file)[0]
                image_filename = f"{sample_name}.png"
                image_path = os.path.join(images_dir, image_filename)
                
                os.makedirs(os.path.join(images_dir, "metadata"), exist_ok=True)
                meta_path = os.path.join(images_dir, "metadata", f"{sample_name}_cam.json")
                with open(meta_path, "w") as jf:
                    json.dump(meta, jf, indent=2)
                                
                try:
                    Image.fromarray(np.uint8(concatenated)).save(image_path)
                except Exception as e:
                    logger.error(f"Failed to save image for {file}: {e}")
                    break

                rotations_dict[sample_name] = rotation
        
            try:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["sample", "rot_x", "rot_y", "rot_z"])
                    for sample, rotation in rotations_dict.items():
                        writer.writerow([sample, rotation[0], rotation[1], rotation[2]])
                logger.info(f"Rotation data saved to {csv_path}.")
            except Exception as e:
                logger.error(f"Failed to save rotation CSV file: {e}")
        else:
            logger.info("Images have already been generated.")
