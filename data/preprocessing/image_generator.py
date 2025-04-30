import logging
import os
import csv
import json
from typing import Tuple, List, Optional

from tqdm import tqdm
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import vtk_to_numpy

logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(
        self,
        raw_mesh_dir: str = 'raw',
        output_dir: str = 'processed',
        random_seed: int = 1337,
        view_angle: float = 30.0
    ):
        """
        Initializes the generator with fixed perspective FOV so all meshes share identical intrinsics.
        """
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.view_angle = view_angle
        self.data_dir = os.getenv("DATA_DIR_PATH")
        if not self.data_dir:
            raise EnvironmentError("DATA_DIR_PATH environment variable not set.")

    def _vtk4x4_to_numpy(self, mat: vtk.vtkMatrix4x4) -> np.ndarray:
        return np.array([[mat.GetElement(r, c) for c in range(4)] for r in range(4)])

    def _camera_dict(
        self, cam: vtk.vtkCamera, renderer: vtk.vtkRenderer
    ) -> dict:
        """
        Records camera intrinsics & extrinsics into a JSON-serializable dict.
        """
        mv = self._vtk4x4_to_numpy(cam.GetModelViewTransformMatrix())
        near_, far_ = cam.GetClippingRange()
        aspect = renderer.GetAspect()[0]
        proj = self._vtk4x4_to_numpy(cam.GetProjectionTransformMatrix(aspect, near_, far_))
        return {
            "position": cam.GetPosition(),
            "focal_point": cam.GetFocalPoint(),
            "view_up": cam.GetViewUp(),
            "clipping_range": [near_, far_],
            "modelview": mv.tolist(),
            "projection": proj.tolist(),
            "proj_modelview": (proj @ mv).tolist()
        }

    def _smooth_mesh(
        self,
        reader: vtk.vtkSTLReader,
        smoothing_iterations: int = 30,
        relaxation_factor: float = 0.1
    ) -> vtk.vtkSmoothPolyDataFilter:
        """
        Applies Laplacian smoothing. Raises if no valid input polydata.
        """
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Mesh has no points to smooth.")
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(reader.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.SetRelaxationFactor(relaxation_factor)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOff()
        smoother.Update()
        return smoother

    def _setup_renderers(
        self,
        actor1: vtk.vtkActor,
        actor2: vtk.vtkActor,
        center: Tuple[float, float, float],
        distance: float,
        max_dim: float
    ) -> Tuple[Tuple[vtk.vtkRenderer, dict], Tuple[vtk.vtkRenderer, dict]]:
        """
        Creates two side-by-side renderers with fixed-view perspective cameras.
        """
        # common background
        r1, r2 = vtk.vtkRenderer(), vtk.vtkRenderer()
        for r, vp in [(r1, (0,0,0.5,1)), (r2, (0.5,0,1,1))]:
            r.SetViewport(*vp)
            r.SetBackground(0,0,0)
            r.AddActor(actor1 if r is r1 else actor2)

        # perspective front
        cam1 = vtk.vtkCamera()
        cam1.SetFocalPoint(center)
        cam1.SetPosition(center[0], center[1], center[2] + distance)
        cam1.SetViewUp(0,1,0)
        cam1.ParallelProjectionOff()
        cam1.SetViewAngle(self.view_angle)
        r1.SetActiveCamera(cam1)

        # perspective side
        cam2 = vtk.vtkCamera()
        cam2.SetFocalPoint(center)
        cam2.SetPosition(center[0] + distance, center[1], center[2])
        cam2.SetViewUp(0,1,0)
        cam2.ParallelProjectionOff()
        cam2.SetViewAngle(self.view_angle)
        r2.SetActiveCamera(cam2)

        # shared light
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(1,1,1)
        light.SetIntensity(1.0)
        r1.AddLight(light)
        r2.AddLight(light)

        return ( (r1, self._camera_dict(cam1, r1)),
                 (r2, self._camera_dict(cam2, r2)) )

    def _render_orthogonal_views(
        self,
        mesh_file_path: str,
        rotation: Tuple[float, float, float] = (0, 0, 0)
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Loads a mesh, smooths it, and renders two 90° apart perspective images.
        Returns gray numpy arrays and metadata.
        """
        if not os.path.exists(mesh_file_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_file_path}")

        # read STL robustly
        reader = vtk.vtkSTLReader()
        reader.SetFileName(mesh_file_path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            # skip invalid mesh
            logger.error(f"Invalid or empty STL: {mesh_file_path}")
            raise RuntimeError("Empty mesh, skipping render.")

        # smooth mesh
        smoother = self._smooth_mesh(reader)

        # mapper & actors
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        actors = []
        for m in (mapper, mapper):
            actor = vtk.vtkActor()
            actor.SetMapper(m)
            actor.GetProperty().SetColor(1,1,1)
            actor.GetProperty().SetAmbient(0.3)
            actor.GetProperty().SetDiffuse(0.7)
            actor.GetProperty().SetSpecular(0.5)
            actor.GetProperty().SetSpecularPower(50)
            actors.append(actor)

        # compute view parameters
        bounds = smoother.GetOutput().GetBounds()
        center = smoother.GetOutput().GetCenter()
        max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        distance = max_dim * 2.5 if max_dim>0 else 100

        for actor in actors:
            actor.SetOrigin(center)
            actor.SetOrientation(rotation)

        (r1_data, r2_data) = self._setup_renderers(
            actors[0], actors[1], center, distance, max_dim
        )
        r1, cam1_meta = r1_data
        r2, cam2_meta = r2_data

        # offscreen render
        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.SetSize(2048,1024)
        rw.AddRenderer(r1)
        rw.AddRenderer(r2)
        r1.ResetCameraClippingRange()
        r2.ResetCameraClippingRange()
        rw.Render()

        # capture
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.Update()
        img = w2i.GetOutput()
        arr = vtk_to_numpy(img.GetPointData().GetScalars())
        h,w,_ = img.GetDimensions()[1], img.GetDimensions()[0], img.GetPointData().GetScalars().GetNumberOfComponents()
        arr = arr.reshape(h, w, -1)[::-1]  # flip y
        gray = np.array(Image.fromarray(arr.astype(np.uint8)).convert("L"))

        half = gray.shape[1]//2
        left, right = gray[:, :half], gray[:, half:]

        meta = {
            "rotation_deg": list(rotation),
            "center": list(center),
            "distance": distance,
            "image_height": gray.shape[0],
            "image_width": half,
            "camera_front": cam1_meta,
            "camera_side": cam2_meta
        }
        return left, right, meta

    def _concatenate_images(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.concatenate((left, right), axis=1)

    def _get_missing_files(self, files: List[str]) -> List[str]:
        """Lists which meshes don't yet have rendered images."""
        images_dir = os.path.join(self.data_dir, self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        existing = {os.path.splitext(f)[0] for f in os.listdir(images_dir)}
        return [f for f in files if os.path.splitext(f)[0] not in existing]

    def process(self, files: List[str]) -> None:
        """
        Render missing meshes and write out images + camera metadata.
        Skips invalid meshes without halting the pipeline.
        """
        np.random.seed(self.random_seed)
        images_dir = os.path.join(self.data_dir, self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        csv_path = os.path.join(self.data_dir, self.output_dir, "rotations.csv")

        # load existing rotations
        rotations = {}
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as cf:
                for row in csv.DictReader(cf):
                    rotations[row['sample']] = (
                        float(row['rot_x']), float(row['rot_y']), float(row['rot_z'])
                    )

        missing = self._get_missing_files(files)
        for fname in tqdm(missing, desc="Rendering orthogonal views"):
            mesh_path = os.path.join(self.data_dir, "processed", "interim", fname)
            rot = tuple(np.random.uniform(0,360,3))
            try:
                left, right, meta = self._render_orthogonal_views(mesh_path, rot)
            except Exception as e:
                logger.error(f"Skipping mesh {fname}: {e}")
                continue

            # save images
            img = self._concatenate_images(left, right).astype(np.uint8)
            sample = os.path.splitext(fname)[0]
            Image.fromarray(img).save(os.path.join(images_dir, f"{sample}.png"))
            os.makedirs(os.path.join(images_dir, "metadata"), exist_ok=True)
            with open(os.path.join(images_dir, "metadata", f"{sample}_cam.json"), 'w') as j:
                json.dump(meta, j, indent=2)

            rotations[sample] = rot

        # write rotations CSV
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(["sample","rot_x","rot_y","rot_z"])
            for s,(x,y,z) in rotations.items():
                writer.writerow([s,x,y,z])
        logger.info("Image generation complete.")
