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
    """Generates a 1 × 8 strip of 224 px view-tiles for every mesh."""

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        raw_mesh_dir: str = "raw",
        output_dir: str = "processed",
        random_seed: int = 1337,
    ):
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.data_dir = os.getenv("DATA_DIR_PATH")
        if not self.data_dir:
            raise EnvironmentError("DATA_DIR_PATH environment variable not set.")

    # --------------------------------------------------------- helper (unchanged)
    def _vtk4x4_to_numpy(self, mat: vtk.vtkMatrix4x4) -> np.ndarray:
        return np.array([[mat.GetElement(r, c) for c in range(4)] for r in range(4)])

    # ------------------------------------------------------------------ cameras
    def _camera_dict(self, cam: vtk.vtkCamera, renderer: vtk.vtkRenderer) -> dict:
        mv_vtk = cam.GetModelViewTransformMatrix()
        mv_np = self._vtk4x4_to_numpy(mv_vtk)

        near_, far_ = cam.GetClippingRange()
        aspect = renderer.GetAspect()[0]
        proj_vtk = cam.GetProjectionTransformMatrix(aspect, near_, far_)
        proj_np = self._vtk4x4_to_numpy(proj_vtk)

        return {
            "position": cam.GetPosition(),
            "focal_point": cam.GetFocalPoint(),
            "view_up": cam.GetViewUp(),
            "parallel_scale": cam.GetParallelScale(),
            "clipping_range": [near_, far_],
            "modelview": mv_np.tolist(),
            "projection": proj_np.tolist(),
            "proj_modelview": (proj_np @ mv_np).tolist(),
        }

    # ------------------------------------------------------------------- smooth
    def _smooth_mesh(
        self,
        reader: vtk.vtkSTLReader,
        smoothing_iterations: int = 30,
        relaxation_factor: float = 0.1,
    ) -> vtk.vtkSmoothPolyDataFilter:
        smooth_filter = vtk.vtkSmoothPolyDataFilter()
        smooth_filter.SetInputConnection(reader.GetOutputPort())
        smooth_filter.SetNumberOfIterations(smoothing_iterations)
        smooth_filter.SetRelaxationFactor(relaxation_factor)
        smooth_filter.FeatureEdgeSmoothingOff()
        smooth_filter.BoundarySmoothingOff()
        smooth_filter.Update()
        return smooth_filter

    # -------------------------------------------------------------- 8-view setup
    def _setup_renderers_8(
        self,
        mapper: vtk.vtkPolyDataMapper,
        center: Tuple[float, float, float],
        distance: float,
        max_dim: float,
        base_az: float,                       # ★ NEW
    ) -> Tuple[List[vtk.vtkRenderer], List[dict]]:

        n_views = 8
        renderers, camera_meta = [], []

        # per-view viewport
        def vp(i: int):
            return (i / n_views, 0.0, (i + 1) / n_views, 1.0)

        # offsets that preserve orthogonality of the first two
        inc = [0, 90, 180, 270, 45, 135, 225, 315]
        azimuths = [(base_az + d) % 360 for d in inc]          # ★ spin applied

        scale = max_dim * 0.6

        for i, az in enumerate(azimuths):
            r = vtk.vtkRenderer()
            r.SetViewport(*vp(i))
            r.SetBackground(1, 1, 1)          # white
            renderers.append(r)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 1, 1)
            actor.GetProperty().SetAmbient(0.3)
            actor.GetProperty().SetDiffuse(0.7)
            actor.GetProperty().SetSpecular(0.5)
            actor.GetProperty().SetSpecularPower(50)
            actor.SetOrigin(center)
            r.AddActor(actor)

            theta = np.radians(az)
            cam = vtk.vtkCamera()
            cam.SetFocalPoint(center)
            cam.SetPosition(
                center[0] + distance * np.sin(theta),
                center[1],
                center[2] + distance * np.cos(theta),
            )
            cam.SetViewUp(0, 1, 0)
            cam.ParallelProjectionOn()
            cam.SetParallelScale(scale)
            r.SetActiveCamera(cam)

            camera_meta.append(self._camera_dict(cam, r))

            light = vtk.vtkLight()
            light.SetLightTypeToSceneLight()
            light.SetPosition(1, 1, 1)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1, 1, 1)
            light.SetIntensity(1.0)
            r.AddLight(light)

        return renderers, camera_meta

    # --------------------------------------------------------- render 8-view row
    def _render_eight_views(
        self, mesh_file_path: str, rotation: Tuple[float, float, float] = (0, 0, 0)
    ) -> Tuple[np.ndarray, dict]:
        """
        Renders eight views (first two = orthogonal front+side) and returns
        a single numpy array shape (224, 1792), plus metadata.
        """
        if not os.path.exists(mesh_file_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file_path}")

        # -- read + smooth ---------------------------------------------------
        reader = vtk.vtkSTLReader()
        reader.SetFileName(mesh_file_path)
        reader.Update()

        smooth_filter = self._smooth_mesh(reader)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smooth_filter.GetOutputPort())

        # -- bounds / camera distance ---------------------------------------
        polydata = smooth_filter.GetOutput()
        bounds = polydata.GetBounds()
        center = polydata.GetCenter()
        max_dim = max(
            bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
        )
        distance = max_dim * 2.5 if max_dim > 0 else 100
        base_az = float(np.random.uniform(0, 360))             # ★ NEW

        # -- renderers & cameras --------------------------------------------
        renderers, cam_meta = self._setup_renderers_8(
            mapper, center, distance, max_dim, base_az          # ★ pass spin
        )

        # -- off-screen window (8×224 × 224) --------------------------------
        n_views = 8
        tile_px = 224
        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.SetSize(n_views * tile_px, tile_px)

        for r in renderers:
            rw.AddRenderer(r)
            r.ResetCameraClippingRange()

        rw.Render()
        

        # -- capture ---------------------------------------------------------
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(rw)
        w2if.Update()
        vtk_image = w2if.GetOutput()
        dims = vtk_image.GetDimensions()  # (width, height, _)
        num_comp = vtk_image.GetPointData().GetScalars().GetNumberOfComponents()
        vtk_array = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        height, width = dims[1], dims[0]
        arr = vtk_array.reshape(height, width, num_comp)
        arr = np.flipud(arr)  # vtk -> numpy orientation

        # Convert to 8-bit grayscale
        gray_image = Image.fromarray(arr.astype(np.uint8)).convert("L")
        strip = np.array(gray_image)

        meta = {
            "rotation_deg": list(rotation),
            "azimuth_base_deg": base_az,                        # ★ NEW
            "center": list(center),
            "distance": distance,
            "tile_px": tile_px,
            "num_views": n_views,
            "cameras": cam_meta,
        }
        return strip, meta
    
    # ------------------------------------------------------- helpers unchanged
    def _get_missing_files(self, files: List[str]) -> List[str]:
        images_dir = os.path.join(self.data_dir, self.output_dir, "images")
        folder_files = os.listdir(images_dir) if os.path.exists(images_dir) else []
        folder_base_names = {os.path.splitext(f)[0] for f in folder_files}
        base_names = {os.path.splitext(f)[0] for f in files}
        missing_base_names = base_names - folder_base_names
        return [f"{base}.stl" for base in missing_base_names]

    # ---------------------------------------------------------------- process
    def process(self, files: List[str], mesh_path: str = None) -> None:
        """
        Generates an 8-view strip for any mesh that does not yet have one.
        Rotation CSV logic is unchanged.
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
                        rotations_dict[row["sample"]] = (
                            float(row["rot_x"]),
                            float(row["rot_y"]),
                            float(row["rot_z"]),
                        )
            except Exception as e:
                logger.error(f"Failed to load existing rotation CSV file: {e}")

        missing_files = self._get_missing_files(files)
        if not missing_files:
            logger.info("Images have already been generated.")
            return

        logger.info(
            f"Found {len(missing_files)} out of {len(files)} files to generate images for."
        )
        os.makedirs(os.path.join(images_dir, "metadata"), exist_ok=True)

        for file in tqdm(missing_files, desc="Generating 8-view strips"):
            curr_mesh_path = os.path.join(mesh_path, file)
            rotation = tuple(np.random.uniform(0, 360, 3))
            sample_name = os.path.splitext(file)[0]

            try:
                strip, meta = self._render_eight_views(curr_mesh_path, rotation)
            except Exception as e:
                logger.error(f"Failed to render images for {file}: {e}")
                break

            # -- save image ---------------------------------------------------
            image_path = os.path.join(images_dir, f"{sample_name}.png")
            try:
                Image.fromarray(np.uint8(strip)).save(image_path)
            except Exception as e:
                logger.error(f"Failed to save image for {file}: {e}")
                break

            # -- save metadata ----------------------------------------------
            meta_path = os.path.join(images_dir, "metadata", f"{sample_name}_cam.json")
            with open(meta_path, "w") as jf:
                json.dump(meta, jf, indent=2)

            rotations_dict[sample_name] = rotation

        # -- write/update rotation CSV ---------------------------------------
        try:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["sample", "rot_x", "rot_y", "rot_z"])
                for sample, rotation in rotations_dict.items():
                    writer.writerow([sample, rotation[0], rotation[1], rotation[2]])
            logger.info(f"Rotation data saved to {csv_path}.")
        except Exception as e:
            logger.error(f"Failed to save rotation CSV file: {e}")
