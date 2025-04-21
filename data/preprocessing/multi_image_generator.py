import os
import json
import csv
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from tqdm import tqdm

class MultiViewImageGenerator:
    def __init__(
        self,
        raw_mesh_dir: str = 'raw',
        output_dir: str = 'processed',
        num_views: int = 4,
        random_seed: int = 1337
    ):
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir
        self.num_views = num_views
        self.random_seed = random_seed
        self.data_dir = os.getenv("DATA_DIR_PATH")
        if not self.data_dir:
            raise EnvironmentError("DATA_DIR_PATH environment variable not set.")

    def _vtk4x4_to_numpy(self, mat: vtk.vtkMatrix4x4) -> np.ndarray:
        return np.array([[mat.GetElement(r, c) for c in range(4)] for r in range(4)])

    def _camera_dict(self, cam: vtk.vtkCamera, renderer: vtk.vtkRenderer) -> dict:
        near_, far_ = cam.GetClippingRange()
        aspect = renderer.GetAspect()[0]
        mv = self._vtk4x4_to_numpy(cam.GetModelViewTransformMatrix())
        proj = self._vtk4x4_to_numpy(cam.GetProjectionTransformMatrix(aspect, near_, far_))
        return {
            "position": cam.GetPosition(),
            "focal_point": cam.GetFocalPoint(),
            "view_up": cam.GetViewUp(),
            "parallel_scale": cam.GetParallelScale(),
            "clipping_range": [near_, far_],
            "modelview": mv.tolist(),
            "projection": proj.tolist(),
            "proj_modelview": (proj @ mv).tolist(),
        }

    def _smooth_mesh(self, reader: vtk.vtkSTLReader, iterations: int = 30, relax: float = 0.1):
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(reader.GetOutputPort())
        smooth.SetNumberOfIterations(iterations)
        smooth.SetRelaxationFactor(relax)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOff()
        smooth.Update()
        return smooth

    def _render_single_view(
        self,
        mesh_path: str,
        rotation: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, dict]:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(mesh_path)
        reader.Update()
        smooth = self._smooth_mesh(reader)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smooth.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1,1,1)
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.7)
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(50)

        poly = smooth.GetOutput()
        center = poly.GetCenter()
        bounds = poly.GetBounds()
        max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        dist = max_dim * 2.5 if max_dim>0 else 100
        actor.SetOrigin(center)
        actor.SetOrientation(rotation)

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0,0,0)
        renderer.AddActor(actor)
        cam = vtk.vtkCamera()
        cam.SetFocalPoint(center)
        cam.SetPosition(center[0], center[1], center[2] + dist)
        cam.SetViewUp(0,1,0)
        cam.ParallelProjectionOn()
        cam.SetParallelScale(max_dim*0.6)
        renderer.SetActiveCamera(cam)
        renderer.ResetCameraClippingRange()

        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.AddRenderer(renderer)
        rw.SetSize(512,512)
        rw.Render()
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(rw)
        w2if.Update()
        img = vtk_to_numpy(w2if.GetOutput().GetPointData().GetScalars())
        h, w = w2if.GetOutput().GetDimensions()[1], w2if.GetOutput().GetDimensions()[0]
        arr = img.reshape(h, w, -1)
        gray = Image.fromarray(np.flipud(arr).astype(np.uint8)).convert('L')
        return np.array(gray), self._camera_dict(cam, renderer)

    def _concatenate_views(self, views: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(views, axis=1)

    def _load_rotations(self, csv_path: str) -> Dict[str, List[Tuple[float,float,float]]]:
        rotations = {}
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row['sample']
                    idx = int(row['view_index'])
                    rot = (float(row['rot_x']), float(row['rot_y']), float(row['rot_z']))
                    rotations.setdefault(key, [None]*self.num_views)[idx] = rot
        return rotations

    def process(self, files: List[str]) -> None:
        np.random.seed(self.random_seed)
        images_dir = os.path.join(self.data_dir, self.output_dir, f"images_{self.num_views}")
        os.makedirs(images_dir, exist_ok=True)
        meta_dir = os.path.join(images_dir, "metadata")
        os.makedirs(meta_dir, exist_ok=True)

        csv_path = os.path.join(self.data_dir, self.output_dir, f"rotations_{self.num_views}.csv")
        existing = self._load_rotations(csv_path)
        new_records = []

        for file in tqdm(files, desc=f"Generating {self.num_views} views"):
            base = os.path.splitext(file)[0]
            mesh_path = os.path.join(self.data_dir, self.raw_mesh_dir, file)
            yaws = np.linspace(0, 360, self.num_views, endpoint=False)
            rotations = [(0.0, float(y), 0.0) for y in yaws]
            images, cams = [], []
            for idx, rot in enumerate(rotations):
                img, cam = self._render_single_view(mesh_path, rot)
                images.append(img)
                cams.append({"rotation_deg": rot, "camera": cam})
                # only regenerate missing
                if base not in existing or existing[base][idx] is None:
                    new_records.append((base, idx, rot))

            concat = self._concatenate_views(images)
            out_name = f"{base}_{self.num_views}views.png"
            Image.fromarray(concat).save(os.path.join(images_dir, out_name))

            meta = {
                "num_views": self.num_views,
                "height": concat.shape[0],
                "width_each": images[0].shape[1],
                "views": cams
            }
            with open(os.path.join(meta_dir, f"{base}_{self.num_views}views.json"), 'w') as jf:
                json.dump(meta, jf, indent=2)

        # merge existing with new and write CSV
        merged = existing.copy()
        for base, idx, rot in new_records:
            merged.setdefault(base, [None]*self.num_views)[idx] = rot

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["sample", "view_index", "rot_x", "rot_y", "rot_z"])
            for sample, rots in merged.items():
                for idx, rot in enumerate(rots):
                    writer.writerow([sample, idx, rot[0], rot[1], rot[2]])
