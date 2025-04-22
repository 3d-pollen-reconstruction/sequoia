import os
import json
import numpy as np
from typing import List, Tuple
from PIL import Image
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from tqdm import tqdm

class NeRFDatasetGenerator:
    def __init__(
        self,
        raw_mesh_dir: str = 'raw',
        output_dir: str = 'nerf_dataset',
        num_views: int = 16,
        img_size: Tuple[int, int] = (512, 512),
        random_seed: int = 1337
    ):
        self.raw_mesh_dir = raw_mesh_dir
        self.output_dir = output_dir
        self.num_views = num_views
        self.img_width, self.img_height = img_size
        np.random.seed(random_seed)
        os.makedirs(self.output_dir, exist_ok=True)

    def _smooth_mesh(self, reader: vtk.vtkSTLReader, iterations: int = 30, relax: float = 0.1) -> vtk.vtkSmoothPolyDataFilter:
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(reader.GetOutputPort())
        smooth.SetNumberOfIterations(iterations)
        smooth.SetRelaxationFactor(relax)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOff()
        smooth.Update()
        return smooth

    def _setup_renderer(self, actor: vtk.vtkActor) -> Tuple[vtk.vtkRenderer, vtk.vtkCamera]:
        # Create renderer and add actor
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0, 0, 0)
        renderer.AddActor(actor)
        # Add headlight so we see geometry
        light = vtk.vtkLight()
        light.SetLightTypeToHeadlight()
        renderer.AddLight(light)
        # Create perspective camera
        cam = vtk.vtkCamera()
        cam.ParallelProjectionOff()
        cam.SetViewAngle(60)  # wider FOV
        renderer.SetActiveCamera(cam)
        # Automatically adjust clipping and position once actor is present
        renderer.ResetCamera()
        renderer.ResetCameraClippingRange()
        return renderer, cam

    def _render_view(self, renderer: vtk.vtkRenderer) -> np.ndarray:
        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.AddRenderer(renderer)
        rw.SetSize(self.img_width, self.img_height)
        rw.Render()
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(rw)
        w2if.Update()
        arr = vtk_to_numpy(w2if.GetOutput().GetPointData().GetScalars())
        w, h = w2if.GetOutput().GetDimensions()[:2]
        img = arr.reshape(h, w, -1)
        img = np.flipud(img).astype(np.uint8)
        return img

    def _compute_intrinsics(self, cam: vtk.vtkCamera) -> Tuple[float, float, float, float]:
        # Use camera's view angle
        fovy = np.deg2rad(cam.GetViewAngle())
        fy = 0.5 * self.img_height / np.tan(0.5 * fovy)
        fx = fy
        cx = self.img_width / 2
        cy = self.img_height / 2
        return fx, fy, cx, cy

    def _camera_to_world(self, cam: vtk.vtkCamera) -> np.ndarray:
        # Invert modelview to get camera-to-world
        mv = cam.GetModelViewTransformMatrix()
        M = np.array([[mv.GetElement(r, c) for c in range(4)] for r in range(4)])
        return np.linalg.inv(M)

    def process(self, mesh_files: List[str]) -> None:
        transforms = {'camera_angle_x': None, 'frames': []}
        for mesh_file in tqdm(mesh_files, desc='Generating NeRF dataset'):
            name = os.path.splitext(os.path.basename(mesh_file))[0]
            mesh_path = os.path.join(self.raw_mesh_dir, mesh_file)

            # Load and smooth mesh
            reader = vtk.vtkSTLReader()
            reader.SetFileName(mesh_path)
            reader.Update()
            smooth_filter = self._smooth_mesh(reader)

            # Create actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(smooth_filter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 1, 1)
            actor.GetProperty().SetAmbient(0.5)  # stronger ambient
            actor.GetProperty().SetDiffuse(0.5)
            actor.GetProperty().SetSpecular(0.2)

            # Compute center and distance
            poly = smooth_filter.GetOutput()
            center = poly.GetCenter()
            bounds = poly.GetBounds()
            diameter = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            dist = diameter * 2.5 if diameter > 0 else 100.0

            # Generate views around 360° at equator
            angles = np.linspace(0, 360, self.num_views, endpoint=False)
            for i, yaw in enumerate(angles):
                theta = np.deg2rad(yaw)
                # Setup renderer and camera
                renderer, cam = self._setup_renderer(actor)
                # Position camera on a circle around object
                cam.SetPosition(
                    center[0] + dist * np.sin(theta),
                    center[1],
                    center[2] + dist * np.cos(theta)
                )
                cam.SetFocalPoint(*center)
                cam.SetViewUp(0, 1, 0)

                # Render and save image
                img = self._render_view(renderer)
                img_name = f"{name}_{i:02d}.png"
                Image.fromarray(img).save(os.path.join(self.output_dir, img_name))

                # Save intrinsics once
                if transforms['camera_angle_x'] is None:
                    fx, fy, cx, cy = self._compute_intrinsics(cam)
                    transforms['camera_angle_x'] = 2 * np.arctan(self.img_width / (2 * fx))

                # Save camera-to-world
                c2w = self._camera_to_world(cam).tolist()
                transforms['frames'].append({
                    'file_path': img_name,
                    'transform_matrix': c2w
                })

        # Write transforms.json
        with open(os.path.join(self.output_dir, 'transforms.json'), 'w') as f:
            json.dump(transforms, f, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate NeRF dataset from STL meshes')
    parser.add_argument('--raw_dir', type=str, default='raw')
    parser.add_argument('--out_dir', type=str, default='nerf_dataset')
    parser.add_argument('--views', type=int, default=16)
    parser.add_argument('--w', type=int, default=512)
    parser.add_argument('--h', type=int, default=512)
    parser.add_argument('files', nargs='+', help='List of mesh filenames (STL)')
    args = parser.parse_args()

    gen = NeRFDatasetGenerator(
        raw_mesh_dir=args.raw_dir,
        output_dir=args.out_dir,
        num_views=args.views,
        img_size=(args.w, args.h)
    )
    gen.process(args.files)