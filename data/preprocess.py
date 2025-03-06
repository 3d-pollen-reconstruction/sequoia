#!/usr/bin/env python
import os
import argparse
import logging
import random
import shutil

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

vtk.vtkObject.GlobalWarningDisplayOff()


class Pipeline:
    def __init__(self, raw_dir: str, processed_images_dir: str, k_folds: int):
        self.raw_dir = raw_dir
        self.processed_images_dir = processed_images_dir
        self.k_folds = k_folds
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.processed_images_dir, exist_ok=True)

    def render_model_views_from_file(self, stl_path: str, rotation: tuple):
        """
        Render two orthogonal views from the given STL model.
        The model is rotated by the provided (rx, ry, rz) angles.
        Returns:
            left_view, right_view: Two grayscale numpy arrays.
        """
        # Read the STL file
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()

        # Check if data was loaded
        polydata = reader.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            raise RuntimeError("STL file contains no points.")

        # Apply smoothing filter
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputConnection(reader.GetOutputPort())
        smoothFilter.SetNumberOfIterations(30)
        smoothFilter.SetRelaxationFactor(0.1)
        smoothFilter.FeatureEdgeSmoothingOff()
        smoothFilter.BoundarySmoothingOff()
        smoothFilter.Update()

        # Setup mapper and actors for two views
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoothFilter.GetOutputPort())

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

        # Setup two renderers with separate viewports
        renderer1 = vtk.vtkRenderer()
        renderer2 = vtk.vtkRenderer()
        renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
        renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)
        renderer1.SetBackground(0, 0, 0)
        renderer2.SetBackground(0, 0, 0)
        renderer1.AddActor(actor1)
        renderer2.AddActor(actor2)

        # Lighting
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(1, 1, 1)
        light.SetFocalPoint(0, 0, 0)
        light.SetColor(1, 1, 1)
        light.SetIntensity(1.0)
        renderer1.AddLight(light)
        renderer2.AddLight(light)

        # Centering the object in view
        polydata = smoothFilter.GetOutput()
        bounds = polydata.GetBounds()  # (xmin,xmax, ymin,ymax, zmin,zmax)
        center = polydata.GetCenter()
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        distance = max_dim * 2.5 if max_dim > 0 else 100

        # Camera for view 1: from the Z axis
        camera1 = vtk.vtkCamera()
        camera1.SetFocalPoint(center)
        camera1.SetPosition(center[0], center[1], center[2] + distance)
        camera1.SetViewUp(0, 1, 0)
        renderer1.SetActiveCamera(camera1)

        # Camera for view 2: from the X axis
        camera2 = vtk.vtkCamera()
        camera2.SetFocalPoint(center)
        camera2.SetPosition(center[0] + distance, center[1], center[2])
        camera2.SetViewUp(0, 1, 0)
        renderer2.SetActiveCamera(camera2)

        # Apply provided rotation to both actors
        actor1.SetOrigin(center)
        actor2.SetOrigin(center)
        actor1.SetOrientation(rotation)
        actor2.SetOrientation(rotation)

        # Offscreen render window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(2048, 1024)  # 1024x1024 for side-by-side views
        renderWindow.SetOffScreenRendering(1)
        renderWindow.AddRenderer(renderer1)
        renderWindow.AddRenderer(renderer2)
        renderer1.ResetCameraClippingRange()
        renderer2.ResetCameraClippingRange()
        renderWindow.Render()

        # Capture the image
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderWindow)
        w2if.Update()
        vtk_image = w2if.GetOutput()
        dims = vtk_image.GetDimensions()  # (width, height, depth)
        num_comp = vtk_image.GetPointData().GetScalars().GetNumberOfComponents()
        vtk_array = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        height, width = dims[1], dims[0]
        arr = vtk_array.reshape(height, width, num_comp)
        arr = np.flipud(arr)

        # Convert to grayscale (luminance conversion) if RGB
        if num_comp >= 3:
            arr = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
        else:
            arr = arr[..., 0]

        # Split into left and right views
        half_width = width // 2
        left_view = arr[:, :half_width]
        right_view = arr[:, half_width:]
        return left_view, right_view

    def process_models(self):
        """
        Processes each STL file in the raw folder by rendering an orthogonal image pair.
        Saves the concatenated image (left | right) as a PNG in the processed images folder.
        Returns a list of saved image file paths.
        """
        stl_files = [f for f in os.listdir(self.raw_dir) if f.lower().endswith('.stl')]
        if not stl_files:
            self.logger.error(f"No STL files found in {self.raw_dir}")
            return []
        processed_paths = []
        for stl_file in tqdm(stl_files, desc="Processing STL models"):
            stl_path = os.path.join(self.raw_dir, stl_file)
            # Generate a random rotation
            rotation = (random.uniform(0, 360),
                        random.uniform(0, 360),
                        random.uniform(0, 360))
            try:
                left_view, right_view = self.render_model_views_from_file(stl_path, rotation)
            except Exception as e:
                self.logger.error(f"Failed to render {stl_file}: {e}")
                continue
            # Concatenate the two views horizontally
            combined = np.concatenate((left_view, right_view), axis=1)
            # Convert grayscale to RGB (by replicating the single channel)
            combined_rgb = np.stack((combined,)*3, axis=-1)
            base_name = os.path.splitext(stl_file)[0]
            image_filename = f"{base_name}_combined.png"
            image_path = os.path.join(self.processed_images_dir, image_filename)
            try:
                Image.fromarray(np.uint8(combined_rgb)).save(image_path)
                processed_paths.append(image_path)
            except Exception as e:
                self.logger.error(f"Failed to save image for {stl_file}: {e}")
        return processed_paths

    def split_into_folds(self, image_paths):
        """
        Splits the list of processed images into k folds.
        Creates subdirectories fold_1, fold_2, …, fold_k under the processed images directory,
        and moves the corresponding images into these folders.
        """
        random.shuffle(image_paths)
        folds = np.array_split(image_paths, self.k_folds)
        for i, fold in enumerate(folds, start=1):
            fold_dir = os.path.join(self.processed_images_dir, f"fold_{i}")
            os.makedirs(fold_dir, exist_ok=True)
            for img_path in fold:
                dest = os.path.join(fold_dir, os.path.basename(img_path))
                shutil.move(img_path, dest)
        self.logger.info("Data splitting into folds complete.")

    def run(self):
        self.logger.info("Starting Pipeline...")
        processed_images = self.process_models()
        if processed_images:
            self.split_into_folds(processed_images)
        self.logger.info("Pipeline completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="3D Pollen Mesh Preprocessing Pipeline")
    parser.add_argument("--raw_dir", type=str,
                        default=os.path.join(os.getenv("DATA_DIR_PATH"), "raw"),
                        help="Directory containing raw STL models.")
    parser.add_argument("--processed_images_dir", type=str,
                        default=os.path.join(os.getenv("DATA_DIR_PATH"), "processed", "images"),
                        help="Directory to save processed image pairs.")
    parser.add_argument("--k_folds", type=int, default=5,
                        help="Number of folds to split the processed images into.")
    args = parser.parse_args()

    pipeline = Pipeline(args.raw_dir, args.processed_images_dir, args.k_folds)
    pipeline.run()
