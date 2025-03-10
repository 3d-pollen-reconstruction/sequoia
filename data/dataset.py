import os
import random
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class PollenDataset(Dataset):
    def __init__(self, data_dir, transform=None, return_3d=False):
        """
        Args:
            data_dir (str): Path of STL models.
            transform (callable, optional): Transformation(s) that get applied to the models.
            return_3d (bool): If true, the path of the 3d models will be returned.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.return_3d = return_3d

        self.exclude_list = [
            "17778_Salix%20alba%20-%20Willow_NIH3D.stl",
            "17796_Fuchsia%20magellanica%20-%20Hardy%20fuchsia_NIH3D.stl",
        ]
        
        self.files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".stl") and f not in self.exclude_list
        ]
        if len(self.files) == 0:
            raise RuntimeError("No STL-files after filtering found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        rotation = (
            random.uniform(0, 360),
            random.uniform(0, 360),
            random.uniform(0, 360),
        )
        left_view, right_view = self.render_model_views_from_file(file_path, rotation)
        
        left_img = Image.fromarray(left_view.astype(np.uint8))
        right_img = Image.fromarray(right_view.astype(np.uint8))
        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        sample = {
            'left_view': left_img,
            'right_view': right_img,
            'rotation': rotation,
            'file_name': file_name
        }
        if self.return_3d:
            sample['3d_model_path'] = file_path

        return sample

    def render_model_views_from_file(self, selected_path, rotation):
        # 1. Loading .stl file
        reader = vtk.vtkSTLReader()
        reader.SetFileName(selected_path)
        reader.Update()

        # 2. Applying smoothing filter to simplify mesh
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputConnection(reader.GetOutputPort())
        smoothFilter.SetNumberOfIterations(30)
        smoothFilter.SetRelaxationFactor(0.1)
        smoothFilter.FeatureEdgeSmoothingOff()
        smoothFilter.BoundarySmoothingOff()
        smoothFilter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoothFilter.GetOutputPort())

        # 3. Creating actors to capture two orthogonal views
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

        # 4. Creating two renderers for both views
        renderer1 = vtk.vtkRenderer()
        renderer2 = vtk.vtkRenderer()
        renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
        renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)
        renderer1.SetBackground(0, 0, 0)
        renderer2.SetBackground(0, 0, 0)
        renderer1.AddActor(actor1)
        renderer2.AddActor(actor2)

        # 5. Adding light sources
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(1, 1, 1)
        light.SetFocalPoint(0, 0, 0)
        light.SetColor(1, 1, 1)
        light.SetIntensity(1.0)
        renderer1.AddLight(light)
        renderer2.AddLight(light)

        # 6. Set up shadow mapping
        shadow_pass = vtk.vtkShadowMapPass()
        opaque_pass = vtk.vtkOpaquePass()
        render_pass_collection = vtk.vtkRenderPassCollection()
        render_pass_collection.AddItem(opaque_pass)
        render_pass_collection.AddItem(shadow_pass)
        sequence_pass = vtk.vtkSequencePass()
        sequence_pass.SetPasses(render_pass_collection)
        camera_pass = vtk.vtkCameraPass()
        camera_pass.SetDelegatePass(sequence_pass)
        renderer1.SetPass(camera_pass)
        renderer2.SetPass(camera_pass)

        # 7. Center pollen grain object and set camera options
        polydata = smoothFilter.GetOutput()
        bounds = polydata.GetBounds()
        center = polydata.GetCenter()
        max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        distance = max_dim * 2.5 if max_dim > 0 else 100

        # Z-Axis cam
        cam1 = vtk.vtkCamera()
        cam1.SetFocalPoint(center)
        cam1.SetPosition(center[0], center[1], center[2] + distance)
        cam1.SetViewUp(0, 1, 0)
        renderer1.SetActiveCamera(cam1)

        # X-Axis cam
        cam2 = vtk.vtkCamera()
        cam2.SetFocalPoint(center)
        cam2.SetPosition(center[0] + distance, center[1], center[2])
        cam2.SetViewUp(0, 1, 0)
        renderer2.SetActiveCamera(cam2)

        # Apply rotations on the actors
        actor1.SetOrigin(center)
        actor2.SetOrigin(center)
        actor1.SetOrientation(rotation)
        actor2.SetOrientation(rotation)

        # 8. Setup offscreen render window (reduced resolution of 1024x512)
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(1024, 512)  # 512x512 pixels for each view, combined for both views 1024x512
        renderWindow.SetOffScreenRendering(1)
        renderWindow.AddRenderer(renderer1)
        renderWindow.AddRenderer(renderer2)
        renderer1.ResetCameraClippingRange()
        renderer2.ResetCameraClippingRange()
        renderWindow.Render()

        # 9. Capture the render window
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderWindow)
        w2if.Update()

        vtk_image = w2if.GetOutput()
        dims = vtk_image.GetDimensions()  # (width, height, depth)
        num_comp = vtk_image.GetPointData().GetScalars().GetNumberOfComponents()
        vtk_array = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        height, width = dims[1], dims[0]
        arr = vtk_array.reshape(height, width, num_comp)
        arr = np.flipud(arr)  # flip vertically

        # Convert to grayscale
        if num_comp >= 3:
            arr = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
        else:
            arr = arr[..., 0]

        # Extract left and right view from the combined image
        half_width = width // 2
        left_view = arr[:, :half_width]
        right_view = arr[:, half_width:]
        return left_view, right_view
