#!/usr/bin/env python

import os
import shutil
import argparse
import numpy as np
import vtk


def make_files(root):
    # Remove and recreate directory structure
    for d in ['imgs', 'train', 'test']:
        path = os.path.join(root, d)
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(os.path.join(root, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(root, 'train', 'pose'), exist_ok=True)
    os.makedirs(os.path.join(root, 'train', 'intrinsics'), exist_ok=True)
    os.makedirs(os.path.join(root, 'test', 'pose'), exist_ok=True)
    os.makedirs(os.path.join(root, 'test', 'intrinsics'), exist_ok=True)


def write_intrinsics(root, N, img_size, fov_deg):
    train_cnt = 0
    test_cnt = 0
    # Compute focal length from FOV and image size
    fov_rad = np.deg2rad(fov_deg)
    fy = 0.5 * img_size / np.tan(fov_rad / 2)
    fx = fy
    cx = img_size / 2
    cy = img_size / 2

    for j in range(N):
        prefix = 'train' if (j + 1) % 10 else 'test'
        cnt = train_cnt if prefix == 'train' else test_cnt
        if prefix == 'train':
            train_cnt += 1
        else:
            test_cnt += 1
        intrinsics = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
        path = os.path.join(root, prefix, 'intrinsics', f'{prefix}_{cnt}.txt')
        with open(path, 'w') as fh:
            for val in intrinsics:
                fh.write(f"{val}\n")


def write_pose(root, prefix, cnt, c2w):
    path = os.path.join(root, prefix, 'pose', f'{prefix}_{cnt}.txt')
    with open(path, 'w') as fh:
        for v in c2w.flatten():
            fh.write(f"{v}\n")


def compute_c2w(camera):
    view_mat = camera.GetViewTransformMatrix()
    inv_mat = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(view_mat, inv_mat)
    M = np.eye(4)
    for i in range(4):
        for j in range(4):
            M[i, j] = inv_mat.GetElement(i, j)
    return M


def main(stl_file, output_dir, rotation_steps=32, radius=1.5, img_size=400, fov=60.0):
    # Fixed image size
    IMG_SIZE = img_size

    # Prepare folders and intrinsics
    make_files(output_dir)
    write_intrinsics(output_dir, rotation_steps, IMG_SIZE, fov)

    # Load and prepare mesh with normals
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.Update()
    polydata = normals.GetOutput()
    center = polydata.GetCenter()
    bounds = polydata.GetBounds()
    diameter = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])

    # Compute camera distance so object fits exactly, then apply radius multiplier
    fov_rad = np.deg2rad(fov)
    base_dist = diameter / (2 * np.tan(fov_rad / 2))
    distance = base_dist * radius

    # Setup actor centered at origin
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8,0.8,0.8)
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.6)
    actor.GetProperty().SetSpecular(0.3)
    actor.SetOrigin(center)
    actor.SetPosition(-center[0], -center[1], -center[2])

    # Renderer & offscreen window
    ren = vtk.vtkRenderer()
    ren.SetBackground(0,0,0)
    ren.AddActor(actor)
    ren.ResetCamera()
    ren.ResetCameraClippingRange()

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(IMG_SIZE, IMG_SIZE)
    ren_win.SetOffScreenRendering(1)
    ren_win.SetAlphaBitPlanes(1)
    ren_win.SetMultiSamples(0)

    # Set camera FOV
    cam = ren.GetActiveCamera()
    cam.SetViewAngle(fov)

    # Add headlight
    light = vtk.vtkLight()
    light.SetLightTypeToHeadlight()
    ren.AddLight(light)

    # Compute golden spiral sample directions
    n = rotation_steps * 2
    idx = np.arange(n)
    gr = (1 + np.sqrt(5)) / 2.0
    theta = 2 * np.pi * idx / gr
    phi = np.arccos(1 - 2 * (idx + 0.5) / n)
    mask = (phi > np.pi/2) & (phi < 3*np.pi/2)
    theta, phi = theta[~mask], phi[~mask]
    x = distance * np.cos(theta) * np.sin(phi)
    y = distance * np.sin(theta) * np.sin(phi)
    z = distance * np.cos(phi)

    # Setup capture
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    writer = vtk.vtkPNGWriter()

    train_cnt=test_cnt=0
    for i in range(rotation_steps):
        cam.SetPosition(x[i], y[i], z[i])
        cam.SetFocalPoint(0,0,0)
        cam.SetViewUp(0,0,1)
        ren.ResetCameraClippingRange()
        ren_win.Render()
        w2if.Modified(); w2if.Update()

        c2w = compute_c2w(cam)
        prefix = 'train' if (i+1)%10 else 'test'
        cnt = train_cnt if prefix=='train' else test_cnt
        if prefix=='train': train_cnt+=1
        else: test_cnt+=1

        # Save outputs
        write_pose(output_dir, prefix, cnt, c2w)
        img_path = os.path.join(output_dir, 'imgs', f"{prefix}_{cnt}.png")
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.SetFileName(img_path)
        writer.Write()

    print("Rendering complete. Outputs in:", output_dir)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stl',    required=True, help='STL file')
    p.add_argument('--out',    required=True, help='Output dir root')
    p.add_argument('--steps',  type=int,   default=32, help='Number of views')
    p.add_argument('--radius', type=float, default=1.5, help='Distance multiplier (>1 zooms out)')
    p.add_argument('--img_size',type=int,   default=400, help='Width/height of image')
    p.add_argument('--fov',    type=float, default=60.0, help='Field of view (deg)')
    args=p.parse_args()
    main(args.stl, args.out, args.steps, args.radius, args.img_size, args.fov)