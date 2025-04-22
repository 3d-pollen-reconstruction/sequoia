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
    # Compute focal length from FOV
    fov_rad = np.deg2rad(fov_deg)
    fy = 0.5 * img_size / np.tan(fov_rad / 2)
    fx = fy
    cx = img_size / 2
    cy = img_size / 2
    for j in range(N):
        prefix = 'train' if (j + 1) % 10 else 'test'
        cnt = train_cnt if prefix == 'train' else test_cnt
        if prefix == 'train': train_cnt += 1
        else: test_cnt += 1
        intr = [fx, 0.0, cx, 0.0,
                0.0, fy, cy, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
        path = os.path.join(root, prefix, 'intrinsics', f"{prefix}_{cnt}.txt")
        with open(path, 'w') as fh:
            for v in intr:
                fh.write(f"{v}\n")


def write_pose(root, prefix, cnt, c2w):
    path = os.path.join(root, prefix, 'pose', f"{prefix}_{cnt}.txt")
    with open(path, 'w') as fh:
        for v in c2w.flatten():
            fh.write(f"{v}\n")


def compute_c2w(cam):
    # Invert view transform to get camera-to-world
    view = cam.GetViewTransformMatrix()
    inv = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(view, inv)
    M = np.eye(4)
    for i in range(4):
        for j in range(4):
            M[i, j] = inv.GetElement(i, j)
    return M


def main(stl_file, output_dir, rotation_steps=32, radius=1.5, img_size=400, fov=60.0):
    # Prepare directories
    make_files(output_dir)
    write_intrinsics(output_dir, rotation_steps, img_size, fov)

    # Load mesh and compute normals
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.Update()
    poly = normals.GetOutput()
    center = poly.GetCenter()
    bounds = poly.GetBounds()
    diameter = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])

    # Center actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.6)
    actor.GetProperty().SetSpecular(0.3)
    actor.SetOrigin(center)
    actor.SetPosition(-center[0], -center[1], -center[2])

    # Renderer with transparent background
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.0, 0.0, 0.0)
    ren.SetBackgroundAlpha(0.0)
    ren.AddActor(actor)
    ren.ResetCamera()
    ren.ResetCameraClippingRange()

    # Compute camera distance so object fits
    fov_rad = np.deg2rad(fov)
    base_dist = diameter / (2 * np.tan(fov_rad / 2))
    dist = base_dist * radius

    # Off-screen render window
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(img_size, img_size)
    ren_win.SetOffScreenRendering(1)
    ren_win.SetAlphaBitPlanes(1)
    ren_win.SetMultiSamples(0)

    # Camera and light
    cam = ren.GetActiveCamera()
    cam.SetViewAngle(fov)
    light = vtk.vtkLight()
    light.SetLightTypeToHeadlight()
    ren.AddLight(light)

    # Spiral camera positions
    n = rotation_steps * 2
    idx = np.arange(n)
    gr = (1 + np.sqrt(5)) / 2.0
    theta = 2 * np.pi * idx / gr
    phi = np.arccos(1 - 2 * (idx + 0.5) / n)
    mask = (phi > np.pi/2) & (phi < 3 * np.pi/2)
    theta, phi = theta[~mask], phi[~mask]
    x = dist * np.cos(theta) * np.sin(phi)
    y = dist * np.sin(theta) * np.sin(phi)
    z = dist * np.cos(phi)

    # Setup RGBA capture
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    w2if.SetInputBufferTypeToRGBA()
    w2if.ReadFrontBufferOff()
    writer = vtk.vtkPNGWriter()

    train_cnt = 0
    test_cnt = 0

    for i in range(rotation_steps):
        # Position camera
        cam.SetPosition(x[i], y[i], z[i])
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetViewUp(0.0, 0.0, 1.0)
        ren.ResetCameraClippingRange()

        # Render and capture
        ren_win.Render()
        w2if.Modified()
        w2if.Update()

        # Save pose
        c2w = compute_c2w(cam)
        prefix = 'train' if (i + 1) % 10 else 'test'
        cnt = train_cnt if prefix == 'train' else test_cnt
        if prefix == 'train': train_cnt += 1
        else: test_cnt += 1
        write_pose(output_dir, prefix, cnt, c2w)

        # Write RGBA image
        img_path = os.path.join(output_dir, 'imgs', f"{prefix}_{cnt}.png")
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.SetFileName(img_path)
        writer.Write()

    print("Rendering complete. Outputs in:", output_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate NeRF-style dataset from STL using VTK')
    p.add_argument('--stl', required=True, help='Input STL file path')
    p.add_argument('--out', required=True, help='Output directory')
    p.add_argument('--steps', type=int, default=32, help='Number of renders')
    p.add_argument('--radius', type=float, default=1.5, help='Radius scale')
    p.add_argument('--img_size', type=int, default=400, help='Image size (px)')
    p.add_argument('--fov', type=float, default=60.0, help='Field of view (deg)')
    args = p.parse_args()
    main(args.stl, args.out, args.steps, args.radius, args.img_size, args.fov)