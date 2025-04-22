#!/usr/bin/env python
"""Generate a NeRF‑style dataset (RGBA PNGs + poses + intrinsics + per‑view bounds)

Key points
----------
* **Unit normalisation** – longest side of the STL bbox → 2 units ⇒ every object lives in [‑1 .. 1]³.
* **Per‑view near/far** saved in a separate *bounds* folder (`train/bounds/train_0.txt`, etc.) so loaders
  that don’t need them can ignore the files and directory counts still match.
* **Fibonacci‑spiral cameras** (upper hemisphere) with an assert that we have ≥ `rotation_steps` views.
* **PNG compression level 9** to save disk space.
* **Automatic sanity checks** at the end to verify intrinsics and camera-to-world poses by reprojection.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import vtk
import sys

# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------

def _clean_and_mkdir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_files(root: Path) -> None:
    """Create a fresh output directory tree."""
    for d in ["imgs", "train", "test"]:
        _clean_and_mkdir(root / d)
    for split in ["train", "test"]:
        for sub in ["pose", "intrinsics", "bounds"]:
            _clean_and_mkdir(root / split / sub)

# -----------------------------------------------------------------------------
# Intrinsics & pose I/O
# -----------------------------------------------------------------------------

def write_intrinsics(root: Path, n_views: int, img_size: int, fov_deg: float) -> None:
    """
    Write row-major 4x4 intrinsics matrices (one per view) in txt format.
    fx = fy = (img_size/2) / tan(fov/2), cx = cy = img_size/2.
    """
    fov_rad = np.deg2rad(fov_deg)
    fy = 0.5 * img_size / np.tan(fov_rad / 2)
    fx = fy
    cx = cy = img_size / 2
    intr = [fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]

    train_cnt = test_cnt = 0
    for j in range(n_views):
        split = "train" if (j + 1) % 10 else "test"
        idx = train_cnt if split == "train" else test_cnt
        if split == "train": train_cnt += 1
        else: test_cnt += 1
        out_file = root / split / "intrinsics" / f"{split}_{idx}.txt"
        out_file.write_text("\n".join(f"{v:.6f}" for v in intr))


def write_matrix(path: Path, mat: np.ndarray) -> None:
    """Write a 4×4 matrix in row-major, one value per line."""
    vals = np.asarray(mat, dtype=float).flatten()
    path.write_text("\n".join(f"{v:.6f}" for v in vals))


def compute_c2w(cam: vtk.vtkCamera) -> np.ndarray:
    """Return the 4×4 camera-to-world matrix (row-major) by inverting VTK's view matrix."""
    view = cam.GetViewTransformMatrix()
    inv = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(view, inv)
    M = np.eye(4, dtype=float)
    for i in range(4):
        for j in range(4):
            M[i, j] = inv.GetElement(i, j)
    return M

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main(
    stl_file: str,
    output_dir: str,
    *,
    rotation_steps: int = 32,
    radius: float = 1.5,
    img_size: int = 400,
    fov: float = 60.0
) -> None:
    root = Path(output_dir).expanduser().resolve()
    make_files(root)

    # Load & normalize mesh
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file); reader.Update()
    normals = vtk.vtkPolyDataNormals(); normals.SetInputConnection(reader.GetOutputPort())
    normals.ComputePointNormalsOn(); normals.Update()
    poly = normals.GetOutput()
    center = np.array(poly.GetCenter())
    bounds = np.array(poly.GetBounds()).reshape(3, 2)
    diameter = np.max(bounds[:,1] - bounds[:,0])
    canonical_size = 2.0
    scale_factor = canonical_size / diameter

    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    actor.SetScale(scale_factor); actor.SetOrigin(*center); actor.SetPosition(-center)
    prop = actor.GetProperty(); prop.SetColor(0.8,0.8,0.8); prop.SetAmbient(0.3); prop.SetDiffuse(0.6); prop.SetSpecular(0.3)

    # Scene setup
    ren = vtk.vtkRenderer(); ren.SetBackground(0,0,0); ren.SetBackgroundAlpha(0); ren.AddActor(actor); ren.ResetCamera()
    ren_win = vtk.vtkRenderWindow(); ren_win.AddRenderer(ren); ren_win.SetSize(img_size,img_size)
    ren_win.SetOffScreenRendering(True); ren_win.SetAlphaBitPlanes(True); ren_win.SetMultiSamples(0)
    cam = ren.GetActiveCamera(); cam.SetViewAngle(fov)
    light = vtk.vtkLight(); light.SetLightTypeToHeadlight(); ren.AddLight(light)

    # Camera trajectory
    fov_rad = np.deg2rad(fov)
    base_dist = canonical_size/(2*np.tan(fov_rad/2)); dist = base_dist*radius
    n = rotation_steps*2; idx_arr = np.arange(n)
    phi = np.arccos(1 - 2*(idx_arr+0.5)/n); theta = 2*np.pi*idx_arr/((1+np.sqrt(5))/2)
    mask = (phi>np.pi/2)&(phi<3*np.pi/2); theta,phi = theta[~mask],phi[~mask]
    assert len(theta)>=rotation_steps, "Not enough camera positions"
    x = dist*np.cos(theta)*np.sin(phi); y = dist*np.sin(theta)*np.sin(phi); z = dist*np.cos(phi)

    # PNG writer
    w2if = vtk.vtkWindowToImageFilter(); w2if.SetInput(ren_win); w2if.SetInputBufferTypeToRGBA(); w2if.ReadFrontBufferOff()
    writer = vtk.vtkPNGWriter(); writer.SetCompressionLevel(9); writer.SetInputConnection(w2if.GetOutputPort())

    # Intrinsics files
    write_intrinsics(root, rotation_steps, img_size, fov)

    # Render loop
    train_cnt = test_cnt = 0
    for i in range(rotation_steps):
        cam.SetPosition(float(x[i]),float(y[i]),float(z[i])); cam.SetFocalPoint(0,0,0); cam.SetViewUp(0,0,1)
        ren.ResetCameraClippingRange(); ren_win.Render(); w2if.Modified(); w2if.Update()
        split = "train" if (i+1)%10 else "test"; idx_out = train_cnt if split=="train" else test_cnt
        if split=="train": train_cnt+=1
        else: test_cnt+=1
        # Pose
        c2w = compute_c2w(cam)
        write_matrix(root/split/"pose"/f"{split}_{idx_out}.txt", c2w)
        # Bounds
        cam_pos = np.linalg.norm([x[i],y[i],z[i]])
        near = cam_pos - canonical_size*0.6; far=cam_pos+canonical_size*0.6
        (root/split/"bounds"/f"{split}_{idx_out}.txt").write_text(f"{near:.6f}\n{far:.6f}\n")
        # Image
        img_path = root/"imgs"/f"{split}_{idx_out}.png"; writer.SetFileName(str(img_path)); writer.Write()

    # Sanity checks
    def sanity_check(view='train', idx=0):
        print(f"Sanity check for {view}_{idx}:")
        # Load intrinsics (K) and pose (c2w)
        K = np.loadtxt(root/view/"intrinsics"/f"{view}_{idx}.txt").reshape(4,4)
        c2w = np.loadtxt(root/view/"pose"/f"{view}_{idx}.txt").reshape(4,4)
        # Compute world-to-camera (w2c) by inverting c2w
        w2c = np.linalg.inv(c2w)
        # Project world origin [0,0,0,1]^T: pixel = K * [I|0] * w2c * [Xw;1]
        # For world origin Xw=0, pixel ~ K @ w2c @ [0,0,0,1] = K @ [0;0;0;1] = [cx; cy; 1]
        uvw = K @ w2c @ np.array([0,0,0,1], dtype=float)
        u, v = uvw[0]/uvw[2], uvw[1]/uvw[2]
        # Principal point from K
        cx, cy = K[0,2], K[1,2]
        print(f"  Projected origin to (u={u:.4f}, v={v:.4f}), principal point = ({cx:.4f}, {cy:.4f})")
        # Check focal lengths
        fx, fy = K[0,0], K[1,1]
        print(f"  Focal lengths fx={fx:.4f}, fy={fy:.4f}, ratio fy/fx={fy/fx:.6f}")
        # Verify close to principal point
        if not (abs(u-cx) < 1e-3 and abs(v-cy) < 1e-3):
            print(f"  WARNING: projection offset = ({u-cx:.4e}, {v-cy:.4e})")
    
    sanity_check('train',0)
    sanity_check('test',0)

    print("Rendering complete. Outputs in:", root)

if __name__=="__main__":
    p=argparse.ArgumentParser(description="Generate a NeRF dataset from STL with sanity checks")
    p.add_argument("--stl",required=True,help="Input STL file")
    p.add_argument("--out",required=True,help="Output dir")
    p.add_argument("--steps",type=int,default=32,help="Number of views")
    p.add_argument("--radius",type=float,default=1.5,help="Camera radius multiplier")
    p.add_argument("--img_size",type=int,default=400,help="Image size (px)")
    p.add_argument("--fov",type=float,default=60.0,help="FOV (deg)")
    args=p.parse_args()
    main(args.stl,args.out,rotation_steps=args.steps,radius=args.radius,img_size=args.img_size,fov=args.fov)
