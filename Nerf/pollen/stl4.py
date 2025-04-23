#!/usr/bin/env python
"""Generate a NeRF‑style dataset (RGBA PNGs + poses + intrinsics) using VTK.

Key points compared to original VTK script:
* **No Bounds Files:** Removed the per-view near/far bounds files and directories.
* **Fixed Intrinsics:** Uses a pre-defined intrinsics matrix (loaded from fox dataset)
  instead of calculating from FOV.
* **Adjusted Scaling:** Normalizes mesh to fit within [-1.5, 1.5] (diameter 3.0)
  to match MeshExtraction notebook scale.
* **OpenCV Pose Convention:** Transforms the computed C2W matrix to OpenCV
  convention (+X right, +Y down, +Z forward) before saving.
* **Default 100 Views:** Generates 100 views (90 train, 10 test) by default.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

# Attempt to import vtk, provide guidance if missing
try:
    import vtk
except ImportError:
    print("Error: VTK library not found.")
    print("Please install it, e.g., using: pip install vtk")
    sys.exit(1)

import numpy as np


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------

def _clean_and_mkdir(path: Path) -> None:
    """Removes directory if it exists, then creates it."""
    if path.exists():
        # Add error handling for directory removal
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Warning: Could not remove directory {path}: {e}")
            print("Please check permissions or close programs using this directory.")
            # Optionally exit if cleanup is critical
            # sys.exit(1)
    # Create directory, parents=True ensures intermediate dirs are made
    path.mkdir(parents=True, exist_ok=True)


def make_files(root: Path) -> None:
    """Create a fresh output directory tree (without bounds)."""
    # Clean/create top-level output and imgs directories
    _clean_and_mkdir(root)
    _clean_and_mkdir(root / "imgs")

    # Clean/create train and test subdirectories
    for split in ["train", "test"]:
        split_path = root / split
        _clean_and_mkdir(split_path)
        # Create pose and intrinsics subdirs
        for sub in ["pose", "intrinsics"]:
             # No need to clean these subdirs as the parent was just cleaned
             (split_path / sub).mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Intrinsics & pose I/O
# -----------------------------------------------------------------------------

def write_intrinsics(
    root: Path,
    n_views: int,
    fox_intrinsics_path: str = '../fox/train/intrinsics/train_0.txt'
) -> bool:
    """
    Writes a fixed 4x4 intrinsics matrix loaded from the specified file
    for all views. Returns True on success, False on failure.
    """
    intrinsics_file = Path(fox_intrinsics_path)
    # Fallback path if relative doesn't work
    if not intrinsics_file.exists():
         intrinsics_file = Path('Nerf - Copy/fox/train/intrinsics/train_0.txt')

    if not intrinsics_file.exists():
        print(f"Error: Fox intrinsics file not found at '{fox_intrinsics_path}' or fallback path.")
        print("Cannot proceed without the reference intrinsics matrix.")
        return False

    try:
        # Load the matrix, assuming space-separated values, potentially in one line or multiple
        intr_flat = np.loadtxt(intrinsics_file)
        if intr_flat.size != 16:
            print(f"Error: Intrinsics file '{intrinsics_file}' does not contain 16 values.")
            return False
        intr_matrix = intr_flat.reshape(4, 4)
        print(f"Successfully loaded intrinsics matrix from {intrinsics_file}:")
        print(intr_matrix)
    except Exception as e:
        print(f"Error loading or reshaping intrinsics from {intrinsics_file}: {e}")
        return False

    # Format the matrix values for writing
    intr_text = "\n".join(f"{v:.8f}" for v in intr_matrix.flatten())

    train_cnt = test_cnt = 0
    for j in range(n_views):
        split = "train" if (j + 1) % 10 else "test"
        idx = train_cnt if split == "train" else test_cnt
        if split == "train": train_cnt += 1
        else: test_cnt += 1

        out_file = root / split / "intrinsics" / f"{split}_{idx}.txt"
        try:
            out_file.write_text(intr_text)
        except Exception as e:
             print(f"Error writing intrinsics file {out_file}: {e}")
             # Decide if one failure should stop the whole process
             # return False
    print(f"Wrote fixed intrinsics to {train_cnt} train and {test_cnt} test files.")
    return True


def write_matrix(path: Path, mat: np.ndarray) -> None:
    """Write a 4×4 matrix in row-major, one value per line."""
    vals = np.asarray(mat, dtype=float).flatten()
    # Use higher precision matching typical NeRF datasets
    path.write_text("\n".join(f"{v:.8f}" for v in vals))


def compute_c2w_opencv(cam: vtk.vtkCamera) -> np.ndarray:
    """
    Computes the 4x4 camera-to-world matrix (C2W) and converts it
    to OpenCV convention (+X right, +Y down, +Z forward).
    """
    # Get the VTK view transform matrix (World-to-Camera, OpenGL convention)
    view_matrix_vtk = cam.GetViewTransformMatrix()
    M_vtk_w2c = np.eye(4, dtype=float)
    for i in range(4):
        for j in range(4):
            M_vtk_w2c[i, j] = view_matrix_vtk.GetElement(i, j)

    # Invert to get Camera-to-World (OpenGL convention)
    # C2W_gl = inv(W2C_gl)
    try:
        M_vtk_c2w_gl = np.linalg.inv(M_vtk_w2c)
    except np.linalg.LinAlgError:
        print("Error: Could not invert VTK view matrix. Skipping pose.")
        return None # Indicate failure

    # Transformation matrix to convert C2W_OpenGL to C2W_OpenCV
    # This flips the Y and Z axes of the camera's local coordinate system
    ogl_to_cv_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0], # Flip Y axis
        [0, 0, -1, 0], # Flip Z axis
        [0, 0, 0, 1]
    ])

    # Apply the transformation: C2W_cv = C2W_gl @ Transform
    M_c2w_cv = M_vtk_c2w_gl @ ogl_to_cv_transform

    return M_c2w_cv

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main(
    stl_file: str,
    output_dir: str,
    *,
    rotation_steps: int = 100, # Default to 100 views (90 train, 10 test)
    radius: float = 1.5,       # Camera distance multiplier (relative to object size/fov)
    img_size: int = 400,       # Image resolution
    fov: float = 60.0          # Camera Field of View (degrees) - Used for distance calc
) -> None:
    root = Path(output_dir).expanduser().resolve()
    make_files(root)

    # --- Intrinsics Setup ---
    # Write the fixed intrinsics loaded from the fox dataset file
    # The path might need adjustment depending on where you run the script from
    fox_intrinsics_path = '../fox/train/intrinsics/train_0.txt' # Adjust if needed
    if not write_intrinsics(root, rotation_steps, fox_intrinsics_path=fox_intrinsics_path):
         print("Failed to write intrinsics. Aborting.")
         return # Stop if intrinsics can't be prepared

    # --- Load & normalize mesh ---
    stl_file_path = Path(stl_file)
    if not stl_file_path.exists():
        print(f"Error: Input STL file not found at '{stl_file}'")
        return

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_file_path))
    reader.Update()

    # Check if reader has output
    if reader.GetOutput() is None or reader.GetOutput().GetNumberOfPoints() == 0:
        print(f"Error: Failed to read geometry from STL file '{stl_file}'")
        return

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.ComputePointNormalsOn()
    # Add consistency check for normals
    normals.ConsistencyOn()
    normals.SplittingOff() # Avoid splitting vertices
    normals.Update()
    poly = normals.GetOutput()

    # Calculate center and scale factor
    center = np.array(poly.GetCenter())
    bounds = np.array(poly.GetBounds()).reshape(3, 2)
    diameter = np.max(bounds[:,1] - bounds[:,0])

    # Target size based on MeshExtraction scale (-1.5 to 1.5 -> diameter 3.0)
    canonical_size = 3.0
    if diameter < 1e-6:
        print("Warning: Mesh diameter is very small. Scaling might be inaccurate.")
        scale_factor = 1.0
    else:
        scale_factor = canonical_size / diameter
    print(f"Mesh Center: {center}, Original Diameter: {diameter:.4f}, Scale Factor: {scale_factor:.4f}")

    # --- VTK Actor Setup ---
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # Apply scaling and centering using actor properties
    actor.SetScale(scale_factor)
    actor.SetOrigin(*center) # Set rotation/scaling origin to the mesh center
    actor.SetPosition(-center * scale_factor) # Translate the centered mesh to the world origin

    # Set visual properties
    prop = actor.GetProperty()
    prop.SetColor(0.8,0.8,0.8) # Grey color
    prop.SetAmbient(0.3)
    prop.SetDiffuse(0.7) # Increase diffuse slightly
    prop.SetSpecular(0.2)
    prop.SetSpecularPower(20) # Add some specular highlight

    # --- Scene Setup ---
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.0, 0.0, 0.0) # Black background
    ren.SetBackgroundAlpha(0.0)     # Transparent background
    ren.AddActor(actor)
    ren.ResetCamera() # Initial camera placement

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(img_size,img_size)
    ren_win.SetOffScreenRendering(True) # Enable offscreen rendering
    ren_win.SetAlphaBitPlanes(True)     # Enable alpha channel
    ren_win.SetMultiSamples(0)          # Disable multisampling for speed

    cam = ren.GetActiveCamera()
    cam.SetViewAngle(fov) # Set FOV

    # Use a headlight attached to the camera for simple, consistent lighting
    light = vtk.vtkLight()
    light.SetLightTypeToHeadlight()
    ren.AddLight(light)

    # --- Camera Trajectory ---
    fov_rad = np.deg2rad(fov)
    # Calculate distance based on the *normalized* size and FOV
    if np.tan(fov_rad / 2.0) < 1e-6:
         base_dist = canonical_size # Avoid division by zero
    else:
         base_dist = (canonical_size / 2.0) / np.tan(fov_rad / 2.0)
    dist = base_dist * radius # Apply radius multiplier
    print(f"Normalized Size: {canonical_size:.4f}, Base Distance: {base_dist:.4f}, Final Distance: {dist:.4f}")

    # Fibonacci spiral positions (upper hemisphere)
    n = rotation_steps * 2 # Generate more points initially
    idx_arr = np.arange(n)
    phi = np.arccos(1 - 2*(idx_arr+0.5)/n) # Inclination from +Z axis
    theta = 2*np.pi*idx_arr/((1+np.sqrt(5))/2) # Azimuth

    # Filter points to keep only upper hemisphere (phi from 0 to pi/2)
    # And ensure we have enough points
    mask = (phi >= 0) & (phi <= np.pi / 2.0)
    theta, phi = theta[mask], phi[mask]
    if len(theta) < rotation_steps:
         print(f"Warning: Generated only {len(theta)} points in upper hemisphere.")
         print("Consider increasing initial point count if more views are needed.")
         # Use all generated upper hemisphere points if fewer than requested
         actual_steps = len(theta)
    else:
         # Select the first 'rotation_steps' points from the filtered set
         theta = theta[:rotation_steps]
         phi = phi[:rotation_steps]
         actual_steps = rotation_steps

    # Convert spherical to Cartesian coordinates (Z is up)
    x = dist * np.sin(phi) * np.cos(theta)
    y = dist * np.sin(phi) * np.sin(theta)
    z = dist * np.cos(phi)
    print(f"Generating {actual_steps} views from camera positions...")

    # --- PNG Writer Setup ---
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    w2if.SetInputBufferTypeToRGBA() # Capture RGBA
    w2if.ReadFrontBufferOff()       # Important for offscreen rendering

    writer = vtk.vtkPNGWriter()
    writer.SetCompressionLevel(9) # Max compression
    writer.SetInputConnection(w2if.GetOutputPort())

    # --- Render Loop ---
    train_cnt = test_cnt = 0
    for i in range(actual_steps):
        # Set camera position, focal point (origin), and up vector (Z)
        cam.SetPosition(float(x[i]), float(y[i]), float(z[i]))
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetViewUp(0.0, 0.0, 1.0) # Z-up convention

        # Reset clipping range for each view - important!
        ren.ResetCameraClippingRange()
        ren_win.Render()

        # Update filter and writer
        w2if.Modified()
        # w2if.Update() # Update happens implicitly with writer.Write()

        # Determine split and index
        # Use modulo 10 for split (every 10th frame is test)
        is_test = (i + 1) % 10 == 0
        split = "test" if is_test else "train"
        idx_out = test_cnt if is_test else train_cnt

        # Pose Matrix (C2W, OpenCV convention)
        c2w_cv = compute_c2w_opencv(cam)
        if c2w_cv is None:
             print(f"Skipping view {i} due to pose calculation error.")
             continue # Skip if pose failed

        write_matrix(root / split / "pose" / f"{split}_{idx_out}.txt", c2w_cv)

        # Image Saving
        img_path = root / "imgs" / f"{split}_{idx_out}.png"
        writer.SetFileName(str(img_path))
        writer.Write()

        # Check if image file was actually created (basic check)
        if not img_path.exists() or img_path.stat().st_size == 0:
             print(f"Warning: Image file {img_path} was not created or is empty after write call.")
             # Optionally try rendering again or handle error
             # ren_win.Render()
             # writer.Write()
             # if not img_path.exists() or img_path.stat().st_size == 0:
             #     print(f"Error: Still failed to write image {img_path}. Skipping view.")
             #     # Clean up potentially failed pose/intrinsics?
             #     continue

        # Increment counters *after* successful processing
        if is_test: test_cnt += 1
        else: train_cnt += 1

        if (i + 1) % 10 == 0:
             print(f"Generated view {i+1}/{actual_steps} (Train: {train_cnt}, Test: {test_cnt})")


    # --- Sanity Checks ---
    def sanity_check(view='train', idx=0):
        """Performs basic reprojection check on saved pose/intrinsics."""
        print(f"\nSanity check for {view}_{idx}:")
        intr_path = root / view / "intrinsics" / f"{view}_{idx}.txt"
        pose_path = root / view / "pose" / f"{view}_{idx}.txt"

        if not intr_path.exists() or not pose_path.exists():
             print(f"  Skipping: Files not found ({intr_path.name}, {pose_path.name})")
             return

        try:
            # Load intrinsics (K) and pose (c2w)
            K = np.loadtxt(intr_path).reshape(4,4)
            c2w = np.loadtxt(pose_path).reshape(4,4)

            # Compute world-to-camera (w2c) by inverting c2w
            w2c = np.linalg.inv(c2w)

            # Project world origin [0,0,0,1]^T: pixel = K[:3,:3] @ w2c[:3,:] @ [Xw;1]
            # For world origin Xw=0, pixel_homogeneous = K[:3,:3] @ w2c[:3, 3]
            origin_world = np.array([0,0,0,1], dtype=float)
            origin_cam = w2c @ origin_world
            pixel_homogeneous = K[:3,:3] @ origin_cam[:3] # Project using 3x3 K

            # Check if projection is valid (z > 0)
            if pixel_homogeneous[2] < 1e-6:
                 print("  Warning: Projected origin has non-positive Z_cam. Check pose.")
                 u, v = -1, -1 # Invalid projection
            else:
                 u = pixel_homogeneous[0] / pixel_homogeneous[2]
                 v = pixel_homogeneous[1] / pixel_homogeneous[2]

            # Principal point from K
            cx, cy = K[0,2], K[1,2]
            print(f"  Projected origin to (u={u:.4f}, v={v:.4f}), Principal point = ({cx:.4f}, {cy:.4f})")

            # Check focal lengths
            fx, fy = K[0,0], K[1,1]
            print(f"  Focal lengths fx={fx:.4f}, fy={fy:.4f}")

            # Verify close to principal point (allow slightly larger tolerance)
            if not (abs(u-cx) < 1.0 and abs(v-cy) < 1.0):
                print(f"  WARNING: Projection offset = ({u-cx:.4e}, {v-cy:.4e})")
        except Exception as e:
            print(f"  Error during sanity check: {e}")

    # Perform checks only if files were generated
    if train_cnt > 0: sanity_check('train', 0)
    if test_cnt > 0: sanity_check('test', 0)

    print(f"\nRendering complete. {train_cnt} train views, {test_cnt} test views.")
    print(f"Outputs saved in: {root}")

if __name__=="__main__":
    p = argparse.ArgumentParser(description="Generate a NeRF dataset from STL using VTK (Fox Format)")
    p.add_argument("--stl", required=True, help="Input STL file path")
    p.add_argument("--out", required=True, help="Output directory path")
    p.add_argument("--steps", type=int, default=100, help="Number of views to generate (default: 100 for 90 train/10 test)")
    p.add_argument("--radius", type=float, default=1.5, help="Camera distance multiplier (relative to object size/fov)")
    p.add_argument("--img_size", type=int, default=400, help="Image size (px)")
    p.add_argument("--fov", type=float, default=60.0, help="Camera Field of View (deg) - used for distance calc")
    args = p.parse_args()

    # Basic validation
    if not Path(args.stl).exists():
         print(f"Error: Input STL file not found: {args.stl}")
         sys.exit(1)
    if args.steps < 10:
         print("Warning: Very few steps requested. Ensure enough for train/test split.")

    main(
        args.stl,
        args.out,
        rotation_steps=args.steps,
        radius=args.radius,
        img_size=args.img_size,
        fov=args.fov
    )
