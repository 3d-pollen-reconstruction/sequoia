
import vtk
import numpy as np
import os
import argparse
import math
import sys  # Import sys for sys.exit
import traceback # For detailed error printing
# Optional: Import gc for garbage collection if memory issues arise
# import gc

# --- Helper Functions ---

def get_fibonacci_sphere_points(samples):
    """Generates points on a sphere using the Fibonacci lattice method."""
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    # Ensure samples is at least 1 to avoid division by zero or invalid range
    if samples < 1:
        samples = 1

    for i in range(samples):
        # Handle the case samples = 1 correctly
        if samples == 1:
            y = 0.0 # Place the single point at the equator for consistency
        else:
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            # Clamp y to avoid math domain error in sqrt for edge cases
            y = max(-1.0, min(1.0, y))

        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)


def vtk_matrix_to_numpy(vtk_matrix):
    """Converts a VTK 4x4 matrix to a NumPy array."""
    mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            mat[i, j] = vtk_matrix.GetElement(i, j)
    return mat


def make_dirs(path):
    """Creates directories if they don't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


# --- Core Dataset Generation Logic for a Single Mesh ---

def generate_dataset_for_single_mesh(stl_file_path, args):
    """Generates the dataset structure for a single input STL file."""
    print(f"\n--- Processing Mesh: {os.path.basename(stl_file_path)} ---")

    # --- 1. Load STL and Get Bounds ---
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file_path)
    try:
        reader.Update()
    except Exception as e:
        print(f"VTK Error reading STL file {stl_file_path}: {e}")
        traceback.print_exc()
        return

    polydata = reader.GetOutput()
    if not polydata or polydata.GetNumberOfPoints() == 0:
        print(f"Error: Could not read STL file or STL file is empty: {stl_file_path}")
        return

    # Check for degenerate polydata (e.g., only lines or vertices)
    if polydata.GetNumberOfPolys() == 0 and polydata.GetNumberOfStrips() == 0:
         print(f"Warning: STL file {stl_file_path} contains no polygons or triangle strips. Skipping.")
         return

    bounds = polydata.GetBounds()
    # Check if bounds are valid
    if not all(math.isfinite(b) for b in bounds) or \
       bounds[1] <= bounds[0] or bounds[3] <= bounds[2] or bounds[5] <= bounds[4]:
        print(f"Error: Invalid bounds computed for STL file: {stl_file_path}. Bounds: {bounds}. Skipping.")
        return

    center = np.array(polydata.GetCenter())
    print(f"Original Bounds: {bounds}")
    print(f"Original Center: {center}")

    # --- 2. Center & Scale ---
    translate = vtk.vtkTransform()
    translate.Translate(-center[0], -center[1], -center[2])

    dims = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    # Handle potential zero dimensions robustly
    dims[dims <= 1e-9] = 1e-9 # Use a small epsilon to avoid division by zero or instability
    max_dim = np.max(dims)

    # Check max_dim again after potential epsilon adjustment
    if max_dim <= 1e-9:
        print(f"Warning: Mesh has zero or near-zero maximum dimension after epsilon adjustment: {stl_file_path}. Skipping scaling.")
        scale_factor = 1.0
    else:
        scale_factor = args.target_size / max_dim
    print(f"Scaling factor: {scale_factor}")

    scale = vtk.vtkTransform()
    scale.Scale(scale_factor, scale_factor, scale_factor)

    combined = vtk.vtkTransform()
    combined.Concatenate(scale) # Apply scale first
    combined.Concatenate(translate) # Then translate

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(polydata)
    tf.SetTransform(combined)
    try:
        tf.Update()
    except Exception as e:
        print(f"Error during transform update for {stl_file_path}: {e}")
        traceback.print_exc()
        return

    transformed_polydata = tf.GetOutput()
    if not transformed_polydata or transformed_polydata.GetNumberOfPoints() == 0:
         print(f"Error: Mesh became empty after transformation: {stl_file_path}")
         return
    transformed_bounds = transformed_polydata.GetBounds()
    # Check transformed bounds
    if not all(math.isfinite(b) for b in transformed_bounds) or \
       transformed_bounds[1] <= transformed_bounds[0] or \
       transformed_bounds[3] <= transformed_bounds[2] or \
       transformed_bounds[5] <= transformed_bounds[4]:
        print(f"Error: Invalid transformed bounds computed: {transformed_bounds}. Skipping.")
        return
    print(f"Transformed Bounds: {transformed_bounds}")

    # --- 3. Rendering Pipeline Setup ---
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tf.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # --- Modify Material Properties Here ---
    prop = actor.GetProperty()
    prop.SetColor(0.9, 0.4, 0.1)
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.3)
    prop.SetSpecularPower(30)
    prop.SetAmbient(0.1)
    prop.SetInterpolationToGouraud()
    # --- End Material Modification ---

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(*args.bg_color)

    # single camera‑mounted headlight for PixelNeRF
    light = vtk.vtkLight()
    light.SetLightTypeToHeadlight()
    renderer.AddLight(light)
    renderer.LightFollowCameraOn()

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(args.img_width, args.img_height)
    try:
        window.Render()
    except Exception as e:
        print(f"Warning: Initial window render failed for {stl_file_path}: {e}. Continuing...")

    # --- 4. Camera & Intrinsics ---
    cam = renderer.GetActiveCamera()
    if not cam:
        print("Error: Failed to get active camera from renderer. Skipping.")
        return
    cam.ParallelProjectionOff()
    cam.SetViewAngle(args.view_angle)

    fov = math.radians(args.view_angle)
    if abs(fov) < 1e-6 or abs(fov - math.pi) < 1e-6:
         fov = np.clip(fov, 1e-6, math.pi - 1e-6)
    tan2 = math.tan(fov/2.0)
    if abs(tan2) <= 1e-9:
        print("Error: tan(FOV/2) is near zero. Cannot compute intrinsics. Check --view_angle.")
        return
    fy = args.img_height / (2.0 * tan2)
    fx = fy
    cx = args.img_width / 2.0
    cy = args.img_height / 2.0
    w = args.img_width
    h = args.img_height
    print(f"Intrinsics: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.2f}, cy={cy:.2f}")

    max_t_dim = np.max([
        transformed_bounds[1]-transformed_bounds[0],
        transformed_bounds[3]-transformed_bounds[2],
        transformed_bounds[5]-transformed_bounds[4]
    ])
    if max_t_dim <= 1e-9:
        max_t_dim = args.target_size
        if max_t_dim <= 1e-9:
             print("Error: target_size is also too small. Cannot estimate distance.")
             return

    cam_dist = (max_t_dim / 2.0) / tan2
    cam_dist *= args.distance_factor
    if cam_dist <= 1e-6:
        cam_dist = 0.1
    print(f"Estimated Camera distance: {cam_dist:.4f}")

    # --- 5. Views & Splits ---
    print(f"Generating {args.num_views} views...")
    dirs = get_fibonacci_sphere_points(args.num_views)
    positions = dirs * cam_dist
    total = args.num_views
    n_tr = int(total * args.train_split)
    n_val = int(total * args.val_split)
    n_te = total - n_tr - n_val
    print(f"Splits: Train={n_tr}, Val={n_val}, Test={n_te}")

    out_base = args.output_dir
    make_dirs(out_base)
    obj_name = os.path.splitext(os.path.basename(stl_file_path))[0]

    # --- Write Intrinsics File (Once per split) ---
    for split_name, count in zip(["train", "val", "test"], [n_tr, n_val, n_te]):
        if count <= 0:
            continue
        split_dir = os.path.join(out_base, f"pollen_{split_name}", obj_name)
        make_dirs(split_dir)
        intrinsics_path = os.path.join(split_dir, "intrinsics.txt")
        try:
            with open(intrinsics_path, 'w') as f:
                f.write(f"{fx:.8f} {cx:.8f} {cy:.8f} 0.\n")
                f.write("0. 0. 0.\n")
                f.write("1.\n")
                f.write(f"{w} {h}\n")
            print(f"Saved intrinsics for {split_name} ({intrinsics_path})")
        except IOError as e:
            print(f"Error writing intrinsics file {intrinsics_path}: {e}")
            # clean up and abort this mesh
            del reader, polydata, translate, scale, combined, tf, transformed_polydata
            del mapper, actor, prop, renderer, window, cam, light
            return

    # --- Setup Image Writer ---
    writer = vtk.vtkPNGWriter()
    to_img = vtk.vtkWindowToImageFilter()
    to_img.SetInput(window)
    to_img.ReadFrontBufferOff()
    to_img.SetInputBufferTypeToRGBA()
    writer.SetInputConnection(to_img.GetOutputPort())

    # --- Render Each View ---
    counters = {"train": 0, "val": 0, "test": 0}
    for i_idx, view_idx in enumerate(np.arange(total)):
        if view_idx < n_tr:
            split = "train"
        elif view_idx < n_tr + n_val:
            split = "val"
        else:
            split = "test"
        idx = counters[split]

        pos = positions[view_idx]
        cam.SetPosition(*pos)
        cam.SetFocalPoint(0,0,0)

        up_vec = np.array([0.0, 1.0, 0.0])
        pos_norm = np.linalg.norm(pos)
        if pos_norm < 1e-9:
             continue
        view_dir = -pos / pos_norm
        if abs(np.dot(view_dir, up_vec)) > (1.0 - 1e-6):
            up_vec = np.array([0.0, 0.0, 1.0 if pos[1]>0 else -1.0])
        cam.SetViewUp(*up_vec)

        obj_radius = max_t_dim / 2.0
        near = max(0.01, cam_dist - obj_radius * 1.5)
        far  = cam_dist + obj_radius * 1.5
        if near >= far:
             far = near + max(0.1, obj_radius * 0.5)
             if near >= far:
                  near = cam_dist * 0.1
                  far  = cam_dist * 10.0
        cam.SetClippingRange(near, far)

        try:
             window.Render()
        except Exception:
             continue

        split_base = os.path.join(out_base, f"pollen_{split}", obj_name)
        pose_dir = os.path.join(split_base, "pose")
        rgb_dir  = os.path.join(split_base, "rgb")
        make_dirs(pose_dir)
        make_dirs(rgb_dir)

        try:
            w2c = cam.GetModelViewTransformMatrix()
            w2c_np = vtk_matrix_to_numpy(w2c)
            pose_p = os.path.join(pose_dir, f"{idx:08d}.txt")
            np.savetxt(pose_p, w2c_np.flatten()[np.newaxis,:], fmt='%.17g')
        except Exception:
             continue

        try:
            img_p = os.path.join(rgb_dir, f"{idx:08d}.png")
            to_img.Modified()
            to_img.Update()
            writer.SetFileName(img_p)
            writer.Write()
        except Exception:
             continue

        counters[split] += 1
        if (i_idx+1)%20==0 or i_idx==total-1:
            print(f"Saved view {i_idx+1}/{total} [{split} idx {idx}]")

    # write near/far for each split
    near = max(0.01, cam_dist - obj_radius * 1.5)
    far  = cam_dist + obj_radius * 1.5
    for split_name in ["train","val","test"]:
        d = os.path.join(out_base, f"pollen_{split_name}", obj_name)
        if os.path.isdir(d):
            nf = os.path.join(d,"near_far.txt")
            try:
                with open(nf,"w") as f:
                    f.write(f"{near:.6f} {far:.6f}\n")
            except Exception:
                pass

    print(f"--- Finished processing {obj_name} ---")
    print(f"Output saved to {out_base}/pollen_train|val|test/{obj_name}")

    # final cleanup
    del reader, polydata, translate, scale, combined, tf, transformed_polydata
    del mapper, actor, prop, renderer, window, cam, light, writer, to_img


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate PixelNeRF-style dataset from STL meshes using VTK.")
    p.add_argument("--mesh_dir", required=True, help="Directory containing input STL files.")
    p.add_argument("--output_dir", default="pollen_dataset_structured", help="Base directory for the output dataset.")
    p.add_argument("--num_views", type=int, default=100, help="Total number of views to render per mesh.")
    p.add_argument("--img_width", type=int, default=128, help="Width of the output images.")
    p.add_argument("--img_height", type=int, default=128, help="Height of the output images.")
    p.add_argument("--view_angle", type=float, default=51.93, help="Camera vertical field of view in degrees.")
    p.add_argument("--target_size", type=float, default=1.5, help="Target maximum dimension for the object after scaling (world units).")
    p.add_argument("--distance_factor", type=float, default=1.8, help="Multiplier for the calculated camera distance (>=1.0).")
    p.add_argument("--train_split", type=float, default=0.8, help="Fraction of views for the training set (0.0 to 1.0).")
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of views for the validation set (0.0 to 1.0).")
    p.add_argument("--bg_color", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Background color (R G B, values 0.0 to 1.0).")
    args = p.parse_args()

    # Input validation
    if not os.path.isdir(args.mesh_dir):
        print(f"Error: Input mesh directory not found: {args.mesh_dir}")
        sys.exit(1)
    if args.train_split < 0 or args.val_split < 0 or args.train_split + args.val_split > 1.0:
        print("Error: Invalid train/val split values. Must be >= 0 and sum <= 1.0.")
        sys.exit(1)
    if args.num_views <= 0 or args.img_width <= 0 or args.img_height <= 0:
        print("Error: Number of views and image dimensions must be positive.")
        sys.exit(1)
    if args.view_angle <= 0 or args.view_angle >= 180:
        print("Error: View angle must be between 0 and 180 degrees.")
        sys.exit(1)
    if args.target_size <= 0:
        print("Error: Target size must be positive.")
        sys.exit(1)
    if args.distance_factor < 1.0:
        print("Warning: Distance factor is less than 1.0, camera might be inside the object.")

    files = [f for f in os.listdir(args.mesh_dir) if f.lower().endswith('.stl')]
    if not files:
        print(f"Error: No STL files found in directory: {args.mesh_dir}")
        sys.exit(1)

    print(f"Found {len(files)} STL files. Starting dataset generation...")
    processed_count = 0
    skipped_count = 0
    total_files = len(files)
    for idx, filename in enumerate(files):
        print(f"\n[{idx+1}/{total_files}] Processing file: {filename}")
        stl_path = os.path.join(args.mesh_dir, filename)
        try:
            generate_dataset_for_single_mesh(stl_path, args)
            processed_count += 1
        except Exception as e:
            print(f"\n---!!! Critical Error processing {filename}: {e} !!!---")
            traceback.print_exc()
            skipped_count += 1

    print("\n--- Dataset Generation Complete ---")
    print(f"Successfully processed: {processed_count} files")
    print(f"Skipped due to errors: {skipped_count} files")
