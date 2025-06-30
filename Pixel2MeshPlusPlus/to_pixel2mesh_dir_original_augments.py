import vtk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import math
import random
from vtk.util import numpy_support
import trimesh
import open3d as o3d
import json
import glob

# === CONFIG ===
MAX_SAMPLES = 207  # Set to None for all files, or an integer for a quick test

# Use absolute path for interim folder
input_folder = r"C:\Users\super\Documents\Github\sequoia\data\processed\augmentation"
output_root_folder = (
    r"C:\Users\super\Documents\Github\sequoia\data\processed\pixel2mesh_original_augmented"
)

def normalize_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    
    center = np.mean(vertices, axis=0)
    vertices -= center

    scale = np.max(np.linalg.norm(vertices, axis=1))
    vertices /= scale

    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # ✅ Check if normalized
    new_center = np.mean(vertices, axis=0)
    max_radius = np.max(np.linalg.norm(vertices, axis=1))

    print(f"   🔎 Mesh center after normalization: {new_center}")
    print(f"   🔎 Max vertex norm (should be ~1): {max_radius:.6f}")
    if np.allclose(new_center, np.zeros(3), atol=1e-3) and np.isclose(max_radius, 1.0, atol=1e-3):
        print("   ✅ Mesh is centered and normalized.")
    else:
        print("   ❌ Mesh normalization issue detected!")

    return mesh



def get_camera_positions(num_views=8, distance=2.5):
    positions = []

    angles = [
        (0, 30),
        (90, 30),
        (180, 30),
        (270, 30),
        (45, 30),
        (135, 30),
        (225, 30),
        (315, 30),
    ]
    for i in range(num_views):
        azimuth, elevation = angles[i]
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        x = distance * math.cos(el_rad) * math.sin(az_rad)
        y = distance * math.sin(el_rad)
        z = distance * math.cos(el_rad) * math.cos(az_rad)
        positions.append(((x, y, z), (azimuth, elevation)))
    return positions


def check_3d_shape(vertices, dat_path):
    centered = vertices - vertices.mean(axis=0)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    rank = np.sum(s > 1e-6)
    print(f"   Vertices rank: {rank} (should be 3 for 3D shape)")
    if rank < 3:
        print(
            f"❌ WARNING: Vertices in {os.path.basename(dat_path)} do not span a 3D shape!"
        )
    else:
        print(f"✅ Vertices span a 3D shape.")


def save_mesh_as_obj(polydata, obj_path):
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(obj_path)
    writer.SetInputData(polydata)
    writer.Update()


def render_multiview_data(
    mesh_path, output_dir, dat_path, image_size=(224, 224), num_views=8
):
    os.makedirs(output_dir, exist_ok=True)
    mesh_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    # === Load and normalize STL mesh ===
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    polydata = reader.GetOutput()
    bounds = polydata.GetBounds()
    center = np.array(
        [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]
    )
    scale = (
        max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) / 2.0
    )
    if scale == 0.0:
        print(
            f"❌ Warning: Mesh {os.path.basename(mesh_path)} has zero size and will be skipped."
        )
        return

    # === Normalize with VTK ===
    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])
    transform.Scale(1.0 / scale, 1.0 / scale, 1.0 / scale)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    normalized_polydata = transform_filter.GetOutput()

    # === Convert to Trimesh & simplify ===
    vertices = numpy_support.vtk_to_numpy(normalized_polydata.GetPoints().GetData())
    faces_raw = numpy_support.vtk_to_numpy(normalized_polydata.GetPolys().GetData())
    faces = faces_raw.reshape(-1, 4)[:, 1:]

    # Pad triangle faces to (N, 4) with 0 for mesh processing (trimesh/open3d)
    if faces.shape[1] == 3:
        faces_proc = np.pad(faces, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    else:
        faces_proc = faces

    try:
        # Use only the first 3 columns for trimesh (triangles)
        mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces_proc[:, :3], process=False)
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces_proc[:, :3])
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(20000)
        mesh_o3d = normalize_mesh(mesh_o3d)  # ✅ Use your function here
        mesh_o3d.compute_vertex_normals()

        vertices = np.asarray(mesh_o3d.vertices)
        faces_simple = np.asarray(mesh_o3d.triangles)
        normals = np.asarray(mesh_o3d.vertex_normals)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        
        # After simplification, pad faces again if needed (for saving)
        if faces_simple.shape[1] == 3:
            faces_save = np.pad(faces_simple, ((0, 0), (0, 1)), mode='constant', constant_values=-1)
        else:
            faces_save = faces_simple
    except Exception as e:
        print(f"❌ Mesh simplification failed: {e}")
        return

    # === Save .obj of simplified mesh ===
    #simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces_simple, process=False)
    #simplified_mesh.export(os.path.join(mesh_dir, "model.obj"))

    # === Use simplified mesh for VTK rendering ===
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(vertices))
    vtk_cells = vtk.vtkCellArray()
    for face in faces_simple:
        vtk_cells.InsertNextCell(3)
        for idx in face:
            vtk_cells.InsertCellPoint(idx)
    vtk_polydata = vtk.vtkPolyData()
    vtk_polydata.SetPoints(vtk_points)
    vtk_polydata.SetPolys(vtk_cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vtk_polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(*image_size)

    # === Render views ===
    rendering_dir = os.path.join(output_dir, "rendering")
    os.makedirs(rendering_dir, exist_ok=True)
    all_camera_meta = []
    camera_positions = get_camera_positions(num_views)
    for i, (position, angles) in enumerate(camera_positions):
        camera = renderer.GetActiveCamera()
        camera.SetPosition(position)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        renderer.ResetCamera()
        renderer.ResetCameraClippingRange()
        render_window.Render()

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.SetInputBufferTypeToRGB()
        window_to_image_filter.ReadFrontBufferOff()
        window_to_image_filter.Update()

        vtk_image = window_to_image_filter.GetOutput()
        dims = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        numpy_image = numpy_support.vtk_to_numpy(vtk_array).reshape(dims[1], dims[0], 3)
        image_path = os.path.join(rendering_dir, f"{i:02d}.png")
        plt.imsave(image_path, np.flipud(numpy_image))
        camera_meta = [
            angles[0],
            angles[1],
            0,
            np.linalg.norm(position),
            camera.GetViewAngle(),
        ]
        all_camera_meta.append(camera_meta)

    np.savetxt(
        os.path.join(rendering_dir, "rendering_metadata.txt"),
        np.array(all_camera_meta),
        fmt="%f",
    )

    # === Save npz file ===
    print("DEBUG points shape:", vertices.shape)
    print("DEBUG normals shape:", normals.shape)
    if normals.shape[0] != vertices.shape[0]:
        print(f"❌ Normals/points count mismatch for {dat_path}")
        return

    pollen_id = os.path.splitext(os.path.basename(dat_path))[0]
    np.savez_compressed(
        dat_path.replace(".dat", ".npz"),
        name=pollen_id,
        points=vertices.astype(np.float32),
        faces=faces_save.astype(np.int32),
        normals=normals.astype(np.float32),
    )
    print(f"✅ Processed {mesh_path} -> {output_dir}")
    check_3d_shape(vertices, dat_path)


def write_split_file(split_list, split_path):
    with open(split_path, "w") as f:
        for item in split_list:
            f.write(item + "\n")


if __name__ == "__main__":
    os.makedirs(output_root_folder, exist_ok=True)
    category_name = "pollen"
    category_folder = os.path.join(output_root_folder, category_name)
    os.makedirs(category_folder, exist_ok=True)

    # Load splits from JSON
    splits_json_path = r"C:\Users\super\Documents\Github\sequoia\data\pollen_augmented\splits.json"
    with open(splits_json_path, "r") as f:
        splits = json.load(f)

    # Recursively find all STL files in input_folder and subfolders
    all_stl = glob.glob(os.path.join(input_folder, "**", "*.stl"), recursive=True)

    # Map: id (from split, without .stl) -> list of STL file paths that start with id
    id_to_stl = {}
    for stl_path in all_stl:
        stl_file = os.path.basename(stl_path)
        stl_base = os.path.splitext(stl_file)[0]
        # Store all possible stl files for each id prefix
        for split_name in ["train", "val", "test"]:
            for split_id in splits.get(split_name, []):
                split_id_base = os.path.splitext(split_id)[0]
                if stl_file.startswith(split_id_base) and stl_file.endswith(".stl"):
                    id_to_stl.setdefault(split_id_base, []).append(stl_path)

    # Prepare split dict: split_name -> list of (id, npz_name, dir_name, mesh_path)
    split_dict = {}
    for split_name in ["train", "val", "test"]:
        split_files = splits.get(split_name, [])
        split_items = []
        for stl_file in split_files:
            split_id = os.path.splitext(stl_file)[0]
            stl_paths = id_to_stl.get(split_id, [])
            if not stl_paths:
                print(f"❌ STL file listed in split but not found: {stl_file}")
                continue
            for mesh_path in stl_paths:
                # Use the full mesh filename (without extension) as postfix_id
                postfix_id = os.path.splitext(os.path.basename(mesh_path))[0]
                npz_name = f"pollen_{postfix_id}_00.npz"
                dir_name = f"{postfix_id}_00"
                split_items.append((postfix_id, npz_name, dir_name, mesh_path))
        split_dict[split_name] = split_items

    # Write split files with .npz filenames (with postfix)
    for split_name in ["train", "val", "test"]:
        split_items = split_dict[split_name]
        write_split_file(
            [npz for _, npz, _, _ in split_items],
            os.path.join(category_folder, f"{split_name}_split.txt"),
        )

    # Process each split
    for split_name in ["train", "val", "test"]:
        split_items = split_dict[split_name]
        split_dir = os.path.join(category_folder, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for pollen_id, npz_name, dir_name, mesh_path in split_items:
            if not os.path.exists(mesh_path):
                print(f"❌ Mesh file not found: {mesh_path}")
                continue
            output_model_dir = os.path.join(split_dir, dir_name)
            npz_path = os.path.join(split_dir, npz_name)
            # Skip if output folder or npz file already exists
            if os.path.exists(output_model_dir) or os.path.exists(npz_path):
                print(f"⏩ Skipping {split_name}/{pollen_id} (already exists)")
                continue
            dat_path = os.path.join(split_dir, npz_name.replace(".npz", ".dat"))
            try:
                print(f"🔄 Processing {split_name}/{pollen_id} ({os.path.basename(mesh_path)})...")
                render_multiview_data(
                    mesh_path, output_model_dir, dat_path, num_views=8
                )
            except Exception as e:
                print(f"❌ Failed to process {pollen_id}: {e}")
