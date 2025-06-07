import vtk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import math
import random
from vtk.util import numpy_support

# === CONFIG ===
MAX_SAMPLES = 10  # Set to None for all files, or an integer for a quick test

# Use absolute path for interim folder
input_folder = r"C:\Users\super\Documents\Github\sequoia\data\processed\interim"
output_root_folder = r"C:\Users\super\Documents\Github\sequoia\data\processed\pixel2mesh_data"

def get_camera_positions(num_views=8, distance=2.5):
    positions = []
    angles = [
        (45, 30), (-45, 30), (135, 30), (-135, 30),
        (45, -30), (-45, -30), (135, -30), (-135, -30)
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
    print("   Vertices rank: {} (should be 3 for 3D shape)".format(rank))
    if rank < 3:
        print("❌ WARNING: Vertices in {} do not span a 3D shape!".format(os.path.basename(dat_path)))
    else:
        print("✅ Vertices span a 3D shape.")

def render_multiview_data(mesh_path, output_dir, dat_path, image_size=(224, 224), num_views=8):
    os.makedirs(output_dir, exist_ok=True)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    polydata = reader.GetOutput()
    bounds = polydata.GetBounds()
    center = np.array([(bounds[0] + bounds[1]) / 2.0, 
                       (bounds[2] + bounds[3]) / 2.0, 
                       (bounds[4] + bounds[5]) / 2.0])
    scale = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) / 2.0
    if scale == 0.0:
        print("❌ Warning: Mesh {} has zero size and will be skipped.".format(os.path.basename(mesh_path)))
        return
    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])
    transform.Scale(1.0 / scale, 1.0 / scale, 1.0 / scale)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    normalized_polydata = transform_filter.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(normalized_polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(*image_size)
    camera_positions = get_camera_positions(num_views)
    all_camera_meta = []
    rendering_dir = os.path.join(output_dir, 'rendering')
    os.makedirs(rendering_dir, exist_ok=True)
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
        image_path = os.path.join(rendering_dir, '{:02d}.png'.format(i))
        plt.imsave(image_path, np.flipud(numpy_image))
        camera_meta = [angles[0], angles[1], 0, np.linalg.norm(position), camera.GetViewAngle()]
        all_camera_meta.append(camera_meta)
    metadata_path = os.path.join(rendering_dir, 'rendering_metadata.txt')
    np.savetxt(metadata_path, np.array(all_camera_meta), fmt='%f')
    vertices = numpy_support.vtk_to_numpy(normalized_polydata.GetPoints().GetData())
    faces_vtk = numpy_support.vtk_to_numpy(normalized_polydata.GetPolys().GetData())
    faces = faces_vtk.reshape(-1, 4)[:, 1:]
    dat_content = (vertices.astype(np.float32), faces.astype(np.int32))
    with open(dat_path, 'wb') as f:
        pickle.dump(dat_content, f, protocol=2)
    print("✅ Processed {} -> {}".format(mesh_path, output_dir))
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

    all_stl = [f for f in os.listdir(input_folder) if f.endswith(".stl")]
    all_dat = ["pollen_" + os.path.splitext(f)[0] + "_00.dat" for f in all_stl]
    all_ids = [os.path.splitext(f)[0] for f in all_stl]

    combined = list(zip(all_ids, all_dat))
    random.shuffle(combined)
    if MAX_SAMPLES is not None:
        combined = combined[:MAX_SAMPLES]
    split_idx = int(0.7 * len(combined))
    train = combined[:split_idx]
    test = combined[split_idx:]

    write_split_file([dat for _, dat in train], os.path.join(category_folder, "train_split.txt"))
    write_split_file([dat for _, dat in test], os.path.join(category_folder, "test_split.txt"))

    splits = [("train", train), ("test", test)]
    for split_name, split_items in splits:
        split_dir = os.path.join(category_folder, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for pollen_id, dat_name in split_items:
            mesh_path = os.path.join(input_folder, "{}.stl".format(pollen_id))
            if not os.path.exists(mesh_path):
                print("❌ Mesh file not found: {}".format(mesh_path))
                continue
            output_model_dir = os.path.join(split_dir, pollen_id)
            dat_path = os.path.join(split_dir, dat_name)
            try:
                print("🔄 Processing {}/{}...".format(split_name, pollen_id))
                render_multiview_data(mesh_path, output_model_dir, dat_path, num_views=8)
            except Exception as e:
                print("❌ Failed to process {}: {}".format(pollen_id, e))