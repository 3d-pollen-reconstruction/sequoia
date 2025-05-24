import argparse
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
import bpy
from mathutils import Vector
import util
import blender_interface

# CLI args
p = argparse.ArgumentParser(description='Render meshes into PixelNeRF-style train/val/test splits.')
p.add_argument('--mesh_dir', type=str, required=False, help='Directory of .obj or .stl meshes.')
p.add_argument('--output_dir', type=str, required=True, help='Base output directory.')
p.add_argument('--num_observations', type=int, default=128, help='Number of views per object for training.')
p.add_argument('--resolution', type=int, default=256, help='Image resolution.')
p.add_argument('--mesh_fpath', type=str, help='Path to a single mesh file to process')
argv = sys.argv[sys.argv.index("--") + 1:]
opt = p.parse_args(argv)

# Output subdirs
train_dir = os.path.join(opt.output_dir, "pollen_train")
val_dir   = os.path.join(opt.output_dir, "pollen_val")
test_dir  = os.path.join(opt.output_dir, "pollen_test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Mesh list (sorted = stable index-based split)
mesh_files = sorted([
    os.path.join(opt.mesh_dir, f)
    for f in os.listdir(opt.mesh_dir)
    if f.lower().endswith(('.obj', '.stl'))
])

# Index-based split (PixelNeRF-style)
n = len(mesh_files)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
splits = {
    "train": mesh_files[:n_train],
    "val":   mesh_files[n_train:n_train + n_val],
    "test":  mesh_files[n_train + n_val:]
}

# Renderer
renderer = blender_interface.BlenderInterface(resolution=opt.resolution)

# Per-split rendering
for split_name, files in splits.items():
    print("Rendering {} meshes for split: {}".format(len(files), split_name))
    split_output = {
        "train": train_dir,
        "val": val_dir,
        "test": test_dir
    }[split_name]

    for mesh_idx, mesh_fpath in enumerate(files):
        mesh_name = os.path.splitext(os.path.basename(mesh_fpath))[0]
        instance_dir = os.path.join(split_output, mesh_name)

        # Import mesh and normalize to fit inside unit sphere
        renderer.import_mesh(mesh_fpath, scale=1., object_world_matrix=None)
        obj = bpy.context.selected_objects[0]

        # Normalize mesh to unit scale
        bbox_corners = [obj.matrix_world * Vector(corner) for corner in obj.bound_box]
        center = sum(bbox_corners, Vector((0.0, 0.0, 0.0))) / 8.0
        radius = max((v - center).length for v in bbox_corners)
        obj.scale = (1.0 / radius, 1.0 / radius, 1.0 / radius)  # uniform scale        
        bpy.context.scene.update()


        # Recompute center after scaling
        bbox_corners = [obj.matrix_world * Vector(corner) for corner in obj.bound_box]
        center = sum(bbox_corners, Vector((0.0, 0.0, 0.0))) / 8.0
        obj_location = -np.array(center).reshape(1, 3)
        sphere_radius = 2.0  # fixed virtual sphere size

        bpy.ops.object.select_all(action='DESELECT')
        obj.select = True
        bpy.ops.object.delete()

        # Generate camera views
        if split_name == 'train':
            cam_locations = util.sample_spherical(opt.num_observations, sphere_radius)
        else:
            cam_locations = util.get_archimedean_spiral(sphere_radius, 250)

        cv_poses = util.look_at(cam_locations, np.zeros((1, 3)))
        blender_poses = [util.cv_cam2world_to_bcam2world(m) for m in cv_poses]

        # Object pose
        rot_mat = np.eye(3)
        hom_coords = np.array([[0., 0., 0., 1.]])
        obj_pose = np.concatenate((rot_mat, obj_location.reshape(3, 1)), axis=-1)
        obj_pose = np.concatenate((obj_pose, hom_coords), axis=0)

        # Import again with normalization applied
        renderer.import_mesh(mesh_fpath, scale=1.0 / radius, object_world_matrix=obj_pose)

        # Render (will skip views that result in empty or invalid output)
        renderer.render(instance_dir, blender_poses, write_cam_params=True, object_radius=sphere_radius)

split_summary = {
    split: [os.path.splitext(os.path.basename(f))[0] for f in files]
    for split, files in splits.items()
}

# Save to JSON file
split_json_path = os.path.join(opt.output_dir, "split_summary.json")
with open(split_json_path, "w") as f:
    json.dump(split_summary, f, indent=4)

print("Saved split information to: {split_json_path}".format(split_json_path=split_json_path))