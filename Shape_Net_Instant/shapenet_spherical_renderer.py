import argparse
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
import bpy
from mathutils import Vector, Matrix
import util
import blender_interface

def validate_views_for_instantmesh(num_views, split_name):
    """InstantMesh validation for different splits"""
    if split_name == "train" and num_views < 16:
        print("Warning: views may be insufficient for InstantMesh training (recommended: 32)")
    elif split_name in ["val", "test"] and num_views < 4:
        print("Warning: views may be insufficient for InstantMesh {split_name}")
    return True

def get_camera_positions_for_split(split_name, sphere_radius, num_views):
    """Get appropriate camera positions based on split"""
    if split_name == "train":
        # Use dense sampling for training (32 views)
        return util.get_archimedean_spiral(sphere_radius, num_views)
    else:
        # Use uniform sampling for validation/test (fewer views)
        if num_views <= 8:
            return util.get_uniform_sphere_cameras(
                num_views, 
                sphere_radius, 
                elevation_range=(-20, 45)
            )
        else:
            return util.get_archimedean_spiral(sphere_radius, num_views)

# CLI args
p = argparse.ArgumentParser(description='Render meshes into InstantMesh-compatible train/val/test splits.')
p.add_argument('--mesh_dir', type=str, required=False, help='Directory of .obj or .stl meshes.')
p.add_argument('--output_dir', type=str, required=True, help='Base output directory.')
p.add_argument('--num_observations_train', type=int, default=32, help='Number of views per object for training.')
p.add_argument('--num_observations_val', type=int, default=8, help='Number of views per object for validation.')
p.add_argument('--num_observations_test', type=int, default=8, help='Number of views per object for testing.')
p.add_argument('--resolution', type=int, default=224, help='Image resolution.')
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
if opt.mesh_fpath:
    # Single mesh processing
    mesh_files = [opt.mesh_fpath]
    splits = {"train": mesh_files}
else:
    # Batch processing
    mesh_files = sorted([
        os.path.join(opt.mesh_dir, f)
        for f in os.listdir(opt.mesh_dir)
        if f.lower().endswith(('.obj', '.stl', '.ply'))
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

# Split configurations for InstantMesh
split_configs = {
    "train": {
        "output_dir": train_dir,
        "num_views": opt.num_observations_train
    },
    "val": {
        "output_dir": val_dir,
        "num_views": opt.num_observations_val
    },
    "test": {
        "output_dir": test_dir,
        "num_views": opt.num_observations_test
    }
}

# Renderer
renderer = blender_interface.BlenderInterface(resolution=opt.resolution)

# Per-split rendering
for split_name, files in splits.items():
    if split_name not in split_configs:
        continue
        
    config = split_configs[split_name]
    num_views = config["num_views"]
    split_output = config["output_dir"]
    
    print("Rendering {} meshes for split: {} ({} views each)".format(len(files), split_name, num_views))
    validate_views_for_instantmesh(num_views, split_name)

    successful_renders = 0
    failed_renders = 0

    for mesh_idx, mesh_fpath in enumerate(files):
        mesh_name = os.path.splitext(os.path.basename(mesh_fpath))[0]
        instance_dir = os.path.join(split_output, mesh_name)
        
        print("Processing {}/{}: {}".format(mesh_idx+1, len(files), mesh_name))

        try:
            # Import mesh and normalize to fit inside unit sphere
            renderer.import_mesh(mesh_fpath, scale=1., object_world_matrix=None)
            obj = bpy.context.selected_objects[0]

            # Normalize mesh to unit scale
            bbox_corners = [obj.matrix_world * Vector(corner) for corner in obj.bound_box]
            center = sum(bbox_corners, Vector((0.0, 0.0, 0.0))) / 8.0
            radius = max((v - center).length for v in bbox_corners)
            
            if radius == 0:
                print("Warning: Mesh {} has zero radius, skipping".format(mesh_name))
                failed_renders += 1
                continue
                
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

            if split_name == "train":
                cam_locations = util.get_instantmesh_train_cameras(num_views, radius=sphere_radius, elev_min=15, elev_max=60)
            else:
                # Lower number of views: maybe uniform azimuth, fixed elevation (e.g. 30°)
                cam_locations = util.get_instantmesh_train_cameras(num_views, radius=sphere_radius, elev_min=25, elev_max=45)


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

            # Add InstantMesh-specific metadata
            metadata = {
                "mesh_name": mesh_name,
                "num_views": num_views,
                "resolution": opt.resolution,
                "sphere_radius": sphere_radius,
                "object_radius": radius,
                "fov_degrees": 50,
                "split": split_name,
                "format": "instantmesh_compatible",
                "blender_version": "2.79"
            }

            metadata_path = os.path.join(instance_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            successful_renders += 1
            print("Successfully rendered: {}".format(mesh_name))

        except Exception as e:
            print("Failed to render {}: {}".format(mesh_name, e))
            failed_renders += 1
            continue

    print("Split {} complete: {} successful, {} failed".format(split_name, successful_renders, failed_renders))

# Create split summary
split_summary = {
    split: {
        "files": [os.path.splitext(os.path.basename(f))[0] for f in files],
        "num_views": split_configs.get(split, {}).get("num_views", 0),
        "count": len(files)
    }
    for split, files in splits.items()
}

# Add global configuration
split_summary["config"] = {
    "train_views": opt.num_observations_train,
    "val_views": opt.num_observations_val,
    "test_views": opt.num_observations_test,
    "resolution": opt.resolution,
    "target_model": "InstantMesh"
}

# Save to JSON file
split_json_path = os.path.join(opt.output_dir, "split_summary.json")
with open(split_json_path, "w") as f:
    json.dump(split_summary, f, indent=4)

print("Saved split information to: {}".format(split_json_path))
print("Rendering complete! Data ready for InstantMesh training.")