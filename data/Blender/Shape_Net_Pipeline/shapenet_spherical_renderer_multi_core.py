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
p.add_argument('--split_name', type=str, help='Split name (train/val/testa) for single-mesh rendering') 
p.add_argument('--modus', type=str, default="train", help='train/val/test')
p.add_argument('--object_name', type=str, help='Object name for saving folder')
p.add_argument('--orthogonal', action='store_true', help='Use orthographic camera')

argv = sys.argv[sys.argv.index("--") + 1:]
opt = p.parse_args(argv)

if opt.mesh_fpath and opt.split_name and opt.object_name:
    renderer = blender_interface.BlenderInterface(resolution=opt.resolution)
    instance_dir = os.path.join(opt.output_dir, "pollen_{}".format(opt.split_name), opt.object_name)
    os.makedirs(instance_dir, exist_ok=True)

    renderer.import_mesh(opt.mesh_fpath, scale=1.0, object_world_matrix=None)
    obj = bpy.context.selected_objects[0]

    bbox_corners = [obj.matrix_world * Vector(corner) for corner in obj.bound_box]
    center = sum(bbox_corners, Vector((0.0, 0.0, 0.0))) / 8.0
    radius = max((v - center).length for v in bbox_corners)
    obj.scale = (1.0 / radius, 1.0 / radius, 1.0 / radius)
    bpy.context.scene.update()

    bbox_corners = [obj.matrix_world * Vector(corner) for corner in obj.bound_box]
    center = sum(bbox_corners, Vector((0.0, 0.0, 0.0))) / 8.0
    obj_location = -np.array(center).reshape(1, 3)
    sphere_radius = 2.0

    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.ops.object.delete()

    if opt.orthogonal:
        cam_locations = util.get_orthogonal_camera_positions(sphere_radius, center=(0, 0, 0))
    elif opt.split_name == 'train':
        cam_locations = util.sample_spherical(opt.num_observations, sphere_radius)
    else:
       cam_locations = util.get_archimedean_spiral(sphere_radius, 250)

    cv_poses = util.look_at(cam_locations, np.zeros((1, 3)))
    blender_poses = [util.cv_cam2world_to_bcam2world(m) for m in cv_poses]

    rot_mat = np.eye(3)
    hom_coords = np.array([[0., 0., 0., 1.]])
    obj_pose = np.concatenate((rot_mat, obj_location.reshape(3, 1)), axis=-1)
    obj_pose = np.concatenate((obj_pose, hom_coords), axis=0)

    renderer.import_mesh(opt.mesh_fpath, scale=1.0 / radius, object_world_matrix=obj_pose)
    renderer.render(instance_dir, blender_poses, write_cam_params=True, object_radius=sphere_radius)
    exit(0)


