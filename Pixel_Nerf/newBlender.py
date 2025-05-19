import bpy
import os
import shutil
import numpy as np
import math
import mathutils
from numpy import arange, pi, sin, cos, arccos
import addon_utils

# === ENABLE STL ADDON ===
addon_utils.enable("io_mesh_stl")

# === CONFIGURATION ===
IMG_RES   = 128
CX = CY   = IMG_RES / 2.0
STL_DIR   = r"C:/Users/super/Documents/Github/pixel-nerf/data/meshes"
DATA_ROOT = r"C:/Users/super/Documents/Github/pixel-nerf/data"
VIEWS     = 128
RADIUS    = 10.0
SPLITS    = {'train': 'pollen_train', 'val': 'pollen_val', 'test': 'pollen_test'}
# Background mode: 'white' or 'transparent'
BG_MODE   = 'transparent'

# === SCENE MANAGEMENT ===
def clear_scene():
    # Remove all objects and meshes
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)

def clear_meshes():
    # Remove only mesh objects
    for o in list(bpy.data.objects):
        if o.type == 'MESH':
            bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)

# === IMPORT HELPER ===
def import_stl(path):
    bpy.ops.import_mesh.stl(
        filepath=path,
        directory=os.path.dirname(path),
        files=[{"name": os.path.basename(path)}],
        filter_glob="*.stl",
        global_scale=1.0,
        use_scene_unit=False,
        use_facet_normal=False,
        axis_forward='Y',
        axis_up='Z'
    )

# === OUTPUT DIRECTORY SETUP ===
def make_dirs(data_root, pid):
    for split in SPLITS.values():
        base = os.path.join(data_root, split, pid)
        for sub in ('rgb', 'pose'):
            folder = os.path.join(base, sub)
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)

# === MESH RADIUS COMPUTATION ===
def compute_radius(obj):
    vs = [obj.matrix_world @ v.co for v in obj.data.vertices]
    return max((v - obj.location).length for v in vs)

# === CAMERA FILE WRITING ===
def write_intrinsics(data_root, pid, fx, fy):
    lines = [
        f"{fx:.6f} {CX:.6f} {CY:.6f} 0.",
        f"0. {fy:.6f} {CY:.6f} 0.",
        "0. 0. 1.",
        f"{IMG_RES} {IMG_RES}"
    ]
    for split in SPLITS.values():
        out = os.path.join(data_root, split, pid)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'intrinsics.txt'), 'w') as f:
            f.write("\n".join(lines))


def write_near_far(data_root, pid, mesh_r):
    near = max(RADIUS - mesh_r, 0.1)
    far  = RADIUS + mesh_r
    print(f"[{pid}] near={near:.3f}, far={far:.3f}")
    for split in SPLITS.values():
        out = os.path.join(data_root, split, pid)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'near_far.txt'), 'w') as f:
            f.write(f"{near:.6f} {far:.6f}")


def write_pose(data_root, pid, split_key, idx, c2w):
    pose_dir = os.path.join(data_root, SPLITS[split_key], pid, 'pose')
    os.makedirs(pose_dir, exist_ok=True)
    with open(os.path.join(pose_dir, f"{idx:06d}.txt"), 'w') as f:
        M = c2w.reshape(4,4)
        for row in M:
            f.write(" ".join(f"{v:.16f}" for v in row) + "\n")

# === CENTER & SCALE MESH ===
def center_and_scale(obj, target_r):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.location = (0,0,0)
    obj.rotation_euler = (0,0,0)
    obj.scale = (1,1,1)
    bpy.context.view_layer.update()

    r = compute_radius(obj)
    if r > 0:
        s = target_r / r
        obj.scale = (s, s, s)
        bpy.context.view_layer.update()
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.location = (0,0,0)
        bpy.context.view_layer.update()

# === RENDER LOOP ===
def rotate_and_render(data_root, pid, obj, cam):
    scene = bpy.context.scene
    # Compute FX/FY from camera FOV
    fx = 0.5 * IMG_RES / math.tan(cam.data.angle / 2)
    fy = fx

    make_dirs(data_root, pid)
    write_intrinsics(data_root, pid, fx, fy)
    mesh_r = compute_radius(obj)
    write_near_far(data_root, pid, mesh_r)

    # Set render settings
    scene.camera = cam
    scene.render.resolution_x = IMG_RES
    scene.render.resolution_y = IMG_RES
    scene.render.image_settings.file_format = 'PNG'

    # Background
    if BG_MODE == 'transparent':
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = 'RGBA'
    else:
        scene.render.film_transparent = False
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get('Background')
        if bg:
            bg.inputs[0].default_value = (1,1,1,1)
        scene.render.image_settings.color_mode = 'RGB'

    # Golden spiral sampling
    # FULL-SPHERE golden spiral with 2*VIEWS samples
    # total samples
    n     = 2 * VIEWS
    idxs  = np.arange(n)

    # golden ratio φ  
    phi_g = (1 + 5**0.5) / 2  

    # polar angle φ ∈ [0,π]
    phi = np.arccos(1 - 2*(idxs + 0.5)/n)

    # azimuth θ stepping by the golden ratio
    theta = 2 * np.pi * idxs / phi_g

    # cartesian coords on sphere radius RADIUS
    x = RADIUS * np.sin(phi) * np.cos(theta)
    y = RADIUS * np.sin(phi) * np.sin(theta)
    z = RADIUS * np.cos(phi)

    counters = {'train':0, 'val':0, 'test':0}

    for i in range(n):
        pos = mathutils.Vector((x[i], y[i], z[i]))
        cam.location = pos
        # True look-at rotation
        dir_vec = (mathutils.Vector((0,0,0)) - pos).normalized()
        cam.rotation_euler = dir_vec.to_track_quat('-Z','Y').to_euler()

        rem = (i + 1) % 10
        split = 'test' if rem == 0 else 'val' if rem == 1 else 'train'
        idx = counters[split]; counters[split] += 1

        M = np.array(cam.matrix_world)
        FLIP = np.diag([1, -1, -1, 1])
        c2w = M @ FLIP
        write_pose(data_root, pid, split, idx, c2w)

        out_png = os.path.join(data_root, SPLITS[split], pid, 'rgb', f"{idx:06d}.png")
        scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)

# === MAIN ENTRY ===
def main():
    clear_scene()
    # Create camera with explicit FOV
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(60)
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    # Lighting
    for name, loc, energy, size in [
        ("Key",  ( RADIUS,  RADIUS,  RADIUS), 1500, RADIUS * 0.8),
        ("Fill", (-RADIUS, -RADIUS,  RADIUS),  500, RADIUS * 0.5),
        ("Rim",  (0,       -RADIUS,  RADIUS),  300, RADIUS * 0.5),
    ]:
        light = bpy.data.lights.new(name + "Light", 'AREA')
        light.energy, light.size = energy, size
        obj = bpy.data.objects.new(name + "Light", light)
        obj.location = loc
        bpy.context.collection.objects.link(obj)

    # Process each STL
    for fn in sorted(os.listdir(STL_DIR)):
        if not fn.lower().endswith('.stl'):
            continue
        clear_meshes()
        import_stl(os.path.join(STL_DIR, fn))
        obj = bpy.context.selected_objects[0]
        pid = os.path.splitext(fn)[0]
        obj.name = pid

        center_and_scale(obj, RADIUS * 0.7)
        rotate_and_render(DATA_ROOT, pid, obj, cam)

    print("✅ All done.")

if __name__ == "__main__":
    main()