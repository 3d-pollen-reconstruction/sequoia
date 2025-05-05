import bpy
import os
import shutil
import numpy as np
import mathutils
from numpy import arange, pi, sin, cos, arccos
import addon_utils

# === ENABLE STL ADDON ===
addon_utils.enable("io_mesh_stl")

# === CONFIGURATION ===
IMG_RES = 128
FX = FY = 131.25
CX = CY = IMG_RES / 2.0
STL_DIR = r"C:/Users/super/Documents/Github/pixel-nerf/data/meshes"
DATA_ROOT = r"C:/Users/super/Documents/Github/pixel-nerf/data"
VIEWS = 128
RADIUS = 10.0
SPLITS = {'train': 'pollen_train', 'val': 'pollen_val', 'test': 'pollen_test'}

# === CLEAR SCENE ===
def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)

# === IMPORT HELPER ===
def import_stl(path):
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    bpy.ops.import_mesh.stl(
        filepath=path,
        directory=directory,
        files=[{"name": filename}],
        filter_glob="*.stl",
        global_scale=1.0,
        use_scene_unit=False,
        use_facet_normal=False,
        axis_forward='Y',
        axis_up='Z'
    )

# === PREPARE OUTPUT DIRECTORIES ===
def make_dirs(data_root, pid):
    for split_folder in SPLITS.values():
        base = os.path.join(data_root, split_folder, pid)
        for sub in ('rgb', 'pose'):
            folder = os.path.join(base, sub)
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)

# === COMPUTE MESH RADIUS ===
def compute_radius(obj):
    vs = [obj.matrix_world @ v.co for v in obj.data.vertices]
    return max((v - obj.location).length for v in vs)

# === WRITE CAMERA FILES ===
def write_intrinsics(data_root, pid):
    lines = [
        f"{FX:.6f} {CX:.6f} {CY:.6f} 0.",
        f"0. {FY:.6f} {CY:.6f} 0.",
        "0. 0. 1.",
        f"{IMG_RES} {IMG_RES}"
    ]
    for split_folder in SPLITS.values():
        out_dir = os.path.join(data_root, split_folder, pid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'intrinsics.txt'), 'w') as f:
            f.write("\n".join(lines))


def write_near_far(data_root, pid, mesh_r):
    near = max(RADIUS - mesh_r, 0.1)
    far = RADIUS + mesh_r
    for split_folder in SPLITS.values():
        out_dir = os.path.join(data_root, split_folder, pid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'near_far.txt'), 'w') as f:
            f.write(f"{near:.6f} {far:.6f}")


def write_pose(data_root, pid, split_key, idx, c2w):
    pose_dir = os.path.join(data_root, SPLITS[split_key], pid, 'pose')
    os.makedirs(pose_dir, exist_ok=True)
    with open(os.path.join(pose_dir, f"{idx:06d}.txt"), 'w') as f:
        M = c2w.reshape(4, 4)
        for row in M:
            f.write(" ".join(f"{v:.16f}" for v in row) + "\n")

# === CENTER & SCALE WITH ORIGIN SETTING ===
def center_and_scale(obj):
    bpy.context.view_layer.objects.active = obj
    obj.location = (0.0, 0.0, 0.0)
    obj.scale = (0.1, 0.1, 0.1)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

# === RENDER LOOP ===
def rotate_and_render(data_root, pid, obj, cam):
    make_dirs(data_root, pid)
    write_intrinsics(data_root, pid)
    mesh_r = compute_radius(obj)
    write_near_far(data_root, pid, mesh_r)

    scene = bpy.context.scene
    scene.camera = cam
    scene.render.resolution_x = IMG_RES
    scene.render.resolution_y = IMG_RES
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Fibonacci sphere sampling
    n = VIEWS * 2
    idxs = arange(n)
    golden = (1 + 5**0.5) / 2
    theta = 2 * pi * idxs / golden
    phi = arccos(1 - 2 * (idxs + 0.5) / n)
    mask = (phi > pi/2) & (phi < 3*pi/2)
    theta, phi = theta[~mask], phi[~mask]
    x = RADIUS * cos(theta) * sin(phi)
    y = RADIUS * sin(theta) * sin(phi)
    z = RADIUS * cos(phi)

    counters = {'train': 0, 'val': 0, 'test': 0}
    for i in range(VIEWS):
        pos = mathutils.Vector((x[i], y[i], z[i]))
        cam.location = pos
        cam.rotation_euler = (-pos.normalized()).to_track_quat('-Z', 'Y').to_euler()
        rem = (i + 1) % 10
        split = 'test' if rem == 0 else 'val' if rem == 1 else 'train'
        idx = counters[split]
        counters[split] += 1
        M = np.array(cam.matrix_world)
        FLIP = np.diag([1, -1, -1, 1])
        c2w = M @ FLIP
        write_pose(data_root, pid, split, idx, c2w)

        out_png = os.path.join(data_root, SPLITS[split], pid, 'rgb', f"{idx:06d}.png")
        scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)

# === MAIN ENTRY POINT ===
def main():
    clear_scene()
    # Create camera
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    # Add lighting
    # Key light
    key_light_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_light_data.energy = 2000
    key_light = bpy.data.objects.new(name="KeyLight", object_data=key_light_data)
    key_light.location = (RADIUS, RADIUS, RADIUS)
    bpy.context.collection.objects.link(key_light)
    # Fill light
    fill_light_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_light_data.energy = 600
    fill_light = bpy.data.objects.new(name="FillLight", object_data=fill_light_data)
    fill_light.location = (-RADIUS, -RADIUS, RADIUS)
    bpy.context.collection.objects.link(fill_light)

    # Process all STLs
    for fn in sorted(os.listdir(STL_DIR)):
        if not fn.lower().endswith('.stl'):
            continue
        stl_path = os.path.join(STL_DIR, fn)
        import_stl(stl_path)
        obj = bpy.context.selected_objects[0]
        pid = os.path.splitext(fn)[0]
        obj.name = pid
        center_and_scale(obj)
        rotate_and_render(DATA_ROOT, pid, obj, cam)

    print("✅ All done.")

if __name__ == "__main__":
    main()