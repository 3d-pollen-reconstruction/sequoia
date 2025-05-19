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
STL_DIR = r"C:/Users/super/Documents/Github/pixel-nerf/data/meshes" # Adjust if your path is different
DATA_ROOT = r"C:/Users/super/Documents/Github/pixel-nerf/data_new"    # Adjust if your path is different
VIEWS = 128
RADIUS = 10.0
SPLITS = {'train': 'pollen_train', 'val': 'pollen_val', 'test': 'pollen_test'}

# === CLEAR SCENE (ALL OBJECTS) ===
def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)

# === CLEAR ONLY MESHES ===
def clear_meshes():
    # Remove mesh objects but keep camera/lights
    for o in list(bpy.data.objects):
        if o.type == 'MESH':
            bpy.data.objects.remove(o, do_unlink=True)
    # Purge unused mesh data
    for m in list(bpy.data.meshes):
        if m.users == 0:
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
    # Ensure the object's matrix_world is up-to-date if it has been transformed
    # For this script, obj.location is (0,0,0) when this is called by rotate_and_render
    # after center_and_scale, so matrix_world is effectively identity if no other parent transforms.
    # If obj.location wasn't (0,0,0), obj.matrix_world @ v.co would be essential.
    if not obj.data.vertices: # Handle case with no vertices
        return 0.0
    vs = [obj.matrix_world @ v.co for v in obj.data.vertices]
    # Assuming obj.location is (0,0,0) as per center_and_scale
    return max(v.length for v in vs) if vs else 0.0


# === WRITE CAMERA FILES ===
def write_intrinsics(data_root, pid):
    # This function now strictly follows the provided documentation format.

    # Line 1: f cx cy 0.
    # Using FX from your script as 'f', CX as 'cx', and CY as 'cy'.
    line1 = f"{FX:.6f} {CX:.6f} {CY:.6f} 0." 

    # Line 2: 0. 0. 0. (Literal from documentation)
    line2 = "0. 0. 0." 

    # Line 3: 1. (Literal from documentation)
    line3 = "1."

    # Line 4: img_height img_width
    line4 = f"{IMG_RES} {IMG_RES}" # Assuming square images, IMG_RES for both

    lines_to_write = [line1, line2, line3, line4]

    for split_folder_name in SPLITS.values(): # e.g., 'pollen_train', 'pollen_val', 'pollen_test'
        out_dir = os.path.join(data_root, split_folder_name, pid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'intrinsics.txt'), 'w') as f:
            f.write("\n".join(lines_to_write))


def write_near_far(data_root, pid, mesh_r):
    near = max(RADIUS - mesh_r, 0.1) # mesh_r will be TARGET
    far = RADIUS + mesh_r
    for split_folder_name in SPLITS.values():
        out_dir = os.path.join(data_root, split_folder_name, pid)
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
def center_and_scale(obj, target_radius):
    # reset transforms
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.location = (0,0,0)
    obj.rotation_euler = (0,0,0)
    obj.scale = (1,1,1)
    bpy.context.view_layer.update() # Ensure transforms are applied before computing radius

    # compute current radius using local coordinates as object is at origin with no rotation/scale
    if not obj.data.vertices:
        print(f"Warning: Object {obj.name} has no vertices.")
        return
    
    # Calculate radius from local vertex coordinates, as object is at origin and unscaled
    r = 0.0
    # Ensure mesh data is available
    if obj.type == 'MESH' and obj.data:
        if len(obj.data.vertices) > 0:
            r = max(v.co.length for v in obj.data.vertices)
    
    if r == 0: 
        print(f"Warning: Object {obj.name} has zero radius before scaling (or is not a mesh with vertices).")
        return

    # scale so new radius == target_radius
    s = target_radius / r
    obj.scale = (s, s, s)
    bpy.context.view_layer.update() # Apply scale

    # re‑origin & bake scale
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True) # Bake scale
    obj.location = (0,0,0) # Re-center object origin at world origin
    bpy.context.view_layer.update()


# === RENDER LOOP ===
def rotate_and_render(data_root, pid, obj, cam):
    make_dirs(data_root, pid)
    write_intrinsics(data_root, pid)
    
    # After center_and_scale, obj is at origin, baked scale.
    # So, compute_radius will give the target_radius if obj.location is (0,0,0).
    # Or, more directly, mesh_r IS the target_radius it was scaled to.
    # To be safe, let's use the TARGET value passed implicitly via obj's new scaled size.
    # The target_radius used in center_and_scale is what mesh_r should be.
    # We retrieve it from the object's current state.
    mesh_r = 0.0
    if obj.type == 'MESH' and obj.data and len(obj.data.vertices) > 0:
         mesh_r = max(v.co.length for v in obj.data.vertices) # Vertices are now in local scaled coords
                                                              # obj.location is (0,0,0)
                                                              # obj.scale is (1,1,1) after apply
    
    if mesh_r == 0: # Fallback if something went wrong, though TARGET should be > 0
        print(f"Error: Mesh radius for {pid} is zero after scaling. Using a fallback for near/far.")
        # This indicates an issue with the object or scaling if TARGET was non-zero.
        # For safety, use a fraction of RADIUS if mesh_r is 0 and TARGET was meant to be positive
        # This part needs to align with the TARGET_FACTOR logic.
        # If TARGET was RADIUS * TARGET_FACTOR, then mesh_r should be that value.
        # Let's assume TARGET was correctly applied.
        # The `mesh_r` used for `write_near_far` should be the `TARGET` value used in `center_and_scale`.
        # We can re-calculate it here for consistency or pass `TARGET` to this function.
        # For simplicity, re-calculate based on the known TARGET_FACTOR if needed,
        # or better, trust that compute_radius on the scaled object gives the value.
        # The call to compute_radius() at the start of this function should give the correct value.
        mesh_r = compute_radius(obj) # This re-evaluates the scaled object.
        
    write_near_far(data_root, pid, mesh_r)

    scene = bpy.context.scene
    scene.camera = cam
    scene.render.resolution_x = IMG_RES
    scene.render.resolution_y = IMG_RES
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    n_total_sphere_points = VIEWS 
    
    idxs = arange(n_total_sphere_points)
    golden = (1 + 5**0.5) / 2
    theta = 2 * pi * idxs / golden
    phi = arccos(1 - 2 * (idxs + 0.5) / n_total_sphere_points)

    x = RADIUS * cos(theta) * sin(phi)
    y = RADIUS * sin(theta) * sin(phi)
    z = RADIUS * cos(phi)

    counters = {'train': 0, 'val': 0, 'test': 0}
    for i in range(n_total_sphere_points): 
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
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    
    key = bpy.data.lights.new("KeyLight", 'AREA')
    key.energy = 1500; key.size = RADIUS * 0.8
    kl = bpy.data.objects.new("KeyLight", key); kl.location = (RADIUS, RADIUS, RADIUS)
    bpy.context.collection.objects.link(kl)
    fill = bpy.data.lights.new("FillLight", 'AREA')
    fill.energy = 500; fill.size = RADIUS * 0.5
    fl = bpy.data.objects.new("FillLight", fill); fl.location = (-RADIUS, -RADIUS, RADIUS)
    bpy.context.collection.objects.link(fl)
    rim = bpy.data.lights.new("RimLight", 'AREA')
    rim.energy = 300; rim.size = RADIUS * 0.5
    rl = bpy.data.objects.new("RimLight", rim); rl.location = (0, -RADIUS, RADIUS)
    bpy.context.collection.objects.link(rl)

    scene = bpy.context.scene
    cam_data = cam.data 

    cam_data.sensor_fit = 'HORIZONTAL' 
    cam_data.sensor_width = 36.0  
    f_in_mm = FX * (cam_data.sensor_width / IMG_RES) 
    cam_data.lens = f_in_mm 
    cam_data.shift_x = 0.0  
    cam_data.shift_y = 0.0  
                               
    scene.render.resolution_x = IMG_RES
    scene.render.resolution_y = IMG_RES
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    for fn in sorted(os.listdir(STL_DIR)):
        if not fn.lower().endswith('.stl'): continue
        clear_meshes()
        stl_path = os.path.join(STL_DIR, fn)
        import_stl(stl_path)
        
        if not bpy.context.selected_objects:
            print(f"Warning: No object selected after importing {fn}. Skipping.")
            continue
        obj = bpy.context.selected_objects[0]
        
        pid = os.path.splitext(fn)[0]
        obj.name = pid
        
        fov_factor = (IMG_RES / 2.0) / FX
        TARGET = RADIUS * fov_factor
        print(f"Processing {pid}, TARGET world radius set to: {TARGET:.4f}")
        center_and_scale(obj, TARGET)
        
        # After center_and_scale, the object's new max radius in local coords is TARGET.
        # mesh_r for write_near_far should be this TARGET value.
        # rotate_and_render will call compute_radius which should correctly get this value.
        rotate_and_render(DATA_ROOT, pid, obj, cam)
        
    print("✅ All done.")

if __name__ == "__main__":
    main()