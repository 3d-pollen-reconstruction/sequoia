import bpy, sys, os
import numpy as np
from math import pi, sin, cos, acos
from mathutils import Vector

# Force-enable STL importer
import addon_utils
addon_utils.enable("io_mesh_stl")
# --- argument parsing after "--" ---
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--mesh',    required=True,  help='Path to STL file')
parser.add_argument('--output',  required=True,  help='Output folder root')
parser.add_argument('--views',   type=int, default=50, help='Number of views')
parser.add_argument('--radius',  type=float, default=10.0, help='Camera radius')
parser.add_argument('--scale',   type=float, default=0.1, help='Uniform scale factor')
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])

mesh_path   = args.mesh
output_root = args.output
N_views     = args.views
radius      = args.radius
scale_factor= args.scale

# --- 1) Clean scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- 2) Import STL ---
if not os.path.isfile(mesh_path):
    raise FileNotFoundError(f"Mesh not found: {mesh_path}")
bpy.ops.import_mesh.stl(filepath=mesh_path)
obj = bpy.context.selected_objects[0]

# --- 3) Center origin on bounds & scale+loc ---
bpy.context.view_layer.objects.active = obj
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0,0,0)
obj.scale    = (scale_factor,)*3
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# --- 4) Make camera ---
cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("CamObj", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam

# set render settings
scene = bpy.context.scene
scene.render.resolution_x = 400
scene.render.resolution_y = 400
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# --- utilities ---
def make_dirs():
    for split in ["train","test"]:
        for sub in ("pose","intrinsics"):
            d = os.path.join(output_root, split, sub)
            os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(output_root,"imgs"), exist_ok=True)

def write_intrinsics():
    f_px = 120 / (36/400)
    intr = [ f_px,0.0,200,0.0,
             0.0,f_px,200,0.0,
             0.0,0.0,1.0,0.0,
             0.0,0.0,0.0,1.0 ]
    for split in ["train","test"]:
        cnt = 0
        for i in range(N_views):
            if ((i+1)%10==0 and split=="test") or ((i+1)%10!=0 and split=="train"):
                path = os.path.join(output_root, split, "intrinsics", f"{split}_{cnt}.txt")
                with open(path,'w') as f:
                    f.write("\n".join(map(str,intr)))
                cnt += 1

def write_pose(split, cnt, mat):
    path = os.path.join(output_root, split, "pose", f"{split}_{cnt}.txt")
    with open(path,'w') as f:
        f.write("\n".join(str(v) for v in mat))

def fibonacci_hemisphere(n, r):
    gr = (1+5**0.5)/2
    tot = n*2
    pts=[]
    for i in range(tot):
        θ = 2*pi * i / gr
        ϕ = acos(1 - 2*(i+0.5)/tot)
        if ϕ>pi/2 and ϕ<3*pi/2: continue
        x,y,z = r*cos(θ)*sin(ϕ), r*sin(θ)*sin(ϕ), r*cos(ϕ)
        pts.append((x,y,z))
        if len(pts)==n: break
    return pts

# --- 5) Prepare output dirs & intrinsics ---
make_dirs()
write_intrinsics()

# --- 6) Rotate & render ---
train_cnt = test_cnt = 0
positions = fibonacci_hemisphere(N_views, radius)

for idx, pos in enumerate(positions):
    # camera pose
    cam.location = pos
    direction = Vector((0,0,0)) - Vector(pos)
    rot_quat = direction.to_track_quat('-Z','Y')
    cam.rotation_euler = rot_quat.to_euler()

    split = "test" if (idx+1)%10==0 else "train"
    cnt   = test_cnt if split=="test" else train_cnt

    # write pose (camera matrix_world flattened)
    c2w = cam.matrix_world
    write_pose(split, cnt, list(c2w))

    # filepath
    img_fp = os.path.join(output_root, "imgs", f"{split}_{cnt}.png")
    scene.render.filepath = img_fp
    bpy.ops.render.render(write_still=True)

    if split=="train": train_cnt+=1
    else:              test_cnt +=1
