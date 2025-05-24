import bpy
import os
import numpy as np
from numpy import arange, pi, sin, cos, arccos
import mathutils



## run first
#subject = bpy.context.selected_objects[0]
#>>> subject.location = (0, 0, 0)
#>>> subject.scale = (0.1, 0.1, 0.1)  # Scale down
#>>> subject.location = (0, 0, 0)     # Center

#bpy.context.view_layer.objects.active = subject
#bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
#subject.location = (0, 0, 0)3D167775


def make_files():
    for split in ["train", "test"]:
        base = os.path.join(output_root, split)
        if os.path.exists(base):
            import shutil
            shutil.rmtree(base)
        os.makedirs(os.path.join(base, "pose"))
        os.makedirs(os.path.join(base, "intrinsics"))

def write_intrinsics(N):
    train_cnt = 0
    test_cnt = 0
    for j in range(N):
        prefix = 'train' if (j + 1) % 10 else 'test'
        cnt = train_cnt if prefix == 'train' else test_cnt
        if prefix == 'train':
            train_cnt += 1
        else:
            test_cnt += 1

        pixel_size = 36 / 400  # mm
        f = 120 / pixel_size
        intrinsics = [
            f, 0.0, 200, 0.0,
            0.0, f, 200, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
        intrinsics_path = os.path.join(output_root, prefix, 'intrinsics', f'{prefix}_{cnt}.txt')
        with open(intrinsics_path, 'w') as f_out:
            f_out.write('\n'.join(map(str, intrinsics)))

def write_pose(prefix, cnt, c2w: np.array):
    pose_path = os.path.join(output_root, prefix, 'pose', f'{prefix}_{cnt}.txt')
    with open(pose_path, 'w') as f_out:
        pose = c2w.reshape(-1).tolist()
        f_out.write('\n'.join(map(str, pose)))

def rotate_and_render(output_file_pattern_string='render%d.png',
                      rotation_steps=100,
                      subject=None,
                      radius=10):

    make_files()
    write_intrinsics(rotation_steps)

    if subject is None:
        raise ValueError("You must provide a camera as 'subject'.")

    bpy.context.scene.camera = subject
    bpy.context.scene.render.resolution_x = 400
    bpy.context.scene.render.resolution_y = 400
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    n = rotation_steps * 2
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = arange(0, n)
    theta = 2 * pi * i / goldenRatio
    phi = arccos(1 - 2 * (i + 0.5) / n)
    cond = (phi > (pi / 2)) & (phi < (3 * pi / 2))
    phi = phi[~cond]
    theta = theta[~cond]
    x, y, z = radius * cos(theta) * sin(phi), radius * sin(theta) * sin(phi), radius * cos(phi)

    train_cnt = 0
    test_cnt = 0

    for step in range(rotation_steps):
        cam_location = mathutils.Vector((x[step], y[step], z[step]))
        subject.location = cam_location
        direction = -cam_location.normalized()
        rot_quat = direction.to_track_quat('-Z', 'Y')
        subject.rotation_euler = rot_quat.to_euler()

        if (step + 1) % 10:
            render_path = os.path.join(output_root, f"imgs/train_{train_cnt}.png")
            write_pose("train", train_cnt, np.array(subject.matrix_world))
            train_cnt += 1
        else:
            render_path = os.path.join(output_root, f"imgs/test_{test_cnt}.png")
            write_pose("test", test_cnt, np.array(subject.matrix_world))
            test_cnt += 1

        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

# === RUN IT ===
camera = bpy.data.objects["Camera"]
for number in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    count_views = number
    output_root = f"C:/Users/super/Documents/Github/sequoia/Nerf - Copy/2D_Pollen/21585_views{count_views}"
    rotate_and_render(subject=camera, rotation_steps=count_views, radius=10)
