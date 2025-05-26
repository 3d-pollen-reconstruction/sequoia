import os
import util
import bpy
from mathutils import Vector


class BlenderInterface():
    def __init__(self, resolution=128, background_color=(1,1,1)):
        self.resolution = resolution

        # Delete the default cube
        bpy.ops.object.delete()

        # Render settings
        self.blender_renderer = bpy.context.scene.render
        self.blender_renderer.use_antialiasing = False
        self.blender_renderer.resolution_x = resolution
        self.blender_renderer.resolution_y = resolution
        self.blender_renderer.resolution_percentage = 100
        self.blender_renderer.image_settings.file_format = 'PNG'
        self.blender_renderer.alpha_mode = 'SKY'

        # Lighting
        world = bpy.context.scene.world
        world.horizon_color = background_color
        world.light_settings.use_environment_light = True
        world.light_settings.environment_color = 'SKY_COLOR'
        world.light_settings.environment_energy = 1.0

        lamp1 = bpy.data.lamps['Lamp']
        lamp1.type = 'SUN'
        lamp1.shadow_method = 'NOSHADOW'
        lamp1.use_specular = False
        lamp1.energy = 1.0

        bpy.ops.object.lamp_add(type='SUN')
        lamp2 = bpy.data.lamps['Sun']
        lamp2.shadow_method = 'NOSHADOW'
        lamp2.use_specular = False
        lamp2.energy = 1.0
        bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
        bpy.data.objects['Sun'].rotation_euler[0] += 180

        bpy.ops.object.lamp_add(type='SUN')
        lamp3 = bpy.data.lamps['Sun.001']
        lamp3.shadow_method = 'NOSHADOW'
        lamp3.use_specular = False
        lamp3.energy = 0.3
        bpy.data.objects['Sun.001'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
        bpy.data.objects['Sun.001'].rotation_euler[0] += 90

        # Camera setup
        self.camera = bpy.context.scene.camera
        self.camera.data.sensor_height = self.camera.data.sensor_width
        util.set_camera_focal_length_in_world_units(self.camera.data, 525./512*resolution)

        bpy.ops.object.select_all(action='DESELECT')

    def import_mesh(self, fpath, scale=1., object_world_matrix=None):
        ext = os.path.splitext(fpath)[-1]
        if ext == '.obj':
            bpy.ops.import_scene.obj(filepath=str(fpath), split_mode='OFF')
        elif ext == '.ply':
            bpy.ops.import_mesh.ply(filepath=str(fpath))

        obj = bpy.context.selected_objects[0]
        util.dump(bpy.context.selected_objects)

        if object_world_matrix is not None:
            obj.matrix_world = object_world_matrix

        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0., 0., 0.)
        
        if len(obj.data.materials) == 0:
            mat = bpy.data.materials.new(name="DefaultGray")
            mat.diffuse_color = (0.6, 0.6, 0.6)
            mat.use_nodes = False
            obj.data.materials.append(mat)
        else:
            for mat in obj.data.materials:
                mat.diffuse_color = (0.6, 0.6, 0.6)

        if scale != 1.:
            bpy.ops.transform.resize(value=(scale, scale, scale))

        # Clean materials
        for m in bpy.data.materials:
            m.use_transparency = False
            m.specular_intensity = 0.0

        for t in bpy.data.textures:
            try:
                t.use_interpolation = False
                t.use_mipmap = False
                t.use_filter_size_min = True
                t.filter_type = "BOX"
            except:
                continue

    def render(self, output_dir, blender_cam2world_matrices, write_cam_params=False, object_radius=1.0):

        if write_cam_params:
            img_dir = os.path.join(output_dir, 'rgb')
            pose_dir = os.path.join(output_dir, 'pose')
            util.cond_mkdir(img_dir)
            util.cond_mkdir(pose_dir)
        else:
            img_dir = output_dir
            util.cond_mkdir(img_dir)

        if write_cam_params:
            # Save intrinsics
            K = util.get_calibration_matrix_K_from_blender(self.camera.data)
            with open(os.path.join(output_dir, 'intrinsics.txt'),'w') as intrinsics_file:
                intrinsics_file.write('%f %f %f 0.\n' % (K[0][0], K[0][2], K[1][2]))
                intrinsics_file.write('0. 0. 0.\n')
                intrinsics_file.write('1.\n')
                intrinsics_file.write('%d %d\n' % (self.resolution, self.resolution))

            # Compute near/far from camera distances
            cam_locs = [mat.to_translation() for mat in blender_cam2world_matrices]
            dists = [(loc - Vector((0.0, 0.0, 0.0))).length for loc in cam_locs]
            near = max(0.01, min(dists) - object_radius)
            far  = max(dists) + object_radius
            with open(os.path.join(output_dir, 'near_far.txt'), 'w') as nf_file:
                nf_file.write('%.6f %.6f\n' % (near, far))

        for i, mat in enumerate(blender_cam2world_matrices):
            self.camera.matrix_world = mat

            if os.path.exists(os.path.join(img_dir, '%06d.png' % i)):
                continue

            self.blender_renderer.filepath = os.path.join(img_dir, '%06d.png' % i)
            bpy.ops.render.render(write_still=True)

            if write_cam_params:
                RT = util.get_world2cam_from_blender_cam(self.camera)
                cam2world = RT.inverted()
                with open(os.path.join(pose_dir, '%06d.txt' % i), 'w') as pose_file:
                    matrix_flat = [cam2world[j][k] for j in range(4) for k in range(4)]
                    pose_file.write(' '.join(map(str, matrix_flat)) + '\n')

        # Clean up
        meshes_to_remove = []
        for ob in bpy.context.selected_objects:
            meshes_to_remove.append(ob.data)

        bpy.ops.object.delete()
        for mesh in meshes_to_remove:
            bpy.data.meshes.remove(mesh)
