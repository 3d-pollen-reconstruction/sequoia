import os
import util
import bpy
import numpy as np
from mathutils import Vector


class BlenderInterface():
    def __init__(self, resolution=320, background_color=(1, 1, 1)):
        self.resolution = resolution

        bpy.ops.object.delete()  # Delete default cube

        self.blender_renderer = bpy.context.scene.render
        self.blender_renderer.use_antialiasing = False
        self.blender_renderer.resolution_x = resolution
        self.blender_renderer.resolution_y = resolution
        self.blender_renderer.resolution_percentage = 100
        self.blender_renderer.image_settings.file_format = 'PNG'
        self.blender_renderer.alpha_mode = 'SKY'
        self.camera = bpy.context.scene.camera
        self.camera.data.sensor_height = self.camera.data.sensor_width
        # InstantMesh often uses FOV around 50 degrees
        self.camera.data.angle = np.radians(50)  # Set FOV to 50 degrees

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
        util.set_camera_focal_length_in_world_units(self.camera.data, 525. / 512 * resolution)

        bpy.ops.object.select_all(action='DESELECT')

    def import_mesh(self, fpath, scale=1., object_world_matrix=None):
        ext = os.path.splitext(fpath)[-1]
        if ext == '.obj':
            bpy.ops.import_scene.obj(filepath=str(fpath), split_mode='OFF')
        elif ext == '.stl':
            bpy.ops.import_mesh.stl(filepath=str(fpath))
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

        img_dir = os.path.join(output_dir, 'rgb') if write_cam_params else output_dir
        pose_dir = os.path.join(output_dir, 'pose') if write_cam_params else None
        util.cond_mkdir(img_dir)
        if pose_dir:
            util.cond_mkdir(pose_dir)
        util.cond_mkdir(os.path.join(output_dir, 'depth'))
        util.cond_mkdir(os.path.join(output_dir, 'normal'))

        if write_cam_params:
            K = util.get_calibration_matrix_K_from_blender(self.camera.data)
            with open(os.path.join(output_dir, 'intrinsics.txt'), 'w') as intrinsics_file:
                intrinsics_file.write('%f %f %f 0.\n' % (K[0][0], K[0][2], K[1][2]))
                intrinsics_file.write('0. 0. 0.\n')
                intrinsics_file.write('1.\n')
                intrinsics_file.write('%d %d\n' % (self.resolution, self.resolution))

            cam_locs = [mat.to_translation() for mat in blender_cam2world_matrices]
            near_far_lines = []
            for loc in cam_locs:
                dist = (loc - Vector((0.0, 0.0, 0.0))).length
                near = max(0.1, dist - object_radius)
                far = dist + object_radius
                near_far_lines.append("{:.6f} {:.6f}".format(near, far))

            with open(os.path.join(output_dir, 'near_far.txt'), 'w') as nf_file:
                nf_file.write("\n".join(near_far_lines))

        # Setup nodes once
        scene = bpy.context.scene
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()

        rlayers = tree.nodes.new(type='CompositorNodeRLayers')
        scene.render.layers["RenderLayer"].use_pass_normal = True
        scene.render.layers["RenderLayer"].use_pass_z = True

        # === DEPTH normalization ===
        depth_map_range = tree.nodes.new(type='CompositorNodeMapRange')
        depth_map_range.name = 'depth_map_range'
        depth_map_range.inputs['From Min'].default_value = 0.1  # placeholder; dynamically updated in loop
        depth_map_range.inputs['From Max'].default_value = 3.0  # placeholder; dynamically updated in loop
        depth_map_range.inputs['To Min'].default_value = 0.0
        depth_map_range.inputs['To Max'].default_value = 1.0

        depth_output = tree.nodes.new(type='CompositorNodeOutputFile')
        depth_output.label = 'Depth Output'
        depth_output.name = 'depth_output'
        depth_output.base_path = os.path.join(output_dir, "depth")
        depth_output.format.file_format = 'PNG'
        depth_output.format.color_depth = '16'
        depth_output.format.color_mode = 'BW'

        tree.links.new(rlayers.outputs['Depth'], depth_map_range.inputs[0])
        tree.links.new(depth_map_range.outputs[0], depth_output.inputs[0])

    

        normal_output = tree.nodes.new(type='CompositorNodeOutputFile')
        normal_output.label = 'Normal Output'
        normal_output.name = 'normal_output'
        normal_output.base_path = os.path.join(output_dir, "normal")
        normal_output.format.file_format = 'PNG'
        tree.links.new(rlayers.outputs['Normal'], normal_output.inputs[0])
        cam_poses = []
        for i, mat in enumerate(blender_cam2world_matrices):
            self.camera.matrix_world = mat
            
            # Compute dynamic near/far for this specific view
            cam_loc = mat.to_translation()
            dist = (cam_loc - Vector((0.0, 0.0, 0.0))).length
            near = max(0.1, dist - object_radius)
            far = dist + object_radius

            # Update depth normalization range in Map Range node
            tree.nodes['depth_map_range'].inputs['From Min'].default_value = near
            tree.nodes['depth_map_range'].inputs['From Max'].default_value = far


            if os.path.exists(os.path.join(img_dir, '%06d.png' % i)):
                continue

            self.blender_renderer.filepath = os.path.join(img_dir, '%06d.png' % i)

            depth_output.file_slots[0].path = "%06d_" % i
            normal_output.file_slots[0].path = "%06d_" % i


            bpy.ops.render.render(write_still=True)
            cam_poses.append(np.array(mat))
            
            if write_cam_params:
                RT = util.get_world2cam_from_blender_cam(self.camera)
                cam2world = RT.inverted()
                # Save both for .npz and optionally .txt
                cam_poses.append(np.array(cam2world))

                if write_cam_params:
                    with open(os.path.join(pose_dir, '%06d.txt' % i), 'w') as pose_file:
                        matrix_flat = [cam2world[j][k] for j in range(4) for k in range(4)]
                        pose_file.write(' '.join(map(str, matrix_flat)) + '\n')
        
        np.savez(os.path.join(output_dir, 'cameras.npz'), cam_poses=np.stack(cam_poses, axis=0))

        # Cleanup
        meshes_to_remove = [ob.data for ob in bpy.context.selected_objects]
        bpy.ops.object.delete()
        for mesh in meshes_to_remove:
            bpy.data.meshes.remove(mesh)
