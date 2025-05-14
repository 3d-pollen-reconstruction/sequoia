import argparse
import os
import sys
import json
import random
import bpy
from mathutils import Vector

class FastPollenAugmentor:
    """
    Optimized pollen mesh augmentation pipeline with resume capability.
    - On abort/restart, skips already processed meshes.
    - Stores progress in 'progress.json' under output_dir.
    """
    PROGRESS_FILE = 'progress.json'

    def __init__(self, mesh_dir, output_dir, num_augmentations=3, decimate_ratio=0.2, seed=42):
        self.mesh_dir = mesh_dir
        self.output_dir = output_dir
        self.num_augmentations = num_augmentations
        self.decimate_ratio = decimate_ratio
        random.seed(seed)
        # Define deformation methods
        self.deformations = {
            'swelling': self._swelling,
            'shriveling': self._shriveling,
            'softening': self._softening,
            'twisting': self._twisting,
            'stretching': self._stretching,
            #'elastic': self._elastic, probably to unnatural
            'spikify': self._spikify,
            'groove': self._groove,
            'wrinkle': self._wrinkle,
            'asymmetry': self._asymmetry,
            'full_combo': self._full_combo,
        }
        self._prepare_workspace()
        self.progress = self._load_progress()

    def _prepare_workspace(self):
        # Create output directories and textures
        for name in self.deformations:
            out = os.path.join(self.output_dir, name)
            if not os.path.exists(out):
                os.makedirs(out)
        tex1 = bpy.data.textures.new('TexSwelling', type='CLOUDS')
        tex2 = bpy.data.textures.new('TexShrivel', type='CLOUDS')
        self.tex_swelling = tex1
        self.tex_shrivel = tex2

    def _load_progress(self):
        path = os.path.join(self.output_dir, self.PROGRESS_FILE)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _save_progress(self):
        path = os.path.join(self.output_dir, self.PROGRESS_FILE)
        with open(path, 'w') as f:
            json.dump(self.progress, f)

    def clear_scene(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def import_and_reduce(self, filepath):
        self.clear_scene()
        bpy.ops.import_mesh.stl(filepath=filepath)
        obj = bpy.context.selected_objects[0]
        # Normalize
        bbox = [obj.matrix_world * Vector(c) for c in obj.bound_box]
        center = sum(bbox, Vector((0,0,0))) / 8.0
        r = max((v-center).length for v in bbox)
        if r > 0:
            obj.scale = (1.0/r, 1.0/r, 1.0/r)
        # Decimate
        mod = obj.modifiers.new('Decimate', type='DECIMATE')
        mod.ratio = self.decimate_ratio
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return obj

    def bake_and_export(self, obj, out_path):
        bpy.ops.object.select_all(action='DESELECT')
        obj.select = True
        bpy.ops.export_mesh.stl(filepath=out_path, use_selection=True)
        bpy.data.objects.remove(obj, do_unlink=True)

    # Deformation methods accept intensity t in [0,1]
    def _swelling(self, obj, t):
        tex = self.tex_swelling.copy()
        tex.noise_scale = 0.3 + t*0.3
        mod = obj.modifiers.new('Displace', type='DISPLACE')
        mod.texture = tex
        mod.strength = 0.3 + t*0.9
        
    def _spikify(self, obj, t):
        tex = bpy.data.textures.new('TexSpike', type='CLOUDS')
        tex.noise_scale = 0.03 + t * 0.07 
        mod = obj.modifiers.new('DisplaceSpike', type='DISPLACE')
        mod.texture = tex
        mod.strength = 0.6 + t * 1.2    
        mod.mid_level = 0.05             
        mod.direction = 'NORMAL'

    def _groove(self, obj, t):
        # Rotate to simulate bending along Z (default deform axis)
        original_rotation = obj.rotation_euler[:]
        obj.rotation_euler = (1.5708, 0.0, 0.0)  # Rotate X by 90° to simulate Z -> Y

        mod = obj.modifiers.new('GrooveTwist', type='SIMPLE_DEFORM')
        mod.deform_method = 'BEND'
        mod.angle = -0.2 - t * 0.6

        # Optionally apply and rotate back
        bpy.context.scene.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
        obj.rotation_euler = original_rotation

        

    def _wrinkle(self, obj, t):
        tex = bpy.data.textures.new('TexWrinkle', type='CLOUDS')
        tex.noise_scale = 0.15 + t * 0.1
        mod = obj.modifiers.new('Wrinkle', type='DISPLACE')
        mod.texture = tex
        mod.strength = 0.03 + t * 0.1
        mod.mid_level = 0.4
        mod.direction = 'NORMAL'
        
    def _asymmetry(self, obj, t):
        mod = obj.modifiers.new('TiltDeform', type='SIMPLE_DEFORM')
        mod.deform_method = 'TAPER'
        mod.factor = 0.05 + t * 0.15

        # Simulate axis: rotate object slightly before deformation
        obj.rotation_euler = (
            random.uniform(-0.1, 0.1),  # simulate tilt in X
            random.uniform(-0.1, 0.1),  # simulate tilt in Y
            random.uniform(-0.1, 0.1)   # simulate twist in Z
        )

    def _shriveling(self, obj, t):
        tex = self.tex_shrivel.copy()
        tex.noise_scale = 0.3 + t*0.3
        mod = obj.modifiers.new('Displace', type='DISPLACE')
        mod.texture = tex
        mod.strength = -(0.1 + t*0.3)

    def _softening(self, obj, t):
        mod = obj.modifiers.new('SmoothLap', type='LAPLACIANSMOOTH')
        mod.iterations = int(15 + t*25)      # deutlich mehr Iterationen
        mod.lambda_factor = 1.2 + t*1.5      # deutlich stärkerer Glättungsfaktor

    def _twisting(self, obj, t):
        mod = obj.modifiers.new('Twist', type='SIMPLE_DEFORM')
        mod.deform_method = 'TWIST'
        mod.angle = 0.1 + t*0.4

    def _stretching(self, obj, t):
        mod = obj.modifiers.new('Taper', type='SIMPLE_DEFORM')
        mod.deform_method = 'TAPER'
        mod.factor = (0.1 + t*0.9) / 3

    def _elastic(self, obj, t):
        lat_data = bpy.data.lattices.new('Lat')
        lat_data.points_u = lat_data.points_v = lat_data.points_w = 3
        lat = bpy.data.objects.new('LatObj', lat_data)
        bpy.context.scene.objects.link(lat)
        lat.location = obj.location
        lat.scale = obj.dimensions
        mod = obj.modifiers.new('Lattice', type='LATTICE')
        mod.object = lat
        bpy.context.scene.objects.active = lat
        bpy.ops.object.mode_set(mode='EDIT')
        amp = 0.02 + t*0.1
        for p in lat.data.points:
            co = p.co_deform
            p.co_deform = (co[0]+random.uniform(-amp,amp), co[1]+random.uniform(-amp,amp), co[2]+random.uniform(-amp,amp))
        bpy.ops.object.mode_set(mode='OBJECT')

    def _full_combo(self, obj, t):
            print("[ℹ️] Running full_combo with all deformations")
            base = obj
            base.name = "BaseCombo"
            deformed_objs = []

            for name, fn in self.deformations.items():
                if fn == self._full_combo:
                    continue
                try:
                    # Copy base mesh
                    dup = base.copy()
                    dup.data = base.data.copy()
                    bpy.context.scene.objects.link(dup)

                    # Apply deformation
                    fn(dup, t)

                    # Apply all modifiers
                    bpy.context.scene.objects.active = dup
                    for mod in list(dup.modifiers):
                        while dup.modifiers[0].name != mod.name:
                            bpy.ops.object.modifier_move_up(modifier=mod.name)
                        bpy.ops.object.modifier_apply(modifier=mod.name)

                    deformed_objs.append(dup)

                except Exception as e:
                    print("[⚠️] Skipping '{}' in full_combo due to error: {}".format(name, str(e)))

            # Join all deformed objects
            if len(deformed_objs) > 1:
                for objx in deformed_objs:
                    objx.select = True
                bpy.context.scene.objects.active = deformed_objs[0]
                bpy.ops.object.join()
                final = bpy.context.active_object
            elif len(deformed_objs) == 1:
                final = deformed_objs[0]
            else:
                print("[❌] full_combo failed to generate any output")
                return None

            # Remove base object
            if base.name in bpy.context.scene.objects:
                bpy.context.scene.objects.unlink(base)
            if base.name in bpy.data.objects:
                bpy.data.objects.remove(base, do_unlink=True)

            return final





    def augment(self):
        files = [f for f in os.listdir(self.mesh_dir) if f.lower().endswith('.stl')]
        for fname in files:
            # Skip mesh if fully done
            mesh_prog = self.progress.get(fname, {})
            base = self.import_and_reduce(os.path.join(self.mesh_dir, fname))

            for name, fn in self.deformations.items():
                completed = mesh_prog.get(name, -1)
                out_dir = os.path.join(self.output_dir, name)

                for i in range(completed + 1, self.num_augmentations):
                    print('Processing {0} {1} ({2}/{3})'.format(fname, name, i + 1, self.num_augmentations))
                    t = float(i) / (self.num_augmentations - 1) if self.num_augmentations > 1 else 0

                    # Duplicate base mesh
                    dup = base.copy()
                    dup.data = base.data.copy()
                    bpy.context.scene.objects.link(dup)

                    # Apply augmentation function and safely capture output
                    result = fn(dup, t)
                    if result is None:
                        result = dup

                    out_name = '{0}_{1}_{2}.stl'.format(os.path.splitext(fname)[0], name, i + 1)
                    self.bake_and_export(result, os.path.join(out_dir, out_name))

                    # Update progress
                    mesh_prog[name] = i
                    self.progress[fname] = mesh_prog
                    self._save_progress()

        print('🎉 All augmentations done.')


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mesh_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--num_augmentations', type=int, default=3)
    p.add_argument('--decimate_ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    aug = FastPollenAugmentor(args.mesh_dir, args.output_dir, args.num_augmentations, args.decimate_ratio, args.seed)
    aug.augment()
