import argparse
import os
import sys
import random
import bpy
from mathutils import Vector

class FastPollenAugmentor:
    """
    Optimized pollen mesh augmentation pipeline for Blender 2.79 (Python 3.5 compatible).
    Progressive intensity: first augmentation mild, last most intense.
    """
    def __init__(self, mesh_dir, output_dir, num_augmentations=3, decimate_ratio=0.2, seed=42):
        self.mesh_dir = mesh_dir
        self.output_dir = output_dir
        self.num_augmentations = num_augmentations
        self.decimate_ratio = decimate_ratio
        random.seed(seed)
        # deformation methods
        self.deformations = {
            'swelling': self._swelling,
            'shriveling': self._shriveling,
            'softening': self._softening,
            'twisting': self._twisting,
            'stretching': self._stretching,
            'elastic': self._elastic,
            'full_combo': self._full_combo,
        }
        self._prepare_workspace()

    def _prepare_workspace(self):
        # create output directories
        for name in self.deformations:
            out = os.path.join(self.output_dir, name)
            if not os.path.exists(out): os.makedirs(out)
        # prepare textures
        tex1 = bpy.data.textures.new('TexSwelling', type='CLOUDS')
        tex2 = bpy.data.textures.new('TexShrivel', type='CLOUDS')
        self.tex_swelling = tex1
        self.tex_shrivel = tex2

    def clear_scene(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def import_and_reduce(self, filepath):
        self.clear_scene()
        bpy.ops.import_mesh.stl(filepath=filepath)
        obj = bpy.context.selected_objects[0]
        # normalize
        bbox = [obj.matrix_world * Vector(c) for c in obj.bound_box]
        center = sum(bbox, Vector((0,0,0))) / 8.0
        r = max((v-center).length for v in bbox)
        if r > 0: obj.scale = (1.0/r, 1.0/r, 1.0/r)
        # decimate
        mod = obj.modifiers.new('Decimate', type='DECIMATE')
        mod.ratio = self.decimate_ratio
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return obj

    def bake_and_export(self, obj, out_path):
        bpy.ops.object.select_all(action='DESELECT')
        obj.select = True
        bpy.ops.export_mesh.stl(filepath=out_path, use_selection=True)
        bpy.data.objects.remove(obj, do_unlink=True)

    # deformation methods with intensity t in [0,1]
    def _swelling(self, obj, t):
        tex = self.tex_swelling.copy()
        tex.noise_scale = 0.3 + t*0.3
        mod = obj.modifiers.new('Displace', type='DISPLACE')
        mod.texture = tex
        mod.strength = 0.1 + t*0.3

    def _shriveling(self, obj, t):
        tex = self.tex_shrivel.copy()
        tex.noise_scale = 0.3 + t*0.3
        mod = obj.modifiers.new('Displace', type='DISPLACE')
        mod.texture = tex
        mod.strength = -2 * (0.1 + t*0.3)  # vorher: -(0.1 + t*0.3), jetzt verdoppelt

    def _softening(self, obj, t):
        mod = obj.modifiers.new('SmoothLap', type='LAPLACIANSMOOTH')
        mod.iterations = int(6 + t*10)  # vorher: 3 + t*5
        mod.lambda_factor = 0.6 + t*0.8 # vorher: 0.3 + t*0.4

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
        self._swelling(obj, t)
        self._shriveling(obj, t)
        self._softening(obj, t)
        self._twisting(obj, t)
        self._stretching(obj, t)
        self._elastic(obj, t)

    def augment(self):
        files = [f for f in os.listdir(self.mesh_dir) if f.lower().endswith('.stl')]
        for fname in files:
            base = self.import_and_reduce(os.path.join(self.mesh_dir, fname))
            for name, fn in self.deformations.items():
                out_dir = os.path.join(self.output_dir, name)
                for i in range(self.num_augmentations):
                    print('Augmenting {0} with {1} ({2}/{3})'.format(fname, name, i+1, self.num_augmentations))
                    t = float(i)/(self.num_augmentations-1) if self.num_augmentations>1 else 0
                    dup = base.copy(); dup.data = base.data.copy()
                    bpy.context.scene.objects.link(dup)
                    fn(dup, t)
                    out_name = '{0}_{1}_{2}.stl'.format(os.path.splitext(fname)[0], name, i+1)
                    self.bake_and_export(dup, os.path.join(out_dir, out_name))
        print('🎉 All augmentations done.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mesh_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--num_augmentations', type=int, default=3)
    p.add_argument('--decimate_ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    aug = FastPollenAugmentor(args.mesh_dir, args.output_dir, args.num_augmentations, args.decimate_ratio, args.seed)
    aug.augment()
