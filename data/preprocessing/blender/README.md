This is a compact implementation of a batched OBJ- and PLY-renderer in blender. The inspiration was drawn
from the "Stanford Shapenet Renderer". This code can be used to render datasets such as the ones used in the
"Scene Representation Networks" paper.

It assumes blender < 2.8, as it uses the blender-internal renderer.

To render a batch of ply files in parallel, use the "find" command in conjunction with xargs:

    find ~/Downloads/02691156/ -name *.ply -print0 | xargs -0 -n1 -P1 -I {} blender --background --python shapenet_spherical_renderer.py -- --output_dir /tmp --mesh_fpath {} --num_observations 50 --sphere_radius 1 --mode=train

Shapnet for PixelNerf:
& "C:\Program Files\Blender2.7\blender.exe" --background --python .\shapenet_spherical_renderer.py --addons "io_mesh_stl" -- --mesh_dir C:/Users/super/Documents/Github/sequoia/data/processed/meshes_obj/ --mode train --output_dir C:/Users/super/Documents/GitHub/shapenet_renderer/data/sequoia/data --num_observations 128

augmentation:
& "C:\Program Files\Blender2.7\blender.exe" --background --python ./data/preprocessing/blender/augmentation.py --addons "io_mesh_stl" -- --mesh_dir C:/Users/super/Documents/Github/sequoia/data/processed/interim/ --output_dir C:/Users/super/Documents/Github/sequoia/data/augmented_pollen/
