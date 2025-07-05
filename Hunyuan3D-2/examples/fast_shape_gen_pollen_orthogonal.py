import os
import time
import torch
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# Directory containing all sample folders
# edit for your input directory of the orthgonal pollen images
base_dir = r"C:\Users\super\Documents\Github\sequoia\Pixel_Nerf\pollen\pollen_test"

# Mapping from filename to view key
filename_to_view = {
    "000000.png": "front",
    "000001.png": "left",
    "000002.png": "right",
    # Add more mappings if needed
}

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv-turbo',
    variant='fp16'
)
pipeline.enable_flashvdm()

for sample_name in os.listdir(base_dir):
    sample_path = os.path.join(base_dir, sample_name, "rgb")
    if not os.path.isdir(sample_path):
        continue

    images = {}
    for fname, view in filename_to_view.items():
        img_path = os.path.join(sample_path, fname)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGBA")
            if image.mode == 'RGB':
                rembg = BackgroundRemover()
                image = rembg(image)
            images[view] = image

    if len(images) < 2:
        print(f"Skipping {sample_name}: not enough views found.")
        continue

    print(f"Processing {sample_name} with views: {list(images.keys())}")
    start_time = time.time()
    mesh = pipeline(
        image=images,
        num_inference_steps=5,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(12345),
        output_type='trimesh'
    )[0]
    print(f"{sample_name} --- {time.time() - start_time:.2f} seconds ---")
    # edit your output directory here
    output_dir = r"C:\Users\super\Documents\Github\sequoia\data\results\Hunyuan3D"
    os.makedirs(output_dir, exist_ok=True)
    #mesh.export(os.path.join(output_dir, f"{sample_name}.glb"))
    mesh.export(os.path.join(output_dir, f"{sample_name}.stl"))