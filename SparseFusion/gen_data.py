import os
import torch
import trimesh
import numpy as np
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, PointLights, TexturesVertex,
    look_at_view_transform
)

def load_and_normalize_stl(file_path, device="cuda"):
    mesh_trimesh = trimesh.load(file_path, force='mesh')
    if not isinstance(mesh_trimesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh from {file_path}")
    mesh_trimesh.vertices -= mesh_trimesh.centroid
    scale = np.max(mesh_trimesh.extents)
    mesh_trimesh.vertices /= scale
    verts = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh_trimesh.faces, dtype=torch.int64, device=device)
    return Meshes(verts=[verts], faces=[faces])

def generate_random_views(num_views=5, dist=2.7, device="cuda"):
    elev = torch.FloatTensor(num_views).uniform_(0, 90)
    azim = torch.FloatTensor(num_views).uniform_(0, 360)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    return R.to(device), T.to(device)

def setup_renderer(R, T, image_size=(256, 256), device="cuda"):
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return renderer, cameras

def render_sparsefusion_input(mesh, num_views=5, image_size=(256, 256), device="cuda"):
    R, T = generate_random_views(num_views=num_views, device=device)
    renderer, cameras = setup_renderer(R, T, image_size=image_size, device=device)
    verts_rgb = torch.ones_like(mesh.verts_padded(), device=device)
    mesh.textures = TexturesVertex(verts_features=verts_rgb)

    images, valid_regions = [], []

    for i in range(num_views):
        image = renderer(mesh.extend(1), cameras=cameras[i:i+1])
        rgb = image[..., :3].permute(0, 3, 1, 2)
        mask = (image[..., 3] > 1e-4).float().unsqueeze(1)
        images.append(rgb)
        valid_regions.append(mask)

    images = torch.cat(images, dim=0)
    valid_regions = torch.cat(valid_regions, dim=0)

    return {
        'images': images.cpu().numpy(),
        'R': R.cpu().numpy(),
        'T': T.cpu().numpy(),
        'f': cameras.focal_length.cpu().numpy(),
        'c': cameras.principal_point.cpu().numpy(),
        'valid_region': valid_regions.cpu().numpy(),
        'image_size': np.array([image_size]*num_views, dtype=np.int32)
    }

def save_to_npz(data_dict, output_path):
    np.savez_compressed(output_path, **data_dict)

def batch_process_directory(input_dir, output_dir, num_views=5, image_size=256, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    stl_files = [f for f in os.listdir(input_dir) if f.endswith(".stl")]

    for f in tqdm(stl_files, desc="Processing STL files"):
        input_path = os.path.join(input_dir, f)
        output_path = os.path.join(output_dir, f.replace(".stl", ".npz"))

        try:
            mesh = load_and_normalize_stl(input_path, device=device)
            data = render_sparsefusion_input(mesh, num_views=num_views, image_size=(image_size, image_size), device=device)
            save_to_npz(data, output_path)
        except Exception as e:
            print(f"Error processing {f}: {e}")

# === MAIN ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Folder with .stl files')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save .npz files')
    parser.add_argument('--views', type=int, default=5)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    batch_process_directory(args.input_dir, args.output_dir, num_views=args.views, image_size=args.size, device=args.device)
