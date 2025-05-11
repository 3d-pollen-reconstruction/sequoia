
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import numpy as np
import imageio
import skimage.measure
import util
from data import get_split_dataset
from model import make_model
from render import NeRFRenderer
import tqdm
import trimesh

# === Setup & Config ===
def extra_args(parser):
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset root (e.g. pollen_test)")
    parser.add_argument("--data_format", type=str, default="pollen", help="Dataset format (e.g. pollen, srn)")
    parser.add_argument("--output", "-O", type=str, default="output", help="Output directory")
    parser.add_argument("--mesh_res", type=int, default=256, help="Marching Cubes resolution")
    parser.add_argument("--mesh_thresh", type=float, default=10.0, help="Density threshold")
    return parser

args, conf = util.args.parse_args(
    extra_args,
    default_conf=None,
    default_expname="pollen"
)
args.resume = True

os.makedirs(args.output, exist_ok=True)
device = util.get_cuda(args.gpu_id[0])

# === Load Model ===
net = make_model(conf["model"]).to(device=device).load_weights(args)
renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=False, eval_batch_size=args.ray_batch_size).to(device=device)
render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

# === Load Data (1 object only) ===
dset = get_split_dataset(args.data_format, args.data_path, want_split="test", training=False)
dataloader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False)
data_iter = iter(dataloader)

data1 = next(data_iter)
data2 = next(data_iter)
data3 = next(data_iter)
data4 = next(data_iter)
data = next(data_iter)


images = data["images"][0]  # (NV, 3, H, W)
poses = data["poses"][0]   # (NV, 4, 4)
focal = data["focal"][0]
c = data.get("c")
c = c[0].to(device=device).unsqueeze(0) if c is not None else None

H, W = images.shape[-2:]
z_near, z_far = dset.z_near, dset.z_far

# === Use exactly 2 source views ===
source = torch.tensor([0, 1], dtype=torch.long)
src_view_mask = torch.zeros(images.shape[0], dtype=torch.bool)
src_view_mask[source] = 1
src_poses = poses[src_view_mask].to(device=device)

# === Encode features ===
focal = focal[None].to(device=device)
net.encode(images[src_view_mask].to(device=device).unsqueeze(0), src_poses.unsqueeze(0), focal, c=c)

# === Mesh Extraction ===
print("Extracting mesh (.stl)...")
res = args.mesh_res
grid = torch.linspace(-1, 1, res, device=device)
xs, ys, zs = torch.meshgrid(grid, grid, grid, indexing='ij')
pts = torch.stack([xs, ys, zs], -1).reshape(-1, 3)

sigma_list = []
chunk_size = 65536
for i in tqdm.trange(0, pts.size(0), chunk_size):
    pts_chunk = pts[i:i+chunk_size]
    viewdirs = torch.zeros((1, pts_chunk.size(0), 3), device=device)
    out = net.forward(pts_chunk.unsqueeze(0), coarse=True, viewdirs=viewdirs)
    sigma_list.append(out[0, :, 3].detach())

sigma_vals = torch.cat(sigma_list, dim=0)
sig = torch.relu(sigma_vals).view(res, res, res).cpu().numpy()
verts, faces, normals, _ = skimage.measure.marching_cubes(sig, level=args.mesh_thresh)
mesh = trimesh.Trimesh(vertices=verts, faces=faces)

mesh_path = os.path.join(args.output, "recon.stl")
mesh.export(mesh_path)
print(f"Mesh saved to {mesh_path}")