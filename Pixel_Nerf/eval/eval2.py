import sys
import os

# Add mesh/export dependencies
import torch
import numpy as np
from skimage import measure
import trimesh
from PIL import Image
import torchvision.transforms as T
import tqdm
import imageio

# Project imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
import util
from model import make_model
from render import NeRFRenderer

# Argument parsing

def extra_args(parser):
    parser.add_argument(
        "--input", "-I", type=str,
        default=os.path.join(ROOT_DIR, "input"),
        help="Directory of normalized input PNGs",
    )
    parser.add_argument(
        "--output", "-O", type=str,
        default=os.path.join(ROOT_DIR, "output"),
        help="Output directory",
    )
    parser.add_argument("--size", type=int, default=128, help="Max dim of input image")
    parser.add_argument(
        "--out_size", type=str, default="128",
        help="Output image size: one or two ints (W H)",
    )
    parser.add_argument("--focal", type=float, default=131.25, help="Focal length")
    parser.add_argument("--radius", type=float, default=2, help="Camera distance")
    parser.add_argument("--z_near", type=float, default=0.01)
    parser.add_argument("--z_far", type=float, default=4)
    parser.add_argument("--elevation", "-e", type=float, default=5.0,
                        help="Elevation angle (neg=above)")
    parser.add_argument("--num_views", type=int, default=50,
                        help="Number of rotated render views")
    parser.add_argument("--fps", type=int, default=25, help="FPS for video")
    parser.add_argument("--gif", action="store_true", help="Export GIF instead of MP4")
    parser.add_argument("--no_vid", action="store_true",
                        help="Skip video export (images only)")
    # Mesh-specific args
    parser.add_argument("--mesh_res", type=int, default=256,
                        help="Marching Cubes grid resolution")
    parser.add_argument("--mesh_thresh", type=float, default=10,
                        help="Density threshold for isosurface")
    return parser

# Parse args and config
args, conf = util.args.parse_args(
    extra_args, default_expname="pollen", default_data_format="pollen"
)
args.resume = True

# Device & model
device = util.get_cuda(args.gpu_id[0])
print(conf["model"])
net = make_model(conf["model"]).to(device=device).load_weights(args)
renderer = NeRFRenderer.from_conf(
    conf["renderer"], eval_batch_size=args.ray_batch_size
).to(device=device)
render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

# Camera & rays setup
z_near, z_far = args.z_near, args.z_far
focal = torch.tensor(args.focal, dtype=torch.float32, device=device)
in_sz = args.size
sz = list(map(int, args.out_size.split()))
W, H = (sz[0], sz[0]) if len(sz) == 1 else (sz[0], sz[1])

# Precompute rays for novel views
render_poses = torch.stack([
    util.coord_from_blender() @ util.pose_spherical(a, args.elevation, args.radius)
    for a in np.linspace(-180, 180, args.num_views + 1)[:-1]
], 0)
rays = util.gen_rays(render_poses, W, H, focal, z_near, z_far).to(device=device)

# Input files
inputs = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith("001.png")]
os.makedirs(args.output, exist_ok=True)
if not inputs:
    print("No _normalize.png found in input dir.")
    sys.exit(1)

# Evaluation loop
with torch.no_grad():
    for idx, img_path in enumerate(inputs):
        im_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx+1}/{len(inputs)}] {im_name}")

        # Load & encode image
        img = Image.open(img_path).convert("RGB")
        img = T.Resize(in_sz)(img)
        img_t = util.get_image_to_tensor_balanced()(img).to(device=device)
        cam_pose = torch.eye(4, device=device)
        cam_pose[2, -1] = args.radius
        net.encode(img_t.unsqueeze(0), cam_pose.unsqueeze(0), focal)

        # === Mesh Extraction ===
        print("  Extracting mesh (.stl)...")
        # Build grid of spatial points
        res = args.mesh_res
        grid = torch.linspace(-1, 1, res, device=device)
        xs, ys, zs = torch.meshgrid(grid, grid, grid, indexing='ij')
        pts = torch.stack([xs, ys, zs], -1).reshape(-1, 3)

                        # Query density (sigma) using the full network in chunks to avoid OOM
        sigma_list = []
        # Process points in manageable chunks
        chunk_size = 65536  # adjust based on available GPU memory
        viewdirs_zero = torch.zeros((1, chunk_size, 3), device=device)
        for i in range(0, pts.size(0), chunk_size):
            pts_chunk = pts[i:i+chunk_size]
            n = pts_chunk.size(0)
            # pad viewdirs for last chunk if smaller than chunk_size
            vd = torch.zeros((1, n, 3), device=device)
            out = net.forward(pts_chunk.unsqueeze(0), coarse=True, viewdirs=vd)
            sigma_list.append(out[0, :n, 3].detach())
        sigma_vals = torch.cat(sigma_list, dim=0)
        # Reshape into volume grid
        sig = torch.relu(sigma_vals).view(res, res, res).cpu().numpy()
        # Generate mesh via Marching Cubes
        verts, faces, normals, _ = measure.marching_cubes(sig, level=args.mesh_thresh)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Export STL file
        stl_path = os.path.join(args.output, f"{im_name}.stl")
        mesh.export(stl_path)
        print(f"  -> STL saved at: {stl_path}")

        # === Optional Video Export ===
        if not args.no_vid:
            frames = []
            for rays_chunk in tqdm.tqdm(torch.split(rays.view(-1,8), 80_000), desc="Rendering"):
                rgb, _ = render_par(rays_chunk[None])
                frames.append(rgb[0])
            vid = (torch.cat(frames)
                  .view(args.num_views, H, W, 3)
                  .cpu().numpy() * 255).astype(np.uint8)
            # Save frames
            frm_dir = os.path.join(args.output, im_name + "_frames")
            os.makedirs(frm_dir, exist_ok=True)
            for i, fr in enumerate(vid):
                imageio.imwrite(os.path.join(frm_dir, f"{i:04}.png"), fr)
            # Encode video
            vid_ext = ".gif" if args.gif else ".mp4"
            vid_path = os.path.join(args.output, im_name + vid_ext)
            imageio.mimwrite(vid_path, vid, fps=args.fps, quality=8)
            print(f"  -> Video saved at: {vid_path}")
        else:
            print("  Video export skipped (--no_vid)")
