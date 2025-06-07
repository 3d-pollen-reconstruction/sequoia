"""
Full evaluation script, including PSNR+SSIM evaluation with multi-GPU support.
Converted into classes, keeping the logic unchanged.
"""

import sys
import os
import open3d as o3d
import torch
import numpy as np
import imageio
import skimage.measure
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2
import tqdm
import ipdb
import warnings
import trimesh
from scipy.spatial.distance import directed_hausdorff
from matplotlib import pyplot as plt
import re

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
pixelnerf_src = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../Pixel_Nerf/src")
)
if pixelnerf_src not in sys.path:
    sys.path.insert(0, pixelnerf_src)

import util
from data import get_split_dataset
from model import make_model
from render import NeRFRenderer
from Eval.TestEvaluation.mesh_utils import MeshUtils


class EvalMetricsRunner:
    def __init__(self):
        self.best_mesh_thresh = None  # Store the best threshold

    def extra_args(self, parser):
        parser.add_argument("--split", type=str, default="test", help="Split of data")
        parser.add_argument("--gen_meshes", action="store_true", help="Generate meshes", default=True)
        parser.add_argument("--mesh_thresh", type=float, help="Threshold")
        parser.add_argument("--find_best_mesh_thresh", action="store_true", default=True, help="Best mesh threshold")
        parser.add_argument("--gt_mesh_dir", type=str, default=r"C:\Users\super\Documents\Github\sequoia\data\processed\interim", help="Directory for GT meshes")  # <-- Add this line
        parser.add_argument(
            "--source", "-P", type=str, default="0 1", help="Source view(s)"
        )
        parser.add_argument(
            "--eval_view_list", type=str, default=None, help="Path to eval view list"
        )
        parser.add_argument(
            "--coarse", action="store_true", help="Coarse network as fine"
        )
        parser.add_argument(
            "--no_compare_gt", action="store_true", help="Skip GT comparison"
        )
        parser.add_argument(
            "--multicat", action="store_true", help="Model for multiple categories"
        )
        parser.add_argument(
            "--viewlist", "-L", type=str, default="", help="Path to source view list"
        )
        parser.add_argument(
            "--output", "-O", type=str, default="eval", help="Output dir"
        )
        parser.add_argument(
            "--include_src",
            action="store_true",
            help="Include source views in calculation",
        )
        parser.add_argument("--scale", type=float, default=1.0, help="Video scale")
        parser.add_argument(
            "--write_depth", action="store_true", help="Write depth image"
        )
        parser.add_argument(
            "--write_compare", action="store_true", help="Write GT comparison image"
        )
        parser.add_argument(
            "--free_pose", action="store_true", help="Used if test poses can change"
        )
        parser.add_argument(
            "--gen_video", action="store_true", help="Generate video", default=False
        )
        return parser

    def generate_novel_views(
        self,
        images,
        poses,
        focal,
        c,
        z_near,
        z_far,
        renderer,
        net,
        device,
        output_dir,
        obj_name,
        H,
        W,
        args,
    ):
        src_view = torch.tensor([0, 1], dtype=torch.long)
        NS = len(src_view)
        num_views = 30
        radius = (z_near + z_far) * 0.5
        elevation = -10.0
        render_poses = torch.stack(
            [
                util.pose_spherical(angle, elevation, radius)
                for angle in np.linspace(-180, 180, num_views + 1)[:-1]
            ],
            0,
        ).to(device)
        render_rays = util.gen_rays(
            render_poses,
            W,
            H,
            focal * args.scale,
            z_near,
            z_far,
            c=c * args.scale if c is not None else None,
        ).to(device)

        net.encode(
            images[src_view].to(device).unsqueeze(0),
            poses[src_view].to(device).unsqueeze(0),
            focal.to(device),
            c=c,
        )
        all_rgb = []
        for rays in tqdm.tqdm(
            torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
        ):
            rgb, _ = renderer(rays[None])
            all_rgb.append(rgb[0].cpu())
        rgb_fine = torch.cat(all_rgb)
        frames = rgb_fine.view(num_views, H, W, 3).numpy()
        video_path = os.path.join(output_dir, f"{obj_name}_novel_views_30.mp4")
        imageio.mimwrite(video_path, (frames * 255).astype(np.uint8), fps=30, quality=8)
        print(f"Novel-view-Video (30 Views) gespeichert unter: {video_path}")

    def calc_mesh_metrics(self):
        return {}

    def run(self):
        args, conf = util.args.parse_args(
            self.extra_args,
            default_conf=r"C:\Users\super\Documents\Github\sequoia\Pixel_Nerf\conf\exp\pollen.conf",
            default_datadir=r"C:\Users\super\Documents\Github\sequoia\data\processed\pixelnerf\pollen",
            default_expname="pollen",
        )
        args.resume = True
        device = util.get_cuda(args.gpu_id[0])
        dset = get_split_dataset(
            args.dataset_format,
            args.datadir,
            want_split=args.split,
            training=False,
            image_size=[
                conf["model"]["img_sidelength"],
                conf["model"]["img_sidelength"],
            ],
        )
        data_loader = torch.utils.data.DataLoader(
            dset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False
        )
        output_dir = args.output.strip()
        has_output = len(output_dir) > 0
        total_psnr = 0.0
        total_ssim = 0.0
        cnt = 0

        if has_output:
            finish_path = os.path.join(output_dir, "finish.txt")
            os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(finish_path):
                with open(finish_path, "r") as f:
                    lines = [x.strip().split() for x in f.readlines()]
                lines = [x for x in lines if len(x) == 4]
                finished = set([x[0] for x in lines])
                total_psnr = sum((float(x[1]) for x in lines))
                total_ssim = sum((float(x[2]) for x in lines))
                cnt = sum((int(x[3]) for x in lines))
                if cnt > 0:
                    print("resume psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
                else:
                    total_psnr = 0.0
                    total_ssim = 0.0
            else:
                finished = set()
            finish_file = open(finish_path, "a", buffering=1)
            print("Writing images to", output_dir)
        else:
            finished = set()
            finish_file = None

        net = make_model(conf["model"]).to(device=device).load_weights(args)
        renderer = NeRFRenderer.from_conf(
            conf["renderer"], lindisp=dset.lindisp, eval_batch_size=args.ray_batch_size
        ).to(device=device)
        if args.coarse:
            pass

        if renderer.n_coarse < 64:
            renderer.n_coarse = 64
        if args.coarse:
            renderer.n_coarse = 64
            renderer.n_fine = 128
            renderer.using_fine = True

        render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()
        z_near = dset.z_near
        z_far = dset.z_far
        use_source_lut = len(args.viewlist) > 0
        if use_source_lut:
            print("Using views from list", args.viewlist)
            with open(args.viewlist, "r") as f:
                tmp = [x.strip().split() for x in f.readlines()]
            source_lut = {
                x[0] + "/" + x[1]: torch.tensor(list(map(int, x[2:])), dtype=torch.long)
                for x in tmp
            }
        else:
            source = torch.tensor(
                sorted(list(map(int, args.source.split()))), dtype=torch.long
            )
        NV = dset[0]["images"].shape[0]

        if args.eval_view_list is not None:
            with open(args.eval_view_list, "r") as f:
                eval_views = torch.tensor(list(map(int, f.readline().split())))
            target_view_mask = torch.zeros(NV, dtype=torch.bool)
            target_view_mask[eval_views] = 1
        else:
            target_view_mask = torch.ones(NV, dtype=torch.bool)

        target_view_mask_init = target_view_mask
        all_rays = None
        src_view_mask = None
        total_objs = len(data_loader)

        best_mesh_thresh = None
        mesh_thresh_candidates = [0.1, 0.5, 1.0, 2.0, 2.5, 3.0, 5.0, 6.0, 7.0, 8.0, 10.0]

        # --- Mesh threshold search: only ONCE for the whole test set ---
        if args.gen_meshes and args.find_best_mesh_thresh:
            from util.recon import marching_cubes
            print("Searching for best mesh threshold on first available GT mesh...")
            best_chamfer = float('inf')
            best_thresh = args.mesh_thresh
            pts_gt = None
            # Find the first object with a GT mesh
            for data in data_loader:
                dpath = data["path"][0]
                obj_basename = os.path.basename(dpath)
                cat_name = os.path.basename(os.path.dirname(dpath))
                obj_name = cat_name + "_" + obj_basename if args.multicat else obj_basename
                gt_mesh_path = os.path.join(args.gt_mesh_dir, f"{obj_name}.stl")
                if os.path.exists(gt_mesh_path):
                    mesh_gt = trimesh.load(gt_mesh_path, process=False)
                    mesh_gt = MeshUtils.normalize_mesh(mesh_gt)
                    mesh_gt = mesh_gt.convex_hull
                    pts_gt, _ = trimesh.sample.sample_surface(mesh_gt, 5000)
                    break
            if pts_gt is not None:
                for thresh in mesh_thresh_candidates:
                    try:
                        output = marching_cubes(
                            occu_net=net,
                            c1=[-1, -1, -1],
                            c2=[1, 1, 1],
                            reso=[128, 128, 128],
                            isosurface=thresh,
                            sigma_idx=3,
                            eval_batch_size=8192,  # <-- GOOD (must be divisible by 512)
                            coarse=args.coarse,
                            device=device,
                        )
                        if output is None:
                            continue
                        if isinstance(output, tuple) and len(output) == 2:
                            verts, tris = output
                        elif isinstance(output, tuple) and len(output) >= 4:
                            verts, tris = output[0], output[1]
                        else:
                            continue
                        mesh_pred = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
                        mesh_pred = MeshUtils.normalize_mesh(mesh_pred)
                        mesh_pred = mesh_pred.convex_hull
                        pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, 5000)
                        chamfer = MeshUtils.chamfer_distance(pts_pred, pts_gt)
                        print(f"Thresh {thresh}: Chamfer {chamfer:.6f}")
                        if chamfer < best_chamfer:
                            best_chamfer = chamfer
                            best_thresh = thresh
                    except Exception as e:
                        print(f"Threshold {thresh} failed: {e}")
                print(f"Best mesh threshold found: {best_thresh} (Chamfer {best_chamfer:.6f})")
                best_mesh_thresh = best_thresh
            else:
                print("No GT mesh found for threshold search. Using default threshold.")
                best_mesh_thresh = args.mesh_thresh

        # --- Main evaluation loop ---
        with torch.no_grad():
            for obj_idx, data in enumerate(data_loader):
                print(
                    "OBJECT",
                    obj_idx,
                    "OF",
                    total_objs,
                    "PROGRESS",
                    obj_idx / total_objs * 100.0,
                    "%",
                    data["path"][0],
                )
                dpath = data["path"][0]
                obj_basename = os.path.basename(dpath)
                cat_name = os.path.basename(os.path.dirname(dpath))
                obj_name = (
                    cat_name + "_" + obj_basename if args.multicat else obj_basename
                )
                if has_output and obj_name in finished:
                    print("(skip)")
                    continue
                images = data["images"][0]
                NV, _, H, W = images.shape

                if args.scale != 1.0:
                    Ht = int(H * args.scale)
                    Wt = int(W * args.scale)
                    if (
                        abs(Ht / args.scale - H) > 1e-10
                        or abs(Wt / args.scale - W) > 1e-10
                    ):
                        warnings.warn(
                            f"Inexact scaling: {args.scale} times ({H}, {W}) not integral"
                        )
                    H, W = Ht, Wt

                if all_rays is None or use_source_lut or args.free_pose:
                    if use_source_lut:
                        obj_id = cat_name + "/" + obj_basename
                        source = source_lut[obj_id]

                    NS = len(source)
                    src_view_mask = torch.zeros(NV, dtype=torch.bool)
                    src_view_mask[source] = 1

                    focal = data["focal"][0]
                    if isinstance(focal, float):
                        focal = torch.tensor(focal, dtype=torch.float32)
                    focal = focal[None]

                    c = data.get("c")
                    if c is not None:
                        c = c[0].to(device=device).unsqueeze(0)

                    poses = data["poses"][0]
                    src_poses = poses[src_view_mask].to(device=device)

                    target_view_mask = target_view_mask_init.clone()
                    if not args.include_src:
                        target_view_mask *= ~src_view_mask
                    novel_view_idxs = target_view_mask.nonzero(as_tuple=False).reshape(
                        -1
                    )
                    poses = poses[target_view_mask]

                    all_rays = (
                        util.gen_rays(
                            poses.reshape(-1, 4, 4),
                            W,
                            H,
                            focal * args.scale,
                            z_near,
                            z_far,
                            c=c * args.scale if c is not None else None,
                        )
                        .reshape(-1, 8)
                        .to(device=device)
                    )
                    poses = None
                    focal = focal.to(device=device)

                rays_spl = torch.split(all_rays, args.ray_batch_size, dim=0)
                n_gen_views = len(novel_view_idxs)

                net.encode(
                    images[src_view_mask].to(device=device).unsqueeze(0),
                    src_poses.unsqueeze(0),
                    focal,
                    c=c,
                )

                all_rgb, all_depth = [], []
                for rays in tqdm.tqdm(rays_spl):
                    rgb, depth = render_par(rays[None])
                    rgb = rgb[0].cpu()
                    depth = depth[0].cpu()
                    all_rgb.append(rgb)
                    all_depth.append(depth)

                all_rgb = torch.cat(all_rgb, dim=0)
                all_depth = torch.cat(all_depth, dim=0)
                all_depth = (all_depth - z_near) / (z_far - z_near)
                all_depth = all_depth.reshape(n_gen_views, H, W).numpy()
                all_rgb = torch.clamp(
                    all_rgb.reshape(n_gen_views, H, W, 3), 0.0, 1.0
                ).numpy()

                if has_output:
                    obj_out_dir = os.path.join(output_dir, obj_name)
                    os.makedirs(obj_out_dir, exist_ok=True)
                    for i in range(n_gen_views):
                        out_file = os.path.join(
                            obj_out_dir, f"{novel_view_idxs[i].item():06}.png"
                        )
                        imageio.imwrite(out_file, (all_rgb[i] * 255).astype(np.uint8))
                        if args.write_depth:
                            out_depth_file = os.path.join(
                                obj_out_dir, f"{novel_view_idxs[i].item():06}_depth.exr"
                            )
                            out_depth_norm_file = os.path.join(
                                obj_out_dir,
                                f"{novel_view_idxs[i].item():06}_depth_norm.png",
                            )
                            depth_cmap_norm = util.cmap(all_depth[i])
                            cv2.imwrite(out_depth_file, all_depth[i])
                            imageio.imwrite(out_depth_norm_file, depth_cmap_norm)

                    try:
                        if args.gen_video:
                            print("ZNear:", z_near)
                            print("ZFar:", z_far)
                            print("Focal:", focal)
                            src_view = torch.tensor([0, 1], dtype=torch.long)
                            num_views = 30
                            radius = (z_near + z_far) * 0.5
                            elevation = -10.0
                            render_poses = torch.stack(
                                [
                                    util.pose_spherical(angle, elevation, radius)
                                    for angle in np.linspace(-180, 180, num_views + 1)[
                                        :-1
                                    ]
                                ],
                                0,
                            ).to(device)
                            render_rays = util.gen_rays(
                                render_poses,
                                W,
                                H,
                                focal * args.scale,
                                z_near,
                                z_far,
                                c=c * args.scale if c is not None else None,
                            ).to(device)
                            net.encode(
                                images[src_view].to(device).unsqueeze(0),
                                data["poses"][0][src_view].to(device).unsqueeze(0),
                                focal.to(device),
                                c=c,
                            )
                            all_rgb_novel = []
                            for rays in tqdm.tqdm(
                                torch.split(
                                    render_rays.view(-1, 8), args.ray_batch_size, dim=0
                                ),
                                desc="Novel 30-View",
                            ):
                                rgb, _ = render_par(rays[None])
                                all_rgb_novel.append(rgb[0].cpu())
                            rgb_fine_novel = torch.cat(all_rgb_novel)
                            frames_novel = rgb_fine_novel.view(
                                num_views, H, W, 3
                            ).numpy()
                            video_path_novel = os.path.join(
                                obj_out_dir, f"{obj_name}_novel_views_30.mp4"
                            )
                            imageio.mimwrite(
                                video_path_novel,
                                (frames_novel * 255).astype(np.uint8),
                                fps=30,
                                quality=8,
                            )
                            print(
                                f"Novel-view-Video (30 Views, aus 2 Input-Views) gespeichert unter: {video_path_novel}"
                            )
                    except Exception as e:
                        print(f"Fehler beim Erstellen des 30-View-Videos: {e}")

                curr_ssim = 0.0
                curr_psnr = 0.0
                if not args.no_compare_gt:
                    images_0to1 = images * 0.5 + 0.5
                    images_gt = images_0to1[target_view_mask]
                    rgb_gt_all = images_gt.permute(0, 2, 3, 1).contiguous().numpy()
                    for view_idx in range(n_gen_views):
                        ssim = compare_ssim(
                            all_rgb[view_idx],
                            rgb_gt_all[view_idx],
                            win_size=5,
                            channel_axis=-1,
                            data_range=1.0,
                        )
                        psnr = compare_psnr(
                            all_rgb[view_idx], rgb_gt_all[view_idx], data_range=1
                        )
                        curr_ssim += ssim
                        curr_psnr += psnr
                        if args.write_compare and has_output:
                            out_file = os.path.join(
                                obj_out_dir,
                                f"{novel_view_idxs[view_idx].item():06}_compare.png",
                            )
                            out_im = np.hstack(
                                (all_rgb[view_idx], rgb_gt_all[view_idx])
                            )
                            imageio.imwrite(out_file, (out_im * 255).astype(np.uint8))
                curr_psnr /= n_gen_views
                curr_ssim /= n_gen_views
                curr_cnt = 1
                total_psnr += curr_psnr
                total_ssim += curr_ssim
                cnt += curr_cnt
                if not args.no_compare_gt:
                    print(
                        "curr psnr",
                        curr_psnr,
                        "ssim",
                        curr_ssim,
                        "running psnr",
                        total_psnr / cnt,
                        "running ssim",
                        total_ssim / cnt,
                    )
                if finish_file:
                    finish_file.write(
                        f"{obj_name} {curr_psnr} {curr_ssim} {curr_cnt}\n"
                    )

                if args.gen_meshes:
                    from util.recon import marching_cubes, save_obj

                    # Use the searched threshold if available, otherwise fallback
                    mesh_thresh_to_use = best_mesh_thresh if best_mesh_thresh is not None else args.mesh_thresh

                    # --- Use mesh_thresh_to_use for all objects ---
                    try:
                        print(f"Extracting mesh using marching_cubes (thresh={mesh_thresh_to_use})...")
                        output = marching_cubes(
                            occu_net=net,
                            c1=[-1, -1, -1],
                            c2=[1, 1, 1],
                            reso=[256, 256, 256],
                            isosurface=mesh_thresh_to_use,
                            sigma_idx=3,
                            eval_batch_size=100000,
                            coarse=args.coarse,
                            device=device,
                        )
                        if output is None:
                            raise ValueError("marching_cubes returned None")
                        if isinstance(output, tuple) and len(output) == 2:
                            verts, tris = output
                        elif isinstance(output, tuple) and len(output) >= 4:
                            verts, tris = output[0], output[1]
                        else:
                            raise ValueError(f"Unexpected output format: {output}")
                        mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
                        mesh = MeshUtils.normalize_mesh(mesh)
                        mesh = mesh.convex_hull
                        pts, _ = trimesh.sample.sample_surface(mesh, 5000)
                        chamfer = MeshUtils.chamfer_distance(pts, pts_gt)
                        print(f"Chamfer distance for {obj_name} with thresh {mesh_thresh_to_use}: {chamfer:.6f}")
                        if finish_file:
                            finish_file.write(
                                f"{obj_name}_mesh {mesh_thresh_to_use} {chamfer}\n"
                            )
                        if has_output:
                            safe_obj_name = safe_filename(obj_name)
                            mesh_out_file = os.path.join(output_dir, f"{args.expname}_{safe_obj_name}.obj")
                            save_obj(mesh_out_file, verts, tris)
                            print(f"Mesh gespeichert unter: {mesh_out_file}")
                    except Exception as e:
                        print(f"Fehler beim Extrahieren der Meshes: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        if finish_file:
            finish_file.close()

        if has_output:
            print("Evaluation abgeschlossen. Ergebnisse in", output_dir)
        else:
            print("Evaluation abgeschlossen.")

def safe_filename(name):
    # Replace any non-ASCII or problematic characters with underscore
    return re.sub(r'[^\w\-_\.]', '_', name)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    runner = EvalMetricsRunner()
    runner.run()
