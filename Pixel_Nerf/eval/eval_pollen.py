# filepath: c:\Users\super\Documents\Github\sequoia\Pixel_Nerf\eval\eval_metrics_refactored.py
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import util
from data import get_split_dataset
from model import make_model
from render import NeRFRenderer

class MeshUtils:
    @staticmethod
    def hausdorff_distance(pts1, pts2):
        hd1 = directed_hausdorff(pts1, pts2)[0]
        hd2 = directed_hausdorff(pts2, pts1)[0]
        return max(hd1, hd2)

    @staticmethod
    def fscore(pts1, pts2, threshold):
        d1 = np.min(np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=-1), axis=1)
        d2 = np.min(np.linalg.norm(pts2[:, None, :] - pts1[None, :, :], axis=-1), axis=1)
        recall = (d1 < threshold).mean()
        precision = (d2 < threshold).mean()
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    @staticmethod
    def chamfer_distance(pts_pred, pts_gt):
        dist_pred_to_gt = np.min(
            np.linalg.norm(pts_pred[:, None, :] - pts_gt[None, :, :], axis=-1), axis=1
        )
        dist_gt_to_pred = np.min(
            np.linalg.norm(pts_gt[:, None, :] - pts_pred[None, :, :], axis=-1), axis=1
        )
        return dist_pred_to_gt.mean() + dist_gt_to_pred.mean()

    @staticmethod
    def normalize_mesh(mesh):
        verts = mesh.vertices - mesh.vertices.mean(axis=0)
        scale = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
        verts = verts / scale
        mesh.vertices = verts
        return mesh

class EvalMetricsRunner:
    def __init__(self):
        pass

    def extra_args(self, parser):
        parser.add_argument("--split", type=str, default="test", help="Split of data")
        parser.add_argument("--gen_meshes", action="store_true", help="Generate meshes")
        parser.add_argument("--mesh_thresh", type=float, default=10.0, help="Threshold")
        parser.add_argument("--source", "-P", type=str, default="0 1", help="Source view(s)")
        parser.add_argument("--eval_view_list", type=str, default=None, help="Path to eval view list")
        parser.add_argument("--coarse", action="store_true", help="Coarse network as fine")
        parser.add_argument("--no_compare_gt", action="store_true", help="Skip GT comparison")
        parser.add_argument("--multicat", action="store_true", help="Model for multiple categories")
        parser.add_argument("--viewlist", "-L", type=str, default="", help="Path to source view list")
        parser.add_argument("--output", "-O", type=str, default="eval", help="Output dir")
        parser.add_argument("--include_src", action="store_true", help="Include source views in calculation")
        parser.add_argument("--scale", type=float, default=1.0, help="Video scale")
        parser.add_argument("--write_depth", action="store_true", help="Write depth image")
        parser.add_argument("--write_compare", action="store_true", help="Write GT comparison image")
        parser.add_argument("--free_pose", action="store_true", help="Used if test poses can change")
        parser.add_argument("--gen_video", action="store_true", help="Generate video", default=False)
        return parser

    def generate_novel_views(self, images, poses, focal, c, z_near, z_far, renderer, net, device, output_dir, obj_name, H, W, args):
        src_view = torch.tensor([0, 1], dtype=torch.long)
        NS = len(src_view)
        num_views = 30
        radius = (z_near + z_far) * 0.5
        elevation = -10.0
        render_poses = torch.stack(
            [util.pose_spherical(angle, elevation, radius) for angle in np.linspace(-180, 180, num_views + 1)[:-1]],
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
        for rays in tqdm.tqdm(torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)):
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
            self.extra_args, default_conf="conf/exp/pollen.conf", default_expname="pollen",
        )
        args.resume = True
        device = util.get_cuda(args.gpu_id[0])
        dset = get_split_dataset(
            args.dataset_format, args.datadir, want_split=args.split, training=False, image_size=[conf["model"]["img_sidelength"],conf["model"]["img_sidelength"]]
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
            source = torch.tensor(sorted(list(map(int, args.source.split()))), dtype=torch.long)
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

        with torch.no_grad():
            for obj_idx, data in enumerate(data_loader):
                print(
                    "OBJECT", obj_idx, "OF", total_objs, "PROGRESS",
                    obj_idx / total_objs * 100.0, "%", data["path"][0],
                )
                dpath = data["path"][0]
                obj_basename = os.path.basename(dpath)
                cat_name = os.path.basename(os.path.dirname(dpath))
                obj_name = cat_name + "_" + obj_basename if args.multicat else obj_basename
                if has_output and obj_name in finished:
                    print("(skip)")
                    continue
                images = data["images"][0]
                NV, _, H, W = images.shape

                if args.scale != 1.0:
                    Ht = int(H * args.scale)
                    Wt = int(W * args.scale)
                    if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
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
                    novel_view_idxs = target_view_mask.nonzero(as_tuple=False).reshape(-1)
                    poses = poses[target_view_mask]

                    all_rays = util.gen_rays(
                        poses.reshape(-1, 4, 4),
                        W,
                        H,
                        focal * args.scale,
                        z_near,
                        z_far,
                        c=c * args.scale if c is not None else None,
                    ).reshape(-1, 8).to(device=device)
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
                                obj_out_dir, f"{novel_view_idxs[i].item():06}_depth_norm.png"
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
                                data["poses"][0][src_view].to(device).unsqueeze(0),
                                focal.to(device),
                                c=c,
                            )
                            all_rgb_novel = []
                            for rays in tqdm.tqdm(
                                torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0),
                                desc="Novel 30-View"
                            ):
                                rgb, _ = render_par(rays[None])
                                all_rgb_novel.append(rgb[0].cpu())
                            rgb_fine_novel = torch.cat(all_rgb_novel)
                            frames_novel = rgb_fine_novel.view(num_views, H, W, 3).numpy()
                            video_path_novel = os.path.join(obj_out_dir, f"{obj_name}_novel_views_30.mp4")
                            imageio.mimwrite(
                                video_path_novel, (frames_novel * 255).astype(np.uint8), fps=30, quality=8
                            )
                            print(f"Novel-view-Video (30 Views, aus 2 Input-Views) gespeichert unter: {video_path_novel}")
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
                            all_rgb[view_idx], rgb_gt_all[view_idx],
                            win_size=5, channel_axis=-1, data_range=1.0
                        )
                        psnr = compare_psnr(
                            all_rgb[view_idx], rgb_gt_all[view_idx], data_range=1
                        )
                        curr_ssim += ssim
                        curr_psnr += psnr
                        if args.write_compare and has_output:
                            out_file = os.path.join(
                                obj_out_dir, f"{novel_view_idxs[view_idx].item():06}_compare.png"
                            )
                            out_im = np.hstack((all_rgb[view_idx], rgb_gt_all[view_idx]))
                            imageio.imwrite(out_file, (out_im * 255).astype(np.uint8))
                curr_psnr /= n_gen_views
                curr_ssim /= n_gen_views
                curr_cnt = 1
                total_psnr += curr_psnr
                total_ssim += curr_ssim
                cnt += curr_cnt
                if not args.no_compare_gt:
                    print(
                        "curr psnr", curr_psnr, "ssim", curr_ssim,
                        "running psnr", total_psnr / cnt,
                        "running ssim", total_ssim / cnt,
                    )
                if finish_file:
                    finish_file.write(f"{obj_name} {curr_psnr} {curr_ssim} {curr_cnt}\n")

                if args.gen_meshes:
                    from util.recon import marching_cubes, save_obj
                    try:
                        print("Extracting mesh using marching_cubes...")
                        output = marching_cubes(
                            occu_net=net,
                            c1=[-1, -1, -1],
                            c2=[1, 1, 1],
                            reso=[256, 256, 256],
                            isosurface=args.mesh_thresh if hasattr(args, 'mesh_thresh') else 10.0,
                            sigma_idx=3,
                            eval_batch_size=100000,
                            coarse=args.coarse,
                            device=device
                        )
                        if output is None:
                            raise ValueError("marching_cubes returned None")
                        if isinstance(output, tuple) and len(output) == 2:
                            verts, tris = output
                        elif isinstance(output, tuple) and len(output) >= 4:
                            verts, tris = output[0], output[1]
                        else:
                            raise ValueError("Unexpected output from marching_cubes")

                        mesh_dir = os.path.join(obj_out_dir, f"{obj_name}_mesh.obj")
                        save_obj(verts, tris, mesh_dir)
                        self.calc_mesh_metrics()

                        mesh_id = obj_name
                        gt_mesh_path = os.path.join("../data/processed/meshes", f"{mesh_id}.stl")
                        print("GT-Mesh-Pfad:", gt_mesh_path)
                        if os.path.exists(gt_mesh_path):
                            mesh_pred = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
                            mesh_gt = trimesh.load(gt_mesh_path, process=False)

                            mesh_pred = MeshUtils.normalize_mesh(mesh_pred)
                            mesh_gt = MeshUtils.normalize_mesh(mesh_gt)
                            mesh_pred = mesh_pred.convex_hull
                            mesh_gt = mesh_gt.convex_hull

                            pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, 5000)
                            pts_gt, _ = trimesh.sample.sample_surface(mesh_gt, 5000)

                            pcd_pred = o3d.geometry.PointCloud()
                            pcd_pred.points = o3d.utility.Vector3dVector(pts_pred)
                            pcd_gt = o3d.geometry.PointCloud()
                            pcd_gt.points = o3d.utility.Vector3dVector(pts_gt)

                            threshold = 0.05
                            reg_p2p = o3d.pipelines.registration.registration_icp(
                                pcd_pred, pcd_gt, threshold, np.eye(4),
                                o3d.pipelines.registration.TransformationEstimationPointToPoint()
                            )
                            trans_init = reg_p2p.transformation
                            pts_pred_aligned = np.asarray(
                                (trans_init @ np.hstack([pts_pred, np.ones((pts_pred.shape[0], 1))]).T).T
                            )[:, :3]

                            chamfer = MeshUtils.chamfer_distance(pts_pred_aligned, pts_gt)
                            vol_pred = mesh_pred.volume
                            vol_gt = mesh_gt.volume
                            vol_diff = abs(vol_pred - vol_gt)
                            vol_rel_diff = abs(vol_pred - vol_gt) / vol_gt if vol_gt != 0 else float('inf')
                            area_pred = mesh_pred.area
                            area_gt = mesh_gt.area
                            area_diff = abs(area_pred - area_gt)
                            area_rel_diff = abs(area_pred - area_gt) / area_gt if area_gt != 0 else float('inf')
                            hausdorff = MeshUtils.hausdorff_distance(pts_pred_aligned, pts_gt)
                            bb_diag = np.linalg.norm(pts_gt.max(axis=0) - pts_gt.min(axis=0))
                            fscore_val = MeshUtils.fscore(pts_pred_aligned, pts_gt, threshold=0.01 * bb_diag)

                            print(f"Chamfer: {chamfer:.6f}")
                            print(f"Volumen GT: {vol_gt:.6f}, Pred: {vol_pred:.6f}, Diff: {vol_diff:.6f}, RelDiff: {vol_rel_diff:.4%}")
                            print(f"Fläche  GT: {area_gt:.6f}, Pred: {area_pred:.6f}, Diff: {area_diff:.6f}, RelDiff: {area_rel_diff:.4%}")
                            print(f"Hausdorff: {hausdorff:.6f}")
                            print(f"F-Score (1% BB): {fscore_val:.4f}")

                            metrics_path = os.path.join(obj_out_dir, f"{mesh_id}_mesh_metrics.txt")
                            with open(metrics_path, "w") as f:
                                f.write(f"Mesh: {mesh_id}\n")
                                f.write(f"Chamfer: {chamfer:.6f}\n")
                                f.write(f"Volumen GT: {vol_gt:.6f}\n")
                                f.write(f"Volumen Pred: {vol_pred:.6f}\n")
                                f.write(f"Volumen Diff: {vol_diff:.6f}\n")
                                f.write(f"Volumen RelDiff: {vol_rel_diff:.4%}\n")
                                f.write(f"Flaeche GT: {area_gt:.6f}\n")
                                f.write(f"Flaeche Pred: {area_pred:.6f}\n")
                                f.write(f"Flaeche Diff: {area_diff:.6f}\n")
                                f.write(f"Flaeche RelDiff: {area_rel_diff:.4%}\n")
                                f.write(f"Hausdorff: {hausdorff:.6f}\n")
                                f.write(f"F-Score (1% BB): {fscore_val:.4f}\n")
                            print(f"Mesh-Metriken gespeichert unter: {metrics_path}")

                            fig = plt.figure(figsize=(6, 6))
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], s=1, c='blue', label='GT')
                            ax.scatter(pts_pred_aligned[:, 0], pts_pred_aligned[:, 1], pts_pred_aligned[:, 2],
                                       s=1, c='red', label='Pred (aligned)')
                            ax.set_title(f'{mesh_id}\nChamfer: {chamfer:.4f}')
                            ax.legend()
                            ax.axis('off')
                            plt.tight_layout()
                            plot_path = os.path.join(obj_out_dir, f"{mesh_id}_3d_compare.png")
                            plt.savefig(plot_path, dpi=200)
                            plt.close(fig)
                            print(f"3D Vergleichsplot gespeichert unter: {plot_path}")
                            print(f"Chamfer Distance zu GT-Mesh für {mesh_id}: {chamfer:.6f}")
                        else:
                            print(f"Kein GT-Mesh gefunden für {mesh_id} unter {gt_mesh_path}")
                        print("Mesh saved to", mesh_dir)
                    except Exception as e:
                        print("Failed to extract mesh:", str(e))

                print("final psnr", total_psnr / cnt, "ssim", total_ssim / cnt)

class Program:
    @staticmethod
    def main():
        runner = EvalMetricsRunner()
        runner.run()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    Program.main()