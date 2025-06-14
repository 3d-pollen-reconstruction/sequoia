# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import sys
import os
import wandb
import matplotlib.pyplot as plt
import imageio
import multiprocessing
import datetime
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
from dotenv import load_dotenv
import warnings
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch

print("🚀 CUDA available:", torch.cuda.is_available())
print("🧠 CUDA device count:", torch.cuda.device_count())
print("📛 CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

torch.autograd.set_detect_anomaly(True)

from dotmap import DotMap
warnings.filterwarnings('ignore', category=UserWarning)

def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="2",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
            "--gamma_delay", type=int, default=0,
            help="Number of scheduler.step() calls to wait before applying gamma decay"
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    return parser


args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=256)

device = util.get_cuda(args.gpu_id[0])
print("Using device", device)
print(conf)
dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir, image_size=[conf["model"]["img_sidelength"],conf["model"]["img_sidelength"]], training=True)
# print image dimensions
print("Image size", dset.image_size)
print(
    "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp)
)

net = make_model(conf["model"]).to(device=device)
print(conf["model"])
net.stop_encoder_grad = args.freeze_enc
if args.freeze_enc:
    print("Encoder frozen")
    net.encoder.eval()

renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp,).to(
    device=device
)

# Parallize
render_par = renderer.bind_parallel(net, args.gpu_id).eval()

nviews = list(map(int, args.nviews.split()))


class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=device)
                )

        self.z_near = dset.z_near
        self.z_far = dset.z_far
        

        self.use_bbox = args.no_bbox_step > 0

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)
        
    def extra_save_state(self):
        # Save renderer state as before
        torch.save(renderer.state_dict(), self.renderer_state_path)
        """ print(f"Saved renderer state to {self.renderer_state_path}")
        # Calculate current epoch based on _iter file and dataset size
        try:
            iter_path = os.path.join(self.args.checkpoints_path, self.args.name, "_iter")
            state = torch.load(iter_path)
            global_step = state.get("iter", 0)
            epoch = global_step // self.num_total_batches
            print(f"[HF Upload] Current epoch: {epoch}")
        except Exception as e:
            print(f"[HF Upload] Could not determine epoch from _iter: {e}")
            return

        # Upload every 30 epochs
        if epoch % 50 != 0 or epoch == 0:
            return

        # Prevent duplicate uploads
        last_uploaded_path = os.path.join(self.args.checkpoints_path, self.args.name, "_last_hf_upload.txt")
        if os.path.exists(last_uploaded_path):
            try:
                with open(last_uploaded_path, "r") as f:
                    last = int(f.read().strip())
                    if last == epoch:
                        return
            except:
                pass  # fallback: continue upload

        # Perform upload
        ckpt_dir = os.path.join(self.args.checkpoints_path, self.args.name)
        print(f"[HF Upload] Uploading checkpoint for epoch {epoch} from {ckpt_dir}...")

        try:
            upload_folder(
                folder_path=ckpt_dir,
                repo_id="Etiiir/PixelNerf_Pollen",
                repo_type="model",
                commit_message=f"Upload at epoch {epoch}",
                path_in_repo=f"checkpoints/epoch_{epoch}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                token=os.environ.get("HF_TOKEN", None),
            )
            with open(last_uploaded_path, "w") as f:
                f.write(str(epoch))
            print(f"[HF Upload] ✅ Epoch {epoch} uploaded.")
        except Exception as e:
            print(f"[HF Upload] ❌ Failed to upload: {e}") """

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []

        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            c = None
            if "c" in data:
                c = data["c"][obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )
            images_0to1 = images * 0.5 + 0.5

            cam_rays = util.gen_rays(
                poses, W, H, focal, self.z_near, self.z_far, c=c
            )  # (NV, H, W, 8)
            rgb_gt_all = images_0to1
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)

            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                pix_inds = torch.randint(0, NV * H * W, (args.ray_batch_size,))

            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                device=device
            )  # (ray_batch_size, 8)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(device)
        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = all_poses = all_images = None

        net.encode(
            src_images,
            src_poses,
            all_focals.to(device=device),
            c=all_c.to(device=device) if all_c is not None else None,
        )

        render_dict = DotMap(render_par(all_rays, want_weights=True,))
        coarse = render_dict.coarse
        fine = render_dict.fine
        using_fine = len(fine) > 0

        loss_dict = {}

        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine

        loss = rgb_loss
        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item()

        return loss_dict

    def train_step(self, data, global_step):
        loss_dict = self.calc_losses(data, is_train=True, global_step=global_step)
        # Log all training metrics to wandb
        if global_step % 144 == 0:
            wandb.log({f"train/{k}": v for k, v in loss_dict.items()}, step=global_step)
        return loss_dict

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        # Log all evaluation metrics to wa
        # ndb
        if global_step % 144 == 0:
            wandb.log({f"val/{k}": v for k, v in losses.items()}, step=global_step)
        return losses

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.randint(0, NV - curr_nviews)
        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        renderer.eval()
        source_views = (
            images_0to1[views_src]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            net.encode(
                test_images.unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
            )
            test_rays = test_rays.reshape(1, H * W, -1)
            render_dict = DotMap(render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)

            if using_fine:
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)
        
        debug_outdir = os.path.join(self.args.checkpoints_path, self.args.name, "vis_debug")
        os.makedirs(debug_outdir, exist_ok=True)
        

        step_str = f"{global_step:06d}"
        imageio.imwrite(os.path.join(debug_outdir, f"rgb_{step_str}.png"), (rgb_psnr * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(debug_outdir, f"alpha_{step_str}.png"), (alpha_fine_np * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(debug_outdir, f"depth_{step_str}.png"), (depth_fine_np / np.max(depth_fine_np + 1e-8) * 255).astype(np.uint8))

        # === Optional: visualize central density slice ===
        try:
            res = 256
            grid = torch.linspace(-0.5, 0.5, res, device=device)
            xs, ys, zs = torch.meshgrid(grid, grid, grid, indexing='ij')
            pts = torch.stack([xs, ys, zs], -1).reshape(-1, 3)

            sigma_vals = []
            for i in range(0, pts.shape[0], 65536):
                p = pts[i:i+65536]
                viewdirs = torch.zeros((1, p.shape[0], 3), device=device)
                out = net(p[None], coarse=True, viewdirs=viewdirs)
                sigma_vals.append(out[0, :, 3])

            sigma = torch.cat(sigma_vals).relu().view(res, res, res).cpu().numpy()
            central_slice = sigma[res // 2]
            plt.imshow(central_slice, cmap='inferno')
            plt.colorbar()
            plt.title("Central Sigma Slice (Z-axis)")
            plt.savefig(os.path.join(debug_outdir, f"sigma_zslice_{step_str}.png"))
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to compute sigma slice: {e}")

        # set the renderer network back to train mode
        renderer.train()
        #wandb.log({
        #    "vis/rgb": wandb.Image((rgb_psnr * 255).astype(np.uint8), caption="RGB Prediction"),
        #    "vis/alpha": wandb.Image((alpha_fine_np * 255).astype(np.uint8), caption="Alpha"),
        #    "vis/depth": wandb.Image((depth_fine_np / np.max(depth_fine_np + 1e-8) * 255).astype(np.uint8), caption="Depth"),
        #    "vis/psnr": psnr,
        #}, step=global_step)
        #wandb.log({
        #    "vis/combined": wandb.Image(vis, caption="Combined Visualization"),
        #}, step=global_step)
        # wandb log central density slice
        if 'sigma' in locals():
            pass
            #wandb.log({
            #    "vis/sigma_slice": wandb.Image(central_slice, caption="Central Sigma Slice (Z-axis)"),
            #}, step=global_step)
            
        # if alpha fine > 1.0 or alpha coarse > 1.0, print warning and exit program dont save checkpoint
        if np.any(alpha_fine_np > 1.0) or np.any(alpha_coarse_np > 1.0):
            warnings.warn(
                "Alpha values greater than 1.0 detected, exiting program to prevent saving bad checkpoint."
            )
            sys.exit(1)

            
        return vis, vals


def main():
    # parse args, set device, build datasets, net, renderer, etc.
    wandb.init(
    entity="sequoia-bat",         # Your wandb team/user
    project="PixelNerf",          # Project name
    group="Projects",             # Optional: group name
    name=args.name if hasattr(args, "name") else None,  # Run name
    config=vars(args) if hasattr(args, "__dict__") else None
)
    trainer = PixelNeRFTrainer()
    trainer.start()

if __name__ == "__main__":
    # On Windows, enable freeze_support if needed:
    multiprocessing.freeze_support()
    main()
