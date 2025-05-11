
# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import warnings
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch
from dotmap import DotMap


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
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--gamma_delay", type=int, default=0,
        help="Number of scheduler.step() calls to wait before applying gamma decay"
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
        help="Use fixed test batch for visualization",
    )
    return parser


# Parse arguments and configuration
args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load datasets
dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)
print(f"dset z_near {dset.z_near}, z_far {dset.z_far}, lindisp {dset.lindisp}")

# Build model and wrap for multi-GPU
net = make_model(conf["model"])
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    net = torch.nn.DataParallel(net)
    print("DataParallel model")
net = net.to(device)

# Freeze encoder if requested
net_model = net.module if isinstance(net, torch.nn.DataParallel) else net
net_model.stop_encoder_grad = args.freeze_enc
if args.freeze_enc:
    print("Encoder frozen")
    net_model.encoder.eval()

# Renderer setup
renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp).to(device)
# Parallelize renderer if needed
render_par = renderer.bind_parallel(net, args.gpu_id).eval()

# Number of views settings
nviews = list(map(int, args.nviews.split()))


class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = f"{self.args.checkpoints_path}/{self.args.name}/_renderer"

        # Loss weights
        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(f"lambda coarse {self.lambda_coarse} and fine {self.lambda_fine}")

        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb_fine"] if "rgb_fine" in conf["loss"] else conf["loss.rgb"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        # Optionally resume renderer state
        if args.resume and os.path.exists(self.renderer_state_path):
            renderer.load_state_dict(torch.load(self.renderer_state_path, map_location=device))

        self.z_near = dset.z_near
        self.z_far = dset.z_far
        self.use_bbox = args.no_bbox_step > 0

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        # Move inputs to device
        all_images = data["images"].to(device)
        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device)
        all_bboxes = data.get("bbox")
        all_focals = data["focal"]
        all_c = data.get("c")

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(f">>> Stopped using bbox sampling @ iter {global_step}")
        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt, all_rays = [], []
        curr_nviews = np.random.choice(nviews)
        image_ord = torch.randint(0, NV, (SB, curr_nviews)) if curr_nviews > 1 else torch.randint(0, NV, (SB, 1))

        for obj_idx in range(SB):
            images = all_images[obj_idx]
            poses = all_poses[obj_idx]
            focal = all_focals[obj_idx]
            c_val = all_c[obj_idx] if all_c is not None else None

            images_0to1 = images * 0.5 + 0.5
            cam_rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c_val)
            rgb_gt_all = images_0to1.permute(0,2,3,1).reshape(-1,3)

            if all_bboxes is not None:
                pix = util.bbox_sample(all_bboxes[obj_idx], args.ray_batch_size)
                pix_inds = pix[...,0]*H*W + pix[...,1]*W + pix[...,2]
            else:
                pix_inds = torch.randint(0, NV*H*W, (args.ray_batch_size,))

            all_rgb_gt.append(rgb_gt_all[pix_inds])
            all_rays.append(cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(device))

        all_rgb_gt = torch.stack(all_rgb_gt)
        all_rays = torch.stack(all_rays)
        image_ord = image_ord.to(device)

        src_images = util.batched_index_select_nd(all_images, image_ord)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)

        # Encode with correct net reference
        net_ref = net.module if isinstance(net, torch.nn.DataParallel) else net
        net_ref.encode(src_images, src_poses, all_focals.to(device),
                       c=all_c.to(device) if all_c is not None else None)

        render_dict = DotMap(render_par(all_rays, want_weights=True))
        coarse, fine = render_dict.coarse, render_dict.fine
        using_fine = len(fine) > 0

        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss = rgb_loss * self.lambda_coarse
        loss_dict = {"rc": rgb_loss.item() * self.lambda_coarse}
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            loss = loss + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine

        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item()
        return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
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

        # set the renderer network back to train mode
        renderer.train()
        return vis, vals


trainer = PixelNeRFTrainer()
trainer.start()
