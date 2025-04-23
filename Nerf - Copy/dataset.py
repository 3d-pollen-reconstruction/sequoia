import torch
import numpy as np
import os
import imageio
from torch.utils.data import DataLoader


def get_rays(datapath, mode='train', n_views=100):
    """
    Load rays from a folder of N-views:
      - Images in datapath/imgs_{n_views}/ as train_i.png/test_i.png
      - Poses in datapath/imgs_{n_views}/{mode}/pose/*.txt
      - Intrinsics in datapath/imgs_{n_views}/{mode}/intrinsics/*.txt
    Returns:
      rays_o: (N, H*W, 3), ray origins
      rays_d: (N, H*W, 3), ray directions
      target_px: (N, H*W, 3), pixel RGB values
    """
    # Directories
    img_root = os.path.join(datapath, f'imgs_{n_views}')
    pose_dir = os.path.join(img_root, mode, 'pose')
    intr_dir = os.path.join(img_root, mode, 'intrinsics')

    # Gather filenames
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.txt')])
    intr_files = sorted([f for f in os.listdir(intr_dir) if f.endswith('.txt')])
    assert len(pose_files) == len(intr_files), "Mismatch pose/intrinsics count"
    N = len(pose_files)

    # Preallocate
    poses = np.zeros((N, 4, 4), dtype=float)
    intrinsics = np.zeros((N, 4, 4), dtype=float)
    images = []

    # Load each view
    for i, fname in enumerate(pose_files):
        # Pose
        poses[i] = np.loadtxt(os.path.join(pose_dir, fname)).reshape(4, 4)
        # Intrinsics
        intrinsics[i] = np.loadtxt(os.path.join(intr_dir, fname)).reshape(4, 4)
        # Image (train_i.png or test_i.png)
        idx = fname.split('_')[1].split('.')[0]
        img_name = f"{mode}_{idx}.png"
        img_path = os.path.join(img_root, img_name)
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        images.append(img[None, ...])

    images = np.concatenate(images, axis=0)
    N, H, W = images.shape[0], images.shape[1], images.shape[2]

    # Handle alpha channel
    if images.shape[3] == 4:
        alpha = images[..., 3:4]
        images = images[..., :3] * alpha + (1 - alpha)

    # Preallocate ray arrays
    rays_o = np.zeros((N, H*W, 3), dtype=float)
    rays_d = np.zeros((N, H*W, 3), dtype=float)
    target_px = images.reshape((N, H*W, 3))

    # Pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Compute rays for each view
    for i in range(N):
        c2w = poses[i]
        f = intrinsics[i, 0, 0]
        dirs = np.stack((u - W/2, -(v - H/2), -np.ones_like(u)*f), axis=-1)
        dirs_world = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
        dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
        rays_d[i] = dirs_world.reshape(-1, 3)
        rays_o[i] = c2w[:3, 3]

    return rays_o, rays_d, target_px
