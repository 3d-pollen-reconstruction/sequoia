import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project src to Python path (adjust if necessary)
project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(project_root, 'src'))
from data import get_split_dataset


def compute_camera_center(M: np.ndarray) -> np.ndarray:
    """Compute camera center C from world-to-camera matrix M: C = -R^T t"""
    R = M[:3, :3]
    t = M[:3, 3]
    return -R.T @ t


def check_origin(M: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute M @ [C;1] to verify it's at origin"""
    hom = np.concatenate([C, [1.0]], axis=0)
    return M @ hom


def is_valid_pose_matrix(M):
    """Check if M is a valid SE(3) pose matrix (R orthonormal, last row = [0,0,0,1])"""
    R = M[:3, :3]
    bottom = M[3, :]
    det_R = np.linalg.det(R)
    should_be_identity = R.T @ R
    is_orthonormal = np.allclose(should_be_identity, np.eye(3), atol=1e-4)
    is_bottom_ok = np.allclose(bottom, np.array([0, 0, 0, 1]), atol=1e-5)
    return is_orthonormal, is_bottom_ok, det_R


def main():
    parser = argparse.ArgumentParser(
        description='Load SRNDataset, sample poses, and plot camera centers with sanity checks.'
    )
    parser.add_argument('--datadir', type=str, required=True,
                        help='Root directory of the dataset, e.g., .../pollen')
    parser.add_argument('--stage', type=str, default='train', choices=['train','val','test'],
                        help='Dataset split to use')
    parser.add_argument('--num_objects', type=int, default=5,
                        help='Number of objects to sample')
    parser.add_argument('--num_views', type=int, default=10,
                        help='Number of views per object to sample')
    args = parser.parse_args()

    dataset = get_split_dataset(dataset_type="srn", datadir=args.datadir, want_split=args.stage,
                                 image_size=(128, 128), world_scale=1.0)

    N = min(args.num_objects, len(dataset))
    print(f"Sampling {N} objects from {len(dataset)} total.")

    centers_all = []

    for i in range(N):
        item = dataset[i]
        poses = item['poses'].cpu().numpy()  # shape [V,4,4]
        V = poses.shape[0]
        K = min(args.num_views, V)
        print(f"\nObject {i} ('{item['path']}'): {V} views, sampling {K}.")

        for j in range(K):
            M = poses[j]
            C = compute_camera_center(M)
            transformed = check_origin(M, C)
            is_orthonormal, is_bottom_ok, det_R = is_valid_pose_matrix(M)
            print(f" View {j}:")
            print(f"   Camera Center C       = {C}")
            print(f"   M @ [C;1]             = {transformed}")
            print(f"   Orthonormal R         = {is_orthonormal}")
            print(f"   Bottom row [0 0 0 1]  = {is_bottom_ok}")
            print(f"   det(R)                = {det_R:.5f} (should be ~1.0)")
            if (not np.allclose(transformed[:3], 0.0, atol=1e-3)) or not is_orthonormal or not is_bottom_ok:
                print("   ❌ Pose check FAILED")
            else:
                print("   ✅ Pose check PASSED")
            centers_all.append(C)
            


                    # Zusätzlich: Zeige Bild mit Maske + BBox für erste View jedes Objekts
            img = item['images'][0].permute(1, 2, 0).numpy()
            mask = item['masks'][0][0].numpy()
            bbox = item['bbox'][0].numpy().astype(int)

            fig, axs = plt.subplots(1, 3, figsize=(10, 4))
            axs[0].imshow(img)
            axs[0].set_title("RGB")
            axs[0].add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                        bbox[2] - bbox[0],
                                        bbox[3] - bbox[1],
                                        edgecolor='red',
                                        facecolor='none',
                                        linewidth=2))
            axs[1].imshow(mask, cmap="gray")
            axs[1].set_title("Maske")

            masked = img * mask[..., None] + (1.0 - mask[..., None])
            axs[2].imshow(masked)
            axs[2].set_title("RGB × Maske")

            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.show()


    centers_all = np.vstack(centers_all)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centers_all[:, 0], centers_all[:, 1], centers_all[:, 2], c='blue', s=20)
    ax.set_title('Sampled Camera Centers')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()