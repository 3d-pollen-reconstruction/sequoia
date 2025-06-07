import os
import sys
import pandas as pd
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Add MeshUtils to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../MeshGenerator/src")))
from mesh_utils import MeshUtils

# Config
PRED_ROOT = r"C:\Users\super\Documents\Github\sequoia\Eval\data"
GT_ROOT = r"C:\Users\super\Documents\Github\sequoia\data\processed\interim"
GT_ROOT_AUG = r"C:\Users\super\Documents\Github\sequoia\data\processed\augmented"
CSV_SAVE_PATH = r"C:\Users\super\Documents\Github\sequoia\Eval\TestEvaluation\mesh_eval_results.csv"

GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

PLOTS_DIR = os.path.join(os.path.dirname(CSV_SAVE_PATH), "test", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
COMPARE_DIR = os.path.join(os.path.dirname(CSV_SAVE_PATH), "test", "compare")
os.makedirs(COMPARE_DIR, exist_ok=True)

class MeshEvaluator:
    def __init__(self, pred_root, gt_root, gt_root_aug, csv_save_path):
        self.pred_root = pred_root
        self.gt_root = gt_root
        self.gt_root_aug = gt_root_aug
        self.csv_save_path = csv_save_path
        self.results = []
        self.done_set = set()
        self._load_existing_results()

    def _load_existing_results(self):
        if os.path.exists(self.csv_save_path):
            df = pd.read_csv(self.csv_save_path)
            self.done_set = set(df['pred_path'].tolist())
            self.results = df.to_dict('records')
        else:
            self.done_set = set()
            self.results = []

    @staticmethod
    def evaluate_mesh(pred_mesh_path, gt_mesh_path, fscore_thresh=0.01):
        try:
            mesh_pred = trimesh.load(pred_mesh_path, process=False)
            mesh_gt = trimesh.load(gt_mesh_path, process=False)
            # Align predicted mesh to GT mesh (no scaling)
            mesh_pred_aligned = MeshUtils.align_icp(mesh_pred, mesh_gt)
            mesh_pred_hull = MeshUtils.normalize_mesh(mesh_pred_aligned.copy()).convex_hull
            mesh_gt_hull = MeshUtils.normalize_mesh(mesh_gt.copy()).convex_hull
            pts_pred, face_idx_pred = trimesh.sample.sample_surface(mesh_pred_hull, 5000)
            pts_gt, face_idx_gt = trimesh.sample.sample_surface(mesh_gt_hull, 5000)
            chamfer = MeshUtils.chamfer_distance(pts_pred, pts_gt)
            hausdorff = MeshUtils.hausdorff_distance(pts_pred, pts_gt)
            fscore = MeshUtils.fscore(pts_pred, pts_gt, threshold=fscore_thresh)

            # --- Additional metrics ---
            # Volume difference
            vol_diff = MeshUtils.volume_difference(mesh_pred, mesh_gt)
            # Surface area difference
            area_diff = MeshUtils.surface_area_difference(mesh_pred, mesh_gt)
            # Edge length stats
            edge_mean_pred, edge_std_pred = MeshUtils.edge_length_stats(mesh_pred)
            edge_mean_gt, edge_std_gt = MeshUtils.edge_length_stats(mesh_gt)
            # Voxel IoU
            voxel_iou = MeshUtils.voxel_iou(mesh_pred, mesh_gt, pitch=0.02)
            # Euler characteristic
            euler_pred = MeshUtils.euler_characteristic(mesh_pred)
            euler_gt = MeshUtils.euler_characteristic(mesh_gt)
            # Normal consistency (sample normals from original meshes)
            normals_pred = mesh_pred.face_normals[face_idx_pred]
            normals_gt = mesh_gt.face_normals[face_idx_gt]
            normal_consistency = MeshUtils.normal_consistency(pts_pred, pts_gt, normals_pred, normals_gt)

            return (
                chamfer, hausdorff, fscore, mesh_pred_hull, mesh_gt_hull,
                vol_diff, area_diff,
                edge_mean_pred, edge_std_pred, edge_mean_gt, edge_std_gt,
                voxel_iou, euler_pred, euler_gt, normal_consistency
            )
        except Exception as e:
            return (
                f"ERROR: {e}", None, None, None, None,
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            )

    def get_gt_path(self, pred_path, use_augmented=False):
        basename = os.path.basename(pred_path)
        obj_name = os.path.splitext(basename)[0]
        gt_dir = self.gt_root_aug if use_augmented else self.gt_root
        return os.path.join(gt_dir, f"{obj_name}.stl")

    def collect_mesh_files(self, model_dir):
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(model_dir)
            for f in files if f.endswith(".stl")
        ]

    def plot_meshes(self, mesh_pred, mesh_gt, model_name, fname):
        # Sample points from both meshes (convex hull)
        pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, 5000)
        pts_gt, _ = trimesh.sample.sample_surface(mesh_gt, 5000)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot predicted point cloud (aligned, deepskyblue)
        ax.scatter(pts_pred[:, 0], pts_pred[:, 1], pts_pred[:, 2], c='deepskyblue', s=1, label='Prediction')

        # Plot GT point cloud (orange)
        ax.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], c='orange', s=1, label='Ground Truth')

        ax.set_title('Prediction (blue) vs Ground Truth (orange) - Overlapped')
        ax.set_axis_off()
        ax.legend(loc='upper right')

        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"{model_name}_{os.path.splitext(fname)[0]}_pc.png")
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        return plot_path

    def plot_meshes_compare(self, mesh_pred, mesh_gt, model_name, fname):
        # Plot the original (untouched) meshes as surfaces, not point clouds
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # Plot predicted mesh (original, not convex hull)
        ax1.set_title('Prediction (Mesh)')
        ax1.plot_trisurf(
            mesh_pred.vertices[:, 0], mesh_pred.vertices[:, 1], mesh_pred.vertices[:, 2],
            triangles=mesh_pred.faces, color='deepskyblue', edgecolor='none', alpha=0.9
        )
        ax1.set_axis_off()

        # Plot GT mesh (original, not convex hull)
        ax2.set_title('Ground Truth (Mesh)')
        ax2.plot_trisurf(
            mesh_gt.vertices[:, 0], mesh_gt.vertices[:, 1], mesh_gt.vertices[:, 2],
            triangles=mesh_gt.faces, color='orange', edgecolor='none', alpha=0.9
        )
        ax2.set_axis_off()

        plt.tight_layout()
        compare_path = os.path.join(COMPARE_DIR, f"{model_name}_{os.path.splitext(fname)[0]}_compare.png")
        plt.savefig(compare_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        return compare_path

    def run(self):
        print(f"Saving CSV results to: {self.csv_save_path}")
        model_folders = [
            os.path.join(self.pred_root, d)
            for d in os.listdir(self.pred_root)
            if os.path.isdir(os.path.join(self.pred_root, d))
        ]
        for model_dir in tqdm(model_folders, desc=f"{GREEN}Model Folders{RESET}", bar_format="{l_bar}%s{bar}%s{r_bar}" % (GREEN, RESET)):
            model_name = os.path.basename(model_dir)
            mesh_files = self.collect_mesh_files(model_dir)
            with tqdm(mesh_files, desc=f"{BLUE}Meshes in {model_name}{RESET}", leave=False, bar_format="{l_bar}%s{bar}%s{r_bar}" % (BLUE, RESET)) as mesh_bar:
                for pred_mesh_path in mesh_bar:
                    if pred_mesh_path in self.done_set:
                        continue
                    fname = os.path.basename(pred_mesh_path)
                    use_augmented = "augmented" in pred_mesh_path
                    gt_mesh_path = self.get_gt_path(pred_mesh_path, use_augmented=use_augmented)
                    if not os.path.exists(gt_mesh_path):
                        result = {
                            "model": model_name,
                            "pred_path": pred_mesh_path,
                            "gt_path": gt_mesh_path,
                            "chamfer": "GT_NOT_FOUND",
                            "hausdorff": "GT_NOT_FOUND",
                            "fscore": "GT_NOT_FOUND",
                            "augmented": use_augmented,
                            "plot_path": "",
                            "compare_path": "",
                            "vol_diff": np.nan,
                            "area_diff": np.nan,
                            "edge_mean_pred": np.nan,
                            "edge_std_pred": np.nan,
                            "edge_mean_gt": np.nan,
                            "edge_std_gt": np.nan,
                            "voxel_iou": np.nan,
                            "euler_pred": np.nan,
                            "euler_gt": np.nan,
                            "normal_consistency": np.nan
                        }
                    else:
                        (
                            chamfer, hausdorff, fscore, mesh_pred_hull, mesh_gt_hull,
                            vol_diff, area_diff,
                            edge_mean_pred, edge_std_pred, edge_mean_gt, edge_std_gt,
                            voxel_iou, euler_pred, euler_gt, normal_consistency
                        ) = self.evaluate_mesh(pred_mesh_path, gt_mesh_path)
                        # Load original meshes (not convex hull)
                        mesh_pred_orig = trimesh.load(pred_mesh_path, process=False)
                        mesh_gt_orig = trimesh.load(gt_mesh_path, process=False)
                        plot_path = ""
                        compare_path = ""
                        if mesh_pred_hull is not None and mesh_gt_hull is not None:
                            try:
                                plot_path = self.plot_meshes(mesh_pred_hull, mesh_gt_hull, model_name, fname)
                            except Exception as e:
                                plot_path = f"ERROR: {e}"
                        if mesh_pred_orig is not None and mesh_gt_orig is not None:
                            try:
                                compare_path = self.plot_meshes_compare(mesh_pred_orig, mesh_gt_orig, model_name, fname)
                            except Exception as e:
                                compare_path = f"ERROR: {e}"
                        result = {
                            "model": model_name,
                            "pred_path": pred_mesh_path,
                            "gt_path": gt_mesh_path,
                            "chamfer": chamfer,
                            "hausdorff": hausdorff,
                            "fscore": fscore,
                            "augmented": use_augmented,
                            "plot_path": plot_path,
                            "compare_path": compare_path,
                            "vol_diff": vol_diff,
                            "area_diff": area_diff,
                            "edge_mean_pred": edge_mean_pred,
                            "edge_std_pred": edge_std_pred,
                            "edge_mean_gt": edge_mean_gt,
                            "edge_std_gt": edge_std_gt,
                            "voxel_iou": voxel_iou,
                            "euler_pred": euler_pred,
                            "euler_gt": euler_gt,
                            "normal_consistency": normal_consistency
                        }
                    tqdm.write(f"{model_name}/{fname}: {result['chamfer']}, {result['hausdorff']}, {result['fscore']}")
                    self.results.append(result)
                    self.done_set.add(pred_mesh_path)
                    self.save_results(incremental=True)
        self.save_results(incremental=False)

    def save_results(self, incremental=False):
        df = pd.DataFrame(self.results)
        df.to_csv(self.csv_save_path, index=False)
        if not incremental:
            print("\nSummary Table:")
            print(df)
            print(f"\nResults saved to: {self.csv_save_path}")

if __name__ == "__main__":
    MeshEvaluator(
        pred_root=PRED_ROOT,
        gt_root=GT_ROOT,
        gt_root_aug=GT_ROOT_AUG,
        csv_save_path=CSV_SAVE_PATH
    ).run()