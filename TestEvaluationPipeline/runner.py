import os
import sys
import pandas as pd
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from dotenv import load_dotenv
from mesh_utils import MeshUtils

# --- Load environment variables from .env at the repo root ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(f"Loading environment variables from {repo_root}/.env")
dotenv_path = os.path.join(repo_root, ".env")
env_loaded = load_dotenv(dotenv_path)




def print_example_env():
    print("\nExample .env file (place at the root of your repository):\n")
    print("""DATA_DIR_PATH = 'C:/Users/example/Documents/GitHub/sequoia/data'
PIXEL_NERF_CONF_DIR = 'C:/Users/example/Documents/GitHub/sequoia/Pixel_Nerf/expconf.conf'

CHECKPOINTS_ROOT = 'C:/Users/example/Documents/Github/sequoia/Pixel_Nerf/checkpoints'
CONF_MAP_PATH = 'C:/Users/example/Documents/Github/sequoia/Eval/MeshGenerator/src/checkpoint_conf_map.json'
DATADIR = 'C:/Users/example/Documents/Github/shapenet_renderer/128_views/pollen_augmented'
OUTPUT_ROOT = 'C:/Users/example/Documents/Github/sequoia/Pixel_Nerf/eval/reconstructed'
PIXELNERF_SCRIPT = 'C:/Users/example/Documents/Github/sequoia/Eval/MeshGenerator/src/pixelnerf.py'

PRED_ROOT = 'C:/Users/example/Documents/Github/sequoia/TestEvaluationPipeline/data'
GT_ROOT = 'C:/Users/example/Documents/Github/sequoia/data/processed/interim'
GT_ROOT_AUG = 'C:/Users/example/Documents/Github/sequoia/data/processed/augmented'
CSV_SAVE_PATH = 'C:/Users/example/Documents/Github/sequoia/Eval/TestEvaluation/mesh_eval_results.csv'
""")

# Read from .env or print example if not found
def get_env_var(key):
    val = os.getenv(key)
    if val is None:
        print(f"Environment variable '{key}' not found.")
        print_example_env()
        exit(1)
    return val.strip("'")

PRED_ROOT = get_env_var("PRED_ROOT")
GT_ROOT = get_env_var("GT_ROOT")
GT_ROOT_AUG = get_env_var("GT_ROOT_AUG")
CSV_SAVE_PATH = get_env_var("CSV_SAVE_PATH")


GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

PLOTS_DIR = os.path.join(os.path.dirname(CSV_SAVE_PATH), "test", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
COMPARE_DIR = os.path.join(os.path.dirname(CSV_SAVE_PATH), "test", "compare")
os.makedirs(COMPARE_DIR, exist_ok=True)

class MeshEvaluator:
    def __init__(self, pred_root, gt_root, gt_root_aug, csv_save_path, plot_plots=True, plot_compare=False, debug=True, voxel_iou_on=False):
        self.pred_root = pred_root
        self.gt_root = gt_root
        self.gt_root_aug = gt_root_aug
        self.csv_save_path = csv_save_path
        self.results = []
        self.done_set = set()
        self.plot_plots = plot_plots
        self.plot_compare = plot_compare
        self.debug = debug
        self.voxel_iou_on = voxel_iou_on
        self._load_existing_results()

    @staticmethod
    def _print_duration(name, start, end, debug):
        pass  # Debug print disabled

    def _load_existing_results(self):
        if os.path.exists(self.csv_save_path):
            df = pd.read_csv(self.csv_save_path)
            self.done_set = set(df['pred_path'].tolist())
            self.results = df.to_dict('records')
        else:
            self.done_set = set()
            self.results = []

    def evaluate_mesh(self, pred_mesh_path, gt_mesh_path, fscore_thresh=0.01):
        thresholds = [0.01, 0.025, 0.05]
        
        metric_names = [
            "Load meshes",
            "ICP alignment",
            "Normalize & convex hull",
            "Sample surface points",
            "Chamfer distance",
            "Hausdorff distance",
            "F-score (1%/5%/10%)"
            "Volume difference",
            "Surface area difference",
            "Edge length stats",
            "Voxel IoU",
            "Euler characteristic",
            "Normal consistency"
        ]
        results = None
        with tqdm(metric_names, desc="Metrics", leave=False) as metric_bar:
            # 1. Load meshes
            mesh_pred = trimesh.load(pred_mesh_path, process=False)
            mesh_gt = trimesh.load(gt_mesh_path, process=False)
            metric_bar.update(1)

            # 2. ICP alignment
            mesh_pred_aligned, pts_pred_aligned = MeshUtils.align_icp(mesh_pred, mesh_gt, n_points=5000)
            metric_bar.update(1)

            # 3. Normalize & convex hull
            mesh_pred_hull = mesh_pred_aligned
            mesh_gt_hull = MeshUtils.normalize_mesh(mesh_gt.copy())
            metric_bar.update(1)

            # 4. Sample surface points
            pts_pred = pts_pred_aligned
            face_idx_pred = np.random.randint(0, len(mesh_pred.faces), size=len(pts_pred))
            pts_gt, face_idx_gt = trimesh.sample.sample_surface(mesh_gt_hull, len(pts_pred))
            metric_bar.update(1)

            # 5. Chamfer distance
            chamfer = MeshUtils.chamfer_distance(pts_pred, pts_gt)
            metric_bar.update(1)

            # 6. Hausdorff distance
            hausdorff = MeshUtils.hausdorff_distance(pts_pred, pts_gt)
            metric_bar.update(1)

            # 7. F-scores
            # Pair-wise distances only once
            d1 = np.min(np.linalg.norm(pts_pred[:, None] - pts_gt[None], axis=-1), axis=1)
            d2 = np.min(np.linalg.norm(pts_gt[:, None] - pts_pred[None], axis=-1), axis=1)
            fscore_1, fscore_2_5, fscore_5 = MeshUtils.fscore_multi(d1, d2, thresholds)
            metric_bar.update(1)

            # 8. Volume difference
            vol_diff = MeshUtils.volume_difference(mesh_pred, mesh_gt)
            metric_bar.update(1)

            # 9. Surface area difference
            area_diff = MeshUtils.surface_area_difference(mesh_pred, mesh_gt)
            metric_bar.update(1)

            # 10. Edge length stats
            edge_mean_pred, edge_std_pred = MeshUtils.edge_length_stats(mesh_pred)
            edge_mean_gt, edge_std_gt = MeshUtils.edge_length_stats(mesh_gt)
            metric_bar.update(1)

            # 11. Voxel IoU (32x32x32 grid) - use aligned and normalized meshes!
            bbox = np.vstack([mesh_pred_hull.bounds, mesh_gt_hull.bounds])
            bbox_min = bbox.min(axis=0)
            bbox_max = bbox.max(axis=0)
            max_dim = np.max(bbox_max - bbox_min)
            pitch = max_dim / 32.0 if max_dim > 0 else 1.0

            # Shift both meshes to the same origin for voxelization
            mesh_pred_vox = mesh_pred_hull.copy()
            mesh_gt_vox = mesh_gt_hull.copy()
            mesh_pred_vox.apply_translation(-bbox_min)
            mesh_gt_vox.apply_translation(-bbox_min)

            voxel_iou = MeshUtils.voxel_iou(mesh_pred_vox, mesh_gt_vox, pitch=pitch)
            metric_bar.update(1)

            # 12. Euler characteristic
            euler_pred = MeshUtils.euler_characteristic(mesh_pred)
            euler_gt = MeshUtils.euler_characteristic(mesh_gt)
            metric_bar.update(1)

            # 13. Normal consistency
            normals_pred = mesh_pred.face_normals[face_idx_pred]
            normals_gt = mesh_gt.face_normals[face_idx_gt]
            normal_consistency = MeshUtils.normal_consistency(pts_pred, pts_gt, normals_pred, normals_gt)
            metric_bar.update(1)

            results = (
                chamfer, 
                hausdorff, 
                fscore_1,
                fscore_2_5,
                fscore_5,
                mesh_pred_hull, 
                mesh_gt_hull,
                vol_diff, 
                area_diff,
                edge_mean_pred, 
                edge_std_pred, 
                edge_mean_gt, 
                edge_std_gt,
                voxel_iou, 
                euler_pred, 
                euler_gt, 
                normal_consistency,
                None
            )
            
        return results


    #def get_gt_path(self, pred_path, use_augmented=False):
    #    basename = os.path.basename(pred_path)
    #    obj_name = os.path.splitext(basename)[0]
    #    gt_dir = self.gt_root_aug if use_augmented else self.gt_root
    #    return os.path.join(gt_dir, f"{obj_name}.stl")
    
    def get_gt_path(self, pred_path, use_augmented=False):
        basename = os.path.basename(pred_path)
        obj_name = os.path.splitext(basename)[0]

        # 🔧 Strip _predict if it exists
        if obj_name.endswith("_predict"):
            obj_name = obj_name[:-8]

        gt_dir = self.gt_root_aug if use_augmented else self.gt_root

        stl_path = os.path.join(gt_dir, f"{obj_name}.stl")
        obj_path = os.path.join(gt_dir, f"{obj_name}.obj")
        ## if path ends with _00 remove this part
        if obj_name.endswith("_00"):
            obj_name = obj_name[:-3]
            stl_path = os.path.join(gt_dir, f"{obj_name}.stl")
            obj_path = os.path.join(gt_dir, f"{obj_name}.obj")
            
        # if path starts with "pollen_" remove this part
        if obj_name.startswith("pollen_"):
            obj_name = obj_name[7:]
            stl_path = os.path.join(gt_dir, f"{obj_name}.stl")
            obj_path = os.path.join(gt_dir, f"{obj_name}.obj")
        #print(f"Looking for GT mesh: {stl_path} or {obj_path}")
        if os.path.exists(stl_path):
            return stl_path
        elif os.path.exists(obj_path):
            return obj_path
        else:
            return obj_path  # fallback path for logging


    def collect_mesh_files(self, model_dir):
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(model_dir)
            for f in files if f.endswith((".stl", ".obj"))
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
                            "normal_consistency": np.nan,
                            "voxel_plot_path": ""
                        }
                    else:
                        (
                            chamfer, hausdorff, f1, f2_5, f5, mesh_pred_hull, mesh_gt_hull,
                            vol_diff, area_diff,
                            edge_mean_pred, edge_std_pred, edge_mean_gt, edge_std_gt,
                            voxel_iou, euler_pred, euler_gt, normal_consistency, voxel_plot_path
                        ) = self.evaluate_mesh(pred_mesh_path, gt_mesh_path)
                        mesh_pred_orig = trimesh.load(pred_mesh_path, process=False)
                        mesh_gt_orig = trimesh.load(gt_mesh_path, process=False)
                        plot_path = ""
                        compare_path = ""
                        if self.plot_plots and mesh_pred_hull is not None and mesh_gt_hull is not None:
                            try:
                                plot_path = self.plot_meshes(mesh_pred_hull, mesh_gt_hull, model_name, fname)
                            except Exception as e:
                                plot_path = f"ERROR: {e}"
                        if self.plot_compare and mesh_pred_orig is not None and mesh_gt_orig is not None:
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
                            "fscore_1": f1,
                            "fscore_2_5": f2_5,
                            "fscore_5": f5,
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
                            "normal_consistency": normal_consistency,
                            "voxel_plot_path": voxel_plot_path
                        }
                    tqdm.write(f"{model_name}/{fname}: chamfer={chamfer:.4g}, "
                               f"haus={hausdorff:.4g}, F1={f1:.3f}, F2.5={f2_5:.3f}, F5={f5:.3f}")
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
        csv_save_path=CSV_SAVE_PATH,
        plot_plots=True,      # Set to False to disable plots in 'plots'
        plot_compare=False,    # Set to False to disable plots in 'compare'
        debug=False,           # Set to False to disable debug timing output
        voxel_iou_on=True    # Set to False to disable voxel IoU calculation
    ).run()