import os
import subprocess
from multiprocessing import Pool
import json
from functools import partial

# === CONFIGURATION ===
blender_path    = r"C:\Program Files\Blender2.7\blender.exe"
script_path     = r"C:\Users\super\Documents\GitHub\shapenet_renderer\shapenet_spherical_renderer_multi_core.py"

augmentation_root = r"C:\Users\super\Documents\GitHub\shapenet_renderer\augmentation"
output_dir        = r"C:\Users\super\Documents\GitHub\shapenet_renderer\128_views\256_res"
split_file        = os.path.join(output_dir, "splits.json")
progress_file     = os.path.join(output_dir, "render_progress.json")

num_observations = "128"
resolution       = "256"
num_processes    = 12

split_camera_style = {
    "train": "spherical",
    "val":   "spiral",
    "test":  "orthogonal"
}


def load_splits(path):
    with open(path, "r") as f:
        return json.load(f)


def collect_augmented_meshes(splits):
    split_sets = {s: set(splits[s]) for s in splits}
    collected = {s: [] for s in splits}

    for aug_type in os.listdir(augmentation_root):
        aug_dir = os.path.join(augmentation_root, aug_type)
        if not os.path.isdir(aug_dir):
            continue

        for fn in os.listdir(aug_dir):
            if not fn.lower().endswith(".stl"):
                continue

            marker = f"_{aug_type}_"
            if marker not in fn:
                continue
            base = fn.split(marker)[0] + ".stl"

            for split in split_sets:
                if base in split_sets[split]:
                    collected[split].append(os.path.join(aug_dir, fn))
                    break
    return collected


def infer_completed_renders():
    """Scan output directory and infer which meshes are already rendered."""
    completed = {"train": [], "val": [], "test": []}
    if not os.path.exists(output_dir):
        return completed

    for split in completed:
        split_path = os.path.join(output_dir, split)
        if not os.path.exists(split_path):
            continue
        for fn in os.listdir(split_path):
            if os.path.isdir(os.path.join(split_path, fn)):
                completed[split].append(fn)
    return completed


def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            print("[INFO] Loaded existing render_progress.json")
            return json.load(f)
    else:
        print("[INFO] render_progress.json not found — inferring from output_dir")
        progress = infer_completed_renders()
        save_progress(progress)
        return progress


def save_progress(progress):
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def render_single_mesh(mesh_path, split_name, cam_style, progress):
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    if mesh_name in progress.get(split_name, []):
        print(f"[SKIP] Already rendered: {mesh_name}")
        return

    for attempt in range(1, 4):
        cmd = [
            blender_path,
            "--background",
            "--python", script_path,
            "--addons", "io_mesh_stl",
            "--",
            "--mesh_fpath", mesh_path,
            "--output_dir", output_dir,
            "--split_name", split_name,
            "--object_name", mesh_name,
            "--num_observations", num_observations,
            "--resolution", resolution,
        ]
        if cam_style == "orthogonal":
            cmd.append("--orthogonal")

        print(f"[INFO] [{split_name}][{cam_style}] {mesh_name} (attempt {attempt})")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print(f"[DONE] {mesh_name}")
            progress[split_name].append(mesh_name)
            save_progress(progress)
            return
        else:
            print(f"[ERROR] {mesh_name} failed (attempt {attempt})")
            print(result.stderr)

    print(f"[FAIL] {mesh_name} after 3 attempts")


if __name__ == "__main__":
    splits = load_splits(split_file)
    mesh_groups = collect_augmented_meshes(splits)
    progress = load_progress()

    for split in ["train", "val", "test"]:
        meshes = mesh_groups[split]
        cam_style = split_camera_style[split]
        print(f"\n=== SPLIT={split} has {len(meshes)} augmented meshes → cam={cam_style}")

        worker = partial(render_single_mesh,
                         split_name=split,
                         cam_style=cam_style,
                         progress=progress)

        with Pool(processes=num_processes) as pool:
            pool.map(worker, meshes)

        print(f"[INFO] Done rendering split {split}")
