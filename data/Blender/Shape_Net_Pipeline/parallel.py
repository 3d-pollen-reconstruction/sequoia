import os
import subprocess
from multiprocessing import Pool
import json
import random
from functools import partial

# === CONFIGURATION ===
blender_path = r"C:\Program Files\Blender2.7\blender.exe"
script_path = r"C:\Users\super\Documents\Github\sequoia\Shape_Net_Pipeline\shapenet_spherical_renderer_multi_core.py"
mesh_dir = r"C:\Users\super\Documents\Github\sequoia\data\processed\interim"
output_dir = r"C:\Users\super\Documents\GitHub\shapenet_renderer\128_views\test_output"
num_observations = "128"
resolution = "256"
num_processes = 12

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
split_file = os.path.join(output_dir, "splits.json")


def generate_splits(mesh_list, split_path):
    random.seed(42)
    random.shuffle(mesh_list)
    n = len(mesh_list)
    splits = {
        "train": mesh_list[:int(0.7 * n)],
        "val": mesh_list[int(0.7 * n):int(0.85 * n)],
        "test": mesh_list[int(0.85 * n):],
    }
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def render_single_mesh(mesh_path, split_name, cam_style, max_retries=3):
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    attempt = 0
    while attempt < max_retries:
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

        print(f"[INFO] Launching Blender for: {mesh_name} [split={split_name}, cam={cam_style}] (attempt {attempt+1})")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print(f"[DONE] Finished: {mesh_name}")
            return
        else:
            print(f"[ERROR] Rendering failed for {mesh_name} (attempt {attempt+1})")
            print("STDERR:\n", result.stderr)
            attempt += 1

    print(f"[FAIL] All attempts failed for {mesh_name}")


if __name__ == "__main__":
    all_mesh_files = [
        os.path.join(mesh_dir, f)
        for f in os.listdir(mesh_dir)
        if f.lower().endswith(('.stl', '.obj'))
    ]

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            splits = json.load(f)
    else:
        mesh_names = [os.path.basename(f) for f in all_mesh_files]
        splits = generate_splits(mesh_names, split_file)

    # Define camera logic per split
    split_camera_style = {
        "train": "spherical",
        "val": "spiral",
        "test": "orthogonal"
    }

    for split_name in ["train","val","test"]:
        print(f"\n===== STARTING SPLIT: {split_name.upper()} =====")
        selected_names = set(splits[split_name])
        mesh_files = [
            os.path.join(mesh_dir, f)
            for f in os.listdir(mesh_dir)
            if f in selected_names
        ]

        print(f"[INFO] Found {len(mesh_files)} mesh files for split: {split_name}")
        cam_style = split_camera_style[split_name]

        # Use partial to freeze args for multiprocessing
        render_fn = partial(render_single_mesh, split_name=split_name, cam_style=cam_style)

        with Pool(processes=num_processes) as pool:
            pool.map(render_fn, mesh_files)

        print(f"[INFO] Completed rendering for split: {split_name}")
