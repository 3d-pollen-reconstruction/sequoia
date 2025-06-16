import os
import json
import subprocess
from tqdm import tqdm
from dotenv import load_dotenv

# --- User settings ---
CHECKPOINTS_ROOT = r"C:/Users/super/Documents/Github/sequoia/Pixel_Nerf/checkpoints"
CONF_MAP_PATH = r"C:/Users/super/Documents/Github/sequoia/Eval/MeshGenerator/src/checkpoint_conf_map.json"
DATADIR = r"C:/Users/super/Documents/Github/shapenet_renderer/128_views/pollen_augmented"
OUTPUT_ROOT = r"C:/Users/super/Documents/Github/sequoia/Pixel_Nerf/eval/reconstructed"
PIXELNERF_SCRIPT = r"C:/Users/super/Documents/Github/sequoia/Eval/MeshGenerator/src/pixelnerf.py"

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
dotenv_path = os.path.join(repo_root, ".env")
load_dotenv(dotenv_path)

# --- User settings from .env ---
CHECKPOINTS_ROOT = os.getenv("CHECKPOINTS_ROOT").strip("'")
CONF_MAP_PATH = os.getenv("CONF_MAP_PATH").strip("'")
DATADIR = os.getenv("DATADIR").strip("'")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT").strip("'")
PIXELNERF_SCRIPT = os.getenv("PIXELNERF_SCRIPT").strip("'")

# --- Load checkpoint to config mapping ---
print(f"Loading checkpoint config map from {CONF_MAP_PATH}")
with open(CONF_MAP_PATH, "r") as f:
    checkpoint_conf_map = json.load(f)

checkpoint_folders = [d for d in os.listdir(CHECKPOINTS_ROOT) if os.path.isdir(os.path.join(CHECKPOINTS_ROOT, d))]
print(f"Found {len(checkpoint_folders)} checkpoint folders.")

# --- Iterate over all checkpoint folders with tqdm ---
for checkpoint_name in tqdm(checkpoint_folders, desc="Checkpoints"):
    checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, checkpoint_name)

    # Determine --source argument from checkpoint name postfix (e.g., pollen_augmentation4 -> 0 1 2 3)
    postfix = ''.join(filter(str.isdigit, checkpoint_name.split('_')[-1]))
    if postfix.isdigit():
        n_views = int(postfix)
        source = ' '.join(str(i) for i in range(n_views))
    else:
        source = "0 1"

    # Get config path from mapping
    conf_path = checkpoint_conf_map.get(checkpoint_name)
    if not conf_path:
        print(f"Warning: No config found for checkpoint '{checkpoint_name}', skipping.")
        continue

    output_dir = os.path.join(OUTPUT_ROOT, checkpoint_name)
    checkpoint_path_arg = checkpoint_dir

    cmd = [
        "python", PIXELNERF_SCRIPT,
        "--datadir", DATADIR,
        "--output", output_dir,
        "--source", f'"{source}"',
        "--gen_meshes",
        "--include_src",
        "--write_compare",
        "--conf", conf_path,
        "--datadir", DATADIR,
        "--checkpoints_path", checkpoint_path_arg
    ]

    print(f"\n=== Running checkpoint: {checkpoint_name} ===")
    print("Command:", ' '.join(cmd))
    result = subprocess.run(' '.join(cmd), shell=True)
    print(f"Finished {checkpoint_name} with return code {result.returncode}")