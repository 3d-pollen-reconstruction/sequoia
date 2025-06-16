import os
import numpy as np
import cv2

# --- CONFIGURATION ---
base_dir = "/home2/etienne.roulet/sequoia/Pixel_Nerf/pollen/pollen_test"
output_txt = "pixel2meshpp_input.txt"
target_res = 224  # Resize from 256 to 224
fixed_number = 25  # As seen in your desired output format

entries = []

# Walk through test folder
for obj_id in sorted(os.listdir(base_dir)):
    obj_path = os.path.join(base_dir, obj_id)
    rgb_dir = os.path.join(obj_path, "rgb")
    pose_dir = os.path.join(obj_path, "pose")
    intr_path = os.path.join(obj_path, "intrinsics.txt")

    if not os.path.exists(rgb_dir) or not os.path.exists(pose_dir) or not os.path.exists(intr_path):
        continue

    # Load original intrinsics (from 256x256)
    with open(intr_path, 'r') as f:
        intr = np.array([list(map(float, line.strip().split())) for line in f])
    fx_orig = intr[0, 0]

    # Scale focal length to 224x224
    fx_scaled = fx_orig * (target_res / 256.0)

    for img_name in sorted(os.listdir(rgb_dir)):
        if not img_name.endswith(".png"):
            continue
        img_idx = os.path.splitext(img_name)[0]
        img_path = os.path.join(rgb_dir, img_name)
        pose_path = os.path.join(pose_dir, f"{img_idx}.txt")

        if not os.path.exists(pose_path):
            continue

        # Read and invert pose (camera-to-world → world-to-camera)
        pose = np.loadtxt(pose_path)
        if pose.shape == (3, 4):
            pose = np.vstack((pose, [0, 0, 0, 1]))
        pose_inv = np.linalg.inv(pose)

        # Flatten extrinsics
        extr_flat = pose_inv.flatten()

        # Optional: resize image
        img = cv2.imread(img_path)
        resized = cv2.resize(img, (target_res, target_res), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, resized)  # overwrite or save separately

        # Compose line for Pixel2Mesh++ .txt
        line = f"{img_path} {fx_scaled:.9f} " + " ".join([f"{v:.9f}" for v in extr_flat]) + f" {fixed_number}"
        entries.append(line)

# Write to output file
with open(output_txt, "w") as f:
    for line in entries:
        f.write(line + "\n")

print(f"✅ Done. {len(entries)} entries written to {output_txt}")