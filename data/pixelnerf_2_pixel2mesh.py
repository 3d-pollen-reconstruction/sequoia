import os
import numpy as np
import cv2

# --- CONFIG ---
base_dir = r"C:\Users\super\Documents\Github\sequoia\data\pollen_augmented\pollen_test"
output_txt = "pixel2meshpp_camera.txt"
target_res = 224
fixed_number = 25
entries = []

for obj_id in sorted(os.listdir(base_dir)):
    obj_path = os.path.join(base_dir, obj_id)
    rgb_dir = os.path.join(obj_path, "rgb")
    pose_dir = os.path.join(obj_path, "pose")
    intr_path = os.path.join(obj_path, "intrinsics.txt")
    rgb_out_dir = os.path.join(obj_path, "rgb_resized")
    os.makedirs(rgb_out_dir, exist_ok=True)

    if not (os.path.exists(rgb_dir) and os.path.exists(pose_dir) and os.path.exists(intr_path)):
        continue

    # Load intrinsics and scale to 224x224
    with open(intr_path, 'r') as f:
        fx, cx, cy = map(float, f.readline().strip().split()[:3])
    scale_factor = target_res / 256.0
    fx_scaled = fx * scale_factor
    cx_scaled = cx * scale_factor
    cy_scaled = cy * scale_factor

    for img_name in sorted(os.listdir(rgb_dir)):
        if not img_name.lower().endswith(".png"):
            continue

        img_idx = os.path.splitext(img_name)[0]
        pose_path = os.path.join(pose_dir, f"{img_idx}.txt")
        img_path = os.path.join(rgb_dir, img_name)
        out_img_path = os.path.join(rgb_out_dir, img_name)

        if not os.path.exists(pose_path):
            continue

        # Resize image to 224x224
        img = cv2.imread(img_path)
        resized = cv2.resize(img, (target_res, target_res), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_img_path, resized)

        # Load and invert pose
        pose = np.loadtxt(pose_path)
        if pose.shape == (3, 4):
            pose = np.vstack([pose, [0, 0, 0, 1]])
        elif pose.size == 16:
            pose = pose.reshape((4, 4))

        cam_to_world = np.linalg.inv(pose)
        cam_position = cam_to_world[:3, 3]
        cam_distance = np.linalg.norm(cam_position)
        scale_value = 1.0 / cam_distance if cam_distance > 1e-5 else 1.0

        # Compose line: fx cx cy scale 25
        line = f"{fx_scaled:.9f} {cx_scaled:.9f} {cy_scaled:.9f} {scale_value:.9f} {fixed_number}"
        entries.append(line)

# Write to output file
with open(output_txt, "w") as f:
    for line in entries:
        f.write(line + "\n")

print(f"✅ Done. {len(entries)} camera entries written to '{output_txt}'")
