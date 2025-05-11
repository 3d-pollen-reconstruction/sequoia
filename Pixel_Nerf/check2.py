import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.spatial.distance import pdist

# === Anzahl Kameras, die geplottet werden sollen ===
NUM_VIEWS = 128
IMG_RES = 128
FX = 131.25

# === Alle Posen finden ===
pose_files_all = sorted(glob.glob('./data/pollen_train/*/pose/*.txt'))

# === Bild- & Posen-Paare validieren ===
pose_files, img_files = [], []
for f in pose_files_all:
    img_path = os.path.join(
        os.path.dirname(f).replace('pose', 'rgb'),
        os.path.basename(f).replace('.txt', '.png')
    )
    if os.path.exists(img_path):
        pose_files.append(f)
        img_files.append(img_path)
    if len(pose_files) >= NUM_VIEWS:
        break

if len(pose_files) < NUM_VIEWS:
    print(f"⚠️  Nur {len(pose_files)} gültige Posen mit Bild gefunden.")
    exit(1)

# === Posen & Bilder laden ===
poses = [np.loadtxt(f).reshape(4, 4) for f in pose_files]
poses = np.stack(poses)  # shape: (N, 4, 4)
images = [np.array(Image.open(f)) for f in img_files]
cam_positions = poses[:, :3, 3]

# === Plot ===
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')

# Ursprung
ax.scatter([0], [0], [0], c='b', label='Origin')

# Kamerapositionen
ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], c='r', label='Camera Positions')

# Übergänge
for i in range(len(cam_positions) - 1):
    p0, p1 = cam_positions[i], cam_positions[i + 1]
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c='g', linestyle='--')

# Blickrichtungen (Z-Achse) und Dot-Produkt zur Kontrolle
print("\n=== View Direction Checks ===")
for i in range(len(poses)):
    pos = cam_positions[i]
    forward = -poses[i][:3, 2]  # -Z axis
    to_origin = -pos
    dot = np.dot(forward, to_origin / np.linalg.norm(to_origin))
    print(f"[{i}] Alignment (forward · to-origin): {dot:.3f}")
    ax.quiver(pos[0], pos[1], pos[2], forward[0], forward[1], forward[2], length=1.0, color='k')
    ax.text(pos[0], pos[1], pos[2], f'{i}', fontsize=10)

ax.set_title("Camera Positions + View Directions")
ax.set_box_aspect((1, 1, 1))
ax.legend()

# === Bilder zeigen ===
for i in range(min(len(images), 4)):
    ax_img = fig.add_subplot(2, 4, 5 + i)
    ax_img.imshow(images[i])
    ax_img.set_title(f"Image {i}")
    ax_img.axis('off')

plt.tight_layout()
plt.show()

# === Weitere Checks ===
print("\n=== Sanity Checks ===")

# Z-Positionen
z_vals = cam_positions[:, 2]
print(f"Z-Positionen: min={z_vals.min():.2f}, max={z_vals.max():.2f}")

# Duplicate Camera Check
min_dist = pdist(cam_positions).min()
print(f"Closest two cameras: {min_dist:.6f} units apart")

# Field of View
fov = 2 * np.arctan(IMG_RES / (2 * FX)) * 180 / np.pi
print(f"Field of view: {fov:.2f} degrees")

# Image intensity check
print("\n=== RGB Image Check ===")
for i, img in enumerate(images):
    mean_rgb = img[..., :3].mean()
    print(f"[{i}] Mean RGB: {mean_rgb:.2f}, shape: {img.shape}")

# Intrinsics check
intrinsics_path = os.path.join(os.path.dirname(pose_files[0]), '..', 'intrinsics.txt')
intrinsics_path = os.path.abspath(intrinsics_path)
if os.path.exists(intrinsics_path):
    print("\n=== Intrinsics ===")
    with open(intrinsics_path) as f:
        print(f.read())
else:
    print("⚠️  intrinsics.txt not found.")

# Near/far check
nf_path = os.path.join(os.path.dirname(pose_files[0]), '..', 'near_far.txt')
nf_path = os.path.abspath(nf_path)
if os.path.exists(nf_path):
    near, far = np.loadtxt(nf_path)
    print(f"\nNear/Far bounds: near={near:.3f}, far={far:.3f}")
else:
    print("⚠️  near_far.txt not found.")
