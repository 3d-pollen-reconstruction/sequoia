#!/usr/bin/env python3
from PIL import Image
import os
import glob

# Passe den Pfad zu Deinem data‐Ordner an:
DATA_ROOT = "data"
SPLITS = ["pollen_train", "pollen_val", "pollen_test"]

for split in SPLITS:
    split_dir = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(split_dir):
        continue
    for pid in os.listdir(split_dir):
        rgb_dir = os.path.join(split_dir, pid, "rgb")
        if not os.path.isdir(rgb_dir):
            continue

        print(f"Processing split={split}, obj={pid}...")
        for fn in glob.glob(os.path.join(rgb_dir, "*.png")):
            img = Image.open(fn)
            if img.mode != "RGBA":
                continue  # schon RGB, überspringen

            # Weißer Hintergrund
            white_bg = Image.new("RGB", img.size, (255, 255, 255))
            # Alpha‐Composite: legt das RGBA‐Bild über Weiß
            composite = Image.alpha_composite(white_bg.convert("RGBA"), img).convert("RGB")
            composite.save(fn)  # überschreibt die alte Datei
