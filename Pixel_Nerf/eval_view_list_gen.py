import numpy as np
from math import acos, degrees
import math

def get_archimedean_spiral(sphere_radius, num_steps=250):
    '''
    https://en.wikipedia.org/wiki/Spiral, section "Spherical spiral". c = a / pi
    '''
    a = 40
    r = sphere_radius

    translations = []

    i = a / 2
    while i < a:
        theta = i / a * math.pi
        x = r * math.sin(theta) * math.cos(-i)
        z = r * math.sin(-theta + math.pi) * math.sin(-i)
        y = r * - math.cos(theta)

        translations.append((x, y, z))
        i += a / (2 * num_steps)

    return np.array(translations)

def compute_angle(v1, v2):
    """Angle in degrees between two vectors."""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return degrees(acos(dot_product))

def generate_eval_pairs(num_views=250, target_angle=90.0, tolerance=5.0):
    cam_positions = get_archimedean_spiral(sphere_radius=2.0, num_steps=num_views)
    used = set()
    pairs = []

    for i, cam_i in enumerate(cam_positions):
        if i in used:
            continue
        best_j = -1
        best_diff = 999

        for j in range(i + 1, len(cam_positions)):
            if j in used:
                continue
            angle = compute_angle(cam_i, cam_positions[j])
            diff = abs(angle - target_angle)
            if diff < best_diff and diff <= tolerance:
                best_j = j
                best_diff = diff

        if best_j >= 0:
            pairs.append((i, best_j))
            used.add(i)
            used.add(best_j)

    return pairs

if __name__ == "__main__":
    pairs = generate_eval_pairs(num_views=250, target_angle=90.0, tolerance=5.0)

    with open("./viewlist/pollen_eval_view_list.txt", "w") as f:
        for i, j in pairs:
            f.write(f"{i} {j}\n")

    print(f"✅ Wrote {len(pairs)} view pairs to eval_view_list.txt")
