"""
Mesh reconstruction tools
"""
import mcubes
import torch
import numpy as np
import util
import tqdm
import warnings


def marching_cubes(
    occu_net,
    c1=[-1, -1, -1],
    c2=[1, 1, 1],
    reso=[128, 128, 128],
    isosurface=0.01,
    sigma_idx=3,
    eval_batch_size=100000,
    coarse=True,
    device=None,
):
    """
    Run marching cubes on network. Uses skimage (more robust).
    """
    import skimage.measure

    with torch.no_grad():
        grid = util.gen_grid(*zip(c1, c2, reso), ij_indexing=True)
        if device is None:
            device = next(occu_net.parameters()).device
        grid = grid.to(device)

        all_sigmas = []
        for chunk in tqdm.tqdm(torch.split(grid, eval_batch_size, dim=0), desc="Evaluating sigma"):
            viewdirs = torch.zeros((1, chunk.size(0), 3), device=device)
            output = occu_net(chunk.unsqueeze(0), coarse=coarse, viewdirs=viewdirs)
            sigma = output[0, :, sigma_idx]
            all_sigmas.append(sigma)

        sigmas = torch.cat(all_sigmas).view(*reso).cpu().numpy()
        print(f"[marching_cubes] Sigma stats: min={sigmas.min():.6f}, max={sigmas.max():.6f}, mean={sigmas.mean():.6f}")

        if np.all(sigmas <= isosurface):
            raise ValueError(f"All sigma values are below isosurface={isosurface}")

        verts, faces, normals, _ = skimage.measure.marching_cubes(sigmas, level=isosurface)
        # Rescale vertices
        c1, c2 = np.array(c1), np.array(c2)
        scale = (c2 - c1) / np.array(reso)
        verts = verts * scale + c1
        return verts, faces



def save_obj(vertices, triangles, path, vert_rgb=None):
    """
    Save OBJ file, optionally with vertex colors.
    This version is faster than PyMCubes and supports color.
    Taken from PIFu.
    :param vertices (N, 3)
    :param triangles (N, 3)
    :param vert_rgb (N, 3) rgb
    """
    file = open(path, "w", encoding="utf-8")
    if vert_rgb is None:
        # No color
        for v in vertices:
            file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    else:
        # Color
        for idx, v in enumerate(vertices):
            c = vert_rgb[idx]
            file.write(
                "v %.4f %.4f %.4f %.4f %.4f %.4f\n"
                % (v[0], v[1], v[2], c[0], c[1], c[2])
            )
    for f in triangles:
        f_plus = f + 1
        file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()
