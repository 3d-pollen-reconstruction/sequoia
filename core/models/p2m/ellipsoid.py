from typing import List

import torch
import numpy as np
import trimesh

def _unique_edges(faces: np.ndarray) -> np.ndarray:
    # faces: (F,3)
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    all_e = np.vstack([e01, e12, e20])
    all_e.sort(axis=1)   # undirected
    return np.unique(all_e, axis=0)


def normalise_adj(edge_index, num_nodes):
    """Return D⁻¹Â with self-loops (PyTorch sparse COO)."""
    row, col = edge_index
    self_loop = torch.arange(num_nodes, device=row.device)
    row = torch.cat([row, self_loop])
    col = torch.cat([col, self_loop])
    val = torch.ones_like(row, dtype=torch.float32)

    deg = torch.bincount(row, minlength=num_nodes).clamp(min=1)
    val = val / deg[row]                       # row-wise D⁻¹Â
    idx = torch.stack([row, col])
    return torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes))


class Ellipsoid:
    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        base_subdivisions: int = 3,
        lap_k: int = 10,
    ):
        """
        Builds three levels of icosphere meshes (subdivisions = base_subdivisions,
        +1 and +2), scales them by (a,b,c), and then:
          - coord           : (V0, 3)  coarse vertices
          - faces[i]        : (Fi,3)   triangle indices at level i
          - edges[i]        : (Ei,2)   unique undirected edges at level i
          - laplace_idx[i]  : (Vi, lap_k) neighbor indices padded with -1
          - adj_mat[i]      : sparse COO (Vi × Vi)
          - unpool_idx      : [edges_coarse, edges_mid]
        """
        # 1) create & scale meshes
        meshes = []
        for s in [base_subdivisions, base_subdivisions + 1, base_subdivisions + 2]:
            m = trimesh.creation.icosphere(subdivisions=s, radius=1.0)
            m.apply_scale([a, b, c])
            meshes.append(m)

        # 2) extract verts & faces as torch
        verts: List[torch.Tensor] = [
            torch.from_numpy(m.vertices).float() for m in meshes
        ]
        faces: List[torch.Tensor] = [
            torch.from_numpy(m.faces.astype(np.int64)).long() for m in meshes
        ]
        self.coord = verts[0]   # coarse points for init_pts
        self.faces = faces      # three levels

        self.edges: List[torch.Tensor] = []
        self.laplace_idx: List[torch.Tensor] = []
        self.adj_mat: List[torch.sparse_coo_tensor] = []

        # 3) for each level build edges, laplacian‐index, adj_mat
        for i, (v, f) in enumerate(zip(verts, faces)):
            V = v.shape[0]

            # 3a) edges
            ue = _unique_edges(f.numpy())            # (Ei,2) as np
            e_t = torch.from_numpy(ue).long()         # to torch
            self.edges.append(e_t)

            # 3b) laplace_idx (k-NN padding)
            nbrs = [[] for _ in range(V)]
            for u, w in ue:
                nbrs[u].append(int(w))
                nbrs[w].append(int(u))

            # --- inside Ellipsoid.__init__ after building nbrs -----------
            laplace = torch.full((V, lap_k+1), -1, dtype=torch.long)
            for vidx, neigh in enumerate(nbrs):
                k = min(len(neigh), lap_k)
                laplace[vidx, :k] = torch.tensor(neigh[:k])
                laplace[vidx, -1] = k                     # <-- neighbour count
            self.laplace_idx.append(laplace)
            
            # 3c) adjacency sparse COO
            rows = []
            cols = []
            for u, w in ue:
                rows += [u, w]
                cols += [w, u]
            
            idx  = torch.tensor([rows, cols], dtype=torch.long)        # (2, E·2)
            A    = normalise_adj(idx, V)                               # D⁻¹Â with self-loops
            self.adj_mat.append(A)

        # 4) unpool: coarse→mid and mid→fine
        #    simply “every edge spawns a midpoint”
        self.unpool_idx = [
            self.edges[0],   # shape (E0,2)
            self.edges[1],   # shape (E1,2)
        ]


    def __repr__(self):
        vs = [v.shape[0] for v in self.laplace_idx]
        es = [e.shape[0] for e in self.edges]
        fs = [f.shape[0] for f in self.faces]
        return (
            f"<Ellipsoid: "
            f"levels V={vs}, E={es}, F={fs}>"
        )
