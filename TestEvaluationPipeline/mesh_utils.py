import copy
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import trimesh
import open3d as o3d

class MeshUtils:
    @staticmethod
    def hausdorff_distance(pts1, pts2):
        hd1 = directed_hausdorff(pts1, pts2)[0]
        hd2 = directed_hausdorff(pts2, pts1)[0]
        return max(hd1, hd2)

    @staticmethod
    def fscore(pts1, pts2, threshold):
        d1 = np.min(np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=-1), axis=1)
        d2 = np.min(np.linalg.norm(pts2[:, None, :] - pts1[None, :, :], axis=-1), axis=1)
        recall = (d1 < threshold).mean()
        precision = (d2 < threshold).mean()
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)
    
    @staticmethod
    def fscore_multi(d1, d2, thresholds):
        """
        Same F-score definition as `fscore`, but re-uses the
        already-computed point–point distances for several thresholds.
        Returns a list with the same order as `thresholds`.
        """
        out = []
        for th in thresholds:
            recall    = (d1 < th).mean()
            precision = (d2 < th).mean()
            f = 0.0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)
            out.append(f)
        return out

    @staticmethod
    def chamfer_distance(pts_pred, pts_gt):
        dist_pred_to_gt = np.min(
            np.linalg.norm(pts_pred[:, None, :] - pts_gt[None, :, :], axis=-1), axis=1
        )
        dist_gt_to_pred = np.min(
            np.linalg.norm(pts_gt[:, None, :] - pts_pred[None, :, :], axis=-1), axis=1
        )
        return dist_pred_to_gt.mean() + dist_gt_to_pred.mean()

    @staticmethod
    def normalize_mesh(mesh):
        verts = mesh.vertices - mesh.vertices.mean(axis=0)
        scale = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
        verts = verts / scale
        mesh.vertices = verts
        return mesh

    @staticmethod
    def volume_difference(mesh_pred, mesh_gt):
        """Absolute difference in mesh volumes."""
        try:
            return abs(mesh_pred.volume - mesh_gt.volume)
        except Exception:
            return np.nan

    @staticmethod
    def surface_area_difference(mesh_pred, mesh_gt):
        """Absolute difference in mesh surface areas."""
        try:
            return abs(mesh_pred.area - mesh_gt.area)
        except Exception:
            return np.nan

    @staticmethod
    def normal_consistency(pts_pred, pts_gt, normals_pred, normals_gt):
        """
        Average cosine similarity between corresponding normals.
        Assumes pts_pred and pts_gt are sampled correspondingly.
        """
        # Normalize normals
        normals_pred = normals_pred / (np.linalg.norm(normals_pred, axis=1, keepdims=True) + 1e-8)
        normals_gt = normals_gt / (np.linalg.norm(normals_gt, axis=1, keepdims=True) + 1e-8)
        # Cosine similarity
        cos_sim = np.abs((normals_pred * normals_gt).sum(axis=1))
        return cos_sim.mean()

    @staticmethod
    def edge_length_stats(mesh):
        """
        Returns mean and std of edge lengths in the mesh.
        """
        edges = mesh.edges_unique
        verts = mesh.vertices
        edge_lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        return edge_lengths.mean(), edge_lengths.std()

    @staticmethod
    def voxel_iou(mesh_pred, mesh_gt, pitch=0.02):
        """
        Voxelizes both meshes and computes IoU.
        """
        m_pred = mesh_pred.copy()
        m_gt = mesh_gt.copy()

        # Center each mesh
        center_pred = (m_pred.bounds[0] + m_pred.bounds[1]) * 0.5
        center_gt   = (m_gt.bounds[0]   + m_gt.bounds[1])   * 0.5
        m_pred.apply_translation(-center_pred)
        m_gt.apply_translation(-center_gt)

        # Voxelize and fill interiors
        vox_pred = m_pred.voxelized(pitch).fill()
        vox_gt   = m_gt.voxelized(pitch).fill()

        # Extract points
        pts_pred = np.array(vox_pred.points)
        pts_gt   = np.array(vox_gt.points)

        # Combine for union grid
        all_pts = np.vstack([pts_pred, pts_gt])
        mins = all_pts.min(axis=0)
        
        # Convert points to integer grid indices
        idx_pred = np.round((pts_pred - mins) / pitch).astype(int)
        idx_gt   = np.round((pts_gt   - mins) / pitch).astype(int)

        # Determine grid size
        max_idx = np.max(np.vstack([idx_pred, idx_gt]), axis=0) + 1
        grid = np.zeros(max_idx, dtype=bool)

        # Fill union
        grid[tuple(idx_pred.T)] = True
        grid[tuple(idx_gt.T)]   = True

        # Compute IoU
        set_pred = {tuple(idx) for idx in idx_pred}
        set_gt   = {tuple(idx) for idx in idx_gt}
        inter = len(set_pred & set_gt)
        union = len(set_pred | set_gt)
        iou = 0.0 if union == 0 else inter / union

        return iou

    @staticmethod
    def euler_characteristic(mesh):
        """
        Returns the Euler characteristic of the mesh.
        """
        try:
            V = len(mesh.vertices)
            E = len(mesh.edges_unique)
            F = len(mesh.faces)
            return V - E + F
        except Exception:
            return np.nan

    @staticmethod
    def align_icp(mesh_source, mesh_target, n_points=5000, max_iterations=10000, threshold=0.05):
        """
        Align mesh_source to mesh_target using Open3D ICP (point-to-plane).
        Returns:
            - mesh_aligned: transformed mesh_source
            - pts_src_aligned: transformed sampled source points
        """
        src = mesh_source.copy()
        tgt = mesh_target.copy()

        # Normalize both meshes: center + scale to unit box
        def normalize_mesh_inplace(mesh):
            mesh.vertices -= mesh.vertices.mean(axis=0)
            scale = np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
            if scale > 0:
                mesh.vertices /= scale
            return mesh

        normalize_mesh_inplace(src)
        normalize_mesh_inplace(tgt)

        # Sample points
        pts_src, _ = trimesh.sample.sample_surface(src, n_points)
        pts_tgt, _ = trimesh.sample.sample_surface(tgt, n_points)

        # Convert to Open3D point clouds
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(pts_src)
        pcd_src.estimate_normals()

        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(pts_tgt)
        pcd_tgt.estimate_normals()

        # Run ICP
        try:
            reg = o3d.pipelines.registration.registration_icp(
                pcd_src, pcd_tgt, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
            )

            if reg.fitness < 1e-4:
                print(f"[WARNING] ICP failed: very low fitness. No alignment applied.")
                reg_trans = np.eye(4)
            else:
                reg_trans = reg.transformation

        except Exception as e:
            print(f"[ERROR] ICP failed due to exception: {e}")
            reg_trans = np.eye(4)

        # Apply transformation to mesh and points
        mesh_aligned = src.copy()
        mesh_aligned.apply_transform(reg_trans)

        pts_src_homo = np.hstack([pts_src, np.ones((pts_src.shape[0], 1))])
        pts_src_aligned = (reg_trans @ pts_src_homo.T).T[:, :3]

        return mesh_aligned, pts_src_aligned
