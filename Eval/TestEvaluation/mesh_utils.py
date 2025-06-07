import numpy as np
from scipy.spatial.distance import directed_hausdorff
import trimesh
import open3d as o3d
import numpy as np
import trimesh

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
        try:
            vox_pred = mesh_pred.voxelized(pitch)
            vox_gt = mesh_gt.voxelized(pitch)
            filled_pred = set(map(tuple, vox_pred.points))
            filled_gt = set(map(tuple, vox_gt.points))
            intersection = len(filled_pred & filled_gt)
            union = len(filled_pred | filled_gt)
            if union == 0:
                return np.nan
            return intersection / union
        except Exception:
            return np.nan

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
    def align_icp(mesh_source, mesh_target, n_points=5000, max_iterations=100, threshold=0.05):
        """
        Align mesh_source to mesh_target using Open3D ICP (point-to-plane).
        Returns:
            - mesh_aligned: transformed mesh_source
            - pts_src_aligned: transformed sampled source points
        """
        import open3d as o3d
        import numpy as np
        import trimesh

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

            print(f"[DEBUG][ICP] Fitness: {reg.fitness:.6f}, RMSE: {reg.inlier_rmse:.6f}")
            print(f"[DEBUG][ICP] Transformation:\n{reg.transformation}")

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



