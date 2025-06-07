import numpy as np
from scipy.spatial.distance import directed_hausdorff
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
    def align_icp(mesh_source, mesh_target, n_points=1000, max_iterations=50):
        """
        Align mesh_source to mesh_target using ICP (rigid, no scaling).
        Returns a transformed copy of mesh_source.
        """
        # Sample points from both meshes
        pts_source, _ = trimesh.sample.sample_surface(mesh_source, n_points)
        pts_target, _ = trimesh.sample.sample_surface(mesh_target, n_points)
        # Run ICP
        matrix, _, _ = trimesh.registration.icp(pts_source, pts_target, scale=False, max_iterations=max_iterations)
        # Transform mesh_source
        mesh_aligned = mesh_source.copy()
        mesh_aligned.apply_transform(matrix)
        return mesh_aligned