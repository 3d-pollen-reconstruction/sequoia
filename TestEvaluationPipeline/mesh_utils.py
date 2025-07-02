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
    def align_icp(
        mesh_source: trimesh.Trimesh,
        mesh_target: trimesh.Trimesh,
        n_points: int = 20_000,
        voxel_size: float = 0.01,
        max_iter=(60, 40, 20),          # coarse ➜ fine
        distance_factor: float = 2.0,
        robust_kernel=o3d.pipelines.registration.TukeyLoss(k=0.05),
    ):
        """
        Coarse-to-fine registration:
        1.  Normalize meshes (center + isotropic scale).
        2.  Down-sample & estimate normals.
        3.  Fast global registration (FPFH + RANSAC) for an initial pose.
        4.  Multi-scale Generalized ICP with robust loss.
        Returns the aligned mesh, sampled aligned points and the 4×4 transform.
        """
        def norm_mesh(m):
            m = copy.deepcopy(m)
            m.vertices -= m.vertices.mean(0)
            span = np.linalg.norm(m.vertices.ptp(0))
            m.vertices /= span
            return m

        src, tgt = norm_mesh(mesh_source), norm_mesh(mesh_target)

        # --- sampling & down-sampling ------------------------------------------------
        def to_pcd(m):
            pcd = o3d.geometry.PointCloud()
            pts, _ = trimesh.sample.sample_surface(m, n_points)
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
            return pcd

        pcd_src = to_pcd(src).voxel_down_sample(voxel_size)
        pcd_tgt = to_pcd(tgt).voxel_down_sample(voxel_size)
        radius_feature = voxel_size*5

        # --- 1. Fast global registration (FPFH-RANSAC) ------------------------------
        pcd_src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_src,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        pcd_tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_tgt,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_src, pcd_tgt, pcd_src_fpfh, pcd_tgt_fpfh,
            mutual_filter=True, max_correspondence_distance=voxel_size*distance_factor,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size*distance_factor)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        trans = result_ransac.transformation

        # --- 2. Multi-scale Generalized ICP refinement ------------------------------
        for stage, it in enumerate(max_iter):
            dist = voxel_size * (2.0 / (stage+1))
            gicp = o3d.pipelines.registration.registration_generalized_icp(
                pcd_src, pcd_tgt, dist, trans,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(robust_kernel),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
            trans = gicp.transformation

        # --- apply transform ---------------------------------------------------------
        mesh_aligned = src.copy()
        mesh_aligned.apply_transform(trans)
        pts_src, _ = trimesh.sample.sample_surface(mesh_source, n_points)
        pts_src_h = np.c_[pts_src, np.ones(len(pts_src))]
        pts_src_aligned = (trans @ pts_src_h.T).T[:, :3]

        return mesh_aligned, pts_src_aligned, trans

