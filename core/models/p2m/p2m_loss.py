import torch
import torch.nn as nn
import torch.nn.functional as F

from chamferdist.chamfer import knn_points

class P2MLoss(nn.Module):
    def __init__(self, options, ellipsoid):
        super().__init__()
        self.options = options

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')

        # We use knn_points instead of ChamferDistance direct
        self.chamfer_dist = None

        # Store laplace indices and edge lists
        # They remain on CPU but will be moved at runtime
        self.laplace_idx = nn.ParameterList([
            nn.Parameter(idx, requires_grad=False)
            for idx in ellipsoid.laplace_idx
        ])
        self.edges = nn.ParameterList([
            nn.Parameter(e, requires_grad=False)
            for e in ellipsoid.edges
        ])

    def edge_regularization(self, pred, edges):
        # Move edges to pred's device
        if edges.device != pred.device:
            edges = edges.to(pred.device)
        return self.l2_loss(
            pred[:, edges[:, 0]],
            pred[:, edges[:, 1]]
        ) * pred.size(-1)

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        # Move lap_idx to inputs device
        if lap_idx.device != inputs.device:
            lap_idx = lap_idx.to(inputs.device)

        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid = indices.clone()
        all_valid[invalid_mask] = 0

        neighs = inputs[:, all_valid]
        neighs[:, invalid_mask] = 0
        nbr_sum = neighs.sum(dim=2)
        nbr_cnt = lap_idx[:, -1].float()

        return inputs - nbr_sum / nbr_cnt[None, :, None]

    def laplace_regularization(self, before, after, level):
        lap1 = self.laplace_coord(before, self.laplace_idx[level])
        lap2 = self.laplace_coord(after,  self.laplace_idx[level])
        lap_loss = self.l2_loss(lap1, lap2) * lap1.size(-1)
        move_loss = (
            self.l2_loss(before, after) * before.size(-1)
            if level > 0 else 0.0
        )
        return lap_loss, move_loss

    def normal_loss(self, gt_normals, idx2, pred_pts, edges):
        # Move edges and idx2 to pred_pts device
        if edges.device != pred_pts.device:
            edges = edges.to(pred_pts.device)
        if idx2.device != pred_pts.device:
            idx2 = idx2.to(pred_pts.device)

        e = F.normalize(
            pred_pts[:, edges[:, 0]] - pred_pts[:, edges[:, 1]],
            dim=2
        )
        nearest = torch.stack([
            normals[i] for normals, i in zip(gt_normals, idx2.long())
        ])
        n = F.normalize(nearest[:, edges[:, 0]], dim=2)
        cos = torch.abs((e * n).sum(dim=2))
        return cos.mean()

    def image_loss(self, gt_img, reconst):
        return F.binary_cross_entropy(reconst, gt_img)

    def forward(self, outputs, targets):
        gt_coord, gt_normals, gt_imgs = (
            targets['points'],
            targets['normals'],
            targets['images']
        )
        pred_list   = outputs['pred_coord']
        before_list = outputs['pred_coord_before_deform']
        lap_const   = [0.2, 1.0, 1.0]

        chamfer_loss = 0.0
        edge_loss    = 0.0
        normal_loss  = 0.0
        lap_loss     = 0.0
        move_loss    = 0.0

        # Reconstruction term
        img_loss = 0.0
        if outputs.get('reconst', None) is not None and self.options.weights.reconst != 0:
            img_loss = self.image_loss(gt_imgs, outputs['reconst'])

        for lvl in range(3):
            P = pred_list[lvl]
           

            # Chamfer via knn_points
            knn1 = knn_points(gt_coord, P, K=5)
            d1 = torch.sqrt(knn1.dists[...,0] + 1e-12)
            knn2 = knn_points(P, gt_coord, K=5)
            d2 = torch.sqrt(knn2.dists[..., 0] + 1e-12)
            idx2 = knn2.idx[...,   0]

            chamfer_loss += (
                self.options.weights.chamfer[lvl] *
                (d1.mean() + self.options.weights.chamfer_opposite * d2.mean())
            )

            normal_loss += self.normal_loss(
                gt_normals, idx2, P, self.edges[lvl]
            )

            edge_loss += self.edge_regularization(P, self.edges[lvl])

            lap, move = self.laplace_regularization(
                before_list[lvl], P, lvl
            )
            lap_loss  += lap_const[lvl] * lap
            move_loss += lap_const[lvl] * move

        loss = (
              chamfer_loss
            + img_loss     * self.options.weights.reconst
            + lap_loss     * self.options.weights.laplace
            + move_loss    * self.options.weights.move
            + edge_loss    * self.options.weights.edge
            + normal_loss  * self.options.weights.normal
        )
        loss = loss * self.options.weights.constant

        return loss, {
            'loss':        loss,
            'loss_chamfer':chamfer_loss,
            'loss_edge':   edge_loss,
            'loss_laplace':lap_loss,
            'loss_move':   move_loss,
            'loss_normal': normal_loss,
        }
