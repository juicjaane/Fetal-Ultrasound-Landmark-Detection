import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_argmax_2d(heatmaps, beta=100.0):
    """
    Differentiable coordinate extraction using soft-argmax.
    heatmaps: (B, C, H, W)
    returns coords: (B, C, 2) in (x, y) format
    """
    B, C, H, W = heatmaps.shape

    heatmaps = heatmaps.view(B, C, -1)
    probs = F.softmax(heatmaps * beta, dim=-1)

    xs = torch.linspace(0, W - 1, W, device=heatmaps.device)
    ys = torch.linspace(0, H - 1, H, device=heatmaps.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    coords = coords.to(heatmaps.device)

    exp_coords = (probs.unsqueeze(-1) * coords).sum(dim=2)
    return exp_coords


class WeightedHeatmapMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        """
        pred, target: (B, C, H, W)
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=(2, 3))  # (B, C)
        loss = loss * self.weights
        return loss.mean()


def angle_between(v1, v2, eps=1e-4):
    """
    Compute angle between two vectors.
    v1, v2: (B, 2)
    returns:
        angles: (B,) degrees
        valid_mask: (B,)
    """
    norm1 = torch.norm(v1, dim=1)
    norm2 = torch.norm(v2, dim=1)

    valid = (norm1 > eps) & (norm2 > eps)
    angles = torch.zeros(v1.shape[0], device=v1.device)

    if valid.any():
        v1n = v1[valid]
        v2n = v2[valid]

        dot = (v1n * v2n).sum(dim=1)
        denom = torch.norm(v1n, dim=1) * torch.norm(v2n, dim=1)
        cos = dot / denom
        cos = torch.clamp(cos, -1.0, 1.0)

        theta = torch.acos(cos) * 180.0 / math.pi
        angles[valid] = theta

    return angles, valid


def angle_consistency_loss(coords, target_angle=90.0):
    """
    Penalize deviation from target angle (orthogonality).
    coords: (B, 4, 2) -> [BPD1, BPD2, OFD1, OFD2]
    """
    bpd_vec = coords[:, 1] - coords[:, 0]
    ofd_vec = coords[:, 3] - coords[:, 2]

    angles, valid = angle_between(bpd_vec, ofd_vec)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=coords.device)

    return ((angles[valid] - target_angle) ** 2).mean()


class HeatmapWithAngleLoss(nn.Module):
    def __init__(self, heatmap_loss, lambda_angle=0.01, warmup_epochs=10):
        super().__init__()
        self.heatmap_loss = heatmap_loss
        self.lambda_angle = lambda_angle
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, pred_hm, gt_hm):
        loss_hm = self.heatmap_loss(pred_hm, gt_hm)

        if self.current_epoch < self.warmup_epochs:
            return loss_hm, {
                "heatmap": loss_hm.detach(),
                "angle": torch.tensor(0.0, device=pred_hm.device)
            }

        coords = soft_argmax_2d(pred_hm)
        loss_angle = angle_consistency_loss(coords)

        total = loss_hm + self.lambda_angle * loss_angle

        return total, {
            "heatmap": loss_hm.detach(),
            "angle": loss_angle.detach()
        }
