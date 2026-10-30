"""Interaction-aware relational identity encoder for dense fish groups."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_norm(vectors: torch.Tensor, eps: float = 1.0e-6):
    """Differentiable vector norm with a finite derivative at the origin.

    Fish frequently have zero displacement between adjacent frames and every
    interaction graph contains zero-length self edges.  ``atan2(0, 0)`` and
    the exact Euclidean norm at the origin have undefined derivatives, which
    used to make the first joint-training backward pass produce NaN gradients.
    Geometry is evaluated in FP32 so this also remains safe under AMP.
    """
    norm_sq = vectors.float().square().sum(dim=-1)
    return torch.sqrt(norm_sq + eps).to(dtype=vectors.dtype)


def _safe_direction(vectors: torch.Tensor, eps: float = 1.0e-6):
    """Return [cos(theta), sin(theta)] without evaluating atan2."""
    denominator = _safe_norm(vectors, eps=eps).unsqueeze(-1)
    return vectors / denominator


def _kinematics(boxes: torch.Tensor, masks: torch.Tensor):
    xy = boxes[..., :2]
    prev = torch.cat([xy[:, :, :1], xy[:, :, :-1]], dim=2)
    previous_visible = torch.cat(
        [torch.zeros_like(masks[:, :, :1]), ~masks[:, :, :-1]], dim=2
    )
    valid = (~masks) & previous_visible
    velocity = (xy - prev) * valid[..., None]
    heading_direction = _safe_direction(velocity)
    return velocity, heading_direction


def _edge_geometry(boxes: torch.Tensor, masks: torch.Tensor):
    velocity, heading_direction = _kinematics(boxes, masks)
    xy = boxes[..., :2]
    delta = xy[..., None, :, :] - xy[..., :, None, :]
    distance = _safe_norm(delta)
    bearing_direction = _safe_direction(delta)
    delta_velocity = velocity[..., None, :, :] - velocity[..., :, None, :]
    source_heading = heading_direction[..., :, None, :]
    target_heading = heading_direction[..., None, :, :]
    delta_heading_sin = (
        source_heading[..., 0] * target_heading[..., 1]
        - source_heading[..., 1] * target_heading[..., 0]
    )
    delta_heading_cos = (source_heading * target_heading).sum(dim=-1)
    relative_speed_sq = delta_velocity.float().square().sum(dim=-1).clamp(min=1.0e-6)
    relative_speed_sq = relative_speed_sq.to(dtype=delta_velocity.dtype)
    ttc = -(delta * delta_velocity).sum(dim=-1) / relative_speed_sq
    ttc = ttc.clamp(-2.0, 2.0)
    closest = delta + ttc[..., None] * delta_velocity
    dca = _safe_norm(closest).clamp(max=2.0)
    radial = _safe_norm(xy - xy.new_tensor([0.5, 0.5]))
    wall_delta = radial[..., None, :] - radial[..., :, None]
    edge = torch.cat([
        delta,
        distance[..., None],
        bearing_direction[..., 1, None], bearing_direction[..., 0, None],
        delta_velocity,
        delta_heading_sin[..., None], delta_heading_cos[..., None],
        ttc[..., None], dca[..., None], wall_delta[..., None],
    ], dim=-1)
    return edge, distance


class InteractionLayer(nn.Module):
    def __init__(self, dim=256, heads=8, edge_dim=12, nearest_k=8):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("INTERACTION_DIM must be divisible by INTERACTION_HEADS")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.nearest_k = nearest_k
        self.qkv = nn.Linear(dim, dim * 3)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, 64), nn.GELU(), nn.Linear(64, heads)
        )
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, nodes, edges, distance, masks):
        # nodes [B,G,T,N,C], edges [B,G,T,N,N,E]
        b, g, t, n, _ = nodes.shape
        q, k, v = self.qkv(nodes).chunk(3, dim=-1)
        reshape = lambda x: x.reshape(b, g, t, n, self.heads, self.head_dim).permute(0, 1, 2, 4, 3, 5)
        q, k, v = reshape(q), reshape(k), reshape(v)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        learned_edge_bias = self.edge_bias(edges).permute(0, 1, 2, 5, 3, 4)
        scores = scores + learned_edge_bias

        key_invalid = masks[..., None, None, :]
        scores = scores.masked_fill(key_invalid, -1.0e4)
        if self.nearest_k > 0 and n > self.nearest_k:
            candidate_distance = distance.masked_fill(masks[..., None, :], float("inf"))
            nearest = torch.topk(candidate_distance, k=self.nearest_k, largest=False).indices
            local = torch.zeros_like(distance, dtype=torch.bool)
            local.scatter_(-1, nearest, True)
            scores = scores.masked_fill(~local[..., None, :, :], -1.0e4)

        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v).permute(0, 1, 2, 4, 3, 5).reshape(b, g, t, n, self.dim)
        nodes = self.norm1(nodes + self.out(context))
        nodes = self.norm2(nodes + self.ffn(nodes))
        nodes = nodes.masked_fill(masks[..., None], 0.0)
        return nodes, learned_edge_bias.mean(dim=3)


class FishInteractionTransformer(nn.Module):
    def __init__(self, input_dim=256, interaction_dim=256, num_layers=2,
                 num_heads=8, nearest_k=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + 8, interaction_dim)
        self.layers = nn.ModuleList([
            InteractionLayer(interaction_dim, num_heads, edge_dim=12, nearest_k=nearest_k)
            for _ in range(num_layers)
        ])

    def forward(self, features, boxes, masks):
        velocity, heading_direction = _kinematics(boxes, masks)
        radial = boxes[..., :2] - boxes.new_tensor([0.5, 0.5])
        geometry = torch.cat([
            boxes, velocity,
            heading_direction[..., 1, None], heading_direction[..., 0, None],
        ], dim=-1)
        nodes = self.input_proj(torch.cat([features, geometry], dim=-1))
        edges, distance = _edge_geometry(boxes, masks)
        scalar_bias = None
        for layer in self.layers:
            nodes, scalar_bias = layer(nodes, edges, distance, masks)
        return nodes, scalar_bias


def interaction_recovery_loss(seq_info: dict):
    unknown = F.normalize(seq_info["unknown_interaction_features"], dim=-1)
    trajectory = F.normalize(seq_info["trajectory_interaction_features"], dim=-1)
    t = min(unknown.shape[2], trajectory.shape[2])
    if t < 2 or unknown.shape[-2] != trajectory.shape[-2]:
        return unknown.sum() * 0.0
    query = unknown[:, :, 1:t]
    previous = trajectory[:, :, :t - 1]
    logits = torch.einsum("bgtnc,bgtmc->bgtnm", query, previous) / 0.07
    previous_masks = seq_info["trajectory_masks"][:, :, :t - 1]
    logits = logits.masked_fill(previous_masks[..., None, :], -1.0e4)
    valid = (~seq_info["unknown_masks"][:, :, 1:t]) & (~previous_masks)
    n = logits.shape[-1]
    labels = torch.arange(n, device=logits.device)[None, None, None, :].expand_as(valid)
    return F.cross_entropy(logits[valid], labels[valid]) if valid.any() else logits.sum() * 0.0
