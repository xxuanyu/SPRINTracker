"""Learnable nonlinear motion model for top-view fish tracking.

The module deliberately does not contain a Kalman filter.  A causal temporal
Transformer encodes fish kinematics and predicts a heteroscedastic Gaussian
for the next centre.  The covariance becomes an additive association bias in
the MOTIP ID decoder.
"""

import math

import torch
import torch.nn as nn


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def boxes_to_fish_state(
        boxes: torch.Tensor,
        masks: torch.Tensor,
        pool_center=(0.5, 0.5),
        pool_radius=0.5,
) -> torch.Tensor:
    """Convert normalized cxcywh histories to 16-D fish states.

    State = [x,y,vx,vy,ax,ay,sin(theta),cos(theta),omega,length,
             width,v_radial,v_tangent,d_wall,speed,observed].
    Derivatives are valid only across two visible consecutive observations;
    this prevents an occlusion gap from being interpreted as an extreme jump.
    """
    observed = ~masks
    xy = boxes[..., :2]
    wh = boxes[..., 2:].clamp(min=0.0)

    prev_xy = torch.cat([xy[:, :, :1], xy[:, :, :-1]], dim=2)
    pair_valid = observed & torch.cat(
        [torch.zeros_like(observed[:, :, :1]), observed[:, :, :-1]], dim=2
    )
    velocity = (xy - prev_xy) * pair_valid[..., None]

    eps = 1.0e-6
    prev_velocity = torch.cat([velocity[:, :, :1], velocity[:, :, :-1]], dim=2)
    acceleration = (velocity - prev_velocity) * pair_valid[..., None]

    # Never differentiate atan2 at (0, 0). First frames, masked frames and
    # temporarily stationary fish all produce zero velocity, and atan2(0, 0)
    # has an undefined backward derivative. That derivative used to poison the
    # DETR box head immediately after it was unfrozen in stage 2.
    speed_sq = velocity.square().sum(dim=-1)
    moving = pair_valid & (speed_sq > eps * eps)
    safe_speed = torch.sqrt(speed_sq.clamp(min=eps * eps))
    heading_sin = torch.where(moving, velocity[..., 1] / safe_speed, torch.zeros_like(safe_speed))
    heading_cos = torch.where(moving, velocity[..., 0] / safe_speed, torch.zeros_like(safe_speed))

    prev_speed_sq = prev_velocity.square().sum(dim=-1)
    prev_moving = torch.cat([torch.zeros_like(moving[:, :, :1]), moving[:, :, :-1]], dim=2)
    turn_valid = moving & prev_moving & (prev_speed_sq > eps * eps)
    turn_cross = prev_velocity[..., 0] * velocity[..., 1] - prev_velocity[..., 1] * velocity[..., 0]
    turn_dot = (prev_velocity * velocity).sum(dim=-1)
    safe_cross = torch.where(turn_valid, turn_cross, torch.zeros_like(turn_cross))
    safe_dot = torch.where(turn_valid, turn_dot, torch.ones_like(turn_dot))
    omega = torch.atan2(safe_cross, safe_dot)

    centre = boxes.new_tensor(pool_center)
    radial_vec = xy - centre
    radius_sq = radial_vec.square().sum(dim=-1)
    away_from_center = radius_sq > eps * eps
    safe_radius = torch.sqrt(radius_sq.clamp(min=eps * eps))
    radius = torch.where(away_from_center, safe_radius, torch.zeros_like(safe_radius))
    radial_unit = torch.where(
        away_from_center[..., None], radial_vec / safe_radius[..., None], torch.zeros_like(radial_vec)
    )
    tangent_unit = torch.stack([-radial_unit[..., 1], radial_unit[..., 0]], dim=-1)
    v_radial = (velocity * radial_unit).sum(dim=-1)
    v_tangent = (velocity * tangent_unit).sum(dim=-1)
    d_wall = boxes.new_tensor(float(pool_radius)) - radius
    speed = torch.where(moving, safe_speed, torch.zeros_like(safe_speed))

    length = torch.maximum(wh[..., 0], wh[..., 1])
    width = torch.minimum(wh[..., 0], wh[..., 1])
    state = torch.cat([
        xy,
        velocity,
        acceleration,
        heading_sin[..., None],
        heading_cos[..., None],
        omega[..., None],
        length[..., None],
        width[..., None],
        v_radial[..., None],
        v_tangent[..., None],
        d_wall[..., None],
        speed[..., None],
        observed.to(boxes.dtype)[..., None],
    ], dim=-1)
    return state * observed[..., None].to(state.dtype)


class FishMotionEncoder(nn.Module):
    """Causal motion Transformer with Gaussian next-position prediction."""

    def __init__(self, embed_dim=96, num_layers=3, num_heads=4, max_length=120,
                 pool_center=(0.5, 0.5), pool_radius=0.5, min_std=0.005):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.pool_center = tuple(pool_center)
        self.pool_radius = float(pool_radius)
        self.min_std = float(min_std)
        self.state_embed = nn.Sequential(
            nn.Linear(16, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)
        )
        self.time_embed = nn.Parameter(torch.zeros(max_length, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(embed_dim))
        self.start_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.gaussian_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 4)
        )
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.normal_(self.start_token, std=0.02)

    def forward(self, boxes: torch.Tensor, masks: torch.Tensor, query_length: int):
        b, g, t, n, _ = boxes.shape
        if t > self.max_length:
            raise ValueError(f"Motion history {t} exceeds MOTION_MAX_LENGTH={self.max_length}")
        states = boxes_to_fish_state(
            boxes, masks, pool_center=self.pool_center, pool_radius=self.pool_radius
        )
        tokens = self.state_embed(states)
        tokens = tokens + self.time_embed[:t][None, None, :, None, :]
        tokens = tokens.permute(0, 1, 3, 2, 4).reshape(b * g * n, t, self.embed_dim)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=boxes.device), diagonal=1)
        encoded = self.encoder(tokens, mask=causal)
        encoded = encoded.reshape(b, g, n, t, self.embed_dim).permute(0, 1, 3, 2, 4)

        # During training query and history have the same time axis.  Prediction
        # at t uses only 0..t-1.  At runtime a single current frame uses the last
        # history context to predict the next centre.
        history_anchors = self._causal_last_observed_centres(boxes, masks)
        if query_length == t:
            start = self.start_token.expand(b * g * n, 1, -1)
            shifted = torch.cat([start, tokens.new_zeros(b * g * n, max(t - 1, 0), self.embed_dim)], dim=1)
            if t > 1:
                shifted[:, 1:] = encoded[:, :, :-1].permute(0, 1, 3, 2, 4).reshape(b * g * n, t - 1, self.embed_dim)
            context = shifted.reshape(b, g, n, t, self.embed_dim).permute(0, 1, 3, 2, 4)
        elif query_length == 1:
            context = encoded[:, :, -1:, :, :]
        else:
            context = encoded[:, :, -1:, :, :].expand(-1, -1, query_length, -1, -1)

        gaussian = self.gaussian_head(context)
        delta = 0.25 * torch.tanh(gaussian[..., :2])
        # Anchor predictions on the last observed/current slot centre.
        if query_length == t:
            prev_xy = history_anchors
        else:
            observed = ~masks
            last = boxes.new_full((b, g, n, 2), 0.5)
            for time_idx in range(t):
                last = torch.where(observed[:, :, time_idx, :, None], boxes[:, :, time_idx, :, :2], last)
            prev_xy = last[:, :, None].expand(-1, -1, query_length, -1, -1)
        mu = (prev_xy + delta).clamp(0.0, 1.0)
        std = torch.nn.functional.softplus(gaussian[..., 2:]) + self.min_std
        return encoded, context, mu, std

    @staticmethod
    def _causal_last_observed_centres(boxes: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Centre before each query time, with no access to the current frame."""
        b, g, t, n, _ = boxes.shape
        last = boxes.new_full((b, g, n, 2), 0.5)
        anchors = []
        for time_idx in range(t):
            anchors.append(last)
            last = torch.where(
                (~masks[:, :, time_idx, :])[..., None],
                boxes[:, :, time_idx, :, :2],
                last,
            )
        return torch.stack(anchors, dim=2)


def gaussian_motion_nll(seq_info: dict) -> torch.Tensor:
    """Masked diagonal-Gaussian NLL in normalized image coordinates."""
    mu = seq_info["motion_mu"]
    std = seq_info["motion_std"]
    target = seq_info.get("unknown_gt_boxes", seq_info["unknown_boxes"])[..., :2]
    valid = ~seq_info["unknown_masks"]
    # Persistent slots stay decoder-visible while occluded.  Motion validity,
    # however, must still use the raw observation mask rather than treating a
    # propagated slot box as a new measurement.
    trajectory_observation_masks = seq_info.get(
        "trajectory_observation_masks", seq_info["trajectory_masks"]
    )
    observed = ~trajectory_observation_masks
    previously_seen = torch.cumsum(observed.to(torch.int64), dim=2) > 0
    previously_seen = torch.cat(
        [torch.zeros_like(previously_seen[:, :, :1]), previously_seen[:, :, :-1]], dim=2
    )
    if previously_seen.shape == valid.shape:
        valid = valid & previously_seen
    if mu.shape[-2] != target.shape[-2]:
        return mu.sum() * 0.0
    var = std.square()
    nll = 0.5 * (((target - mu).square() / var) + 2.0 * torch.log(std) + math.log(2.0 * math.pi))
    nll = nll.sum(dim=-1)
    return nll[valid].mean() if valid.any() else nll.sum() * 0.0
