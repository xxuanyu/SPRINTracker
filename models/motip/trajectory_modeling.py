# Copyright (c) Ruopeng Gao. All Rights Reserved.

"""Unified trajectory modeling for closed-pool fish tracking.

The three cues are intentionally ordered rather than independently voting:
PersistentFishMemory keeps born identities alive, FishMotionEncoder predicts
their plausible motion, and FishInteractionTransformer reasons about the
resulting group geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ffn import FFN
from .fish_motion import FishMotionEncoder
from .persistent_memory import PersistentFishMemory
from .fish_interaction import FishInteractionTransformer


class TrajectoryModeling(nn.Module):
    def __init__(
            self,
            detr_dim: int,
            ffn_dim_ratio: int,
            feature_dim: int,
            motion_dim: int = 96,
            motion_layers: int = 3,
            motion_heads: int = 4,
            motion_max_length: int = 120,
            pool_center=(0.5, 0.5),
            pool_radius: float = 0.5,
            quality_iou_threshold: float = 0.25,
            quality_update_threshold: float = 0.65,
            memory_momentum: float = 0.90,
            interaction_dim: int = 256,
            interaction_layers: int = 2,
            interaction_heads: int = 8,
            interaction_nearest_k: int = 8,
            detach_detr_boxes: bool = True,
    ):
        super().__init__()
        self.detr_dim = detr_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.feature_dim = feature_dim
        self.pool_center = tuple(pool_center)
        self.pool_radius = float(pool_radius)
        self.detach_detr_boxes = bool(detach_detr_boxes)

        # Shared appearance adapter; names remain checkpoint-compatible.
        self.adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

        # 1) Video-long slots and quality-gated identity memory.
        self.persistent_memory = PersistentFishMemory(
            feature_dim=feature_dim,
            quality_iou_threshold=quality_iou_threshold,
            quality_update_threshold=quality_update_threshold,
            memory_momentum=memory_momentum,
        )

        # 2) Nonlinear causal motion and uncertainty.
        self.motion_encoder = FishMotionEncoder(
            embed_dim=motion_dim,
            num_layers=motion_layers,
            num_heads=motion_heads,
            max_length=motion_max_length,
            pool_center=pool_center,
            pool_radius=pool_radius,
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.LayerNorm(32)
        )
        self.scene_encoder = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.LayerNorm(32)
        )
        self.tracklet_fusion = nn.Sequential(
            nn.Linear(feature_dim + motion_dim + 32 + 32, feature_dim),
            nn.GELU(), nn.LayerNorm(feature_dim),
        )
        self.detection_fusion = nn.Sequential(
            nn.Linear(feature_dim + 32 + 32, feature_dim),
            nn.GELU(), nn.LayerNorm(feature_dim),
        )

        # 3) Spatial-motion-interaction graph reasoning.
        self.interaction_encoder = FishInteractionTransformer(
            input_dim=feature_dim,
            interaction_dim=interaction_dim,
            num_layers=interaction_layers,
            num_heads=interaction_heads,
            nearest_k=interaction_nearest_k,
        )
        self.trajectory_fusion = nn.Sequential(
            nn.Linear(feature_dim + interaction_dim, feature_dim),
            nn.GELU(), nn.LayerNorm(feature_dim),
        )
        self.unknown_fusion = nn.Sequential(
            nn.Linear(feature_dim + interaction_dim, feature_dim),
            nn.GELU(), nn.LayerNorm(feature_dim),
        )

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, seq_info):
        raw_trajectory_masks = seq_info["trajectory_masks"]
        raw_trajectory_boxes = seq_info["trajectory_boxes"]
        raw_unknown_boxes = seq_info["unknown_boxes"]

        trajectory_features = seq_info["trajectory_features"]
        trajectory_features = trajectory_features + self.adapter(trajectory_features)
        trajectory_features = self.norm(trajectory_features)
        trajectory_features = trajectory_features + self.ffn(trajectory_features)
        trajectory_features = self.ffn_norm(trajectory_features)

        # Auxiliary geometry cannot inject high-curvature gradients into DETR.
        memory_boxes = (
            raw_trajectory_boxes.detach()
            if self.detach_detr_boxes else raw_trajectory_boxes
        )
        unknown_geometry_boxes = (
            raw_unknown_boxes.detach()
            if self.detach_detr_boxes else raw_unknown_boxes
        )

        memory = self.persistent_memory(
            trajectory_features, memory_boxes, raw_trajectory_masks
        )
        persistent_features = memory["features"]
        persistent_boxes = memory["boxes"]
        persistent_masks = memory["masks"]

        # Decoder visibility and true measurement visibility are different:
        # born slots remain decoder-visible during an occlusion, while motion
        # still treats their propagated boxes as unobserved predictions.
        seq_info["trajectory_observation_masks"] = raw_trajectory_masks
        seq_info["memory_quality_logits"] = memory["quality_logits"]
        seq_info["memory_quality_target"] = memory["quality_target"]
        seq_info["slot_state_logits"] = memory["state_logits"]
        seq_info["slot_state_target"] = memory["state_target"]
        seq_info["long_memory_before"] = memory["long_memory_before"]

        query_length = raw_unknown_boxes.shape[2]
        motion_hist, motion_context, mu, std = self.motion_encoder(
            persistent_boxes,
            raw_trajectory_masks,
            query_length=query_length,
        )
        trajectory_shape = self.shape_encoder(persistent_boxes)
        unknown_shape = self.shape_encoder(unknown_geometry_boxes)
        trajectory_scene = self.scene_encoder(self._scene_geometry(persistent_boxes))
        unknown_scene = self.scene_encoder(self._scene_geometry(unknown_geometry_boxes))
        motion_trajectory_features = self.tracklet_fusion(torch.cat([
            persistent_features, trajectory_shape, motion_hist, trajectory_scene
        ], dim=-1))
        motion_unknown_features = self.detection_fusion(torch.cat([
            seq_info["unknown_features"], unknown_shape, unknown_scene
        ], dim=-1))

        # Uncertainty-aware current-detection-to-persistent-slot compatibility.
        diff = unknown_geometry_boxes[..., :2].unsqueeze(-2) - mu.unsqueeze(-3)
        variance = std[..., None, :, :].square().clamp(min=1.0e-6)
        distance = (diff.square() / variance).sum(dim=-1).clamp(max=80.0)
        history_length = persistent_boxes.shape[2]
        motion_bias = -distance[..., None, :].expand(
            -1, -1, -1, -1, history_length, -1
        )
        b, g, tq, nq, th, ns = motion_bias.shape
        seq_info["motion_attention_bias"] = motion_bias.reshape(
            b * g, tq * nq, th * ns
        )
        seq_info["motion_mu"] = mu
        seq_info["motion_std"] = std
        seq_info["motion_context"] = motion_context

        # Occluded born slots participate using propagated tokens and boxes.
        trajectory_interaction, _ = self.interaction_encoder(
            motion_trajectory_features, persistent_boxes, persistent_masks
        )
        unknown_interaction, unknown_edge_bias = self.interaction_encoder(
            motion_unknown_features,
            unknown_geometry_boxes,
            seq_info["unknown_masks"],
        )
        final_trajectory_features = self.trajectory_fusion(torch.cat([
            motion_trajectory_features, trajectory_interaction
        ], dim=-1))
        final_unknown_features = self.unknown_fusion(torch.cat([
            motion_unknown_features, unknown_interaction
        ], dim=-1))

        seq_info["trajectory_features"] = final_trajectory_features
        seq_info["trajectory_boxes"] = persistent_boxes
        seq_info["trajectory_masks"] = persistent_masks
        seq_info["unknown_features"] = final_unknown_features
        seq_info["trajectory_interaction_features"] = trajectory_interaction
        seq_info["unknown_interaction_features"] = unknown_interaction
        seq_info["interaction_self_bias"] = unknown_edge_bias.reshape(
            -1, unknown_edge_bias.shape[-2], unknown_edge_bias.shape[-1]
        )

        query = F.normalize(unknown_interaction, dim=-1).reshape(
            b * g, tq * nq, -1
        )
        key = F.normalize(trajectory_interaction, dim=-1).reshape(
            b * g, th * ns, -1
        )
        seq_info["interaction_cross_bias"] = torch.matmul(
            query, key.transpose(-1, -2)
        ).clamp(-1.0, 1.0)
        return seq_info

    def _scene_geometry(self, boxes: torch.Tensor) -> torch.Tensor:
        centre = boxes.new_tensor(self.pool_center)
        radial = boxes[..., :2] - centre
        radius_sq = radial.float().square().sum(dim=-1)
        safe_radius = torch.sqrt(radius_sq + 1.0e-6).to(boxes.dtype)
        bearing_sin = radial[..., 1] / safe_radius
        bearing_cos = radial[..., 0] / safe_radius
        d_wall = boxes.new_tensor(self.pool_radius) - safe_radius
        return torch.stack([d_wall, bearing_sin, bearing_cos], dim=-1)
