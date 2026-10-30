"""Persistent fish slots with short-term dynamics and gated identity memory."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_iou_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    xy1 = boxes[..., :2] - boxes[..., 2:] / 2
    xy2 = boxes[..., :2] + boxes[..., 2:] / 2
    left_top = torch.maximum(xy1[..., :, None, :], xy1[..., None, :, :])
    right_bottom = torch.minimum(xy2[..., :, None, :], xy2[..., None, :, :])
    inter = (right_bottom - left_top).clamp(min=0).prod(dim=-1)
    area = boxes[..., 2:].clamp(min=0).prod(dim=-1)
    union = area[..., :, None] + area[..., None, :] - inter
    iou = inter / union.clamp(min=1.0e-6)
    eye = torch.eye(iou.shape[-1], dtype=torch.bool, device=iou.device)
    return iou.masked_fill(eye, 0.0)


class PersistentFishMemory(nn.Module):
    """Causal dual memory.

    Long memory is updated only by visible, low-overlap observations selected by
    a learned quality gate.  Missing-but-born slots receive a latent token from
    long memory plus constant-velocity short memory and are never deleted.
    """

    def __init__(self, feature_dim=256, quality_iou_threshold=0.25,
                 quality_update_threshold=0.65, memory_momentum=0.90):
        super().__init__()
        self.feature_dim = feature_dim
        self.quality_iou_threshold = float(quality_iou_threshold)
        self.quality_update_threshold = float(quality_update_threshold)
        self.memory_momentum = float(memory_momentum)
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim + 5, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.visible_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2 + 4, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim)
        )
        self.occluded_fusion = nn.Sequential(
            nn.Linear(feature_dim + 9, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim)
        )
        self.state_head = nn.Linear(feature_dim, 3)
        self.status_embed = nn.Embedding(3, 3)

    def forward(self, features: torch.Tensor, boxes: torch.Tensor, masks: torch.Tensor):
        b, g, t, n, c = features.shape
        observed = ~masks
        overlaps = _pairwise_iou_cxcywh(boxes).max(dim=-1).values
        quality_logits = self.quality_head(torch.cat([features, boxes, overlaps[..., None]], dim=-1)).squeeze(-1)
        quality_target = observed & (overlaps < self.quality_iou_threshold)

        long_memory = features.new_zeros(b, g, n, c)
        last_box = boxes.new_zeros(b, g, n, 4)
        velocity = boxes.new_zeros(b, g, n, 4)
        ever_seen = torch.zeros(b, g, n, dtype=torch.bool, device=features.device)
        output_features, output_boxes, memory_before_update = [], [], []
        state_targets = []

        for time_idx in range(t):
            curr_observed = observed[:, :, time_idx]
            curr_box = boxes[:, :, time_idx]
            curr_feature = features[:, :, time_idx]
            predicted_box = (last_box + velocity).clamp(0.0, 1.0)
            state = torch.where(
                curr_observed,
                torch.zeros_like(curr_observed, dtype=torch.long),
                torch.where(ever_seen, torch.ones_like(curr_observed, dtype=torch.long),
                            torch.full_like(curr_observed, 2, dtype=torch.long)),
            )
            memory_before_update.append(long_memory)

            visible_token = self.visible_fusion(torch.cat(
                [curr_feature, long_memory, curr_box], dim=-1
            ))
            occluded_token = self.occluded_fusion(torch.cat([
                long_memory, predicted_box, velocity[..., :2],
                self.status_embed(state),
            ], dim=-1))
            token = torch.where(curr_observed[..., None], visible_token, occluded_token)
            output_features.append(token)
            output_boxes.append(torch.where(curr_observed[..., None], curr_box, predicted_box))
            state_targets.append(state)

            first_observation = curr_observed & ~ever_seen
            quality = torch.sigmoid(quality_logits[:, :, time_idx])
            if self.training:
                update_strength = quality * quality_target[:, :, time_idx].to(quality.dtype)
            else:
                update_strength = (
                    (quality > self.quality_update_threshold) & quality_target[:, :, time_idx]
                ).to(quality.dtype)
            update_strength = torch.where(first_observation, torch.ones_like(update_strength), update_strength)
            candidate = F.normalize(curr_feature.float(), dim=-1).to(curr_feature.dtype)
            mixed = self.memory_momentum * long_memory + (1.0 - self.memory_momentum) * candidate
            mixed = torch.where(first_observation[..., None], candidate, mixed)
            long_memory = torch.where(
                curr_observed[..., None],
                (1.0 - update_strength[..., None]) * long_memory + update_strength[..., None] * mixed,
                long_memory,
            )

            measured_velocity = curr_box - last_box
            measured_velocity = torch.where(ever_seen[..., None], measured_velocity, torch.zeros_like(measured_velocity))
            velocity = torch.where(curr_observed[..., None], measured_velocity, velocity * 0.85)
            last_box = torch.where(curr_observed[..., None], curr_box, predicted_box)
            ever_seen = ever_seen | curr_observed

        output_features = torch.stack(output_features, dim=2)
        output_boxes = torch.stack(output_boxes, dim=2)
        memory_before_update = torch.stack(memory_before_update, dim=2)
        state_targets = torch.stack(state_targets, dim=2)
        # Born slots remain attention-visible during an occlusion. Only slots
        # that have never appeared in the clip stay padded.
        ever_seen_by_time = torch.cumsum(observed.to(torch.int64), dim=2) > 0
        persistent_masks = masks & ~ever_seen_by_time
        return {
            "features": output_features,
            "boxes": output_boxes,
            "masks": persistent_masks,
            "quality_logits": quality_logits,
            "quality_target": quality_target,
            "state_logits": self.state_head(output_features),
            "state_target": state_targets,
            "long_memory_before": memory_before_update,
        }


def persistent_auxiliary_losses(seq_info: dict):
    quality_logits = seq_info["memory_quality_logits"]
    quality_target = seq_info["memory_quality_target"].to(quality_logits.dtype)
    quality_loss = F.binary_cross_entropy_with_logits(quality_logits, quality_target)

    state_logits = seq_info["slot_state_logits"]
    state_target = seq_info["slot_state_target"]
    state_loss = F.cross_entropy(state_logits.reshape(-1, 3), state_target.reshape(-1))

    reconnect_mask = seq_info.get("reconnect_mask")
    if reconnect_mask is None or not reconnect_mask.any():
        reconnect_loss = state_logits.sum() * 0.0
    else:
        query = F.normalize(seq_info["unknown_features"], dim=-1)
        memory = F.normalize(seq_info["long_memory_before"], dim=-1)
        logits = torch.einsum("bgtnc,bgtmc->bgtnm", query, memory) / 0.07
        n = logits.shape[-1]
        labels = torch.arange(n, device=logits.device)[None, None, None, :].expand_as(reconnect_mask)
        valid = reconnect_mask & ~seq_info["unknown_masks"]
        reconnect_loss = F.cross_entropy(logits[valid], labels[valid]) if valid.any() else logits.sum() * 0.0
    return quality_loss, state_loss, reconnect_loss
