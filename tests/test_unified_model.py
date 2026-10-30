import torch

from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder
from models.motip.fish_motion import gaussian_motion_nll
from models.motip.fish_interaction import interaction_recovery_loss
from models.motip.persistent_memory import persistent_auxiliary_losses


def test_all_three_branches_forward_and_backward():
    torch.manual_seed(13)
    b, g, t, n, c = 1, 1, 6, 5, 256

    # Stationary and overlapping centres exercise every formerly undefined
    # zero-vector geometry path. Slot 2 is hidden and later reconnected.
    boxes = torch.full((b, g, t, n, 4), 0.5)
    boxes[..., 2:] = 0.05
    trajectory_masks = torch.zeros(b, g, t, n, dtype=torch.bool)
    trajectory_masks[:, :, 2:5, 2] = True
    unknown_masks = torch.zeros_like(trajectory_masks)
    reconnect_mask = torch.zeros_like(trajectory_masks)
    reconnect_mask[:, :, 5, 2] = True

    trajectory_features = torch.randn(b, g, t, n, c, requires_grad=True)
    unknown_features = torch.randn(b, g, t, n, c, requires_grad=True)
    labels = torch.arange(n)[None, None, None].expand(b, g, t, n)
    times = torch.arange(t)[None, None, :, None].expand(b, g, t, n)
    seq_info = {
        "trajectory_features": trajectory_features,
        "trajectory_boxes": boxes.clone(),
        "trajectory_masks": trajectory_masks,
        "trajectory_id_labels": labels,
        "trajectory_times": times,
        "unknown_features": unknown_features,
        "unknown_boxes": boxes.clone(),
        "unknown_gt_boxes": boxes.clone(),
        "unknown_masks": unknown_masks,
        "unknown_id_labels": labels,
        "unknown_times": times,
        "reconnect_mask": reconnect_mask,
    }

    modeling = TrajectoryModeling(
        c, 2, c,
        motion_max_length=t,
        interaction_nearest_k=4,
        detach_detr_boxes=True,
    )
    decoder = IDDecoder(
        c, 256, 2, 2, 32, 32, t, False, True,
        use_interaction_bias=True,
        use_motion_bias=True,
    )
    seq_info = modeling(seq_info)
    logits, _, _ = decoder(seq_info, use_decoder_checkpoint=False)

    motion_loss = gaussian_motion_nll(seq_info)
    interaction_loss = interaction_recovery_loss(seq_info)
    memory_losses = persistent_auxiliary_losses(seq_info)
    total = logits.mean() + motion_loss + interaction_loss + sum(memory_losses)
    total.backward()

    assert torch.isfinite(total)
    assert logits.shape == (b, g, t, n, 33)
    assert not seq_info["trajectory_masks"][:, :, 2:5, 2].any()
    assert seq_info["slot_state_target"][:, :, 2:5, 2].eq(1).all()
    assert seq_info["motion_attention_bias"].shape == (b * g, t * n, t * n)
    assert seq_info["interaction_cross_bias"].shape == (b * g, t * n, t * n)
    assert seq_info["interaction_self_bias"].shape == (b * g * t, n, n)

    for feature in (trajectory_features, unknown_features):
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
    for parameter in list(modeling.parameters()) + list(decoder.parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_runtime_single_frame_query():
    b, g, history, n, c = 1, 1, 6, 5, 256
    trajectory_boxes = torch.rand(b, g, history, n, 4)
    trajectory_boxes[..., 2:] = trajectory_boxes[..., 2:] * 0.08 + 0.02
    trajectory_masks = torch.zeros(b, g, history, n, dtype=torch.bool)
    trajectory_masks[:, :, 4:, 2] = True
    unknown_boxes = torch.rand(b, g, 1, n, 4)
    unknown_boxes[..., 2:] = unknown_boxes[..., 2:] * 0.08 + 0.02
    unknown_masks = torch.zeros(b, g, 1, n, dtype=torch.bool)
    seq_info = {
        "trajectory_features": torch.randn(b, g, history, n, c),
        "trajectory_boxes": trajectory_boxes,
        "trajectory_masks": trajectory_masks,
        "trajectory_id_labels": torch.arange(n)[None, None, None].expand(
            b, g, history, n
        ),
        "trajectory_times": torch.arange(history)[None, None, :, None].expand(
            b, g, history, n
        ),
        "unknown_features": torch.randn(b, g, 1, n, c),
        "unknown_boxes": unknown_boxes,
        "unknown_masks": unknown_masks,
        "unknown_times": torch.full((b, g, 1, n), history, dtype=torch.long),
    }
    modeling = TrajectoryModeling(c, 2, c, motion_max_length=history)
    decoder = IDDecoder(
        c, 256, 2, 2, 32, 32, history + 1, False, True,
        use_interaction_bias=True,
        use_motion_bias=True,
    )
    modeling.eval()
    decoder.eval()
    with torch.no_grad():
        seq_info = modeling(seq_info)
        logits, labels, masks = decoder(seq_info, use_decoder_checkpoint=False)
    assert labels is None
    assert logits.shape == (b, g, 1, n, 33)
    assert masks.shape == (b, g, 1, n)
    assert seq_info["motion_attention_bias"].shape == (
        b * g, n, history * n
    )
    assert seq_info["interaction_cross_bias"].shape == (
        b * g, n, history * n
    )
    assert torch.isfinite(logits).all()


if __name__ == "__main__":
    test_all_three_branches_forward_and_backward()
    test_runtime_single_frame_query()
    print("unified Motion + Persistent + Interaction test: OK")
