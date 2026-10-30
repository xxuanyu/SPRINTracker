import torch

from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder
from models.motip.fish_interaction import interaction_recovery_loss


def test_interaction_biases_decoder_and_backpropagates():
    b, g, t, n, c = 1, 2, 6, 7, 256
    # Deliberately include stationary fish, overlapping fish and graph self
    # edges.  These all produced atan2(0, 0) in the original implementation:
    # the forward loss was finite but its gradients were NaN.
    boxes = torch.full((b, g, t, n, 4), 0.5)
    boxes[..., 2:] = 0.05
    masks = torch.zeros(b, g, t, n, dtype=torch.bool)
    trajectory_features = torch.randn(b, g, t, n, c, requires_grad=True)
    unknown_features = torch.randn(b, g, t, n, c, requires_grad=True)
    trajectory_boxes = boxes.clone().requires_grad_()
    unknown_boxes = boxes.clone().requires_grad_()
    info = {
        "trajectory_features": trajectory_features,
        "trajectory_boxes": trajectory_boxes,
        "trajectory_masks": masks,
        "trajectory_id_labels": torch.arange(n)[None, None, None].expand(b, g, t, n),
        "trajectory_times": torch.arange(t)[None, None, :, None].expand(b, g, t, n),
        "unknown_features": unknown_features,
        "unknown_boxes": unknown_boxes,
        "unknown_masks": masks.clone(),
        "unknown_id_labels": torch.arange(n)[None, None, None].expand(b, g, t, n),
        "unknown_times": torch.arange(t)[None, None, :, None].expand(b, g, t, n),
        "unknown_gt_boxes": boxes.clone(),
        "reconnect_mask": torch.zeros_like(masks),
    }
    modeling = TrajectoryModeling(c, 2, c, interaction_nearest_k=4)
    decoder = IDDecoder(c, 256, 2, 2, 32, 32, t, False, True, True)
    info = modeling(info)
    logits, _, _ = decoder(info, use_decoder_checkpoint=False)
    loss = logits.mean() + interaction_recovery_loss(info)
    loss.backward()
    assert info["interaction_cross_bias"].shape == (b * g, t * n, t * n)
    assert info["interaction_self_bias"].shape == (b * g * t, n, n)
    assert torch.isfinite(loss)
    for tensor in (trajectory_features, unknown_features):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    # Unified mode intentionally stops geometric auxiliary gradients before
    # the DETR box head; DETR boxes are trained by the normal detection loss.
    assert trajectory_boxes.grad is None
    assert unknown_boxes.grad is None
    for parameter in list(modeling.parameters()) + list(decoder.parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


if __name__ == "__main__":
    test_interaction_biases_decoder_and_backpropagates()
    print("interaction smoke test: OK")
