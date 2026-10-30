import random
import sys
import torch
from configs.util import load_super_config
from data import build_dataset
from models.motip import build
from utils.misc import yaml_to_dict


def make_seq_info(annotations, feature_dim=256):
    g, _, n = annotations[0]["trajectory_id_labels"].shape
    t = len(annotations)
    stack = lambda name: torch.stack([a[name][:, 0] for a in annotations], dim=1)[None]
    traj_masks, unk_masks = stack("trajectory_id_masks"), stack("unknown_id_masks")
    traj_boxes = torch.zeros(1, g, t, n, 4); unk_boxes = torch.zeros_like(traj_boxes)
    for time_idx, ann in enumerate(annotations):
        for group in range(g):
            for prefix, boxes, masks in [("trajectory", traj_boxes, traj_masks), ("unknown", unk_boxes, unk_masks)]:
                idx = ann[f"{prefix}_ann_idxs"][group, 0]
                valid = ~masks[0, group, time_idx]
                boxes[0, group, time_idx, valid] = ann["bbox"][idx[valid]]
    return {
        "trajectory_id_labels": stack("trajectory_id_labels"), "trajectory_times": stack("trajectory_times"),
        "trajectory_masks": traj_masks, "trajectory_boxes": traj_boxes,
        "trajectory_features": torch.randn(1, g, t, n, feature_dim, requires_grad=True),
        "unknown_id_labels": stack("unknown_id_labels"), "unknown_times": stack("unknown_times"),
        "unknown_masks": unk_masks, "unknown_boxes": unk_boxes,
        "unknown_features": torch.randn(1, g, t, n, feature_dim, requires_grad=True),
    }


def prepare_with_synthetic_detr(annotations, feature_dim):
    from train import prepare_for_motip
    max_objects = max(len(a["bbox"]) for a in annotations)
    outputs = torch.randn(len(annotations), max_objects, feature_dim, requires_grad=True)
    boxes = torch.zeros(len(annotations), max_objects, 4)
    indices = []
    for time_idx, ann in enumerate(annotations):
        count = len(ann["bbox"]); boxes[time_idx, :count] = ann["bbox"]
        idx = torch.arange(count); indices.append((idx, idx))
    detr = {"pred_logits": torch.zeros(len(annotations), max_objects, 1),
            "pred_boxes": boxes, "outputs": outputs}
    return prepare_for_motip(detr, [annotations], indices)


if __name__ == "__main__":
    random.seed(7); torch.manual_seed(7)
    cfg = yaml_to_dict(sys.argv[1]); cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    cfg.update({"AUG_NUM_GROUPS": 1, "AUG_RESIZE_SCALES": [256], "AUG_MAX_SIZE": 384,
                "AUG_RANDOM_RESIZE": [256], "AUG_RANDOM_CROP_MIN": 224, "AUG_RANDOM_CROP_MAX": 256})
    dataset = build_dataset(cfg)
    _, annotations, _ = dataset[{"dataset": "SportsMOT", "split": "train", "sequence": "seq1",
                                  "frame_idxs": list(range(8))}]
    model, _ = build(cfg); info = prepare_with_synthetic_detr(annotations, cfg["FEATURE_DIM"])
    info = model.trajectory_modeling(info)
    logits, _, _ = model.id_decoder(info, use_decoder_checkpoint=False)
    from models.motip.fish_interaction import interaction_recovery_loss
    loss = logits.mean() + interaction_recovery_loss(info); loss.backward()
    assert torch.isfinite(loss)
    print("real fish clip interaction test: OK", tuple(logits.shape))
