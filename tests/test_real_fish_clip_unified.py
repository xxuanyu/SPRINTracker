import random
import sys

import torch

from configs.util import load_super_config
from data import build_dataset
from models.motip import build
from models.motip.fish_interaction import interaction_recovery_loss
from models.motip.fish_motion import gaussian_motion_nll
from models.motip.persistent_memory import persistent_auxiliary_losses
from utils.misc import yaml_to_dict


def prepare_with_synthetic_detr(annotations, feature_dim):
    from train import prepare_for_motip

    max_objects = max(len(annotation["bbox"]) for annotation in annotations)
    outputs = torch.randn(
        len(annotations), max_objects, feature_dim, requires_grad=True
    )
    boxes = torch.zeros(len(annotations), max_objects, 4)
    indices = []
    for time_idx, annotation in enumerate(annotations):
        count = len(annotation["bbox"])
        boxes[time_idx, :count] = annotation["bbox"]
        index = torch.arange(count)
        indices.append((index, index))
    detr = {
        "pred_logits": torch.zeros(len(annotations), max_objects, 1),
        "pred_boxes": boxes,
        "outputs": outputs,
    }
    return prepare_for_motip(detr, [annotations], indices)


if __name__ == "__main__":
    random.seed(7)
    torch.manual_seed(7)
    config = yaml_to_dict(sys.argv[1])
    config = load_super_config(config, config["SUPER_CONFIG_PATH"])
    config.update({
        "AUG_NUM_GROUPS": 1,
        "AUG_RESIZE_SCALES": [256],
        "AUG_MAX_SIZE": 384,
        "AUG_RANDOM_RESIZE": [256],
        "AUG_RANDOM_CROP_MIN": 224,
        "AUG_RANDOM_CROP_MAX": 256,
    })
    dataset = build_dataset(config)
    _, annotations, _ = dataset[{
        "dataset": "SportsMOT",
        "split": "train",
        "sequence": "seq1",
        "frame_idxs": list(range(8)),
    }]
    model, _ = build(config)
    seq_info = prepare_with_synthetic_detr(annotations, config["FEATURE_DIM"])
    seq_info = model.trajectory_modeling(seq_info)
    logits, _, _ = model.id_decoder(seq_info, use_decoder_checkpoint=False)
    losses = [
        gaussian_motion_nll(seq_info),
        interaction_recovery_loss(seq_info),
        *persistent_auxiliary_losses(seq_info),
    ]
    loss = logits.mean() + sum(losses)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    print("real fish clip unified test: OK", tuple(logits.shape))

