"""Create a strict-loadable FishMOTIP-Unified bootstrap checkpoint.

Shared DETR/ID parameters and interaction modules come from the completed
Interaction model. Persistent-memory parameters and motion-specific parameters
are overlaid from their respective completed models. No tensor is averaged:
each learned subsystem keeps the checkpoint in which it was actually trained.
"""

import argparse
import gc
import os

import torch

from configs.util import load_super_config
from models.motip import build
from utils.misc import yaml_to_dict


DEFAULT_MOTION = (
    "./pretrained/motion/outputs/"
    "fish_motion_stage2_joint/checkpoint_39.pth"
)
DEFAULT_PERSISTENT = (
    "./pretrained/persistent/outputs/"
    "fish_persistent_stage2_joint/checkpoint_39.pth"
)
DEFAULT_INTERACTION = (
    "./pretrained/interaction/outputs/"
    "fish_interaction_stage2_joint/checkpoint_39.pth"
)
DEFAULT_OUTPUT = (
    "./checkpoints/"
    "fishmotip_unified_bootstrap.pth"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/fish_unified_stage1.yaml")
    parser.add_argument("--motion", default=DEFAULT_MOTION)
    parser.add_argument("--persistent", default=DEFAULT_PERSISTENT)
    parser.add_argument("--interaction", default=DEFAULT_INTERACTION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_config(path):
    config = yaml_to_dict(path)
    return load_super_config(config, config["SUPER_CONFIG_PATH"])


def read_model_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["model"] if "model" in checkpoint else checkpoint


def overlay(merged, source_path, predicate, source_name, covered):
    source = read_model_state(source_path)
    copied = []
    shape_mismatch = []
    for key, target in merged.items():
        if not predicate(key) or key not in source:
            continue
        if source[key].shape != target.shape:
            shape_mismatch.append((key, tuple(source[key].shape), tuple(target.shape)))
            continue
        merged[key] = source[key].detach().clone()
        copied.append(key)
        covered.add(key)
    del source
    gc.collect()
    if shape_mismatch:
        raise RuntimeError(f"{source_name} shape mismatches: {shape_mismatch[:8]}")
    print(f"[{source_name}] copied {len(copied)} tensors from {source_path}")


def main():
    args = parse_args()
    config = read_config(args.config)
    model, _ = build(config)
    merged = model.state_dict()
    covered = set()

    # Interaction is the common backbone because it contains the latest NaN
    # fixes and the global relational decoder trained through epoch 39.
    overlay(merged, args.interaction, lambda _: True, "interaction/base", covered)
    overlay(
        merged,
        args.persistent,
        lambda key: key.startswith("trajectory_modeling.persistent_memory."),
        "persistent-memory",
        covered,
    )
    motion_prefixes = (
        "trajectory_modeling.motion_encoder.",
        "trajectory_modeling.shape_encoder.",
        "trajectory_modeling.scene_encoder.",
        "trajectory_modeling.tracklet_fusion.",
        "trajectory_modeling.detection_fusion.",
        "id_decoder.motion_bias_scales",
    )
    overlay(
        merged,
        args.motion,
        lambda key: key.startswith(motion_prefixes),
        "motion",
        covered,
    )

    uncovered = sorted(set(merged) - covered)
    if uncovered:
        raise RuntimeError(
            "Unified bootstrap contains parameters not initialized from the "
            f"completed branches: {uncovered[:30]}"
        )
    model.load_state_dict(merged, strict=True)

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        "model": merged,
        "states": {"start_epoch": 0, "global_step": 0},
        "source_checkpoints": {
            "motion": args.motion,
            "persistent": args.persistent,
            "interaction": args.interaction,
        },
    }, args.output)
    print(f"[SUCCESS] wrote strict unified checkpoint: {args.output}")
    print(f"[SUCCESS] initialized {len(covered)}/{len(merged)} model tensors")


if __name__ == "__main__":
    main()

