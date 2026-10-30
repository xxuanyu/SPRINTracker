# Evaluation and reported result

The included summary was retrieved from the server evaluation of `fish_unified_19seq_v1_stage2/checkpoint_13.pth`, ID threshold 0.02, SportsMOT-format fish validation split. The validation sequences are seq3, seq7, seq14, and seq18. TrackEval reports 23,964 ground-truth detections and 114 ground-truth identities across these sequences.

```bash
accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=no   submit_and_evaluate.py   --config-path ./configs/fish_unified_19seq_v1_stage2.yaml   --inference-model /absolute/path/to/checkpoint_13.pth   --outputs-dir ./outputs/reproduce_checkpoint13   --inference-group id_0.02 --id-thresh 0.02
```

The command requires the original compatible checkpoint and validation data, neither of which is bundled. Preserve the actual evaluated split and configuration when comparing methods. The raw summary is evidence of a saved run, not a claim that the portable release has been rerun in a clean environment.

HOTA, DetA, and AssA come from HOTA evaluation; MOTA, MOTP, IDSW, and Frag come from CLEAR; IDF1, IDFP, and IDFN come from Identity. Use `IDSW` for the table column named `IDs`. The Count metric named `IDs` counts predicted identities and is a different quantity.

The evaluation wrapper defaults to several thresholds for development. This sweep is model selection, not independent validation of generalization. Report final held-out results only with fixed checkpoints and thresholds. No unverified baseline or ablation results are included in this release.
