# Fish dataset preparation

The code uses the SportsMOT loader with fish-specific labels. It expects six-digit JPEG frame names and nine comma-separated ground-truth fields.

```text
datasets/
└── SportsMOT/
    ├── train/
    │   └── sequence_name/
    │       ├── img1/000001.jpg
    │       ├── gt/gt.txt
    │       └── seqinfo.ini
    ├── val/
    │   └── sequence_name/...
    ├── train_seqmap.txt
    └── val_seqmap.txt
```

Ground-truth rows follow `frame,id,left,top,width,height,mark,class,visibility`. Frame indexing begins at 1. The fish loader treats loaded annotations as one class and does not implement generic MOT ignore-region semantics; prepare annotations accordingly. The evaluation loader may apply its own preprocessing, so verify the evaluated GT counts.

`seqinfo.ini` must describe the actual image dimensions and sequence length. Sequence maps contain a `name` header followed by sequence names. Keep entire sequences in a single split to prevent frame leakage.

Set `DATA_ROOT` to the parent of `SportsMOT`. Update `PERSISTENT_NUM_SLOTS` for each sequence according to independently known fish counts; the bundled map is specific to the original experiment. `POOL_CENTER` and `POOL_RADIUS` are image-normalized geometry parameters, not metres. Recalibrate them for different crops or cameras. Physical behavior measurements need a separate pixel-to-metre calibration.
