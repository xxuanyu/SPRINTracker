# FishMOTIP-Unified

这是一个面向封闭圆形水池、多条外观高度相似鱼类的统一 MOTIP 算法。项目不是对三个 tracker 的结果投票，而是在同一条端到端关联数据流中依次融合：

1. **Persistent Fish Slot**：固定物理身份槽、短期动态记忆、质量门控长期身份记忆以及跨遮挡重连接。
2. **Motion-Conditioned Association**：非线性 Fish Motion Transformer、异方差位置分布与 Mahalanobis attention bias。
3. **Interaction-Aware Global ID Decoder**：鱼群局部图、空间/速度/朝向/TTC 关系编码、self-attention bias 与 cross-attention relational bias。

## 统一数据流

```text
Deformable DETR object features / boxes
        |
Shared appearance adapter
        |
Persistent Fish Memory
  |-- visible: quality-gated long-memory update
  |-- occluded: slot token + predicted box propagation
        |
Fish Motion Encoder
  |-- motion/shape/pool-scene feature fusion
  |-- Gaussian next-position prediction -> motion bias
        |
Fish Interaction Transformer
  |-- current-object self relation bias
  |-- history/current relational cross bias
        |
Unified ID Decoder
  temporal PE + motion bias + interaction bias
        |
Persistent Runtime Tracker
```

## 联合目标函数

```text
L = L_det + lambda_id L_id
          + lambda_m L_motion
          + lambda_i L_interaction
          + lambda_q L_memory_quality
          + lambda_s L_slot_state
          + lambda_r L_reconnect
```

所有权重都在 `configs/fish_unified_stage1.yaml` 与 `configs/fish_unified_stage2.yaml` 中显式配置。

## 初始化策略

`tools/merge_unified_checkpoints.py` 会生成严格可加载的 bootstrap checkpoint：

- DETR、共享 ID Decoder、Interaction 模块：Interaction `checkpoint_39.pth`
- Persistent memory：Persistent `checkpoint_39.pth`
- Motion/shape/scene/motion bias：Motion `checkpoint_39.pth`

各子模块保留其完成训练时的参数，不对互不对应的权重做平均。

## 三卡训练

先执行短验证。这个命令只训练 2 个 iteration，用来确认三卡 DDP、数据和反向传播正常：

```bash
cd .
FISH_NUM_PROCESSES=3 bash train_unified.sh smoke
```

完整训练（先 8 epoch Stage 1，再 30 epoch Stage 2）：

```bash
cd .
FISH_NUM_PROCESSES=3 bash train_unified.sh all
```

训练分为：

- Stage 1：8 epoch，冻结 DETR，学习三个已训练子模块之间的统一表示。
- Stage 2：30 epoch，使用两个可训练检测帧进行端到端联合微调。

也可以分开执行：

```bash
FISH_NUM_PROCESSES=3 bash train_unified.sh stage1
FISH_NUM_PROCESSES=3 bash train_unified.sh stage2
```

训练脚本会自动检查并复用 bootstrap 权重；如果某个 Stage 已存在
`checkpoint_*.pth`，会同时恢复模型、优化器、学习率调度器和 epoch 状态继续训练。
完整 Stage 已结束时会自动跳过。只有确实需要从头覆盖训练时才设置
`FISH_FORCE_RESTART=1`。

## 评估

```bash
cd .
FISH_EVAL_PROCESSES=2 bash evaluate_unified.sh latest
```

评估脚本会扫描多个 ID threshold，并保留每个阈值对应的 HOTA、MOTA、IDF1、IDS、AssA 等结果。
`latest` 会自动选择 Stage 2 最新的 `checkpoint_*.pth`。若尚未训练出权重，脚本会在
启动分布式评估之前明确报错并给出训练命令，不再产生多个 rank 的冗长 traceback。

指定某个已存在权重也可以：

```bash
FISH_EVAL_PROCESSES=2 bash evaluate_unified.sh \
  ./outputs/fish_unified_stage2_joint/checkpoint_29.pth
```

## 数值稳定性

- Motion 与 Interaction 几何分支避免 `atan2(0,0)` 和零向量不可导范数。
- 默认阻断复杂几何辅助损失到 DETR box head 的梯度。
- 分布式训练中只要任意 rank 梯度非有限，所有 rank 同步跳过该次更新。
- `tests/test_unified_model.py` 专门覆盖静止鱼、重叠鱼、长遮挡和重连接。
