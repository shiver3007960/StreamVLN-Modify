# StreamVLN Future Visual Tokens Plan

最近更新：2026-06-09 CST

## 基线协议

所有结果统一用 released StreamVLN checkpoint 的 R2R val_unseen eval 对齐：

| run | ckpt | SR | SPL | OS | NE |
| --- | --- | ---: | ---: | ---: | ---: |
| official | released | 57.86 | 51.30 | 65.25 | 4.78 |

评估路径：

```text
/mnt/inspurfs/evla2_t/lizhen/results/StreamVLN/future_visual_tokens/r2r-val-unseen-eval-alloc6367298-20260609-145325
```

## 已完成方案：Gated Future Fusion

当前已实现的第一版 future baseline：

```text
1. FutureVisualPredictor 使用 196 个 learnable queries。
2. queries cross-attend 历史 memory tokens + 当前 observation tokens，预测 t+4 future visual tokens。
3. 训练监督为同一轨迹、同一视角的 t+4 RGB 帧经过 StreamVLN vision tower / projector / pooling 后的视觉特征。
4. action joint 阶段不把 future tokens 直接拼入 LLM 序列，而是用 FutureToCurrentFusion：
   current_tokens = LN(current_tokens + sigmoid(gate) * cross_attn(current_tokens, predicted_future))
5. 损失为 action CE + 0.1 * future representation loss。
```

训练分两段：

```text
Stage 1: 只训练 future_predictor，future_pretrain_only=True，future_fusion=False。
Stage 2: 从 Stage 1 checkpoint-4000 初始化，训练 future_predictor, future_fusion, mm_mlp_adapter, LLM LoRA。
```

关键 checkpoint：

```text
Stage 1 best:
  /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s1-48g-alloc6367298-ddp-20260608-230023/checkpoint-4000

Stage 2:
  /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618/checkpoint-{3000,4000,4978}
```

最终 R2R val_unseen 结果：

| run | ckpt | SR | SPL | OS | NE | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| official | released | 57.86 | 51.30 | 65.25 | 4.78 | baseline 正常 |
| gated future | checkpoint-3000 | 45.51 | 41.06 | 52.47 | 6.05 | 明显负收益 |
| gated future | checkpoint-4000 | 46.22 | 41.98 | 52.47 | 5.87 | 明显负收益 |
| gated future | checkpoint-4978 | 46.33 | 41.76 | 52.09 | 5.93 | 明显负收益 |

当前判断：

```text
负收益不能直接归因于 future 预测无效。
混杂因素包括：
  1. old trajectory 数据续训是否本身会让官方 ckpt 掉点；
  2. gated fusion 是否污染当前 observation tokens；
  3. LoRA / projector / future module joint 微调是否扰动原 policy。
```

## 下一轮排查目标

目标是控制无关变量，只判断 future 信息是否有用。

优先级：

```text
1. 先整理当前工作区，删除冗余临时文件，保留可复现实验脚本和必要日志指针，然后推送 GitHub。
2. 跑 no-future old-data continuation control。
3. 跑 direct-cat future baseline。
4. 比较 official / no-future continuation / direct-cat future / gated future。
```

## Control A：Official Ckpt + 老数据续训

目的：

```text
验证官方 ckpt 在相同 old trajectory 数据上继续训练 1 epoch 是否本身掉点。
如果 no-future control 也显著掉点，说明当前负收益很可能来自续训数据/协议，而不是 future 设计本身。
```

训练策略：

```text
init: released StreamVLN checkpoint
future: disabled
data: data/trajectory_data/R2R, data/trajectory_data/RxR, data/trajectory_data/EnvDrop
trainable: mm_mlp_adapter + LLM LoRA
frozen: vision_tower, full language model except LoRA
lora: r=8, alpha=16, dropout=0.05, target q_proj,v_proj
epoch: 1
lr: 2e-5
warmup_ratio: 0.075
scheduler: cosine
```

资源计划：

```text
1. 先用 48 卡 allocation 中 1 个 8GPU node smoke，验证训练、保存、恢复、eval loader。
2. smoke 通过后，用 48GPU 跑完整 1 epoch。
```

控制变量要求：

```text
后续 direct-cat future 训练的 action 微调部分应尽量复用本 control 的数据、LR、epoch、LoRA 设置和 total batch。
```

## Control B：Direct-Cat Future Baseline

目的：

```text
替代 gated fusion，避免 predicted future 直接改写当前 observation tokens。
让模型通过 attention 自己决定是否使用 future tokens。
```

架构：

```text
1. 复用 Stage 1 future_predictor checkpoint-4000 初始化。
2. 对每个当前 observation，先保留原 current visual tokens。
3. 在 current visual tokens 后追加 learnable future marker。
4. 再追加 predicted future tokens + learnable future type embedding。
5. 不启用 FutureToCurrentFusion。
```

输入形态：

```text
[current observation visual tokens]
[future_marker_embed]
[predicted future visual tokens + future_type_embed]
```

训练策略：

```text
init: released StreamVLN checkpoint + Stage 1 future_predictor checkpoint-4000
data: same as Control A
trainable: future_predictor, future marker/type embedding, mm_mlp_adapter, LLM LoRA
frozen: vision_tower, full language model except LoRA
lora: r=8, alpha=16, dropout=0.05, target q_proj,v_proj
epoch: 1
lr: 2e-5 for action path; lower LR may be used for reused future_predictor if param groups are implemented
future_loss_weight: 0.1
```

资源与 batch 对齐：

```text
Control A full run: 48GPU。
Direct-cat run: 16GPU。

两者必须尽量保持 total batch 一致。
若 Control A 使用 per_device_train_batch_size=1, grad_accum=2, global batch=96，
则 Direct-cat 16GPU 优先使用 per_device_train_batch_size=1, grad_accum=6, global batch=96。

如果 direct-cat 显存不足，则先 smoke 后再调整，但调整必须同步记录，并说明和 Control A 的差异。
```

验证：

```text
1. 8GPU smoke：训练、保存、恢复、eval loader 均可用。
2. 16GPU full：完成 1 epoch。
3. R2R val_unseen eval 后和 Control A 同表比较。
```

## 后续诊断

只在 Control A / B 结果出来后进入：

```text
1. disable future: 加载 future ckpt，但 eval 时不插入 future，判断是否是 action 微调本身破坏 policy。
2. oracle future: 用真实未来帧视觉 token 替代 predicted future，判断预测质量上限。
3. random/shuffle future: 判断模型是否真的依赖 future token。
4. official Stage2 data: 如果 old-data continuation 掉点，补齐 DAgger/co-training 后再训练。
```

## Artifact 约定

```text
文档:
  experiments/streamvln/future_visual_tokens/

脚本:
  experiments/streamvln/future_visual_tokens/scripts/

小日志:
  experiments/streamvln/future_visual_tokens/logs/<run>/

大 checkpoint / result:
  /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/<run>/
  /mnt/inspurfs/evla2_t/lizhen/results/StreamVLN/future_visual_tokens/<run>/
```

## 回滚开关

所有实现必须支持：

```text
--use_future_tokens false
```

关闭后应走原始 StreamVLN 数据、模型和 eval 路径。
