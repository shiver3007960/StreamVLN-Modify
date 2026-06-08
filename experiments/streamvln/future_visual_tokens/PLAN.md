# StreamVLN Future Visual Tokens Plan

最近更新：2026-06-08 CST

## Baseline

已复现的 official StreamVLN checkpoint：

```text
R2R val_unseen 8GPU eval:
  SR/SPL/OS/NE = 57.86 / 51.30 / 65.25 / 4.78
  result job: 6318610
  protocol: released checkpoint + current StreamVLN eval
```

所有新结果必须用同 split、同 checkpoint/eval 协议比较。

## Stage 0: 代码接口确认

目标：

```text
确认 future 插入点、训练样本对齐、token budget。
```

动作：

```text
1. 在 dataset 中定位每个训练 round 的当前帧 t 和未来帧 t+k。
2. 在 model 中定位 encode_rgbd 后的 image_features / memory_features。
3. 明确 future module default-off，不影响原始 eval。
```

验证：

```text
无 future flag 时，单卡 smoke eval 输出和当前 baseline 路径一致。
```

## Stage 1: Predicted Future 模块预训练

目标：

```text
先让可学习 future 模块具备预测未来视觉 token 的能力，避免一开始把随机 future token 注入 action 链路。
```

动作：

```text
1. 加入 QFormer-style future predictor：
   learnable future queries cross-attend 当前/历史视觉 token。
2. 输出 196 个 predicted future tokens，和一张图像的 14x14 token 数一致。
3. target 为未来帧经过 StreamVLN vision tower/projector/pooling 后的 196 tokens，target detach。
4. 冻结 StreamVLN 主体，优先只训练 future predictor / target loss。
```

验证：

```text
1. 2GPU smoke 能训练、保存 ckpt、resume。
2. future loss 正常下降，cosine / MSE 不发散。
3. 关闭 future flag 时原始 StreamVLN 路径不变。
```

Review gate：

```text
如果 future predictor 不能稳定学习未来视觉特征，先调整 horizon / predictor / loss，不进入 joint 微调。
```

## Stage 2: Future-Aware Action Joint 微调

目标：

```text
让 predicted future tokens 进入 action 决策链路，并和导航动作 loss 一起训练。
```

动作：

```text
1. 复用 Stage 1 的 future predictor 初始化。
2. 用 future-to-current cross-attn/gate 将 196 个 future tokens 融入当前 image tokens。
3. loss = action CE + lambda_future * future representation loss。
4. 训练 future predictor / fusion / LoRA / mm_projector，避免全参训练 OOM。
```

建议初始实现：

```text
future token count: 196
horizon: t+4 优先
fusion: current_tokens <- cross-attn(future_tokens) + gate
trainable: future predictor / fusion / LoRA / mm_projector，避免全参 2GPU OOM
```

验证：

```text
1. 2GPU smoke：能训练、保存 ckpt、resume。
2. future loss 正常下降，action loss 不 NaN。
3. future dropout/shuffle 对 loss 或 eval 有影响。
```

说明：

```text
相比直接 joint 训练，我更建议 Stage 1 + Stage 2 两段式：
  随机 future token 直接进 action 链路容易被模型忽略或扰乱动作学习；
  先训 future predictor 可以先验证监督信号质量；
  joint 阶段再验证 future 是否真正进入决策。
```

## Stage 3: Oracle / Control 诊断

目标：

```text
判断效果瓶颈来自 future 预测质量，还是来自注入/消费路径。
```

动作：

```text
1. Oracle future: 用真实未来帧的 196 tokens 替换 predicted future tokens。
2. Zero future: 将 future tokens 置零。
3. Shuffle future: batch 内打乱 future tokens。
4. Standalone future tokens vs fused current tokens 对比。
```

Review gate：

```text
如果 oracle future 有收益而 predicted future 没收益，优先改 predictor。
如果 oracle future 也无收益，优先改注入点或动作消费路径。
```

## Stage 4: R2R Pilot / Medium Eval

目标：

```text
判断 predicted future 是否值得 official-scale 训练。
```

动作：

```text
1. 先跑 pilot，不用 train loss 直接下结论。
2. 对关键 checkpoint 做 8GPU R2R val_unseen eval。
3. 同时保留 no-future continuation control。
```

成功标准：

```text
SR/SPL 接近或超过 official baseline；
future controls 证明模型实际依赖 future；
没有明显增加不可接受的显存/吞吐成本。
```

## Stage 5: 扩展实验

只在 Stage 2/3/4 有正信号后进入：

```text
1. token budget: 196 vs 49/98。
2. horizon: t+4 vs t+8。
3. injection: standalone future tokens vs fused current tokens。
4. target: current view future vs lookdown/local-geometry future。
```

## Artifact 约定

```text
文档:
  experiments/streamvln/future_visual_tokens/

可复用脚本:
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
