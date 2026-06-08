# StreamVLN Future Visual Tokens Status

最近更新：2026-06-08 CST

## 当前阶段

```text
stage: planning
status: pending user review
```

## Checklist

```text
Stage 0 代码接口确认: pending
Stage 1 Predicted future 模块预训练: pending
Stage 2 Future-aware action joint 微调: pending
Stage 3 Oracle / control 诊断: pending
Stage 4 R2R pilot / medium eval: pending
Stage 5 扩展实验: pending
```

## 当前有效基线

```text
StreamVLN released checkpoint
R2R val_unseen 8GPU eval:
  SR/SPL/OS/NE = 57.86 / 51.30 / 65.25 / 4.78
```

## 下一步

```text
等待用户 review IDEA.md / PLAN.md。
确认后进入 Stage 0，先做 default-off 的接口和 shape smoke。
```

## 已整合脚本

```text
experiments/streamvln/future_visual_tokens/scripts/run_baseline_r2r_single_gpu_smoke.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_baseline_r2r_val_unseen_8gpu_eval.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_baseline_train_2gpu_save_resume_smoke.sbatch
```

## 风险

```text
196 future tokens 显存/序列长度成本更高，smoke 时需要先确认显存。
如果 oracle future 后续也不敏感，说明注入点或 action 消费路径需要重设。
全参训练显存风险高，默认优先 LoRA / projector / future module。
```
