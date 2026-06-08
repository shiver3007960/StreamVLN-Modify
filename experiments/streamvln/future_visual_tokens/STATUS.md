# StreamVLN Future Visual Tokens Status

最近更新：2026-06-08 CST

## 当前阶段

```text
stage: planning
status: implementation drafted
```

## Checklist

```text
Stage 0 代码接口确认: done
Stage 1 Predicted future 模块预训练: drafted
Stage 2 Future-aware action joint 微调: drafted
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
先做语法和小 batch shape smoke。
通过后提交代码，再跑 2GPU smoke 验证训练/保存/resume。
```

## 已整合脚本

```text
experiments/streamvln/future_visual_tokens/scripts/run_baseline_r2r_single_gpu_smoke.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_baseline_r2r_val_unseen_8gpu_eval.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_baseline_train_2gpu_save_resume_smoke.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_future_stage1_predictor_pretrain_8gpu.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_future_stage2_joint_8gpu.sbatch
```

## 已草拟代码改动

```text
streamvln/dataset/vln_action_dataset.py:
  为每个当前采样帧读取 t+4 future_images 和 future_valid mask。

streamvln/model/stream_video_vln.py:
  增加 196-token FutureVisualPredictor。
  增加 future-to-current fusion。
  支持 future_pretrain_only 和 joint loss。

streamvln/args.py / streamvln_train.py:
  增加 future 相关参数，并支持 mm_tunable_parts 解冻 future 模块。
```

## 风险

```text
196 future tokens 显存/序列长度成本更高，smoke 时需要先确认显存。
如果 oracle future 后续也不敏感，说明注入点或 action 消费路径需要重设。
全参训练显存风险高，默认优先 LoRA / projector / future module。
```
