# StreamVLN Future Visual Tokens Status

最近更新：2026-06-09 CST

## 当前阶段

```text
stage: Stage 2 Future-aware action joint 微调
status: 48GPU full run active, step 1000 checkpoint/eval validated
```

## Checklist

```text
Stage 0 代码接口确认: done
Stage 1 Predicted future 模块预训练: done
Stage 2 Future-aware action joint 微调: smoke passed
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
1. 继续监控 full run 到完成。
2. 后续用 saved checkpoints 做 R2R/RxR val_unseen VLN eval。
```

## Stage 2 配置

```text
init checkpoint: /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s1-48g-alloc6367298-ddp-20260608-230023/checkpoint-4000
trainable: future_predictor, future_fusion, mm_mlp_adapter, mm_lora_layer
frozen: vision_tower, full language model except LoRA
data: data/trajectory_data/R2R, data/trajectory_data/RxR, data/trajectory_data/EnvDrop
lr: 2e-5
future_loss_weight: 0.1
48GPU batch: per_device_train_batch_size 1, gradient_accumulation_steps 2, global batch 96
gradient_checkpointing: false
reference script: scripts/streamvln_train_slurm.sh
```

## 最新 smoke

```text
run: streamvln-future-s2-smoke-6371130
scale: smoke, 8GPU, 20 update steps
status: completed
artifact: /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s2-smoke-6371130
eval:
  step 10 eval_loss: 0.3387698233127594
  step 20 eval_loss: 0.33962497115135193
validated:
  train/backward completed with gradient_checkpointing=false
  checkpoint-10 and checkpoint-20 saved optimizer/scheduler/rng
  checkpoint-10 and checkpoint-20 saved adapter_model.safetensors
  checkpoint-10 and checkpoint-20 saved non_lora_trainables.bin with future_predictor, future_fusion, mm_projector
fixes from failed smoke:
  LoRA target_modules now respects lora_target_modules, using q_proj,v_proj.
  local checkpoint tokenizer loading has a fallback branch.
```

## 当前 Stage 2 Full Run

```text
run: streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618
scale: full Stage 2 joint finetune, 48GPU, 1 epoch
status: running
job: allocation 6367298, eailab_system, 6 nodes / 48 GPUs
artifact: /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618
logs:
  experiments/streamvln/future_visual_tokens/logs/streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618/train.out
  experiments/streamvln/future_visual_tokens/logs/streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618/train.err
validated:
  step 500 eval_loss: 0.2923053205013275
  step 1000 eval_loss: 0.2814578711986542
  checkpoint-1000 saved adapter_model.safetensors, optimizer.pt, scheduler.pt, trainer_state.json, 48 rng_state files
  checkpoint-1000 non_lora_trainables.bin contains future_predictor, future_fusion, mm_projector
```

## 当前运行

```text
run: streamvln-future-s1-48g-alloc6367298-ddp-20260608-230023
scale: full Stage 1 predictor pretrain
status: completed 1 epoch
job: allocation 6367298, eailab_system, 6 nodes / 48 GPUs
artifact: /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s1-48g-alloc6367298-ddp-20260608-230023
logs:
  experiments/streamvln/future_visual_tokens/logs/streamvln-future-s1-48g-alloc6367298-ddp-20260608-230023/train.out
  experiments/streamvln/future_visual_tokens/logs/streamvln-future-s1-48g-alloc6367298-ddp-20260608-230023/train.err
eval: future_eval_size=1024, eval_steps=500
save: save_steps=1000, save_total_limit=2
validated:
  step 500 eval_loss: 1.285919427871704
  step 1000 eval_loss: 1.274842381477356
  checkpoint-1000: present with trainer_state.json, optimizer.pt, scheduler.pt, 48 rng_state files, and 4 model safetensors shards
final:
  final checkpoint: checkpoint-4978
  train_loss: 1.209248219019153
  train_runtime: 7371.2115 sec
  final saved checkpoints: checkpoint-4000, checkpoint-4978
  eval trend: 1.2859 -> 1.2748 -> 1.2698 -> 1.2678 -> 1.2630 -> 1.2610 -> 1.2612 -> 1.2582 -> 1.2606
launch config:
  deepspeed: disabled
  distributed: torchrun DDP
  ddp_find_unused_parameters: True
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 2
  gradient_checkpointing: False
previous failed run:
  streamvln-future-s1-48g-6364948 failed around step 20 with NCCL ALLREDUCE timeout.
  streamvln-future-s1-48g-alloc6367298-20260608-221805 reached tqdm 0/4978 but did not advance to first loss; cancelled inside allocation.
  streamvln-future-s1-48g-alloc6367298-ddp-20260608-224902 failed quickly because DDP had unused trainable future_predictor params with find_unused_parameters=False.
```

## 已整合脚本

```text
experiments/streamvln/future_visual_tokens/scripts/run_baseline_r2r_single_gpu_smoke.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_baseline_r2r_val_unseen_8gpu_eval.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_baseline_train_2gpu_save_resume_smoke.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_future_stage1_predictor_pretrain_8gpu.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_future_stage1_predictor_pretrain_48gpu_system.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_future_stage2_joint_8gpu.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_future_stage2_joint_8gpu_smoke_system.sbatch
experiments/streamvln/future_visual_tokens/scripts/run_future_stage2_joint_48gpu_alloc.sh
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
