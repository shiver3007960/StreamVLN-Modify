# StreamVLN Future Visual Tokens Status

最近更新：2026-06-09 CST

## 当前阶段

```text
stage: gated future baseline eval completed; entering failure diagnosis
status: wait for cleanup/push, then run no-future continuation control and direct-cat future baseline
```

## Checklist

```text
Stage 0 代码接口确认: done
Stage 1 future_predictor 预训练: done
Stage 2 gated fusion joint 微调: done
Stage 3 R2R val_unseen eval: done
Stage 4 失败原因排查: pending
Stage 5 direct-cat future baseline: pending
```

## 当前结论

```text
当前 gated future fusion 方案明显负收益。
该方案是：future_predictor 先预测 196 个 t+4 future tokens，再通过 gated cross-attn residual 加回 current observation tokens。
R2R val_unseen 上三个 future ckpt 的 SR/SPL 都比 official 低约 10-12 点以上。
```

最终 R2R val_unseen：

| run | ckpt | SR | SPL | OS | NE |
| --- | --- | ---: | ---: | ---: | ---: |
| official | released | 57.86 | 51.30 | 65.25 | 4.78 |
| gated future | checkpoint-3000 | 45.51 | 41.06 | 52.47 | 6.05 |
| gated future | checkpoint-4000 | 46.22 | 41.98 | 52.47 | 5.87 |
| gated future | checkpoint-4978 | 46.33 | 41.76 | 52.09 | 5.93 |

结果路径：

```text
/mnt/inspurfs/evla2_t/lizhen/results/StreamVLN/future_visual_tokens/r2r-val-unseen-eval-alloc6367298-20260609-145325
```

## 有效 artifacts

```text
Stage 1 predictor best init:
  /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s1-48g-alloc6367298-ddp-20260608-230023/checkpoint-4000

Stage 2 gated future ckpts:
  /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618/checkpoint-3000
  /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618/checkpoint-4000
  /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618/checkpoint-4978

Stage 2 trainer state / proxy eval loss:
  /mnt/inspurfs/evla2_t/lizhen/checkpoints/StreamVLN/future_visual_tokens/streamvln-future-s2-48g-alloc6367298-ddp-20260609-091618/trainer_state.json
```

## 下一步

```text
1. 清理冗余临时文件，只保留必要脚本、文档和可复现日志指针。
2. 推送当前版本到 GitHub。
3. 写并 smoke no-future old-data continuation control：
   official ckpt + old trajectory data + LoRA + 1 epoch protocol。
4. smoke 通过后，用 48GPU 跑 no-future control full run。
5. 实现 direct-cat future baseline：
   current visual tokens 后追加 future marker + predicted future tokens + future type embedding。
6. direct-cat 用 32GPU 跑 1 epoch，total batch 尽量和 48GPU no-future control 对齐。
```

## 排查假设

```text
H1: official ckpt 用 old trajectory data 续训本身会掉点。
H2: gated fusion 污染 current observation tokens。
H3: future predictor 预测质量不足。
H4: LoRA/projector/action joint 微调扰动了原 policy。
```

优先判断：

```text
先跑 H1 no-future continuation control。
再跑 H2 direct-cat future baseline。
oracle future / shuffle future 等诊断放到下一轮。
```

## 当前代码状态

```text
已实现:
  streamvln/dataset/vln_action_dataset.py: future_images / future_valid
  streamvln/model/stream_video_vln.py: FutureVisualPredictor / FutureToCurrentFusion / future loss
  streamvln/args.py: future 相关参数
  streamvln/streamvln_train.py: future 模块解冻与训练保存
  streamvln/streamvln_eval.py: 支持 PEFT adapter + non_lora_trainables eval 加载

待实现:
  no-future continuation control launcher
  direct-cat future marker/type embedding
  direct-cat train/eval launcher
```

## 工作区注意

```text
当前有未提交改动，需要先清理和 review：
  experiments/streamvln/future_visual_tokens/STATUS.md
  experiments/streamvln/future_visual_tokens/PLAN.md
  streamvln/streamvln_eval.py
  experiments/streamvln/future_visual_tokens/scripts/run_r2r_val_unseen_eval_48gpu_alloc.sh

不要删除大结果和 checkpoint。
冗余日志/旧失败脚本只在确认不影响复现后再清理。
```
