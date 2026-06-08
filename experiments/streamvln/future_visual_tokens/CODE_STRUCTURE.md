# StreamVLN 代码结构速查

## 总体结构

- `streamvln/`：StreamVLN 自己的 VLN 模型、训练、评估、DAgger、轨迹生成逻辑。
- `llava/`：继承自 LLaVA-NeXT / LLaVA-Video 的基础模型、vision tower、projector、trainer。
- `scripts/`：Slurm/torchrun 启动脚本。
- `config/`：Habitat eval / DAgger / co-training 配置。
- `data/`：本地软链到数据盘，包含 episode、scene、trajectory、dagger、co-training 数据。

## 模型定义

主模型文件：

- `streamvln/model/stream_video_vln.py`

关键类：

- `StreamVLNModel`：继承 `LlavaQwenModel`，设置 vision tower、patch feature、`num_history` 等。
- `StreamVLNForCausalLM`：继承 `Qwen2ForCausalLM` 和 `LlavaMetaForCausalLM`，是真正训练/推理的模型类。

关键函数：

- `encode_rgbd()`：把 RGB-D / history frames 编成当前视觉 token 和 memory token。
- `prepare_inputs_labels_for_multimodal()`：把文本中的 image / memory token 替换成视觉 embedding，并构造 label。

底层 LLaVA 组件：

- `llava/model/language_model/llava_qwen.py`：Qwen 语言模型包装。
- `llava/model/llava_arch.py`：LLaVA 多模态通用逻辑。
- `llava/model/multimodal_encoder/`：vision tower。
- `llava/model/multimodal_projector/`：视觉特征到 LLM hidden size 的 projector。
- `llava/model/multimodal_resampler/`：可选 token resampler。

## 数据处理

VLN 轨迹训练数据：

- `streamvln/dataset/vln_action_dataset.py`
- 关键类：`VLNActionDataset`

它从 `--video_folder` 指定的多个目录读取：

- `annotations.json`
- `images/` 或轨迹帧目录

主要逻辑：

- 将每条导航轨迹按 `num_frames` 切成训练样本。
- 将动作转成文本 token：`STOP`、`↑`、`←`、`→`。
- 取当前 frames 和 history frames，组成多轮 VLN prompt。

多任务 / co-training 数据：

- `streamvln/streamvln_train.py`
- `LazySupervisedDataset`：处理 LLaVA-Video / ScanQA 等 JSON、JSONL、YAML 数据。
- `streamvln/dataset/mmc4_dataset.py`
- `LazyMMC4Dataset`：处理 MMC4 图文数据。
- `make_supervised_data_module()`：构造训练 dataset。

数据组合逻辑：

- 默认只使用 `VLNActionDataset`。
- 若 `--multi_task_training True`，额外加入 QA、ScanQA、MMC4。
- 多个 dataset 用 `CombineDataset` 合并。

## 训练入口

主训练文件：

- `streamvln/streamvln_train.py`

主函数：

- `train()`

训练流程：

1. 解析 `ModelArguments`、`DataArguments`、`TrainingArguments`。
2. `get_model()` 加载 `StreamVLNForCausalLM`。
3. 初始化 tokenizer、vision tower、image processor。
4. 根据 `--mm_tunable_parts` 决定训练哪些模块。
5. 构造 dataset 和 collator。
6. 用 `LLaVATrainer` 开始训练。
7. 若 `output_dir` 下已有 `checkpoint-*`，自动 resume。
8. 保存 checkpoint / model state。

可训练模块由 `--mm_tunable_parts` 控制：

- `mm_vision_tower`：训练视觉塔。
- `mm_mlp_adapter`：训练 mm projector。
- `mm_language_model`：训练语言模型主体。
- `mm_lora_layer`：只训练 LoRA 层。

Trainer 文件：

- `llava/train/llava_trainer.py`

主要改动：

- 自定义 sampler。
- 支持 `group_by_task`。
- 给 `mm_projector`、`vision_tower` 设置特殊学习率。
- 自定义部分 checkpoint 保存逻辑。

## Stage 1 训练

启动脚本：

- `scripts/streamvln_train_slurm.sh`

含义：

- 从 LLaVA-Video checkpoint 开始。
- 使用预采集的 VLN observation-action trajectory。
- 做导航行为克隆训练。

默认数据：

- `data/trajectory_data/R2R`
- `data/trajectory_data/RxR`
- `data/trajectory_data/EnvDrop`

默认训练模块：

- `mm_vision_tower`
- `mm_mlp_adapter`
- `mm_language_model`

一句话：Stage 1 是用 R2R/RxR/EnvDrop 轨迹数据训练模型看历史视觉输入并输出导航动作。

## Stage 2 训练

启动脚本：

- `scripts/streamvln_stage_two_train_slurm.sh`

含义：

- 从 Stage 1 checkpoint 继续训练。
- 加入 DAgger 数据。
- 加入通用多模态 co-training 数据。

VLN 数据：

- `data/trajectory_data/R2R`
- `data/trajectory_data/RxR`
- `data/dagger_data/R2R`
- `data/dagger_data/RxR`
- `data/dagger_data/EnvDrop`

Co-training 数据：

- `data/co-training_data/LLaVA-Video-178K`
- `data/co-training_data/ScanNet`
- `data/co-training_data/MMC4-core`
- 配置文件：`config/co-training_data.yaml`

关键开关：

- `--multi_task_training True`
- `--group_by_task True`

一句话：Stage 2 是在 Stage 1 导航模型基础上，用 DAgger 纠偏数据和通用视频/3D/图文数据混合训练，提高泛化和多模态能力。

## 评估与数据采集

评估：

- `streamvln/streamvln_eval.py`
- 启动脚本：`scripts/streamvln_eval_multi_gpu.sh`
- 负责 Habitat 环境中的 VLN-CE eval。

DAgger 采集：

- `streamvln/streamvln_dagger.py`
- 启动脚本：`scripts/streamvln_dagger_collect.sh`
- 用当前模型在 Habitat 中 rollout，并结合 teacher / oracle 生成纠偏数据。

轨迹生成：

- `streamvln/streamvln_trajectory_generation.py`
- 启动脚本：`scripts/streamvln_trajectory_generation.sh`
- 用 Habitat episode 生成 observation-action trajectory。

## 阅读建议

优先看：

1. `streamvln/model/stream_video_vln.py`
2. `streamvln/dataset/vln_action_dataset.py`
3. `streamvln/streamvln_train.py`
4. `scripts/streamvln_train_slurm.sh`
5. `scripts/streamvln_stage_two_train_slurm.sh`
6. `streamvln/streamvln_eval.py`

## 概念速查

### EnvDrop / DAgger / R2R / RxR

- `R2R`、`RxR`：标准 VLN benchmark / episode 数据，包含导航指令和路径。
- `EnvDrop`：R2R 系上的增强训练数据，来自 environmental dropout 和 back-translation，不是一个全新真实 benchmark。
- `Environmental Dropout`：训练时随机扰动/遮掉环境视觉特征，减少模型对训练房间局部视觉模式的死记。
- `Back-translation`：从已有 trajectory/path 反向生成 synthetic instruction，扩充 instruction-trajectory pairs。
- `DAgger`：一种 imitation learning 数据聚合方法，不是固定数据集。当前 policy rollout 后，让 expert/oracle 给偏离状态标动作，再加入训练集，缓解 behavior cloning 的 distribution shift。

简化理解：

```text
R2R / RxR: 标准 VLN 数据
EnvDrop: R2R 风格增强数据
DAgger: 模型自己 rollout 后收集的纠偏训练数据
```

### LLaVA / Qwen / StreamVLN

- `Qwen2`：语言模型 backbone。
- `LLaVA`：VLM 架构，把 vision encoder 的特征通过 projector 接到 LLM。
- `LLaVA-Video-7B-Qwen2`：以 Qwen2 为语言模型的 video VLM，支持多帧视频输入。
- `StreamVLN`：在 LLaVA-Video-Qwen2 上加入 VLN 的 history / memory / action generation 逻辑。

简化结构：

```text
video frames -> vision tower -> mm_projector -> Qwen2 LLM -> action text
```

所以 StreamVLN 不是纯 Qwen，也不是原生 QwenVL，而是基于 Qwen2 backbone 的 LLaVA-Video VLM。

### 为什么不用原生 QwenVL

理论上可以尝试，但不是直接替换 checkpoint：

- StreamVLN 已基于 LLaVA-Video 的输入拼接、trainer、video frame packing、checkpoint 格式实现。
- VLN 需要特殊的 current frames、history frames、memory token、action token 逻辑。
- 换成 QwenVL 需要重接数据 collator、视觉 token 插入方式、eval agent 和 checkpoint loading。

因此这是工程路线选择，不代表 QwenVL 不适合。

## OOM 经验

2 卡全参 smoke 曾 OOM，根因是模型全参训练显存过高，不是数据集本身。

失败配置：

```text
--mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model
```

日志中约：

```text
Total parameters: ~8030M
Trainable parameters: ~8030M
```

OOM 发生在 DeepSpeed optimizer step，需要额外分配约 15GB，而每张 A800 80G 已占约 76GB。

原因：

- 7B/8B VLM 全参训练。
- ZeRO-2 不切模型参数本体。
- AdamW optimizer state、gradient、activation、通信/flatten buffer 都很大。
- 多帧视频和长上下文进一步增加 activation 压力。

后续 2 卡 smoke 改成：

```text
LoRA + mm_projector
```

trainable params 降到约 37M，因此能成功训练、保存、resume。

和 DualVLN 类方法相比，StreamVLN 显存高主要因为它是大 Video VLM 端到端训练；DualVLN 若采用 frozen VLM + policy head 或离线视觉特征，天然更省显存。

## 优化器与微调范式

### 优化器

VLN/VLA 里优化器通常不是一阶关键变量。优先级通常低于：

- 数据质量和覆盖。
- 模型结构。
- 训练范式。
- action representation。
- eval protocol。
- 显存和吞吐。

现阶段需要了解：

- `AdamW`：默认首选。
- `8-bit Adam / PagedAdamW`：主要为省显存。
- `Adafactor`：省显存，但不是当前 Qwen/LLaVA 路线首选。
- `SGD`：大 VLM/VLA 微调里较少作为默认选择。

建议：复现 StreamVLN 时先沿用官方 AdamW + DeepSpeed 配置，不优先做优化器 ablation。

### 微调范式

微调范式比优化器更重要，建议优先掌握：

- `Full fine-tuning`：全参微调，效果上限高，显存/算力成本高。StreamVLN 官方 stage-one/stage-two 属于这类。
- `LoRA / QLoRA`：参数高效，适合 debug、小资源 ablation、快速迭代。
- `Frozen VLM + policy head`：冻结 VLM，只训练 policy/action head，最显存友好，但端到端适配能力较弱。
- `Partial fine-tuning`：只训 projector、最后几层 LLM、部分 vision tower 等，介于 full FT 和 LoRA 之间。
- `Two-stage / staged tuning`：先训小模块或 BC，再 DAgger/RL/解冻更多模块。

不需要现阶段大规模比较所有 PEFT 变体。优先理解：

```text
Full FT vs LoRA vs Partial FT vs Frozen VLM + policy head
```
