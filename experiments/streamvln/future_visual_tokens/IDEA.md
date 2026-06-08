# StreamVLN Future Visual Tokens

创建日期：2026-06-08

## 核心目标

构造一个带 future representation 的 StreamVLN baseline，验证导航动作生成是否能从预测的未来视觉状态中受益。

核心原则：

```text
future representation 必须进入 action 决策链路。
不能只做训练时 auxiliary loss 后声称模型使用了未来。
所有效果结论必须对比同协议的 StreamVLN official checkpoint eval。
```

## 动机

StreamVLN 当前主要根据指令、历史观测、当前观测生成动作序列。局部遮挡、拐角、楼梯、门口等场景中，当前帧未必暴露后续可行动空间，因此模型可能需要一个显式的未来状态表征。

希望验证：

```text
由当前/历史观测预测出的 future visual tokens，
如果被注入当前视觉 token 或 action prompt 前，
是否能改善 VLN 决策。
```

## StreamVLN 中的落点

StreamVLN 每个 `<image>` 经 vision tower、projector、2D pooling 后约为 `14x14=196` 个视觉 token；`<memory>` 约为 `num_history * 196` 个历史 token。

第一版不把 future token 只作为很短的 prefix，而是优先让它进入当前视觉表示：

```text
history/current visual tokens
-> future predictor
-> predicted future visual tokens
-> future-to-current fusion
-> fused current visual tokens
-> Qwen/LLaVA action generation
```

## 第一版假设

```text
196 个 future tokens 是第一版 token budget：
  和 StreamVLN 一张当前图像的 14x14 visual tokens 对齐；
  避免 future 表征因 token 数太少被当前/历史视觉 token 淹没；
  便于直接用未来帧 visual feature 作为监督目标。
```

future target 使用 StreamVLN 自身视觉编码路径产生的未来帧特征，detach 后作为监督，不额外引入新的视觉 teacher。

第一版直接训练 predicted future 模块，不先做 oracle future。oracle future 放到后续诊断中，用来判断失败来自 future 预测质量还是注入/消费路径。

## 不做的声明

本实验第一版不声称：

```text
模型已经具备完整 world model；
future predictor 单独 loss 下降就代表导航提升；
predicted future 一定优于 oracle future 或直接加大视觉 token budget。
```

## 成功标准

最低成功标准：

```text
1. 训练、保存、resume、eval 路径跑通。
2. future dropout / shuffle 会影响 loss 或 eval，说明模型确实使用 future。
3. official R2R val_unseen 同协议指标不低于 baseline，并最好有 SR/SPL 改善。
```

失败判据：

```text
oracle future 不提升；
predicted future 对 dropout/shuffle 不敏感；
同协议 eval 稳定低于 official baseline。
```
