# V-JEPA 2 Native Predictor Baseline 介绍

## 1. 目标

该 recipe 为 ABDUCTIVE/context-causal 系列提供时间任务对齐的 baseline：输入完整 current clip 的 16 个采样帧，预测完整 future clip 的 16 个采样帧。

旧实验只使用 current 前 8 帧预测 future 后 8 帧。本实现将两段完整 clip 拼成 32 帧 token grid，使可见帧数、预测帧数和时间范围与 state-only 方法一致。

## 2. 方法

模型包含：

1. 从 `vith.pt[target_encoder]` 加载并冻结的 V-JEPA 2 ViT-H encoder；
2. 保持官方接口和结构、在 Physion++ 上随机初始化训练的 native masked-token predictor。

```text
current 16 ──frozen encoder──> context latent
                                      │
                                      ▼
                         native masked predictor
                                      │
                                      ▼
current 16 + future 16 ──frozen encoder──> full target representation
                                             └─ select future-16 target latent
```

默认损失为原生 V-JEPA 风格的 latent L1；配置也支持 `loss.type: mse`，便于和使用 MSE 的其他 world model 做显式匹配。优化器只包含 predictor。

## 3. 为什么冻结 encoder

这样能把比较重点放在未来预测器：native baseline 使用原生 masked predictor，state-only 使用自定义 world evolver，context causal experts 再加入历史上下文、`z_event`、`z_local` 和 expert routing。

## 4. 公平性边界

| 方法 | 直接可见 clip | 目标 |
|---|---|---|
| Native V-JEPA 2 | current 16 | future 16 |
| ABDUCTIVE state-only | current 16 | future 16 |

完整 `context_causal_experts` 还读取多个 context clips 和 previous clip，因此它与 native 的差异仍包含额外历史信息。要单独归因因果变量，还需要 history-matched state-only 消融。

## 5. Split

```text
data_v1         → predictor 训练
readout_data_v1 → latent validation、选 checkpoint
testdata_v1     → 保留给最终下游 OCP 评测
```

本 recipe 不训练 OCP readout，也不读取接触或物理属性标签。
