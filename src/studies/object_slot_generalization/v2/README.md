# Object-Consistent Slot Encoder v2

独立实验目录，不修改 V12/V23 或 `object_slot_generalization`。v2 将 Entity slot 设计为可泛化对象状态提取器：

- context-only，真实 future 不进入 Entity extractor；
- objectness gate，先抑制背景 patch；
- patch 对 slot 的竞争式 ownership；
- 每个 slot 独立局部 feature contribution reconstruction；
- slot load balancing，防止单 slot 吸收全部 patch；
- permutation-invariant matching 接口；
- 可选的 objectness、centroid 和局部 feature supervision。

当前版本先完成模型级 smoke test。后续数据适配应严格分成：

1. Physion++：使用 object valid/state 与投影区域监督；
2. CLEVRER：使用渲染物体 mask、属性和轨迹监督；
3. EK100：只使用 object-only noun transfer，避免 relation/global feature 泄漏。

Smoke test：

```bash
cd /home/zqliu/Work-Space/ABDUCTIVE-WORLD/recipe_formal
PYTHONPATH=. python object_slot_generalization_v2/scripts/smoke_test.py --device cuda
```
