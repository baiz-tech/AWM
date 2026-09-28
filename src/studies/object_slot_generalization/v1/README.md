# Object Slot Generalization

独立于 V12/V23 的对象 slot 泛化实验目录。当前第一阶段实现了：

- 仅使用 context patch tokens 的 Entity extractor；
- 对 slot 维度竞争归一化的 spatial assignment；
- 局部 patch feature reconstruction；
- 每个 slot 独立的 patch contribution reconstruction；
- presence / centroid / assignment entropy / temporal consistency loss 接口；
- slot usage load-balancing，抑制单 slot 吸收全部 patch 的退化解；
- 跨帧 slot matching 诊断工具。

Entity extractor 不接收真实 future，也不包含 Dynamic 或 Relation 分支。后续可在此目录内加入 Physion++、CLEVRER、EK100 的数据适配器和冻结 backbone cache，不会修改原实验目录。

## Smoke test

```bash
cd /home/zqliu/Work-Space/ABDUCTIVE-WORLD/recipe_formal
PYTHONPATH=. python object_slot_generalization/scripts/smoke_test.py --device cuda
```

## 设计判据

slot 是否具备可泛化对象信息，需要同时检查：

1. spatial grounding：slot assignment 与物体区域的 IoU/foreground mass；
2. temporal persistence：跨帧 Hungarian matching、ID switch rate；
3. object-only transfer：只使用 slots 预测 object/attribute/noun；
4. local sufficiency：仅由单个 slot 重建对应局部 feature。
