# 模型与训练架构

## 1. 总览

```text
Physion video
 └─ dataset.py → current[16], future[16]
      ├─ frozen encoder → context tokens
      ├─ frozen encoder → full 32-frame target representation
      ├─ select future-16 target tokens (stop-gradient)
      └─ native predictor → predicted target tokens → L1（默认）
```

## 2. 初始化

`model.py` 调用 `app.vjepa.utils.init_video_model()`：

```text
model_name=vit_huge, max_num_frames=32
patch_size=16, tubelet_size=2
pred_depth=12, pred_embed_dim=384, pred_num_heads=12
num_mask_tokens=1, use_rope=true, use_sdpa=true
```

只加载 checkpoint 的 encoder，不加载原 predictor：

```python
encoder.backbone.load_state_dict(checkpoint["target_encoder"], strict=False)
```

因此它是 same-data predictor training，而不是 off-the-shelf predictor evaluation。

## 3. Forward

```python
combo = torch.cat([current, future], dim=2)  # [B,C,32,H,W]
with torch.no_grad():
    context = encoder([combo], masks=[[context_mask]])[0][0]
    full_target = encoder([combo])[0]
    full_target = layer_norm(full_target)
    target = apply_masks(full_target, [target_mask])
predicted = predictor([[context]], [[context_mask]], [[target_mask]])[0][0]
```

这是原生 V-JEPA teacher 语义：目标分支先编码完整 32 帧，再选出 future 16 对应 token。predictor 只接收 context latent 和目标位置 mask，不接收 target latent。

## 4. Token 数

```text
spatial tokens  = (256/16)^2 = 256
temporal tokens = 32/2 = 16
total            = 4096
context          = 2048 tokens（前16 RGB帧）
target           = 2048 tokens（后16 RGB帧）
```

## 5. 损失与优化

默认 L1 用于反向传播；YAML 可切换为 MSE。日志始终额外记录 MSE 和 cosine similarity。优化器默认使用 AdamW、warmup+cosine learning rate、gradient clip 1.0 和 BF16 autocast。

## 6. DDP 与 checkpoint

`torchrun` 每个 rank 使用 `DistributedSampler`，DDP 只同步 predictor 的可训练梯度。只有 rank 0 从 10 GB 预训练 checkpoint 读取 encoder，随后向其他 rank 广播，避免重复磁盘读取和主机内存峰值。验证由全部 rank 分片执行并通过 `all_reduce` 聚合。rank 0 原子写入 checkpoint，内容包括 predictor、optimizer、scaler、epoch、global step、source checkpoint epoch、历史最佳 validation loss/epoch、协议名和完整配置；冻结 encoder 不重复保存。最低 validation prediction loss 对应的权重保存为 `best.pt`，最近 epoch 保存为 `latest.pt`。
