# 使用说明

## 1. 前置检查

从仓库根目录运行，并确认 checkpoint 与数据存在：

```bash
cd /home/shyang/workspace/work/vjepa2-baiz
ls /data/ABDUCTIVE-WORLD/pretrain/checkpoints/vith.pt
ls /data/ABDUCTIVE-WORLD/physion_v2/extracted/data_v1
ls /data/ABDUCTIVE-WORLD/physion_v2/extracted/readout_data_v1
```

环境需包含 PyTorch、Decord、Pandas、PyYAML 和仓库依赖。

## 2. 8-GPU

```bash
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_8gpu.sh
```

训练结束后执行 OCP 测试和时序检索测试：

```bash
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/eval_full_future_ocp.sh
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/eval_temporal_retrieval_8gpu.sh
```

OCP 会从真实 Current 闭环预测到每个视频末尾，采用与 `vjepa2_ef4`
相同的全 Future contact 标签和 pooled 特征定义；真实 Future 不进入模型。

默认设备 `0..7`，每卡 batch 2，global batch 16。覆盖设备：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_8gpu.sh
```

脚本默认将单机 rendezvous 绑定到 `127.0.0.1:29508`，避免 `torchrun --standalone` 使用无法回连的主机名。端口被占用时可以覆盖：

```bash
MASTER_PORT=29608 bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_8gpu.sh
```

gap32 实验使用独立入口：

```bash
bash recipe/vjepa2_naive/scripts/physionpp/gap32/train_8gpu.sh
```

## 3. 4-GPU

```bash
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_4gpu.sh
```

默认设备 `0..3`，每卡 batch 2，global batch 8。4/8 GPU 使用独立输出目录，不会互相覆盖。两者 global batch 不同，不能视为完全相同的优化预算；若需要 effective global batch 16，需实现梯度累积或在显存允许时提高每卡 batch。

4-GPU 默认 rendezvous 地址为 `127.0.0.1:29504`，也可以通过 `MASTER_ADDR`、`MASTER_PORT` 覆盖。

## 4. 额外参数

```bash
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_4gpu.sh   --max-train-videos 32 --max-eval-videos 16
```

支持 `--resume PATH`、`--max-train-videos N`、`--max-eval-videos N`，以及采样覆盖参数：

```bash
# 使用同名 PKL 的 start_frame_for_prediction（配置默认值）
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_4gpu.sh \
  --sampling-mode prediction_start \
  --current-frame-step 2 \
  --future-frame-step 4 \
  --clip-gap 32

# 不读取 PKL，随机选择合法锚点
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_4gpu.sh --sampling-mode random
```

对应 YAML：

```yaml
data:
  sampling_mode: prediction_start
  current_frame_step: 2
  future_frame_step: 4
  clip_gap: 32
```

命令行值优先于 YAML。训练脚本都会把额外参数原样传给 `train.py`。

所有 `.sh` 都直接使用 `nohup` 在后台启动实际 Python/torchrun 进程，不经过 Python launcher。脚本会在对应输出目录创建日志和 PID 文件，并拒绝重复启动仍存活的 PID。训练日志为 `train.log`/`train.pid`，评估日志为 `eval.log`/`eval.pid`。

新训练若发现输出目录已经存在，会先将旧目录移动到同级 `legacy/`，归档名为原目录名加 `YYYYMMDD_HHMMSS`；同一秒重名时继续追加数字后缀。传入 `--resume` 时保留原训练目录。评估任务同样会归档已有的评估输出目录。查看实时日志：

```bash
tail -f outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_4gpu/train.log
```

## 5. Smoke test

复制配置为 `configs/physionpp/physion-vith-16to16-smoke.yaml`，设置 `epochs: 1, ipe: 2, batch_size: 1, eval_batch_size: 1, max_eval_batches: 2`，然后：

```bash
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29501 \
  --nproc_per_node=1 \
  recipe/vjepa2_naive/train.py \
  --config recipe/vjepa2_naive/configs/physionpp/physion-vith-16to16-smoke.yaml \
  --max-train-videos 8 --max-eval-videos 8
```

ViT-H 的 32 帧 grid 显存占用较大，OOM 时优先将每卡 batch 降为 1。

## 6. 恢复

```bash
bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_4gpu.sh   --resume outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_4gpu/latest.pt

bash recipe/vjepa2_naive/scripts/physionpp/no_gap/train_8gpu.sh   --resume outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/latest.pt
```

## 7. 输出

```text
outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_{4gpu|8gpu}/
├── config.yaml
├── metrics.csv
├── train.log
├── train.pid
├── best.pt
├── latest.pt
└── epoch-*.pt
```

`metrics.csv` 包含 split、epoch、iteration、step、优化损失、MSE、cosine 和 LR。默认优化损失是 L1，MSE 始终作为诊断指标记录；验证不包含 OCP accuracy。`best.pt` 对应最低 validation prediction loss，OCP 和时序检索默认使用该 checkpoint；`latest.pt` 始终保存最近完成的 epoch。恢复训练后会继续沿用 checkpoint 中记录的历史最佳 loss 和 epoch。

## 8. 常见问题

- 找不到模块：从仓库根运行，或设置 `PYTHONPATH=/home/shyang/workspace/work/vjepa2-baiz`。
- 找不到视频：检查 root、split 和 `**/*_img.mp4`。
- Checkpoint 权重不匹配：确认是 V-JEPA 2 ViT-H 且含 `target_encoder` 或 `encoder`。
- CUDA OOM：降低 YAML 中 train/eval batch；32 帧 token 数是旧 16 帧 native 实验的两倍。
