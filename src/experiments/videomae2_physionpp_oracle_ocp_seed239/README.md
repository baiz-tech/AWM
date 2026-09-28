# VideoMAE V2 Physion++ baseline

> **迁移后入口(note)**:下面的命令来自迁移前的源仓库,其中 `recipe.*` 模块路径
> 在本仓库已不存在,仅作历史记录保留。当前用法以
> `scripts/<dataset>/<experiment>/<task>.sh`(`analyses` 为
> `scripts/analyses/<analysis>/<task>.sh`)为准,完整顺序见 `docs/experiment.md`。


官方仓库：<https://github.com/OpenGVLab/VideoMAEv2>；权重说明：
<https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md>。

推荐先使用可直接下载的官方 ViT-base distilled K710 权重：

```bash
bash recipe/recipe_formal/VideoMAE2/scripts/download_official.sh \
  /data/shared/models/videomae-v2
```

ViT-giant 预训练及 K400/K600/K710 微调权重需要填写官方
[Download Request Form](https://docs.google.com/forms/d/e/1FAIpQLSd1SjKMtD8piL9uxGEUwicerxd46bs12QojQt92rzalnoI3JA/viewform?usp=sf_link)。

注意：官方公开的 `.pth` 是原生 VideoMAE V2 checkpoint，不是
`transformers.VideoMAEModel.from_pretrained()` 目录。现有 `model.py` 的 HF
loader 不能直接读取该文件；需要使用官方仓库的 model definition/loader，
或增加一个只做 key mapping 的 adapter。不要把 `.pth` 路径直接填入
`model_name_or_path` 后假设它能加载。

## 两阶段运行

配置中的 `official_repo` 已指向 recipe 内的 `repo/`。协议与 Orca 对齐：
`data_v1` 训练 OCP head，`readout_data_v1` 选择最佳 epoch 和 threshold，
`testdata_v1` 只做最终测试；Current 为 16 帧 step=2，Future 为真实视频的
16 帧 step=4。先缓存 frozen encoder 的 Current 与真实 Future latent：

```bash
PYTHONPATH="$PWD" python -m recipe.recipe_formal.VideoMAE2.cache_latents \
  --config recipe/recipe_formal/VideoMAE2/config.yaml \
  --split data_v1 \
  --output-dir /data/shyang/outputs/vjepa2-baiz/VideoMAE2/cache

PYTHONPATH="$PWD" python -m recipe.recipe_formal.VideoMAE2.cache_latents \
  --config recipe/recipe_formal/VideoMAE2/config.yaml \
  --split readout_data_v1 \
  --output-dir /data/shyang/outputs/vjepa2-baiz/VideoMAE2/cache

PYTHONPATH="$PWD" python -m recipe.recipe_formal.VideoMAE2.cache_latents \
  --config recipe/recipe_formal/VideoMAE2/config.yaml \
  --split testdata_v1 \
  --output-dir /data/shyang/outputs/vjepa2-baiz/VideoMAE2/cache
```

再用缓存训练 readout，并在 test split 评测：

```bash
PYTHONPATH="$PWD" python -m recipe.recipe_formal.VideoMAE2.train_ocp \
  --cache-root /data/shyang/outputs/vjepa2-baiz/VideoMAE2/cache \
  --output-dir outputs/runs/VideoMAE2/joint_oracle
```

缓存协议保存 `context_latent`、`future_latent`、Future mask、OCP label 和
原视频路径。该实现明确使用真实 Future RGB，是 `joint_real_future_oracle`
上界；不应解释为 causal prediction。

使用 8 张 GPU 一键缓存三个 split 并训练/测试 OCP：

```bash
bash recipe/recipe_formal/VideoMAE2/scripts/cache_train_test_8gpu.sh
```

默认后台日志写入 `/data/shyang/outputs/vjepa2-baiz/VideoMAE2/orca_aligned/logs/`。

先克隆官方代码并设置 `config.yaml` 中的 `model.official_repo`：

```bash
git clone --depth 1 https://github.com/OpenGVLab/VideoMAEv2.git /data/shared/models/VideoMAEv2
```

然后将 `official_repo` 改为 `/data/shared/models/VideoMAEv2`。recipe 会调用
官方 `vit_base_patch16_224`，加载 `.pth` 中的 `module` state dict，并严格
检查除分类 head 外的 missing/unexpected keys。
