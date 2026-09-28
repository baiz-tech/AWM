# Orca Physion++ Oracle-Future OCP

冻结 `/data/shared_model/Orca-4B`，分别对 Current 与真实 Future 视频提取 video-token mean latent，再拼接训练最小线性 OCP readout。此实验不使用 Orca NFP、dynamics decoder 或 shallow probe；future latent 是 ground-truth future video 的 oracle 表征，不代表未来预测能力。

运行：`bash recipe/orca/scripts/physionpp/run_all.sh`。缓存支持 `SPLIT`、`OUTPUT_DIR`、`PHYSION_ROOT`、`CHECKPOINT_DIR`、`MAX_SAMPLES` 环境变量。
