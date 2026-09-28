# 实验运行指南

本文说明**当前仓库里的实验分别怎么跑**、彼此依赖什么、产物写到哪里。
配套文档:

- `docs/manual/requirement.md` —— 目录/命名/config 结构的规范(权威);
- `docs/manual/example_config.yaml` —— config 字段的完整样例;
- `docs/manual/resource.md` —— **本机**实测资源(数据、权重、GPU 状态、端口),
  路径相关的问题先查这里;
- `tools/launcher/README.md` —— launcher 的完整模板变量与实现说明。

本文只描述已经接好 launcher 的入口。当前共 **18 个 config / 103 个可启动 task**,
每个 task 一个 `scripts/**/<task>.sh`,全部通过 `--dry-run` 可解析。

---

## 1. launcher 契约:一个 task 怎么跑

```bash
bash scripts/<dataset>/<experiment>/<task>.sh [选项]
```

脚本本身只有三行(`CONFIG` / `TASK` / source launcher),参数全部来自 config 的
`tasks.<task>.experiment.cli_defaults`,由 `src/core/run_context.py` 注入模块。
常用选项:

| 选项 | 作用 |
|---|---|
| `--dry-run` | 只解析 config、打印所有派生路径与实际命令,**不创建目录、不启动进程**。正式运行前必做 |
| `--output-mode both\|data\|workspace` | 覆盖输出位置。默认 `both` |
| `--existing-output archive\|delete\|raise` | 输出目录已存在时的行为。默认 `archive`(移入同级 `legacy/` 并加时间戳);`raise` 直接报错退出 |
| `--background true\|false` | 是否后台启动。大多数 task 默认 `true`,日志写 `<task 输出目录>/logs/` |
| `--resume` | 从 `launch.resume.checkpoint` → `<task 输出目录>/checkpoints/latest.pt` → `.../best.pt` 找 checkpoint,写入 `CHECKPOINT` 环境变量 |
| 其余参数 | 原样透传给模块,例如 `--max-videos 8`、`--max-samples 100`、`--device cpu`,用于 smoke test 或单卡调试 |

输出位置(AWM 项目,dataset 即上表目录名):

```text
data 模式 / both 的主副本:   /data/shyang/outputs/awm/<dataset>/<experiment>/<task>/
workspace 模式 / both 的镜像: outputs/runs/<dataset>/<experiment>/<task>/
跨 run 分析:                  /data/shyang/outputs/awm/analyses/<analysis>/
                              outputs/analyses/<analysis>/
```

每个 task 输出目录下都会留下复现信息:`command.txt`、`configs/<原始 config>`、
`configs/<task>.resolved.yaml`(模块真正读到的内容)、`manifest.json`、
`environment.json`、`logs/<task>.log`、`logs/<task>.exitcode`、`logs/<task>.status.json`。

> 机器专用路径不写死在通用代码里,而是集中在各 config 的
> `launch.extra_config.paths`。换机器时改这里,或用 `MASTER_PORT` /
> `CUDA_VISIBLE_DEVICES` / `NPROC_PER_NODE` 等环境变量覆盖。

---

## 2. 运行前准备

1. **环境**:`/data/shared/envs/vjepa2-312/bin/python`(Python 3.12 + torch 2.6)。
   依赖见 `requirements.txt`;`paper_figures` 额外需要 `matplotlib`(当前环境未装)。
2. **数据与权重**:对照 `docs/manual/resource.md` §4。本机已有 Physion++、CLEVRER
   与 V-JEPA 2 ViT-H 权重;**缺** EK100、Orca-4B、以及论文引用的历史 run 产物。
3. **先 `--dry-run`**:确认 `experiment_root`、各上游输入路径、实际命令符合预期。
4. **按链条顺序跑**:每个 task 只检查自己显式声明的输入是否存在,不会自动补跑上游。

---

## 3. 依赖关系总览

```text
Physion++ (AWM)   train_predictor ─┬─ cache_train ──┐
                                  ├─ cache_readout ┼─ train_probe ─ eval_ocp / eval_ocp_calibrated
                                  └─ cache_test ───┘        │
                                                             └─ 离线分析(extract_physion_*)

CLEVRER (AWM)     prepare_targets_{train,validation} ─ cache_latents/{train,validation} ─ train_probe
                        │                                                                   │
                        └──────────────────────► eval_probe / qa_eval ◄─────────────────────┘
                                                       │
                                                       └─ clevrer_intervention / clevrer_relation_controls

EK100 (AWM)       prepare_indices_{train,validation} ─ cache_{train,probe_validation,final_validation}
                                 ─ train_readout ─ train_adapter ─ evaluate
                                 ─ fair_attribution: train_e0..e4 ─ evaluate ─ parameter_report

V-JEPA 2 (Physion++)  需要 Physion native predictor(见 §8.1)─ cache_latents* ─ train_probe
                       ─ train_ocp_visual_only ─ train_ocp_readout ─ evaluate_ocp_interventions
V-JEPA 2 (CLEVRER)    train(naive world model)─ qa_eval
V-JEPA 2 (EK100)      prepare_targets ─ cache_latents ─ train_decoder ─ evaluate_decoder
VideoMAE v2           physionpp: cache_* ─ train_ocp      ek100: cache_features* ─ train / finetune
Orca                  physionpp: cache_* ─ train_ocp      ek100: cache_latents_* ─ train_readout
```

---

## 4. 主线一:Physion++(AWM)

`configs/physionpp/awm_physionpp_fullpatch_probe_seed239/`

```bash
D=scripts/physionpp/awm_physionpp_fullpatch_probe_seed239
bash $D/train_predictor.sh        # ① 随机初始化 native latent predictor(论文只用来产生 Ẑ_f)
bash $D/cache_train.sh            # ② 三个 split 的 (Z_c, Ẑ_f)
bash $D/cache_readout.sh
bash $D/cache_test.sh
bash $D/train_probe.sh            # ③ HASP 结构化 probe(Entity / Dynamic / Relation)
bash $D/eval_ocp.sh               # ④ 主表 Physion++ 行:AUROC / balanced acc / acc
bash $D/eval_ocp_calibrated.sh    # 可选:校准版本的 OCP 指标
```

要点:

- `train_predictor` 使用 `launch.config_template`(结构特殊的 predictor config),
  其 resolved config 落在 `<task>/configs/train_predictor.resolved.yaml`——
  离线分析 `physion_predictor_regen_ablation` / `physion_relation_instance_ablation`
  会把它当作 `--predictor-config` 读。
- 三个 cache task 共享 `cache_latents/` artifact 根(模块按 `--split` 写子目录),
  这是有意的;`train_probe` 同时需要 train 与 readout 两个 cache。
- `train_probe` 的 `best.pt` 是后续几乎所有 Physion++ 分析与干预的输入。

## 5. 主线二:CLEVRER(AWM)

`configs/clevrer/awm_clevrer_fullpatch_probe_seed239/`

```bash
D=scripts/clevrer/awm_clevrer_fullpatch_probe_seed239
bash $D/prepare_targets_train.sh        # ① 物体属性 / 轨迹 / 配对 / 接触 targets
bash $D/prepare_targets_validation.sh
bash $D/cache_latents_train.sh          # ② 冻结 16x16 context + predicted-future latent(共享 cache_latents/ 根)
bash $D/cache_latents_validation.sh
bash $D/train_probe.sh                  # ③ 结构化 probe
bash $D/eval_probe.sh                   # ④ 物体 / 轨迹 / 配对 / 接触指标
bash $D/qa_eval.sh                      # ⑤ 预测型 QA:option / question accuracy(主表 CLEVRER 行)
bash $D/inspect_scene.sh                # 可选:单场景检查
```

要点:

- `qa_eval` 是最长的一步(导出轨迹 → 训练 QA head → validation/test 评测),
  内部按阶段调用 `src/data/clevrer/qa_pipeline.py`;其 QA checkpoint 固定在
  `<task>/qa_eval/qa_checkpoints/current_future/best.pt`,供 `clevrer_intervention` 使用。
- `cache_latents/{train,validation}` 与 `prepare_targets_*/targets_*.pt` 是
  `clevrer_relation_controls`、`clevrer_dynamic_selectivity`、
  `clevrer_relation_instance_ablation` 的直接输入。

## 6. 主线三:EK100(AWM)

`configs/ek100/awm_ek100_multiscale_adapter_seed239/`

```bash
D=scripts/ek100/awm_ek100_multiscale_adapter_seed239
bash $D/prepare_indices_train.sh        # ① 视频级不重叠的 train / probe_validation 索引
bash $D/prepare_indices_validation.sh
bash $D/cache_train.sh                  # ② 8x8 空间分辨率的冻结 token 缓存
bash $D/cache_probe_validation.sh
bash $D/cache_final_validation.sh
bash $D/train_readout.sh                # ③ multiscale readout(在 probe_validation 上选 checkpoint)
bash $D/train_adapter.sh                # ④ shared temporal adapter
bash $D/evaluate.sh                     # ⑤ 官方 validation 上的 verb / noun / action 指标
```

参数对齐归因矩阵 E0–E4(需先完成上面的 ③④):

```bash
D=scripts/ek100/awm_ek100_fair_attribution_seed239
for v in e0 e1 e2 e3 e4; do bash $D/train_$v.sh; done
bash $D/evaluate.sh
bash $D/parameter_report.sh
```

五个 variant 的 head 写入同一个 artifact 根(`{experiment_root}/E0/best.pt` …
`{experiment_root}/E4/best.pt`),`evaluate` 才能在单个 `--checkpoints` 目录下找齐;
`train_e*` 各自的 `{large_output_dir}` 只保存 log 与运行元数据。

## 7. 控制实验

只用当前 latent、不给预测未来的 CLEVRER 对照:

```bash
D=scripts/clevrer/awm_clevrer_current_only_seed239
bash $D/prepare_targets_train.sh; bash $D/prepare_targets_validation.sh
bash $D/cache_latents_train.sh;  bash $D/cache_latents_validation.sh
bash $D/train_probe.sh;          bash $D/eval_probe.sh
```

## 8. 基线

### 8.1 V-JEPA 2 matched baseline

```bash
# CLEVRER:naive world model + 预测型 QA(主表 'V-JEPA 2' 行)
D=scripts/clevrer/vjepa2_clevrer_naive_world_model_seed239
bash $D/train.sh
bash $D/qa_eval.sh                 # use_probe_tokens: false,不含 structured probe

# CLEVRER:dynamics-decoder 变体
D=scripts/clevrer/vjepa2_clevrer_dynamics_decoder_seed239
bash $D/prepare_targets_train.sh; bash $D/prepare_targets_validation.sh
bash $D/cache_latents_train.sh;  bash $D/cache_latents_validation.sh
bash $D/train_probe.sh;          bash $D/eval_probe.sh

# CLEVRER:all-future 变体(4 个 cache)
D=scripts/clevrer/vjepa2_clevrer_dynamics_decoder_all_seed239
bash $D/prepare_targets_train.sh; bash $D/prepare_targets_validation.sh
bash $D/cache_latents_predictive_train.sh;    bash $D/cache_latents_predictive_validation.sh
bash $D/cache_latents_nonpredictive_train.sh; bash $D/cache_latents_nonpredictive_validation.sh
bash $D/train_probe.sh

# Physion++:HASP probe + 独立 OCP readout + slot 诊断 + 输入干预
D=scripts/physionpp/vjepa2_physionpp_dynamics_decoder_seed239
bash $D/prepare_targets.sh; bash $D/prepare_targets_validation.sh
bash $D/cache_latents.sh;   bash $D/cache_latents_readout.sh   # cache_latents_test.sh 可选
bash $D/train_probe.sh
bash $D/train_ocp_visual_only.sh        # 只看视觉 latent 的对照
bash $D/train_ocp_readout.sh            # visual + structured probe
bash $D/evaluate_interpretability.sh    # slot 对应与语义 readout
bash $D/evaluate_ocp_interventions.sh   # target / irrelevant / random slot 干预

# EK100:dynamics decoder
D=scripts/ek100/vjepa2_ek100_dynamics_decoder_seed239
bash $D/prepare_targets.sh; bash $D/cache_latents.sh
bash $D/train_decoder.sh;   bash $D/evaluate_decoder.sh
```

> Physion++ 的 V-JEPA 2 路径需要一份 Physion native predictor
> (`protocol: current_16_to_future_16`)。默认取
> `launch.extra_config.paths.physion_predictor_checkpoint`,即 §4 的
> `awm_physionpp_fullpatch_probe_seed239/train_predictor/best.pt`;先跑 §4 的 ①,
> 或把它改成你自己的 predictor。

### 8.2 Orca

不提供兼容的 future predictor,因此 Physion++ 上用**真实 future 视频的编码**代替预测
future;CLEVRER 不报告。

```bash
D=scripts/physionpp/orca_physionpp_oracle_ocp_seed239
bash $D/cache_train.sh; bash $D/cache_readout.sh; bash $D/cache_test.sh; bash $D/train_ocp.sh

D=scripts/ek100/orca_ek100_frozen_readout_seed239
bash $D/cache_latents_train.sh; bash $D/cache_latents_validation.sh; bash $D/train_readout.sh
```

### 8.3 VideoMAE v2

```bash
D=scripts/physionpp/videomae2_physionpp_oracle_ocp_seed239
bash $D/cache_train.sh; bash $D/cache_readout.sh; bash $D/cache_test.sh; bash $D/train_ocp.sh

D=scripts/ek100/videomae2_ek100_finetune_seed239
bash $D/prepare_indices_train.sh; bash $D/prepare_indices_validation.sh
bash $D/cache_features.sh                     # train split
bash $D/cache_features_probe_validation.sh    # 选 checkpoint 用的 held-out split
bash $D/cache_features_final_validation.sh    # 官方 validation
bash $D/train.sh                              # 冻结特征上的头
bash $D/finetune.sh                           # 端到端微调(主表 'VideoMAE v2' 行)
```

> 两个 VideoMAE v2 config 的 `model.official_repo` 指向**未 vendor** 的上游仓库:
>
> ```bash
> git clone https://github.com/OpenGVLab/VideoMAEv2 external/VideoMAEv2
> ```
>
> `model.checkpoint` 指向官方 `.pth`。缺任一路径时 `build_encoder` 直接报错。

## 9. 分析与论文表/图

分析实验**不修改训练代码或 checkpoint**,只读冻结 cache,因此训练完成后可随时重跑。
四个 analyses config 的输入 run 根都写在各自的 `launch.extra_config.paths`,换 run
只改那里。

### 9.1 `hasp_offline_analysis`(论文的离线分析表)

```bash
D=scripts/analyses/hasp_offline_analysis

# ① 状态导出(后续所有分析的前置)
bash $D/extract_physion_train.sh
bash $D/extract_physion_validation.sh
bash $D/extract_ek100.sh

# ② 信息量分解
bash $D/physion_factor_probe.sh
bash $D/physion_slot_time_probe.sh
bash $D/physion_pair_probe.sh
bash $D/entity_specialization.sh
bash $D/probe_matrix.sh

# ③ 动态选择性
bash $D/physion_dynamic_selectivity.sh
bash $D/clevrer_dynamic_selectivity.sh

# ④ 干预
bash $D/physion_input_ablation.sh
bash $D/physion_predictor_regen_ablation.sh
bash $D/physion_relation_instance_ablation.sh
bash $D/clevrer_relation_instance_ablation.sh

# ⑤ 定性可视化
bash $D/visualize_ek100.sh
```

依赖要点:

- `physion_factor_probe` / `physion_slot_time_probe` / `physion_pair_probe` /
  `entity_specialization` / `probe_matrix` 读 ① 导出的
  `{data_experiment_root}/extract_physion_{train,validation}`;
- `physion_dynamic_selectivity` / `physion_input_ablation` 直接读 Physion cache +
  `train_probe/best.pt`;
- `physion_predictor_regen_ablation` / `physion_relation_instance_ablation` 额外需要
  `{physion_run}/train_predictor/best.pt` 与
  `{physion_run}/train_predictor/configs/train_predictor.resolved.yaml`;
- `clevrer_dynamic_selectivity` / `clevrer_relation_instance_ablation` 需要 CLEVRER 的
  cache、targets、`train_probe/best.pt`(后者还需要 `world_checkpoint` 与
  `clevrer_config`)。

### 9.2 `clevrer_intervention`(论文表 `query_intervention`)

```bash
D=scripts/analyses/clevrer_intervention
bash $D/export_predictive_latents.sh       # 形式化 predictive latent 轨迹(观测帧 96..124 + 预测帧 128..156)
bash $D/eval_validation_probe_mask_qa.sh   # baseline / relevant_mask / irrelevant_mask
```

前置:CLEVRER AWM 的 `train_probe` 与 `qa_eval` 都已完成。
结果在 `.../eval_validation_probe_mask_qa/validation_probe_mask_qa.json` 的
`metrics.{baseline,relevant_mask,irrelevant_mask}`。

### 9.3 `clevrer_relation_controls`(论文图 `interaction_prediction`)

```bash
D=scripts/analyses/clevrer_relation_controls
bash $D/extract_shards.sh    # 8 卡导出 pair 级 Entity/Dynamic/Relation 特征
bash $D/merge_shards.sh      # 合并 shard 并拟合各成分 readout
```

`merge_shards` 读的是 `{experiment_root}/extract_shards/shards`,所以必须先跑前者。
结果在 `.../merge_shards/metrics.json` 的 `metrics.<variant>.{contact.auroc,ttc.mae}`。

### 9.4 `paper_figures`(论文图 `entity_intervention` / `interaction_prediction`)

```bash
bash scripts/analyses/paper_figures/render_all.sh
```

读 9.3 的 `metrics.json` 与 Physion++ V-JEPA 2 的
`evaluate_ocp_interventions/ocp_interventions.json`,输出两张 PDF。
**需要 `matplotlib`**(`pip install matplotlib`),缺库时明确报 ImportError,不会写空文件。
论文的架构图与四张定性 `entity_grounding_*` 图是手绘/定性可视化,不由脚本生成。

### 9.5 论文表格 / 图 → task 对照

| 论文表格 / 图 | 产出 task |
|---|---|
| `main_results` | §4/§5/§6 三个主实验 + §8 全部基线 |
| `physion_slot_diagnostics` | `hasp_offline_analysis:physion_slot_time_probe`;`vjepa2_physionpp_dynamics_decoder_seed239:evaluate_interpretability` |
| `native_factor_information`, `native_factors` | `physion_factor_probe` |
| `relation_information`, `future_evidence_pair` | `physion_pair_probe` |
| `dynamic_information`, `granularity_specificity` | `physion_dynamic_selectivity` |
| `clevrer_future_events`, `clevrer_future_predictability` | `clevrer_dynamic_selectivity` |
| `input_temporal_intervention`, `input_object_intervention` | `physion_input_ablation` |
| `relation_selective_intervention`, `relation_intervention` | `physion_relation_instance_ablation` |
| `physion_future_input`, `physion_future_semantics` | `physion_predictor_regen_ablation` |
| `clevrer_relation_interaction` | `clevrer_relation_instance_ablation` |
| `query_intervention` | `clevrer_intervention:eval_validation_probe_mask_qa` |
| `physion_confusion_intervention`, `physion_logit_intervention` | `evaluate_ocp_interventions` |
| Figure `entity_intervention` | `paper_figures:render_all` |
| Figure `interaction_prediction` | `paper_figures:render_all` |
| `entity_information` | `entity_specialization` |
| `cross_task_reuse` | `probe_matrix` |
| Figure `Ilustration_awm`, `entity_grounding_*` | 手绘 / 定性可视化,无脚本 |

> 该对照是**结构性映射**,不是逐格数值的自动化复现:每个数字仍需跑完对应 task
> 后从 JSON 里读取并核对。

另有三个不进入论文表格/图的探索性 study,**未接入 launcher**,只能直接
`python -m` 调用(见各自 `README.md`):
`src/studies/object_slot_generalization/{v1,v2,v3}`、
`src/studies/clevrer_interaction_conditioned_future`。

---

## 10. 全部 config 与 task 清单

| config | task(按推荐顺序) |
|---|---|
| `physionpp/awm_physionpp_fullpatch_probe_seed239` | `train_predictor` → `cache_train`/`cache_readout`/`cache_test` → `train_probe` → `eval_ocp`,`eval_ocp_calibrated` |
| `clevrer/awm_clevrer_fullpatch_probe_seed239` | `prepare_targets_train`/`prepare_targets_validation` → `cache_latents_train`/`cache_latents_validation` → `train_probe` → `eval_probe`,`qa_eval`,`inspect_scene` |
| `clevrer/awm_clevrer_current_only_seed239` | 同 CLEVRER 主实验,不含 `qa_eval`/`inspect_scene` |
| `ek100/awm_ek100_multiscale_adapter_seed239` | `prepare_indices_train`/`prepare_indices_validation` → `cache_train`/`cache_probe_validation`/`cache_final_validation` → `train_readout` → `train_adapter` → `evaluate` |
| `ek100/awm_ek100_fair_attribution_seed239` | `train_e0`…`train_e4` → `evaluate` → `parameter_report` |
| `physionpp/vjepa2_physionpp_dynamics_decoder_seed239` | `prepare_targets`/`prepare_targets_validation` → `cache_latents`/`cache_latents_readout`/`cache_latents_test` → `train_probe` → `train_ocp_visual_only` → `train_ocp_readout` → `evaluate_interpretability`,`evaluate_ocp_interventions` |
| `clevrer/vjepa2_clevrer_naive_world_model_seed239` | `train` → `qa_eval` |
| `clevrer/vjepa2_clevrer_dynamics_decoder_seed239` | `prepare_targets_*` → `cache_latents_*` → `train_probe` → `eval_probe` |
| `clevrer/vjepa2_clevrer_dynamics_decoder_all_seed239` | `prepare_targets_*` → 4 个 cache → `train_probe` |
| `ek100/vjepa2_ek100_dynamics_decoder_seed239` | `prepare_targets` → `cache_latents` → `train_decoder` → `evaluate_decoder` |
| `physionpp/orca_physionpp_oracle_ocp_seed239` | `cache_train`/`cache_readout`/`cache_test` → `train_ocp` |
| `ek100/orca_ek100_frozen_readout_seed239` | `cache_latents_train`/`cache_latents_validation` → `train_readout` |
| `physionpp/videomae2_physionpp_oracle_ocp_seed239` | `cache_train`/`cache_readout`/`cache_test` → `train_ocp` |
| `ek100/videomae2_ek100_finetune_seed239` | `prepare_indices_*` → `cache_features`/`cache_features_probe_validation`/`cache_features_final_validation` → `train`,`finetune` |
| `analyses/hasp_offline_analysis` | 15 个 task,见 §9.1 |
| `analyses/clevrer_intervention` | `export_predictive_latents` → `eval_validation_probe_mask_qa` |
| `analyses/clevrer_relation_controls` | `extract_shards` → `merge_shards` |
| `analyses/paper_figures` | `render_all` |

---

## 11. 验证与排错

发布前/改动后应跑的自检:

```bash
cd makeup                                    # 本包根目录(所有相对路径的基准)
python tools/import_check.py                 # 全模块导入
python -m unittest discover -s tests -t .    # unittest 测试
python tools/run_pytest_style_tests.py       # pytest 风格的测试函数(环境无 pytest)
python tools/check_task_defaults.py          # 每个 task 的 cli_defaults 模块都接受、模板都能渲染
for s in $(find scripts -name '*.sh'); do bash "$s" --dry-run >/dev/null || echo "FAIL $s"; done
```

常见问题:

- **`--dry-run` 通过但真实运行缺文件**:`--dry-run` 不校验数据存在。按链条顺序确认
  上游 task 已完成,错误信息里会给出期望路径。
- **`resume requested but no checkpoint found`**:`--resume` 只按
  `launch.resume.checkpoint` → `<task>/checkpoints/latest.pt` → `.../best.pt` 查找,
  不扫描目录。先跑完上游,或在 config 里显式写死。
- **`output already exists`**:默认 `archive` 会把旧目录移到同级 `legacy/` 加时间戳;
  `raise` 直接失败;`delete` 会删除解析出的 task 输出目录(有防误删保护)。想保留旧结果
  请换 experiment 名。
- **复用已有产物**:把 `launch.extra_config.paths` 里的 run 根指到已有目录,先
  `--dry-run` 核对派生路径。
- **改了数据协议 / 模型结构**:按 `docs/manual/requirement.md` §3 新建 experiment,
  不要改现有 config 的语义。

---

## 12. 当前环境跑不了什么

`docs/manual/resource.md` 记录的是实测状态。就本机而言:

- **无可用 GPU**:所有 GPU 端到端 smoke test 与真实训练/评测均未执行;
- **EK100 / Orca-4B 数据与权重缺失**:相关实验无法在本机运行;
- **论文引用的历史 run 产物不存在**:离线分析需先在本机跑出对应 cache/probe;
- **`matplotlib` 未安装**:`paper_figures` 的渲染需要先安装;
- **VideoMAE v2 需要手动克隆上游仓库**(见 §8.3)。
