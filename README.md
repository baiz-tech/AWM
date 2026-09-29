# Abductive World Modeling (AWM)

Official implementation and experiment suite for **Abductive World Modeling via Causal Representation Learning**.

AWM turns a predictive future into structured evidence about the present. Given an observed video context, a frozen V-JEPA 2 backbone first predicts a future latent state. AWM then reasons jointly over the current and predicted representations and infers a structured state that explains how the scene evolves.

## Paper and resources

- **Paper:** [Abductive World Modeling via Causal Representation Learning](docs/paper/awm_paper.pdf)
- **Main figure:** [HASP architecture](docs/figures/awm_architecture.pdf)
- **Code:** this repository

The paper studies physical prediction, event reasoning, and action understanding. The released code contains the corresponding training, evaluation, intervention, and paper-figure pipelines.

## Method

The central principle is **predict forward, then abduce backward**:

1. A predictive video backbone maps the observed context to a latent future prediction.
2. The **Entity Attributor** organizes visual evidence into entity-level states: what exists.
3. The **Dynamic Attributor** adds temporal evidence: how each entity changes.
4. The **Relation Attributor** reasons over entity pairs: how entities interact.
5. The resulting **Hierarchical Abductive State Pyramid (HASP)** is used by lightweight task readouts for downstream prediction and reasoning.

The hierarchy preserves lower-level information while adding higher-order structure. This makes the representation inspectable and enables targeted interventions on entity, dynamic, and relation evidence.

![AWM overview](docs/figures/awm_architecture.pdf)

## Reported tasks

| Task | Dataset | Main AWM result |
| --- | --- | --- |
| Physical outcome prediction | Physion++ | +10.7% AUROC over the V-JEPA 2 backbone-only baseline |
| Event reasoning | CLEVRER | +16.8% question accuracy |
| Action understanding | EPIC-KITCHENS-100 (EK100) | +68.0% action Top-1 accuracy |

These numbers follow the protocols described in the paper and are reproduced by the experiment launchers in this repository.

## Repository layout

```text
configs/       Reproducible experiment and analysis configurations
scripts/       Dataset-specific launchers and analysis entry points
src/           Models, datasets, training, evaluation, and figure renderers
docs/exp/      Environment, data, checkpoint, and reproduction instructions
external/      Vendored V-JEPA 2 components and their upstream license
```

## Quick start

The supported environment is Linux with Python 3.12 and a CUDA-capable NVIDIA GPU. Create an isolated environment and install the pinned framework versions plus repository dependencies:

```bash
conda create -n awm python=3.12 pip -y
conda activate awm
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The repository does not redistribute datasets, V-JEPA 2 weights, or CLEVRER predictor checkpoints. Configure their local paths in a private copy of the relevant YAML configuration. Start with the full reproduction guide:

```bash
less docs/exp/README.md
```

Every launcher supports `--dry-run`; use it before submitting a long GPU job:

```bash
bash scripts/physionpp/awm_physionpp_fullpatch_probe_seed239/train_predictor.sh --dry-run
bash scripts/clevrer/awm_clevrer_fullpatch_probe_seed239/prepare_targets_train.sh --dry-run
bash scripts/ek100/awm_ek100_multiscale_adapter_seed239/train_adapter.sh --dry-run
```

## Paper figures

The two data-driven figures can be regenerated after their upstream metrics are available:

```bash
bash scripts/analyses/paper_figures/render_all.sh --dry-run
bash scripts/analyses/paper_figures/render_all.sh
```

See [`src/studies/paper_figures/README.md`](src/studies/paper_figures/README.md) for input JSON schemas, protocols, and figure-specific commands. The architecture diagram and qualitative grounding panels are distributed as paper assets; the intervention and relation-control plots are rendered from recorded metrics.

## Citation

```bibtex
@inproceedings{liu2027abductive,
  title     = {Abductive World Modeling via Causal Representation Learning},
  author    = {Liu, Ziqi and Yang, Songhan and Zhou, Linfan and Liu, Jiatong
               and Peng, Lijun and Wan, Long and Bai, Yinqi},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```

## License

This repository is released under the Apache License 2.0; see [LICENSE](LICENSE). Vendored V-JEPA 2 code retains its upstream license in [`external/vjepa2/LICENSE`](external/vjepa2/LICENSE).
