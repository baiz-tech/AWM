# V-JEPA2 v5 decoder on EPIC-KITCHENS-100

This recipe keeps the v5 `UniversalDynamicsDecoder` trunk and replaces the
copied Physion++ probe with EPIC-KITCHENS/VISOR supervision.

## Protocol

For every EPIC action event, the annotation midpoint separates current and
future. Current contains 16 uniformly sampled frames from the event first half,
limited to the four seconds before the midpoint. Future contains 16 uniformly
sampled supervision times from the event second half. Only current RGB is fed to
the frozen model:

```text
current RGB -> official V-JEPA2 encoder -> context latent
context latent -> official V-JEPA2 predictor -> predicted-future latent
context + predicted future -> unchanged v5 decoder -> z_dyn [B,8,8,256]
```

The encoder and predictor are both loaded from
`/data/ABDUCTIVE-WORLD/pretrain/checkpoints/vith.pt` and frozen. The checkpoint
contains ten native predictor mask tokens, so the EK100 config uses
`num_mask_tokens: 10` and requires strict key/shape compatibility.

## Supervision

Aligned VISOR polygons provide eight object slots with presence, VISOR category,
16-step normalized center/area/bounding-box/velocity state, and pair distance.
EPIC event annotations provide independent verb, noun, and observed training
action-pair labels. VISOR categories and EPIC nouns are separate vocabularies.
There are no synthetic collision, first-contact, or OCP labels.

Training uses only events with usable future VISOR labels. Train and validation
follow `EPIC_100_train.csv` and `EPIC_100_validation.csv`; action vocabulary is
built from train only, and unknown validation action pairs are masked.

## Run

Inspect all resolved paths and commands:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder_ek100/scripts/epic100/run_all.sh \
  --output-mode both --dry-run
```

Launch target generation, frozen latent caching, and 8-GPU training in the
background:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder_ek100/scripts/epic100/run_all.sh \
  --output-mode both
```

Use a new `RUN_NAME` for a new experiment. `--resume` resumes cache generation
and the same run's `latest.pt`. Set `MAX_EVENTS` only for smoke tests.

Output modes follow the repository convention:

- `both`: large artifacts under `/data/shyang/outputs/vjepa2-baiz/...`, small logs in `outputs/runs/...`;
- `data`: complete output under `/data`;
- `workspace`: complete output under `outputs/runs/`.

The copied `scripts/physionpp/` and `scripts/clevrer/` directories are legacy
source material and are not part of the EK100 entry point. New work should use
only `scripts/epic100/`.
