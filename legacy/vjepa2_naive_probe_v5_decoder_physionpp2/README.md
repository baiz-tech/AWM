# V-JEPA2 Physion++ full-patch dynamics decoder v1

This recipe is the Physion++ counterpart of the CLEVRER structured decoder.
It freezes the V-JEPA encoder and native Physion++ predictor, trains a shared
`UniversalDynamicsDecoder` plus a Physion++ structured probe, and supervises
segmentation-derived 2D object boxes, 2D motion, pair distance, collision,
first-contact, and scene-level OCP.

The implementation is isolated under `scripts/physionpp/`; the copied
`scripts/clevrer/` files are historical references and are not used by this
recipe.

## Smoke test and training

```bash
python -m recipe.vjepa2_naive_probe_v5_decoder_physionpp2.scripts.physionpp.prepare_targets \
  --dataset-root /data/ABDUCTIVE-WORLD/physion_v2/extracted \
  --split data_v1 --output /tmp/physionpp2_targets.pt --max-videos 2

bash recipe/vjepa2_naive_probe_v5_decoder_physionpp2/scripts/physionpp/run_all.sh \
  --output-mode workspace --dry-run
```

`run_all.sh` performs target generation, frozen encoder/predictor latent
caching, and 8-GPU probe training in the background. The default predictor is
`outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/best.pt`.

GT segmentation boxes are generated from paired `_id.json` RLE masks, not by
thresholding compressed RGB frames. The probe visualization uses H.264 output.

This recipe trains a `decoder_0810`-style universal dynamics decoder. It
produces `z_dyn [B,8,8,256]`; CLEVRER shallow probes read only `z_dyn` and use
the existing object, trajectory, distance, contact, and first-contact labels.

Each 128-frame video is exported as three non-overlapping samples by default:
window starts `0,32,64`. Their current/future ranges are `0..30 → 32..62`,
`32..62 → 64..94`, and `64..94 → 96..126` with raw frame step 2. Override
`WINDOW_STARTS` when a different window policy is required.

This experiment freezes the existing naive CLEVRER world model, caches its
complete `16x16` patch grid, trains an object-centric structured probe, and
exports the supervised object, object-time, pair, and pair-time predictions into
the shared CLEVRER v2 QA pipeline.

The detailed architecture, target definitions, losses, and QA integration are
documented in [`docs/probe_architecture.md`](docs/probe_architecture.md).

World-model checkpoint:

```text
/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/
clevrer_vith_16to16_stride2_8gpu/best.pt
```

The probe receives context and predicted-future tensors with shape
`[B,8,256,1280]` each. Six object queries attend to all 4096 context/future
patch tokens. Hungarian matching removes CLEVRER object-id ordering ambiguity;
trajectory validity, object presence, attributes, pair validity, contact, and
first-contact/no-contact are supervised explicitly.

## Output modes

Every public stage accepts `--output-mode both|data|workspace`; the default is
`both`.

- `both`: large caches/checkpoints go to `/data`; workspace keeps
  logs and small JSON/manifest files.
- `data`: complete output goes only to `/data`.
- `workspace` (default): complete output goes only to `outputs/runs/`.

Inspect paths without starting work:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/run_all.sh \
  --output-mode workspace --dry-run
```

Run target generation, full-patch cache export, and structured-probe training
sequentially in one background job:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/run_all.sh \
  --output-mode workspace
```

The full cache is expected to require roughly 157 GB for train+validation.
At the time this recipe was added, `/data` had only about 101 GB free, so
`workspace` is the runnable choice until `/data` has enough capacity. The cache
stage uses 8 GPUs by default and is resumable:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/cache_full_latents.sh \
  --output-mode workspace --resume
```

After `clevrer_structured_probe_seed239_stable_v2/best.pt` exists, inspect and start QA:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/qa_eval_8gpu.sh \
  --output-mode workspace --dry-run

bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/qa_eval_8gpu.sh \
  --output-mode workspace
```

The QA readout sequence is `CLS + 256 current/future visual tokens + 357
structured probe tokens + question/choice tokens`. Its run and evaluation names
are `clevrer_vith_16to16_v2_structured_supervised_route_8gpu` and
`qa_structured_supervised_route`, so it does not overwrite the earlier gated
structured readout.

For a minimal smoke test, use a new temporary output root and small scene
limits, for example `MAX_SCENES=8`, `MAX_TRAIN_SCENES=8`,
`MAX_VALIDATION_SCENES=8`, `EPOCHS=1`, and `NUM_GPUS=1`.

## Probe evaluation

All three launchers run in the background and support
`--output-mode both|data|workspace`. Evaluate matched object, trajectory, pair,
contact, and first-contact metrics on all 5,000 validation scenes:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/evaluate_structured_probe.sh \
  --output-mode workspace
```

Create a deterministic random scene report:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/report_structured_probe_sample.sh \
  --output-mode workspace
```

Create the corresponding report, source video, and H.264 overlay:

```bash
bash recipe/vjepa2_naive_probe_v5_decoder/scripts/clevrer/visualize_structured_probe_sample.sh \
  --output-mode workspace
```

Set `SCENE_ID=10000` to inspect a chosen validation scene. Set `MAX_SCENES`
for a smaller numerical evaluation. Predictions are aligned once per scene
with the same assignment used during training; visualisation does not rematch
objects independently at each frame.
