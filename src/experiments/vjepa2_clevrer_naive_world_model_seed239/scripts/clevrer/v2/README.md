# Naive CLEVRER QA v2

Inspect and launch the visual-only v2 QA pipeline from the repository root:

```bash
bash recipe/vjepa2_naive/scripts/clevrer/v2/qa_eval_8gpu.sh --dry-run
bash recipe/vjepa2_naive/scripts/clevrer/v2/qa_eval_8gpu.sh
```

The launcher accepts `--output-mode both|data|workspace` (default: `workspace`).
`both` keeps the large trajectories and QA checkpoints in `/data` and mirrors
only logs, JSON results, and status metadata to the workspace. `data` writes
only to `/data`; `workspace` writes the complete QA run under `outputs/runs/...`.

This run exports `[16,256,1280]` tokens from the last current window and one
predicted future window, then uses 16 learned spatial tokens per temporal step.
It does not load or provide any explicit probe output.
