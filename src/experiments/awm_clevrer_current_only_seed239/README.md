# CLEVRER current-only predictability

This experiment trains the dynamics decoder and `CLEVRERShallowProbes` from
current latent only. The probes predict the same future targets as the baseline:
`s+32, s+34, ..., s+62` for each window start `s in {0,32,64}`.

The experiment reuses the baseline cache and targets. Training reads only
`context_tokens` from each baseline cache file and never reads its
`future_tokens`; no new latent cache is generated. The decoder receives
`[B,8,256,1280]`, builds a `[B,2048,256]` memory, and produces
`z_dyn [B,8,8,256]` for the unchanged structured probe heads.

Run the complete pipeline with:

```bash
bash recipe/recipe_formal/reproduce_v1_clevrer_only_analysePredictability/scripts/clevrer/run_all.sh --output-mode workspace --dry-run
```
