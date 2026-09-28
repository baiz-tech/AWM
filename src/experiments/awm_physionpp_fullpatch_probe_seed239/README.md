# V12 full-patch structured probe — Physion++ only

This is an isolated reproduction entrypoint for the best V12/full-patch
structured Physion++ experiment.  It uses only Physion++ videos/targets and
randomly initializes the native predictor, structured probe, and OCP readout.
The only external weight allowed is the generic V-JEPA ViT-H encoder.

Data protocol:

```text
data_v1          -> structured-probe train
readout_data_v1  -> checkpoint/readout selection
testdata_v1      -> final OCP evaluation
```

The historical result to reproduce was approximately 72.19% OCP AUROC.  This
directory does not load any V12/V20/V23 task checkpoint.

The implementation is intentionally staged.  `config.yaml` is the single
source of paths and protocol values; `scripts/run_8gpu.sh` is a dry-run-safe
launcher and will be connected to the local training/evaluation modules in the
next implementation step.
