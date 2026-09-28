# V23-style shared/multiscale adapter — EK100 only

This is the isolated EK100 counterpart of the historical V23 series.  It does
not perform Physion++ joint training and does not load V12/V20/V23 task
weights.  Only the generic V-JEPA encoder is imported; the native predictor,
EK100 cache, multiscale readout, temporal/shared adapter, and verb/noun/
action heads are trained from scratch on EK100 annotations only.

The protocol is:

```text
EK100 train annotations
  -> train-only encoder cache
  -> EK100 multiscale readout pretraining
  -> EK100-only shared temporal adapter training
  -> official EK100 validation evaluation
```

The historical joint V23 result is retained only as a reference, not as an
initialization target.  Report this run separately from the joint V23 number.
