# EPIC100 decoder protocol

## Facts and invariants

- Decoder input shapes are `context/future [B,8,256,1280]`.
- The unchanged v5 trunk outputs `z_dyn [B,8,8,256]` at default width.
- Real future RGB is used only to select aligned VISOR labels; it is never
  encoded as a decoder input.
- The official encoder and predictor share the same `vith.pt` checkpoint and
  are frozen before latent generation.
- Object matching is sample-local and permutation invariant.

## Targets

`state [8,16,8]` channels are:

```text
center_x, center_y, mask_area, bbox_width, bbox_height,
velocity_x, velocity_y, log_mask_area
```

Coordinates and box dimensions are normalized by the aligned 256x256 crop.
`state_valid` excludes missing VISOR frames. Tracks are heuristic because VISOR
does not provide stable cross-frame instance IDs: same-category instances are
associated by spatial distance, then event noun, hand regions, duration, and
area determine which eight tracks are retained.

Event heads predict train-vocabulary indices for verb, noun, and observed
verb/noun action pairs. They read only `z_dyn`, as do all object heads.

## Loss

```text
L_object = presence + category + 2 center + geometry
         + 0.5 velocity + 0.1 log_area + pair_distance
L_event  = verb + noun + action
L_total  = L_object + event_weight * L_event
```

Validation checkpoint selection uses total validation loss. Formal reporting
should additionally use `evaluate_decoder.py` for event Top-1/Top-5/MCR and
matched VISOR category accuracy.
