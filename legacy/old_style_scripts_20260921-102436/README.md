# Archived pre-migration shell scripts

These 77 `.sh` files were copied verbatim from the internal working repository
during the `makeup/` migration. They are the **old-style** entry points: each one
calls `torchrun` / `python -m recipe.recipe_formal.*` directly and hardcodes
machine paths and output directories.

They are superseded by the launcher-based entry points:

```text
configs/<dataset>/<experiment>/config.yaml     experiment definition
scripts/<dataset>/<experiment>/<task>.sh       thin launcher (CONFIG + TASK)
tools/launcher/launch_from_config.sh           shared launcher
```

Nothing in the active tree references them (`grep -rn "src/experiments/.*/scripts"
tests/ configs/ tools/ src/core src/data src/training` is empty), and the module
names they invoke (`recipe.recipe_formal.*`) no longer exist in this repository.
They are kept here only as a record of how the original runs were launched.

The 18 scripts that already delegate to `tools/launcher/launch_from_config.sh`
were **not** moved; they remain next to their experiments as variant entry points.
