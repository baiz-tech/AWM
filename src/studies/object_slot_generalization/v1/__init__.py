"""Independent object-slot generalization experiments.

This package intentionally does not import or modify the V12/V23 training
implementations.  It provides a context-only, competitive slot extractor and
the diagnostics needed to test whether slots are spatially grounded,
temporally persistent, and transferable.
"""

from .model import ObjectSlotModel, SlotOutput

__all__ = ["ObjectSlotModel", "SlotOutput"]
