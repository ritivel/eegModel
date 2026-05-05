"""Task registry — every module under `tasks/` self-registers when imported.

Add a new task: drop a `tasks/<name>.py` that:
  1. Defines `NAME`, `TASK_TYPE` ('binary' | 'multiclass' | 'regression'),
     `NUM_CLASSES`, `LIT_ANCHORS` (dict of model_name → published primary metric),
     `SAMPLE_RATE`, `WINDOW_SAMPLES`, `CHANNELS` (list of channel names; None for iid).
  2. Implements `run(encoder, derived_root, *, strategy, device, seed, **opts) -> dict`.
  3. Calls `register_task(NAME, sys.modules[__name__])` at the bottom.

That's it — no class hierarchy, no plugin metadata, no decorators.
"""

from __future__ import annotations
