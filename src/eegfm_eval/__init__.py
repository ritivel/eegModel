"""eegfm_eval — downstream-task evaluation harness for EEG foundation models.

Independent of `eegfm` (the pretraining package). Usage:

    from eegfm_eval import run, list_tasks, list_strategies

    # Run one task (linear-probe TUAB on a checkpoint we trained):
    result = run(
        checkpoint="path/to/ckpt.pt",
        task="tuab",
        strategy="lp",
        derived_root="/opt/dlami/nvme/eeg/derived",
    )

    # Or via the CLI:
    #   eegfm-eval --checkpoint ckpt.pt --task tuab --strategy lp

Design principles:
- one task = one self-contained file under `tasks/`, ~150-300 lines
- one strategy = one function in `strategies.py`
- one encoder source = one factory in `adapter.py`
- shared helpers (metrics, splits, bootstrap, dataloaders) live in `_common.py`
- no inheritance towers; functions over classes where reasonable
- every result is a JSON-serialisable dict with `lit_anchors` so any number we
  produce can be compared to a published baseline at a glance
"""

from __future__ import annotations

# Re-exports are lazy so that `import eegfm_eval` works without torch
# (we want to grep / inspect the package on a CPU-only laptop).
__all__ = ["Encoder", "load_encoder", "list_tasks", "list_strategies", "run"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"Encoder", "load_encoder"}:
        from .adapter import Encoder, load_encoder
        return {"Encoder": Encoder, "load_encoder": load_encoder}[name]
    if name in {"list_tasks", "list_strategies", "run"}:
        from .runner import list_strategies, list_tasks, run
        return {"list_tasks": list_tasks, "list_strategies": list_strategies, "run": run}[name]
    raise AttributeError(f"module 'eegfm_eval' has no attribute {name!r}")
