"""Top-level runner — dispatches `(task, strategy)` to the right code path.

`run(...)` is the function the CLI and Python users call. Tasks register
themselves by importing — see `tasks/__init__.py` for the registry.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any, Literal

from .adapter import load_encoder

_TASKS_REGISTRY: dict[str, Any] = {}        # task_name -> module


def register_task(name: str, module: Any) -> None:
    """Called by `tasks/<name>.py` at import time."""
    _TASKS_REGISTRY[name] = module


def list_tasks() -> list[str]:
    """All registered task names."""
    _ensure_tasks_loaded()
    return sorted(_TASKS_REGISTRY)


def list_strategies() -> list[str]:
    return ["lp", "ft"]


def _ensure_tasks_loaded() -> None:
    """Import every module under `eegfm_eval.tasks` so they self-register."""
    if _TASKS_REGISTRY:
        return
    import pkgutil
    from . import tasks
    for mod in pkgutil.iter_modules(tasks.__path__):
        if mod.name.startswith("_"):
            continue
        importlib.import_module(f"eegfm_eval.tasks.{mod.name}")


def run(
    *,
    task: str,
    strategy: Literal["lp", "ft"] = "lp",
    checkpoint: Path | str | None = None,
    encoder_kind: str = "eegfm",
    derived_root: Path | str | None = None,
    output: Path | str | None = None,
    device: str = "cuda",
    seed: int = 0,
    **task_opts,
) -> dict:
    """Run one (task, strategy) atom against an encoder.

    Args:
        task: registered task name (e.g. 'tuab', 'tuev', 'physionet_mi').
        strategy: 'lp' or 'ft'.
        checkpoint: path to encoder checkpoint (None for `random_init` etc).
        encoder_kind: which adapter factory ('eegfm', 'random_init', 'labram_base', ...).
        derived_root: where preprocessed parquet shards live for the task.
        output: optional file to write the result JSON to.
        device, seed, **task_opts: passed to encoder + task as appropriate.

    Returns the result dict.
    """
    _ensure_tasks_loaded()
    if task not in _TASKS_REGISTRY:
        raise ValueError(f"unknown task {task!r}; valid: {list_tasks()}")
    task_mod = _TASKS_REGISTRY[task]

    encoder = load_encoder(encoder_kind, checkpoint=checkpoint, device=device)

    t0 = time.time()
    result = task_mod.run(encoder, derived_root=Path(derived_root) if derived_root else None,
                          strategy=strategy, device=device, seed=seed, **task_opts)
    result["task"] = task
    result["strategy"] = strategy
    result["encoder"] = {
        "name": encoder.spec.name,
        "kind": encoder_kind,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "n_params": encoder.spec.n_params,
        "pretraining": encoder.spec.pretraining,
    }
    result["lit_anchors"] = getattr(task_mod, "LIT_ANCHORS", {})
    result["wallclock_s"] = round(time.time() - t0, 2)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2, default=str))
    return result
