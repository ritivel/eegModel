"""smoke — synthetic-data task, used to validate the harness wiring.

No real data on disk required. Generates random Gaussian "EEG" + random
binary labels, runs the encoder on it, and reports linear-probe accuracy.

Used by:

    pytest src/eegfm_eval/tests/test_smoke.py
    eegfm-eval --random-init --task smoke --strategy lp

A *correctly wired* harness should report `~0.50 ± 0.05` BAC on this task
(random labels by construction). Anything else means the wiring is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..adapter import Encoder
from ..runner import register_task

NAME = "smoke"
TASK_TYPE = "binary"
NUM_CLASSES = 2
PRIMARY_METRIC = "bac"
LIT_ANCHORS: dict[str, float] = {}  # synthetic — no published baseline
SAMPLE_RATE = 500
WINDOW_SAMPLES = 2000


class _SyntheticDataset(Dataset):
    """Random Gaussian (B, 1, T) samples with random binary labels."""

    def __init__(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        self.x = rng.standard_normal((n, 1, WINDOW_SAMPLES)).astype(np.float32)
        self.y = (rng.standard_normal((n,)) > 0).astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.x[i]), int(self.y[i])


def run(
    encoder: Encoder,
    derived_root: Path | None = None,
    *,
    strategy: str = "lp",
    device: str = "cuda",
    seed: int = 0,
    n_train: int = 200,
    n_test: int = 100,
    **_unused,
) -> dict:
    train = _SyntheticDataset(n=n_train, seed=seed)
    test = _SyntheticDataset(n=n_test, seed=seed + 1)

    if strategy == "lp":
        from ..strategies import linear_probe
        return linear_probe(encoder, train, test,
                             num_classes=2, task_type="binary",
                             batch_size=32, device=device, seed=seed,
                             num_workers=0, n_bootstrap=200)
    if strategy == "ft":
        from ..strategies import fine_tune
        return fine_tune(encoder, train, test, test,
                          num_classes=2, task_type="binary",
                          batch_size=16, epochs=2, lr=1e-3,
                          device=device, seed=seed, num_workers=0,
                          n_bootstrap=200)
    raise NotImplementedError(f"smoke supports lp|ft, not {strategy!r}")


register_task(NAME, sys.modules[__name__])
