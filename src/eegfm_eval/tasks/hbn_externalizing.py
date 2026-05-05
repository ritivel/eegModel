"""HBN externalizing factor — NeurIPS 2025 EEG Foundation Challenge 2.

Per-subject regression: predict the CBCL-derived externalizing factor from
all of a subject's EEG (any task, any channel, aggregated subject-level).

Source of truth: https://eeg2025.github.io/ + arXiv:2506.19141.

Splits (release-based, subject-disjoint by construction):
    train: HBN releases R1, R2, R3, R4, R6, R7, R8, R9, R10, R11
    val:   R5
    test:  R12 (withheld until further notice; if not present, we use R5 for both val and test
                with a clear flag in the result so it's not mistaken for a clean test number)

Metric: nRMSE = sqrt(mean((y_true - y_pred)^2)) / std(y_true)
    with population std (ddof=0). Mean-predict baseline = 1.0.

Lit anchors (all FT, per-subject):
    JLShen winner               nRMSE 0.97843
    MBZUAI [dsml.kz]            nRMSE 0.98519
    MIN~C² (MIND-CICO)          nRMSE 0.98817
    Mean-predict baseline       nRMSE 1.000
    (Only 3 teams of 1183 broke 0.99 — this task is hard.)

Strategy:
    - LP: extract features for all of a subject's windows, mean-pool into one
      feature vector per subject, then RidgeCV on the externalizing scalar.
    - FT: train per-window with a regression head, aggregate test-time
      predictions by mean-pooling across each subject's windows.

Loading is via the `eegdash` library — it provides an `EEG2025R{n}` class
per release that returns Braindecode-compatible BaseConcatDatasets, with
HBN's BIDS events.tsv and participants.tsv pre-parsed.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .._common import compute_metrics
from ..adapter import Encoder
from ..runner import register_task

NAME = "hbn_externalizing"
TASK_TYPE = "regression"
NUM_CLASSES = 1
PRIMARY_METRIC = "nrmse"
LIT_ANCHORS: dict[str, float] = {
    "neurips2025_winner_jlshen_nrmse": 0.97843,
    "neurips2025_2nd_mbzuai_nrmse": 0.98519,
    "neurips2025_3rd_minc2_nrmse": 0.98817,
    "mean_predict_baseline_nrmse": 1.000,
}

SAMPLE_RATE = 100                 # competition data is downsampled to 100 Hz
WINDOW_S = 4
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_S    # 400 samples

# Per-subject prediction; per-window encoding then subject-mean-pool aggregate.
TRAIN_RELEASES = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
VAL_RELEASES = [5]
TEST_RELEASES = [12]              # withheld; we'll fall back to R5 if unavailable


# ---------------------------------------------------------------------------
# Dataset — each item is one window with subject-level externalizing label
# ---------------------------------------------------------------------------


class HBNExternalizingDataset(Dataset):
    """Lazy-loads windows via EEGDash; aggregates labels at subject level.

    `releases`: list of int release IDs to include.
    `cache_dir`: where EEGDash caches the BDFs (~30-100 GB per release).
    `tasks`: which HBN task names to include (default = all 6).
    `max_subjects`: cap subjects (None = all).
    """

    def __init__(
        self,
        releases: list[int],
        cache_dir: Path,
        *,
        tasks: tuple[str, ...] = (
            "RestingState", "surroundSupp",
            "DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent",
            "contrastChangeDetection",
            "seqLearning8target", "seqLearning6target", "symbolSearch",
        ),
        max_subjects: int | None = None,
    ):
        from eegdash import dataset as eegdash_dataset       # type: ignore[import-untyped]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.releases = releases

        # Concatenate one EEGDash dataset per release.
        all_subj_ds: list = []
        for r in releases:
            cls_name = f"EEG2025R{r}"
            cls = getattr(eegdash_dataset, cls_name, None)
            if cls is None:
                raise RuntimeError(f"eegdash has no {cls_name}; install eegdash>=0.6")
            ds_r = cls(cache_dir=str(self.cache_dir),
                       query={"task": {"$in": list(tasks)}})
            all_subj_ds.extend(ds_r.datasets)
        if max_subjects is not None:
            all_subj_ds = all_subj_ds[: max_subjects * 6]  # rough cap (6 tasks/subj)

        # Flatten to (subject_id, raw, externalizing_label) tuples.
        self._items: list[tuple[str, "object", float]] = []
        for sub_ds in all_subj_ds:
            sub_id = getattr(sub_ds, "subject", None) or sub_ds.description.get("subject")
            extn = sub_ds.description.get("externalizing", float("nan"))
            if sub_id is None or np.isnan(extn):
                continue
            self._items.append((str(sub_id), sub_ds, float(extn)))

        # Pre-window each recording lazily on access. Index = (item_idx, window_idx).
        self._index: list[tuple[int, int]] = []
        self._n_windows_per_item: list[int] = []
        for i, (_sub, ds, _y) in enumerate(self._items):
            raw = ds.raw if hasattr(ds, "raw") else ds.load()
            n_samp = raw.n_times
            n_w = n_samp // WINDOW_SAMPLES
            self._n_windows_per_item.append(n_w)
            for w in range(n_w):
                self._index.append((i, w))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        item_idx, w = self._index[idx]
        sub, ds, y = self._items[item_idx]
        raw = ds.raw if hasattr(ds, "raw") else ds.load()
        x = raw.get_data(start=w * WINDOW_SAMPLES, stop=(w + 1) * WINDOW_SAMPLES)
        x = np.nan_to_num(x.astype(np.float32) * 1e6, nan=0.0)   # V → µV
        return torch.from_numpy(x), float(y), sub


# ---------------------------------------------------------------------------
# Subject-level aggregation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _extract_per_subject(
    encoder: Encoder, ds: HBNExternalizingDataset, *, batch_size: int, device: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the encoder, mean-pool features by subject. Returns (X, y, subj_ids)."""
    encoder.eval()
    encoder.to(device)
    feats_by_sub: dict[str, list[np.ndarray]] = defaultdict(list)
    label_by_sub: dict[str, float] = {}
    # Iterate without DataLoader to keep the per-window subject mapping tight.
    for i in range(0, len(ds), batch_size):
        rows = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        x = torch.stack([r[0] for r in rows]).to(device, non_blocking=True)
        ys = [r[1] for r in rows]
        subs = [r[2] for r in rows]
        f = encoder.encode(x).detach().cpu().numpy().astype(np.float32)
        for k in range(f.shape[0]):
            feats_by_sub[subs[k]].append(f[k])
            label_by_sub[subs[k]] = ys[k]

    subjects = sorted(feats_by_sub)
    X = np.stack([np.mean(np.stack(feats_by_sub[s], 0), axis=0) for s in subjects], axis=0)
    y = np.asarray([label_by_sub[s] for s in subjects], dtype=np.float32)
    return X, y, subjects


# ---------------------------------------------------------------------------
# Task entrypoint
# ---------------------------------------------------------------------------


def run(
    encoder: Encoder,
    derived_root: Path | None = None,
    *,
    strategy: str = "lp",
    device: str = "cuda",
    seed: int = 0,
    cache_subdir: str = "hbn_neurips2025",
    train_releases: tuple[int, ...] = tuple(TRAIN_RELEASES),
    val_releases: tuple[int, ...] = tuple(VAL_RELEASES),
    test_releases: tuple[int, ...] = tuple(TEST_RELEASES),
    max_subjects: int | None = None,
    batch_size: int = 64,
    epochs: int = 30, lr: float = 5e-4,
    n_bootstrap: int = 1000,
    **_unused,
) -> dict:
    """Run HBN externalizing eval. LP is per-subject feature mean-pool + RidgeCV."""
    if derived_root is None:
        raise ValueError("hbn_externalizing requires --derived-root for the EEGDash cache")
    cache_dir = Path(derived_root) / cache_subdir

    train_ds = HBNExternalizingDataset(list(train_releases), cache_dir, max_subjects=max_subjects)
    val_ds = HBNExternalizingDataset(list(val_releases), cache_dir, max_subjects=max_subjects)
    try:
        test_ds = HBNExternalizingDataset(list(test_releases), cache_dir, max_subjects=max_subjects)
        test_split_status = "official_R12"
    except Exception:                                         # noqa: BLE001
        print(f"[hbn_externalizing] R{test_releases} unavailable; using R{val_releases} as test (NOT clean!)")
        test_ds = val_ds
        test_split_status = "fallback_to_val (R12 not yet public)"

    if strategy != "lp":
        raise NotImplementedError(
            "hbn_externalizing only supports 'lp' for now (per-subject regression). "
            "FT requires per-window training with subject-level test aggregation — wire later.")

    Xtr, ytr, _ = _extract_per_subject(encoder, train_ds, batch_size=batch_size, device=device)
    Xte, yte, subjects_te = _extract_per_subject(encoder, test_ds, batch_size=batch_size, device=device)

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    reg = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0), cv=5)
    reg.fit(Xtr_s, ytr)
    y_pred = reg.predict(Xte_s)

    metrics = compute_metrics(yte, y_pred=y_pred, task_type="regression",
                               n_bootstrap=n_bootstrap, seed=seed)

    return {
        "strategy": "lp",
        "head": {"type": "RidgeCV", "alpha": float(reg.alpha_)},
        "n_train_subjects": int(Xtr.shape[0]),
        "n_test_subjects": int(Xte.shape[0]),
        "feature_dim": int(Xtr.shape[1]),
        "test_split_status": test_split_status,
        "train_releases": list(train_releases),
        "val_releases": list(val_releases),
        "test_releases": list(test_releases),
        "metrics": metrics,
    }


register_task(NAME, sys.modules[__name__])
