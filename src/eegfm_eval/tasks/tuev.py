"""TUEV v2.0.1 — 6-class epileptiform-event classification.

Differs from TUAB in two ways:
1. Window length is 5 s (1000 samples @ 200 Hz), not 10 s.
2. Labels come from sibling .rec files — one (channel, start, end, label)
   row per window. We do not infer labels from filename. Each .rec row
   defines exactly one training sample whose window is centred on the
   event (~2 s pre-event, ~1 s event, ~2 s post-event). LaBraM's
   `BuildEvents()` is the reference; we replicate its slicing.

6 classes (label IDs from .rec, 1-indexed in the file; we 0-index here):
    0 SPSW  — Spike and Sharp Wave
    1 GPED  — Generalized Periodic Epileptiform Discharges
    2 PLED  — Periodic Lateralized Epileptiform Discharges
    3 EYEM  — Eye Movement
    4 ARTF  — Artifact
    5 BCKG  — Background

Lit anchors (full fine-tune):
    LaBraM-Base   BAC 0.6409 ± 0.0065   κ 0.6637 ± 0.0093
    CBraMod       BAC 0.6671 ± 0.0107
    REVE-Base     BAC 0.6759 ± 0.0229
    Random-init   BAC ~0.17 (chance for 6-class)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..runner import register_task
from .tuab import CHANNELS, _select_channels_in_order, _subject_of

NAME = "tuev"
TASK_TYPE = "multiclass"
NUM_CLASSES = 6
PRIMARY_METRIC = "cohen_kappa"
LIT_ANCHORS: dict[str, float] = {
    "labram_base_bac": 0.6409,
    "labram_base_kappa": 0.6637,
    "cbramod_bac": 0.6671,
    "reve_base_bac": 0.6759,
    "random_init_floor": 1.0 / NUM_CLASSES,
}
SAMPLE_RATE = 200
WINDOW_S = 5
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_S    # 1000

CLASS_NAMES = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]


# ---------------------------------------------------------------------------


def prepare_tuev(raw_root: Path, cache_root: Path) -> dict:
    """Convert raw TUEV .edf + .rec → cached numpy arrays.

    Same shape as `prepare_tuab`. NEDC ships an official train/eval split;
    we further hold out 20% of NEDC's train *patients* for val (deterministic,
    lexicographic), giving the CBraMod-fixed-split target of 68,445 / 15,487 / 29,421.
    """
    import json
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    edf_root = Path(raw_root) / "edf"
    train_pairs = _list_tuev_edfs(edf_root / "train")
    test_pairs = _list_tuev_edfs(edf_root / "eval")

    train_subjects = sorted({_subject_of(p) for p, _ in train_pairs})
    n_val_subj = max(1, int(round(len(train_subjects) * 0.20)))
    val_subjs = set(train_subjects[:n_val_subj])
    train_subjs = set(train_subjects) - val_subjs
    test_subjs = sorted({_subject_of(p) for p, _ in test_pairs})

    print(f"[tuev] {len(train_pairs)} train EDFs ({len(train_subjs)} subjects), "
          f"holding out {len(val_subjs)} for val; {len(test_pairs)} test EDFs.")

    train_x, train_y, train_s = _process_split(train_pairs, keep_subjects=train_subjs)
    val_x, val_y, val_s = _process_split(train_pairs, keep_subjects=val_subjs)
    test_x, test_y, test_s = _process_split(test_pairs, keep_subjects=set(test_subjs))

    np.savez_compressed(cache_root / "train.npz", x=train_x, y=train_y, subj=np.array(train_s))
    np.savez_compressed(cache_root / "val.npz",   x=val_x,   y=val_y,   subj=np.array(val_s))
    np.savez_compressed(cache_root / "test.npz",  x=test_x,  y=test_y,  subj=np.array(test_s))

    manifest = {
        "spec": {
            "bandpass": [0.1, 75.0], "notch_hz": 50.0, "sample_rate": SAMPLE_RATE,
            "window_s": WINDOW_S, "window_samples": WINDOW_SAMPLES,
            "channels": CHANNELS, "norm": "÷0.1mV (i.e. *0.01)",
        },
        "split_strategy": "NEDC train → 80/20 patient-disjoint train/val (lexicographic); NEDC eval → test",
        "n_train": int(train_x.shape[0]), "n_val": int(val_x.shape[0]),
        "n_test": int(test_x.shape[0]),
        "n_train_subjects": len(train_subjs),
        "n_val_subjects": len(val_subjs),
        "n_test_subjects": len(test_subjs),
        "labram_fixed_split_target": [68445, 15487, 29421],
        "label_distribution": _per_split_label_dist(train_y, val_y, test_y),
        "class_names": CLASS_NAMES,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[tuev] cache written: {manifest['n_train']:,}/{manifest['n_val']:,}/{manifest['n_test']:,} "
          f"(target {manifest['labram_fixed_split_target']})")
    return manifest


def _list_tuev_edfs(split_root: Path) -> list[tuple[Path, Path]]:
    """Walk NEDC's TUEV layout; pair each .edf with its sibling .rec."""
    out: list[tuple[Path, Path]] = []
    for edf in sorted(split_root.rglob("*.edf")):
        rec = edf.with_suffix(".rec")
        if not rec.exists():
            continue
        out.append((edf, rec))
    return out


def _parse_rec(rec_path: Path) -> list[tuple[int, float, float, int]]:
    """One row per event: (channel_idx, start_s, end_s, label_id_0indexed)."""
    rows: list[tuple[int, float, float, int]] = []
    for line in rec_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        try:
            ch_idx, t0, t1, lbl = int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
            rows.append((ch_idx, t0, t1, lbl - 1))           # .rec uses 1-indexed labels
        except ValueError:
            continue
    return rows


def _process_split(
    pairs: list[tuple[Path, Path]],
    keep_subjects: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """For each (edf, rec): preprocess EDF, then extract one window per .rec row."""
    import mne
    mne.set_log_level("ERROR")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ss: list[str] = []
    for edf_path, rec_path in pairs:
        sub = _subject_of(edf_path)
        if sub not in keep_subjects:
            continue
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        except Exception as e:                               # noqa: BLE001
            print(f"[tuev] skip {edf_path.name}: {e}")
            continue
        if not _select_channels_in_order(raw, CHANNELS):
            print(f"[tuev] skip {edf_path.name}: missing channels")
            continue
        raw.filter(l_freq=0.1, h_freq=75.0, method="fir", verbose="ERROR")
        raw.notch_filter(freqs=[50.0], verbose="ERROR")
        raw.resample(SAMPLE_RATE, npad="auto", verbose="ERROR")
        data = raw.get_data() * 1e6 * 0.01
        data = np.nan_to_num(data, nan=0.0, posinf=10.0, neginf=-10.0)
        n_samp = data.shape[1]
        # Extract one window per .rec row, centred on the event with 2-sec
        # padding either side (LaBraM's BuildEvents convention).
        for ch_idx, t0, t1, lbl in _parse_rec(rec_path):
            mid = int(round((t0 + t1) / 2 * SAMPLE_RATE))
            half = WINDOW_SAMPLES // 2
            lo = max(0, mid - half)
            hi = lo + WINDOW_SAMPLES
            if hi > n_samp:
                hi = n_samp
                lo = max(0, hi - WINDOW_SAMPLES)
            window = data[:, lo:hi]
            if window.shape[1] != WINDOW_SAMPLES:
                continue
            xs.append(window.astype(np.float16)[None, :, :])
            ys.append(int(lbl))
            ss.append(sub)
    x = np.concatenate(xs, axis=0) if xs else np.empty((0, len(CHANNELS), WINDOW_SAMPLES), np.float16)
    y = np.asarray(ys, dtype=np.int8)
    return x, y, ss


def _per_split_label_dist(*ys: np.ndarray) -> list[dict]:
    out = []
    for y in ys:
        d = {n: int((y == i).sum()) for i, n in enumerate(CLASS_NAMES)}
        out.append(d)
    return out


# ---------------------------------------------------------------------------


class TUEVDataset(Dataset):
    """Reads pre-cached numpy arrays. Yields (x: (23, 1000) float32, y: int)."""

    def __init__(self, cache_root: Path, split: str):
        path = Path(cache_root) / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"TUEV cache missing: {path}. Run `prepare_tuev(raw_root, cache_root)` first.")
        data = np.load(path, allow_pickle=True)
        self.x = data["x"]
        self.y = data["y"]
        self.subj = data["subj"]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.x[i].astype(np.float32))
        return x, int(self.y[i])


# ---------------------------------------------------------------------------


def run(
    encoder,
    derived_root: Path | None = None,
    *,
    strategy: str = "lp",
    device: str = "cuda",
    seed: int = 0,
    cache_subdir: str = "tuev_labram_200hz",
    raw_subdir: str = "raw/tuev/v2.0.1",
    epochs: int = 50, batch_size: int = 64, lr: float = 5e-4,
    **_unused,
) -> dict:
    if derived_root is None:
        raise ValueError("tuev requires --derived-root")
    cache_root = Path(derived_root) / cache_subdir

    if not (cache_root / "train.npz").exists():
        raw_root = Path(derived_root).parent / raw_subdir
        if raw_root.exists():
            print(f"[tuev] cache missing, preparing from {raw_root}")
            prepare_tuev(raw_root, cache_root)
        else:
            raise FileNotFoundError(
                f"TUEV cache missing at {cache_root} and raw not at {raw_root}. "
                f"Run prepare_tuev(...) manually or pre-stage the cache.")

    train_ds = TUEVDataset(cache_root, "train")
    val_ds = TUEVDataset(cache_root, "val")
    test_ds = TUEVDataset(cache_root, "test")

    if strategy == "lp":
        from ..strategies import linear_probe
        return linear_probe(encoder, train_ds, test_ds, val_ds=val_ds,
                             num_classes=NUM_CLASSES, task_type="multiclass",
                             batch_size=batch_size, device=device, seed=seed)
    if strategy == "ft":
        from ..strategies import fine_tune
        return fine_tune(encoder, train_ds, val_ds, test_ds,
                          num_classes=NUM_CLASSES, task_type="multiclass",
                          batch_size=batch_size, epochs=epochs, lr=lr,
                          weight_decay=0.05, warmup_epochs=5,
                          layer_decay=0.65, drop_path=0.1, label_smoothing=0.1,
                          device=device, seed=seed)
    raise NotImplementedError(f"tuev supports lp|ft, not {strategy!r}")


register_task(NAME, sys.modules[__name__])
