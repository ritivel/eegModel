"""TUAB v3.0.1 — binary anomaly detection (normal vs abnormal scalp EEG).

Reads raw .edf from NEDC's TUAB v3.0.1 distribution and re-preprocesses to
the LaBraM specification (0.1-75 Hz bandpass + 50 Hz notch + 200 Hz
resample + ÷ 0.1 mV normalisation + 10-second non-overlapping windows).

NEDC ships an official train / eval split. We split the train side 80 / 20
by *patient* (subject-disjoint) into our train / val. The CBraMod-deterministic
fixed-split reports 297,103 / 75,407 / 36,945 windows; our exact counts will
match if both NEDC v3.0.1 mirror is identical and the patient list is sorted
the same way (we sort lexicographically, same as CBraMod).

References:
- LaBraM/run_class_finetuning.py for channel order + recipe.
- LaBraM/dataset_maker/make_TUAB.py for split logic.
- CBraMod/preprocessing/README.md for fixed-split sample counts.

Lit anchors (full fine-tune):
    LaBraM-Base   AUROC 0.9022 ± 0.0009     BAC 0.8140 ± 0.0019
    CBraMod       AUROC ~0.86               BAC 0.8289 ± 0.0022   (fixed-split eval per REVE)
    REVE-Base     AUROC ~0.85               BAC 0.8315 ± 0.0014
    Random-init Mamba-2 (our floor) ~ 0.50 AUROC
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..runner import register_task

NAME = "tuab"
TASK_TYPE = "binary"
NUM_CLASSES = 2
PRIMARY_METRIC = "auroc"
LIT_ANCHORS: dict[str, float] = {
    "labram_base_auroc": 0.9022,
    "cbramod_bac": 0.8289,
    "reve_base_bac": 0.8315,
    "random_init_floor": 0.50,
}
SAMPLE_RATE = 200
WINDOW_S = 10
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_S    # 2000

# 23-channel TUH subset, LaBraM convention. Order matters — model channel
# embeddings index by position in this list.
CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2",
]


# ---------------------------------------------------------------------------
# Preprocessing (one-shot, cached) — run via `prepare_tuab(raw_root, cache_root)`
# ---------------------------------------------------------------------------


def prepare_tuab(raw_root: Path, cache_root: Path) -> dict:
    """Convert raw TUAB EDFs → cached numpy memmap matching LaBraM's recipe.

    Run once per machine (idempotent — skips if cache exists). Cache layout:

        cache_root/
            train.npz   (x: (N_train, 23, 2000) f16, y: (N_train,) i8, subj: list[str])
            val.npz     (subject-disjoint 20% of NEDC's train set)
            test.npz    (NEDC's official eval set)
            manifest.json  (split sizes, patient lists, preprocessing spec)

    Args:
        raw_root: NEDC root, e.g. /opt/dlami/nvme/eeg/raw/tuab/v3.0.1
                  Expects edf/train/{normal,abnormal}/01_tcp_ar/*.edf
                  and    edf/eval/{normal,abnormal}/01_tcp_ar/*.edf
        cache_root: where to write the cached arrays (≈ 30 GB).

    Returns the manifest dict.
    """
    import json
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    edf_root = Path(raw_root) / "edf"
    train_files = _list_tuab_edfs(edf_root / "train")
    test_files = _list_tuab_edfs(edf_root / "eval")

    # Patient-level 80/20 split of NEDC's train side, sorted lexicographically
    # for determinism. Subject ID is the first 8 chars of the EDF stem
    # (e.g. "aaaaaaae" in "aaaaaaae_s001_t000.edf") per the TUH convention.
    train_subjects = sorted({_subject_of(p) for p in train_files})
    n_val_subj = max(1, int(round(len(train_subjects) * 0.20)))
    val_subjs = set(train_subjects[:n_val_subj])
    train_subjs = set(train_subjects) - val_subjs
    test_subjs = sorted({_subject_of(p) for p in test_files})

    print(f"[tuab] {len(train_files)} train EDFs ({len(train_subjs)} subjects), "
          f"holding out {len(val_subjs)} for val; {len(test_files)} test EDFs.")

    train_x, train_y, train_s = _process_split(train_files, keep_subjects=train_subjs)
    val_x, val_y, val_s = _process_split(train_files, keep_subjects=val_subjs)
    test_x, test_y, test_s = _process_split(test_files, keep_subjects=set(test_subjs))

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
        "labram_fixed_split_target": [297103, 75407, 36945],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[tuab] cache written: {manifest['n_train']:,}/{manifest['n_val']:,}/{manifest['n_test']:,} "
          f"(target {manifest['labram_fixed_split_target']})")
    return manifest


def _list_tuab_edfs(split_root: Path) -> list[tuple[Path, int]]:
    """Walk NEDC's TUAB layout: split/{normal,abnormal}/01_tcp_ar/*.edf → (path, label)."""
    out: list[tuple[Path, int]] = []
    for label_name, label in [("normal", 0), ("abnormal", 1)]:
        for tcp_dir in (split_root / label_name).rglob("01_tcp_ar"):
            for edf in sorted(tcp_dir.glob("*.edf")):
                out.append((edf, label))
    return out


def _subject_of(item: tuple[Path, int] | Path) -> str:
    """Extract the TUH subject ID from an EDF basename (first 8 chars of stem)."""
    p = item[0] if isinstance(item, tuple) else item
    return p.stem.split("_")[0][:8]


def _process_split(
    files: list[tuple[Path, int]],
    keep_subjects: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Apply LaBraM preprocessing to each EDF, window into 10-s chunks, stack."""
    import mne
    mne.set_log_level("ERROR")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ss: list[str] = []
    for edf_path, label in files:
        sub = _subject_of((edf_path, label))
        if sub not in keep_subjects:
            continue
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        except Exception as e:                               # noqa: BLE001
            print(f"[tuab] skip {edf_path.name}: {e}")
            continue
        ok = _select_channels_in_order(raw, CHANNELS)
        if not ok:
            print(f"[tuab] skip {edf_path.name}: missing required channels")
            continue
        raw.filter(l_freq=0.1, h_freq=75.0, method="fir", verbose="ERROR")
        raw.notch_filter(freqs=[50.0], verbose="ERROR")
        raw.resample(SAMPLE_RATE, npad="auto", verbose="ERROR")
        data = raw.get_data() * 1e6 * 0.01                    # V → µV → ÷0.1 mV
        # Drop any NaNs by clipping to ±10 (LaBraM also clips implicitly via /0.1mV)
        data = np.nan_to_num(data, nan=0.0, posinf=10.0, neginf=-10.0)
        n_samp = data.shape[1]
        n_win = n_samp // WINDOW_SAMPLES
        if n_win == 0:
            continue
        windows = data[:, : n_win * WINDOW_SAMPLES].reshape(len(CHANNELS), n_win, WINDOW_SAMPLES)
        windows = np.transpose(windows, (1, 0, 2)).astype(np.float16)   # (n_win, 23, 2000)
        xs.append(windows)
        ys.extend([label] * n_win)
        ss.extend([sub] * n_win)
    x = np.concatenate(xs, axis=0) if xs else np.empty((0, len(CHANNELS), WINDOW_SAMPLES), np.float16)
    y = np.asarray(ys, dtype=np.int8)
    return x, y, ss


def _select_channels_in_order(raw, channel_names: list[str]) -> bool:
    """Subset + reorder MNE Raw to `channel_names`. Returns False if any missing.

    Handles TUH naming variants ("EEG FP1-REF", "FP1", "EEG FP1") by stripping
    the prefix and "-REF" suffix and matching case-insensitively.
    """
    avail = {_canonical(name): name for name in raw.ch_names}
    targets = [_canonical(c) for c in channel_names]
    missing = [c for c in targets if c not in avail]
    if missing:
        return False
    raw.pick([avail[c] for c in targets])
    raw.reorder_channels([avail[c] for c in targets])
    return True


def _canonical(name: str) -> str:
    """LaBraM's normalisation: strip prefix + '-REF' suffix, uppercase."""
    s = name.split(" ")[-1].split("-")[0].upper()
    return s


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class TUABDataset(Dataset):
    """Reads pre-cached numpy arrays. Yields (x: (23, 2000) float32, y: int)."""

    def __init__(self, cache_root: Path, split: str):
        path = Path(cache_root) / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"TUAB cache missing: {path}. Run `prepare_tuab(raw_root, cache_root)` first.")
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
# Task entrypoint
# ---------------------------------------------------------------------------


def run(
    encoder,
    derived_root: Path | None = None,
    *,
    strategy: str = "lp",
    device: str = "cuda",
    seed: int = 0,
    cache_subdir: str = "tuab_labram_200hz",
    raw_subdir: str = "raw/tuab/v3.0.1",
    epochs: int = 50, batch_size: int = 64, lr: float = 5e-4,
    **_unused,
) -> dict:
    """Run TUAB eval (LP or FT) and return metrics + lit anchors."""
    if derived_root is None:
        raise ValueError("tuab requires --derived-root pointing at $EEG_DATA_ROOT/derived")
    cache_root = Path(derived_root) / cache_subdir

    # Auto-prepare if cache missing AND raw root visible
    if not (cache_root / "train.npz").exists():
        raw_root = Path(derived_root).parent / raw_subdir
        if raw_root.exists():
            print(f"[tuab] cache missing, preparing from {raw_root}")
            prepare_tuab(raw_root, cache_root)
        else:
            raise FileNotFoundError(
                f"TUAB cache missing at {cache_root} and raw not at {raw_root}. "
                f"Run `prepare_tuab(...)` manually or pre-stage the cache.")

    train_ds = TUABDataset(cache_root, "train")
    val_ds = TUABDataset(cache_root, "val")
    test_ds = TUABDataset(cache_root, "test")

    if strategy == "lp":
        from ..strategies import linear_probe
        return linear_probe(encoder, train_ds, test_ds, val_ds=val_ds,
                             num_classes=2, task_type="binary",
                             batch_size=batch_size, device=device, seed=seed)
    if strategy == "ft":
        from ..strategies import fine_tune
        return fine_tune(encoder, train_ds, val_ds, test_ds,
                          num_classes=2, task_type="binary",
                          batch_size=batch_size, epochs=epochs, lr=lr,
                          weight_decay=0.05, warmup_epochs=5,
                          layer_decay=0.65, drop_path=0.1,
                          device=device, seed=seed)
    raise NotImplementedError(f"tuab supports lp|ft, not {strategy!r}")


register_task(NAME, sys.modules[__name__])
