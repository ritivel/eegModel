"""PhysioNet-MI (Schalk et al. 2004) — 4-class motor imagery.

Source: https://physionet.org/content/eegmmidb/1.0.0/

109 subjects × ~14 runs each. Motor-imagery runs are 4, 6, 8, 10, 12, 14
(2 imagined-fist runs + 2 imagined-both runs across the 6 MI runs).
Each run is ~2 minutes of 64-channel EEG @ 160 Hz.

Per-trial labels from EDF event annotations:
    T0: rest               → drop (we don't include rest in 4-class)
    T1 (runs 4,8,12):  imagine left fist     → label 0
    T1 (runs 6,10,14): imagine both fists    → label 2
    T2 (runs 4,8,12):  imagine right fist    → label 1
    T2 (runs 6,10,14): imagine both feet     → label 3

CBraMod's exact split (subject-disjoint, lexicographic):
    train: subjects S001..S070
    val:   S071..S089
    test:  S090..S109   (109 - excluded subjects per CBraMod, see notes)

CBraMod / LaBraM both resample to 200 Hz, use 4-second windows = 800 samples.

Lit anchors (full fine-tune, Cohen's κ / BAC):
    LaBraM-Base   κ 0.4912 ± 0.0192   BAC 0.6173 ± 0.0122
    CBraMod       κ 0.5222 ± 0.0169   BAC 0.6417 ± 0.0091
    REVE-Base     BAC 0.6480 ± 0.0140
    Random-init   κ ≈ 0    BAC ≈ 0.25 (chance, 4-class)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..runner import register_task

NAME = "physionet_mi"
TASK_TYPE = "multiclass"
NUM_CLASSES = 4
PRIMARY_METRIC = "cohen_kappa"
LIT_ANCHORS: dict[str, float] = {
    "labram_base_kappa": 0.4912,
    "labram_base_bac": 0.6173,
    "cbramod_kappa": 0.5222,
    "cbramod_bac": 0.6417,
    "reve_base_bac": 0.6480,
    "random_init_floor": 0.25,
}
SAMPLE_RATE = 200
WINDOW_S = 4
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_S    # 800

# Standard 64-channel ordering for PhysioNet-MI EDFs (BCI2000 convention).
# We don't subset; the model gets all 64 channels.
EXPECTED_RUNS = [4, 6, 8, 10, 12, 14]
LEFT_RIGHT_RUNS = {4, 8, 12}     # T1 = left fist, T2 = right fist
FISTS_FEET_RUNS = {6, 10, 14}    # T1 = both fists, T2 = both feet


# ---------------------------------------------------------------------------


def prepare_physionet_mi(raw_root: Path, cache_root: Path) -> dict:
    """Convert raw PhysioNet-MI EDFs → cached numpy arrays."""
    import json
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    raw_root = Path(raw_root)
    # Subject directories: S001/, S002/, ..., S109/
    subj_dirs = sorted([d for d in raw_root.glob("S*") if d.is_dir()])
    if not subj_dirs:
        raise FileNotFoundError(f"no S<NN>/ subject dirs under {raw_root}")
    train_subj = [d for d in subj_dirs if 1 <= int(d.name[1:]) <= 70]
    val_subj = [d for d in subj_dirs if 71 <= int(d.name[1:]) <= 89]
    test_subj = [d for d in subj_dirs if 90 <= int(d.name[1:]) <= 109]

    print(f"[physionet_mi] {len(train_subj)} train / {len(val_subj)} val / {len(test_subj)} test subjects")

    train_x, train_y, train_s = _process_subjects(train_subj)
    val_x, val_y, val_s = _process_subjects(val_subj)
    test_x, test_y, test_s = _process_subjects(test_subj)

    np.savez_compressed(cache_root / "train.npz", x=train_x, y=train_y, subj=np.array(train_s))
    np.savez_compressed(cache_root / "val.npz",   x=val_x,   y=val_y,   subj=np.array(val_s))
    np.savez_compressed(cache_root / "test.npz",  x=test_x,  y=test_y,  subj=np.array(test_s))

    manifest = {
        "spec": {"resample_hz": SAMPLE_RATE, "window_s": WINDOW_S,
                 "n_channels": int(train_x.shape[1]) if train_x.size else 64},
        "split_strategy": "Subject-disjoint: 001-070 train, 071-089 val, 090-109 test (CBraMod convention)",
        "n_train": int(train_x.shape[0]), "n_val": int(val_x.shape[0]), "n_test": int(test_x.shape[0]),
        "label_distribution": _label_dist(train_y, val_y, test_y),
        "class_names": ["left_fist", "right_fist", "both_fists", "both_feet"],
        "cbramod_split_target": [9837, "varies", "varies"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[physionet_mi] cache: {manifest['n_train']:,}/{manifest['n_val']:,}/{manifest['n_test']:,}")
    return manifest


def _process_subjects(subj_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import mne
    mne.set_log_level("ERROR")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ss: list[str] = []
    for sub_dir in subj_dirs:
        sub = sub_dir.name
        for run in EXPECTED_RUNS:
            edf = sub_dir / f"{sub}R{run:02d}.edf"
            if not edf.exists():
                continue
            try:
                raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
            except Exception as e:                           # noqa: BLE001
                print(f"[physionet_mi] skip {edf.name}: {e}")
                continue
            raw.resample(SAMPLE_RATE, npad="auto", verbose="ERROR")
            data = raw.get_data() * 1e6                       # V → µV (no /0.1 mV here; CBraMod doesn't normalise)
            data = np.nan_to_num(data, nan=0.0, posinf=200.0, neginf=-200.0).astype(np.float32)
            n_chan, n_samp = data.shape
            # Annotations: T0 (rest), T1, T2.
            for ann in raw.annotations:
                desc = ann["description"]
                if desc == "T0":
                    continue
                start = int(round(ann["onset"] * SAMPLE_RATE))
                if start + WINDOW_SAMPLES > n_samp:
                    continue
                window = data[:, start : start + WINDOW_SAMPLES].astype(np.float16)
                if desc == "T1":
                    label = 0 if run in LEFT_RIGHT_RUNS else 2
                elif desc == "T2":
                    label = 1 if run in LEFT_RIGHT_RUNS else 3
                else:
                    continue
                xs.append(window[None, :, :])
                ys.append(label)
                ss.append(sub)
    x = np.concatenate(xs, axis=0) if xs else np.empty((0, 64, WINDOW_SAMPLES), np.float16)
    y = np.asarray(ys, dtype=np.int8)
    return x, y, ss


def _label_dist(*ys: np.ndarray) -> list[dict]:
    out = []
    for y in ys:
        d = {n: int((y == i).sum()) for i, n in enumerate(["left_fist", "right_fist", "both_fists", "both_feet"])}
        out.append(d)
    return out


# ---------------------------------------------------------------------------


class PhysioNetMIDataset(Dataset):
    def __init__(self, cache_root: Path, split: str):
        path = Path(cache_root) / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(f"PhysioNet-MI cache missing: {path}. Run prepare_physionet_mi(...).")
        data = np.load(path, allow_pickle=True)
        self.x = data["x"]
        self.y = data["y"]
        self.subj = data["subj"]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        return torch.from_numpy(self.x[i].astype(np.float32)), int(self.y[i])


# ---------------------------------------------------------------------------


def run(
    encoder,
    derived_root: Path | None = None,
    *,
    strategy: str = "lp",
    device: str = "cuda",
    seed: int = 0,
    cache_subdir: str = "physionet_mi_200hz",
    raw_subdir: str = "raw/physionet_mi",
    epochs: int = 50, batch_size: int = 64, lr: float = 5e-4,
    **_unused,
) -> dict:
    if derived_root is None:
        raise ValueError("physionet_mi requires --derived-root")
    cache_root = Path(derived_root) / cache_subdir

    if not (cache_root / "train.npz").exists():
        raw_root = Path(derived_root).parent / raw_subdir
        if raw_root.exists():
            print(f"[physionet_mi] cache missing, preparing from {raw_root}")
            prepare_physionet_mi(raw_root, cache_root)
        else:
            raise FileNotFoundError(f"PhysioNet-MI cache missing at {cache_root} and raw not at {raw_root}.")

    train_ds = PhysioNetMIDataset(cache_root, "train")
    val_ds = PhysioNetMIDataset(cache_root, "val")
    test_ds = PhysioNetMIDataset(cache_root, "test")

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
    raise NotImplementedError(f"physionet_mi supports lp|ft, not {strategy!r}")


register_task(NAME, sys.modules[__name__])
