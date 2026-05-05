"""HBN CCD response-time prediction — NeurIPS 2025 Challenge 1.

Per-trial regression: predict `rt_from_stimulus` (in seconds) from a 2-second
EEG window starting 500 ms after the contrast-change stimulus onset of the
Contrast Change Detection (CCD) task.

Source of truth: https://eeg2025.github.io/ + arXiv:2506.19141 §1.4 + EEGDash.

Splits (release-based, subject-disjoint):
    train: HBN releases R1, R2, R3, R4, R6, R7, R8, R9, R10, R11
    val:   R5
    test:  R12 (withheld; falls back to R5 with a flag if not present)

EEG window: t ∈ [0.5 s, 2.5 s] post-stimulus-onset (CCD task).
    NOT a SuS pre-trial epoch — that was the older proposal design and got
    superseded in the executed competition. EEGDash's
    `annotate_trials_with_target(target_field='rt_from_stimulus', epoch_length=2.0)`
    creates the right annotations; we then window from each annotation onset.

Sample rate: 100 Hz (the competition data is downsampled). Window = 200 samples.

Metric: nRMSE = RMSE / std(y_true), population std (ddof=0).
    Mean-predict baseline = 1.0.

Lit anchors (full FT, per-trial RT regression):
    KU Leuven (KUL_EEG) winner    nRMSE 0.88668
    Sigma Nova                    nRMSE 0.90932
    MIN~C² (MIND-CICO)            nRMSE 0.91026
    Top-10 range                  0.887 to 0.930
    Mean-predict baseline         nRMSE 1.000
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..adapter import Encoder
from ..runner import register_task

NAME = "hbn_ccd_rt"
TASK_TYPE = "regression"
NUM_CLASSES = 1
PRIMARY_METRIC = "nrmse"
LIT_ANCHORS: dict[str, float] = {
    "neurips2025_winner_kul_eeg_nrmse": 0.88668,
    "neurips2025_2nd_sigma_nova_nrmse": 0.90932,
    "neurips2025_3rd_minc2_nrmse": 0.91026,
    "neurips2025_10th_jlshen_nrmse": 0.92991,
    "mean_predict_baseline_nrmse": 1.000,
}

SAMPLE_RATE = 100              # competition data is 100 Hz
WINDOW_S = 2.0                 # 2 s starting 0.5 s post-stimulus
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_S)   # 200
EPOCH_OFFSET_S = 0.5           # window starts 500 ms post-stimulus

TRAIN_RELEASES = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
VAL_RELEASES = [5]
TEST_RELEASES = [12]


# ---------------------------------------------------------------------------


class HBNCCDDataset(Dataset):
    """Yields (window: (C, T) float32, rt: float, subject_id: str).

    Windows are pre-extracted at construction time (eager) — for CCD this is
    typically only 5-50 trials/subject so total memory is bounded (~hundreds
    of thousands of trials at 128ch × 200 samples × float32 ≈ 30 GB across
    all releases — OK on Lambda 1×A100 with 1 TB NVMe).
    """

    def __init__(
        self,
        releases: list[int],
        cache_dir: Path,
        *,
        max_subjects: int | None = None,
    ):
        from eegdash import dataset as eegdash_dataset       # type: ignore[import-untyped]
        from eegdash.hbn import build_trial_table  # type: ignore[import-untyped]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.releases = releases

        all_subj_ds: list = []
        for r in releases:
            cls = getattr(eegdash_dataset, f"EEG2025R{r}", None)
            if cls is None:
                raise RuntimeError(f"eegdash has no EEG2025R{r}")
            ds_r = cls(cache_dir=str(self.cache_dir),
                       query={"task": "contrastChangeDetection"})
            all_subj_ds.extend(ds_r.datasets)
        if max_subjects is not None:
            all_subj_ds = all_subj_ds[:max_subjects]

        self._x: list[np.ndarray] = []
        self._y: list[float] = []
        self._s: list[str] = []
        for sub_ds in all_subj_ds:
            sub_id = getattr(sub_ds, "subject", None) or sub_ds.description.get("subject", "?")
            try:
                raw = sub_ds.raw if hasattr(sub_ds, "raw") else sub_ds.load()
            except Exception as e:                            # noqa: BLE001
                print(f"[hbn_ccd_rt] skip {sub_id}: {e}")
                continue
            try:
                trials = build_trial_table(raw.annotations.to_data_frame())
            except Exception:
                continue
            data = raw.get_data()                              # (C, T)
            data = np.nan_to_num(data * 1e6, nan=0.0).astype(np.float32)  # V → µV
            sr = raw.info["sfreq"]
            for _, row in trials.iterrows():
                rt = float(row.get("rt_from_stimulus", float("nan")))
                if not np.isfinite(rt) or rt <= 0:
                    continue
                stim_t = float(row.get("stimulus_onset", float("nan")))
                if not np.isfinite(stim_t):
                    continue
                start = int(round((stim_t + EPOCH_OFFSET_S) * sr))
                end = int(start + round(WINDOW_S * sr))
                if start < 0 or end > data.shape[1]:
                    continue
                window = data[:, start:end]
                # Resample if the loaded sfreq isn't already 100 Hz
                if window.shape[1] != WINDOW_SAMPLES:
                    from scipy.signal import resample_poly
                    window = resample_poly(window, WINDOW_SAMPLES, window.shape[1], axis=1)
                self._x.append(window.astype(np.float16))
                self._y.append(rt)
                self._s.append(str(sub_id))
        self._x_arr = (np.stack(self._x, axis=0)
                       if self._x else np.empty((0, 1, WINDOW_SAMPLES), np.float16))
        self._y_arr = np.asarray(self._y, dtype=np.float32)
        self._s_arr = np.asarray(self._s)
        self._x = []  # release Python list memory; stack stays

    def __len__(self) -> int:
        return int(self._y_arr.shape[0])

    def __getitem__(self, i: int):
        return torch.from_numpy(self._x_arr[i].astype(np.float32)), float(self._y_arr[i])

    @property
    def subject_ids(self) -> list[str]:
        return self._s_arr.tolist()


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
    if derived_root is None:
        raise ValueError("hbn_ccd_rt requires --derived-root")
    cache_dir = Path(derived_root) / cache_subdir

    train_ds = HBNCCDDataset(list(train_releases), cache_dir, max_subjects=max_subjects)
    val_ds = HBNCCDDataset(list(val_releases), cache_dir, max_subjects=max_subjects)
    try:
        test_ds = HBNCCDDataset(list(test_releases), cache_dir, max_subjects=max_subjects)
        if len(test_ds) == 0:
            raise RuntimeError("R12 returned 0 trials")
        test_split_status = "official_R12"
    except Exception as e:                                    # noqa: BLE001
        print(f"[hbn_ccd_rt] R{test_releases} unavailable ({e}); falling back to R{val_releases} as test")
        test_ds = val_ds
        test_split_status = f"fallback_to_val (R12 not public yet): {e}"

    if strategy == "lp":
        from ..strategies import linear_probe
        result = linear_probe(encoder, train_ds, test_ds, val_ds=val_ds,
                              num_classes=1, task_type="regression",
                              batch_size=batch_size, device=device, seed=seed,
                              n_bootstrap=n_bootstrap)
    elif strategy == "ft":
        from ..strategies import fine_tune
        result = fine_tune(encoder, train_ds, val_ds, test_ds,
                           num_classes=1, task_type="regression",
                           batch_size=batch_size, epochs=epochs, lr=lr,
                           weight_decay=0.05, warmup_epochs=5,
                           layer_decay=0.65, drop_path=0.1,
                           device=device, seed=seed, n_bootstrap=n_bootstrap)
    else:
        raise NotImplementedError(f"hbn_ccd_rt supports lp|ft, not {strategy!r}")

    result["test_split_status"] = test_split_status
    result["train_releases"] = list(train_releases)
    result["val_releases"] = list(val_releases)
    result["test_releases"] = list(test_releases)
    result["n_train_trials"] = len(train_ds)
    result["n_test_trials"] = len(test_ds)
    return result


register_task(NAME, sys.modules[__name__])
