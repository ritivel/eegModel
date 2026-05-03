"""Frozen-feature evaluation suite (§4.3 Protocol A — primary).

This is the eval framework used by every mini-experiment after Check D
sets the random-init floor. The contract is:

    extract_features(model, derived_root, ...)
        → (features [N, D], labels dict {subject_id, task, attention,
                                          externalizing, ...})
    run_protocol_a(features, labels)
        → dict of {metric_name: {mean, ci_low_95, ci_high_95, raw}}

For mini-experiment 01 specifically, `run_random_init_probe` does:
    1. Build the §4.2-default `EEGSSLModel`, freeze it (don't train).
    2. Extract mean-pooled encoder features on a subject-disjoint subset
       of the warehouse parquet (LNSO style — no subject overlap between
       fit and eval).
    3. Train sklearn linear/ridge probes on each §4.3 Protocol A target.
    4. Train a sklearn k-NN top-1 on the cohort.
    5. Bootstrap 95% CIs (1000 resamples) on the test scores.
    6. Return a CheckResult with everything wrapped up.

Per `01_sanity_baselines/README.md`, this number is the *ablation floor*:
every later pretrained encoder must clearly beat these numbers, otherwise
the SSL signal didn't transfer (or the eval extraction is broken).

Mean-pooled-over-time features per the 2026-05-03 reaffirmation of the
mean-pool default (vs CLS-token probing); see model.py
`encode_features` and `01_sanity_baselines/README.md` "What gets carried
forward" for the rationale.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from . import data, model
from .sanity import CheckResult


# =============================================================================
# Feature extraction
# =============================================================================


@dataclass
class ExtractedFeatures:
    """Frozen-encoder features + per-window labels.

    `features` is (N, D=encoder.d_model) float32.
    `subject_ids` is length-N list of HBN subject IDs (used for LNSO splits).
    Other label fields are length-N numpy arrays.
    """
    features: np.ndarray                          # (N, D)
    subject_ids: list[str]                        # (N,)
    task_label: np.ndarray                        # int8 (0..5)
    attention: np.ndarray                         # CBCL attention z-score (NaN if missing)
    externalizing: np.ndarray                     # CBCL externalizing z-score
    p_factor: np.ndarray                          # CBCL p-factor
    internalizing: np.ndarray                     # CBCL internalizing
    age: np.ndarray
    site: list[str]


@torch.no_grad()
def extract_features(
    encoder_model: nn.Module,
    derived_root: Path,
    *,
    max_subjects: int = 50,
    max_windows_per_shard: int = 20,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    progress: bool = True,
) -> ExtractedFeatures:
    """Run the encoder over a subset of the warehouse and pool features.

    Bulk per-shard extraction: read the whole parquet at once, slice to
    `max_windows_per_shard` rows, stack signals into a single numpy
    array, push to GPU as one batch (or a few). This is ~50x faster than
    the row-by-row IterableDataset path when feature extraction is
    GPU-bound on a small model. The `data.ParquetWindowDataset`
    IterableDataset is still the right choice for actual training (where
    the row-by-row + multi-worker shuffle matters); for one-shot eval
    extraction, vectorise.

    Args:
        encoder_model: a built (and possibly pretrained, possibly random-init)
            `EEGSSLModel`.
        derived_root: path to derived/<pipeline>/ on local NVMe.
        max_subjects: cap the number of HBN subjects to read. 50 ≈ 5 GB
            of parquet, ~10k iid windows after `max_windows_per_shard=20`.
        max_windows_per_shard: cap iid windows per recording-shard.
        batch_size: GPU batch size for feature extraction.
        device: "cuda" or "cpu".
        seed: RNG seed for shard / subject selection.
        progress: print a progress line per few shards.

    Returns:
        ExtractedFeatures with everything you need to run the linear probe.
    """
    import pyarrow.parquet as pq

    encoder_model = encoder_model.to(device).eval()

    all_shards = data.list_shards(derived_root)
    subject_ids_avail = sorted({s.subject_id for s in all_shards})
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(subject_ids_avail,
                          size=min(max_subjects, len(subject_ids_avail)),
                          replace=False).tolist())
    chosen_shards = [s for s in all_shards if s.subject_id in keep]
    if progress:
        print(f"[extract_features] {len(keep)} subjects, {len(chosen_shards)} shards")

    feat_chunks: list[np.ndarray] = []
    subject_ids: list[str] = []
    task_label: list[int] = []
    attention: list[float] = []
    externalizing: list[float] = []
    p_factor: list[float] = []
    internalizing: list[float] = []
    age: list[float] = []
    site: list[str] = []

    t0 = time.time()
    n_done = 0
    for shard_i, entry in enumerate(chosen_shards):
        try:
            table = pq.read_table(entry.path, columns=[
                "subject_id", "site", "task_label",
                "attention", "externalizing", "p_factor", "internalizing",
                "age", "signal",
            ])
        except Exception as e:                           # noqa: BLE001
            print(f"[extract_features] WARN: skipping {entry.path.name}: {e}")
            continue
        n_rows = table.num_rows
        if n_rows == 0:
            continue
        # Pick first max_windows_per_shard rows (deterministic; no shuffle
        # inside a shard since the parquet rows are already (channel, window)
        # iid expansion ordered which is fine for a probe-floor estimate).
        n_take = min(max_windows_per_shard, n_rows)
        table_slice = table.slice(0, n_take)

        # Stack signals into (n_take, T) float32. The signal column is
        # list<float16>; we bypass to_pylist (which Python-iterates and is
        # ~50x slower than the underlying buffer access) and instead grab
        # the flat backing buffer directly from the ListArray, then reshape.
        sig_col = table_slice.column("signal")
        ca = sig_col.combine_chunks() if hasattr(sig_col, "combine_chunks") else sig_col
        # ca is a ListArray (or ChunkedArray with a single chunk after combine).
        if hasattr(ca, "chunks"):
            ca = ca.chunks[0]
        flat = ca.values.to_numpy(zero_copy_only=False)   # (n_take * T,) fp16
        if flat.size == 0:
            continue
        T_samples = flat.size // n_take
        signals = flat.astype(np.float32).reshape(n_take, T_samples)
        if signals.ndim != 2:
            print(f"[extract_features] WARN: weird signal shape {signals.shape} "
                  f"in {entry.path.name}; skipping")
            continue

        # Pull metadata columns (also small, just n_take entries)
        sub_arr = table_slice.column("subject_id").to_pylist()
        site_arr = table_slice.column("site").to_pylist()
        task_arr = np.asarray(table_slice.column("task_label").to_pylist(), dtype=np.int8)
        attn_arr = np.asarray(table_slice.column("attention").to_pylist(), dtype=np.float32)
        extn_arr = np.asarray(table_slice.column("externalizing").to_pylist(), dtype=np.float32)
        pfac_arr = np.asarray(table_slice.column("p_factor").to_pylist(), dtype=np.float32)
        intn_arr = np.asarray(table_slice.column("internalizing").to_pylist(), dtype=np.float32)
        age_arr = np.asarray(table_slice.column("age").to_pylist(), dtype=np.float32)

        # Forward pass in batches if shard exceeds batch_size (rare with n=20)
        x_full = torch.from_numpy(signals).to(device, non_blocking=True)
        feats_chunks_for_shard = []
        for i in range(0, x_full.size(0), batch_size):
            x = x_full[i : i + batch_size]
            feats = encoder_model.encode_features(x)
            feats_chunks_for_shard.append(feats.detach().cpu().numpy().astype(np.float32))
        feat_chunks.append(np.concatenate(feats_chunks_for_shard, axis=0))

        subject_ids.extend(sub_arr)
        site.extend(site_arr)
        task_label.extend(task_arr.tolist())
        attention.extend(attn_arr.tolist())
        externalizing.extend(extn_arr.tolist())
        p_factor.extend(pfac_arr.tolist())
        internalizing.extend(intn_arr.tolist())
        age.extend(age_arr.tolist())

        n_done += signals.shape[0]
        if progress and (shard_i % 50 == 0 or shard_i == len(chosen_shards) - 1):
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            print(f"[extract_features] shard {shard_i + 1}/{len(chosen_shards)} "
                  f"  windows={n_done}  rate={rate:.0f}/s")

    if not feat_chunks:
        raise RuntimeError("no windows extracted — empty dataset?")

    features = np.concatenate(feat_chunks, axis=0)
    if progress:
        print(f"[extract_features] done: {features.shape[0]} windows in "
              f"{time.time() - t0:.1f}s "
              f"-> features shape {features.shape}")

    return ExtractedFeatures(
        features=features,
        subject_ids=subject_ids,
        task_label=np.asarray(task_label, dtype=np.int8),
        attention=np.asarray(attention, dtype=np.float32),
        externalizing=np.asarray(externalizing, dtype=np.float32),
        p_factor=np.asarray(p_factor, dtype=np.float32),
        internalizing=np.asarray(internalizing, dtype=np.float32),
        age=np.asarray(age, dtype=np.float32),
        site=site,
    )


# =============================================================================
# Subject-disjoint splits (LNSO)
# =============================================================================


def lnso_split(
    subject_ids: list[str],
    *,
    test_frac: float = 0.30,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-N-Subjects-Out split: returns (train_idx, test_idx).

    No subject appears in both splits. Critical for any clinical eval
    (HBN factor regression, ADHD-binary, etc.) — within-subject data
    leakage trivially inflates AUROC by 10–30 pp.
    """
    rng = np.random.default_rng(seed)
    unique = sorted(set(subject_ids))
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_frac)))
    test_subs = set(unique[:n_test])
    sub_arr = np.asarray(subject_ids)
    test_idx = np.where(np.isin(sub_arr, list(test_subs)))[0]
    train_idx = np.where(~np.isin(sub_arr, list(test_subs)))[0]
    return train_idx, test_idx


# =============================================================================
# Bootstrap CIs
# =============================================================================


def bootstrap_ci(
    fn,
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    **fn_kwargs,
) -> dict[str, float]:
    """Bootstrap on the *test* indices.

    `fn` is a callable taking `idx: np.ndarray` (a resampled-with-replacement
    index array) and returning a single scalar. We resample the test set
    1000 times and return mean / ci_low / ci_high.
    """
    rng = np.random.default_rng(seed)
    base_idx = np.arange(fn_kwargs.pop("n", 0)) if "n" in fn_kwargs else None
    if base_idx is None:
        raise ValueError("must pass n=<test set size>")
    samples = []
    for _ in range(n_bootstrap):
        bi = rng.choice(base_idx, size=base_idx.size, replace=True)
        try:
            samples.append(float(fn(bi)))
        except Exception:                  # noqa: BLE001 — bootstrap can hit weird edge cases
            continue
    if not samples:
        return {"mean": float("nan"), "ci_low_95": float("nan"), "ci_high_95": float("nan")}
    arr = np.asarray(samples)
    alpha = (1 - ci) / 2
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_low_95": float(np.quantile(arr, alpha)),
        "ci_high_95": float(np.quantile(arr, 1 - alpha)),
        "n_bootstrap": len(samples),
    }


# =============================================================================
# §4.3 Protocol A — frozen probing (primary)
# =============================================================================


def _drop_nan(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index mask for non-NaN entries + the cleaned y."""
    mask = ~np.isnan(y)
    return mask, y[mask]


def run_protocol_a(
    extracted: ExtractedFeatures,
    *,
    test_frac: float = 0.30,
    seed: int = 0,
    n_bootstrap: int = 200,                     # 200 is plenty for sanity floor
    knn_k: int = 5,
    knn_subset: int = 10_000,
) -> dict[str, Any]:
    """Run all §4.3 Protocol A targets on the extracted features.

    Returns dict keyed by target name with metric/CI sub-dicts:
        externalizing_r2, externalizing_mae,
        attention_r2, attention_mae,
        attention_binary_auroc,
        task6_bac, task6_wf1,
        knn_top1_task6,
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                  mean_absolute_error, r2_score, roc_auc_score)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    train_idx, test_idx = lnso_split(extracted.subject_ids,
                                      test_frac=test_frac, seed=seed)
    X = extracted.features
    n_train, n_test = train_idx.size, test_idx.size
    print(f"[protocol_a] LNSO split: train={n_train} test={n_test} "
          f"(features dim={X.shape[1]})")

    # Standardise features on train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    out: dict[str, Any] = {}

    # ---------- A.1a — externalizing-factor regression ----------
    y = extracted.externalizing
    mask_train, y_train = _drop_nan(y[train_idx])
    mask_test, y_test = _drop_nan(y[test_idx])
    if mask_train.sum() >= 50 and mask_test.sum() >= 50:
        reg = LinearRegression().fit(X_train[mask_train], y_train)
        y_pred = reg.predict(X_test[mask_test])
        r2_full = r2_score(y_test, y_pred)
        mae_full = mean_absolute_error(y_test, y_pred)

        def _r2(idx):
            return r2_score(y_test[idx], y_pred[idx])
        def _mae(idx):
            return mean_absolute_error(y_test[idx], y_pred[idx])
        out["externalizing_r2"] = {"point": float(r2_full),
                                   **bootstrap_ci(_r2, n_bootstrap=n_bootstrap, seed=seed,
                                                  n=mask_test.sum())}
        out["externalizing_mae"] = {"point": float(mae_full),
                                    **bootstrap_ci(_mae, n_bootstrap=n_bootstrap, seed=seed,
                                                   n=mask_test.sum())}
    else:
        out["externalizing_r2"] = {"reason": "insufficient labeled samples",
                                   "n_train": int(mask_train.sum()),
                                   "n_test": int(mask_test.sum())}

    # ---------- A.1b — attention-factor regression ----------
    y = extracted.attention
    mask_train, y_train = _drop_nan(y[train_idx])
    mask_test, y_test = _drop_nan(y[test_idx])
    if mask_train.sum() >= 50 and mask_test.sum() >= 50:
        reg = LinearRegression().fit(X_train[mask_train], y_train)
        y_pred = reg.predict(X_test[mask_test])
        r2_full = r2_score(y_test, y_pred)
        mae_full = mean_absolute_error(y_test, y_pred)

        def _r2(idx):
            return r2_score(y_test[idx], y_pred[idx])
        def _mae(idx):
            return mean_absolute_error(y_test[idx], y_pred[idx])
        out["attention_r2"] = {"point": float(r2_full),
                               **bootstrap_ci(_r2, n_bootstrap=n_bootstrap, seed=seed,
                                              n=mask_test.sum())}
        out["attention_mae"] = {"point": float(mae_full),
                                **bootstrap_ci(_mae, n_bootstrap=n_bootstrap, seed=seed,
                                               n=mask_test.sum())}
    else:
        out["attention_r2"] = {"reason": "insufficient labeled samples"}

    # ---------- A.1c — attention-binary AUROC (threshold z>0.5) ----------
    y_attn = extracted.attention
    mask_train_b = ~np.isnan(y_attn[train_idx])
    mask_test_b = ~np.isnan(y_attn[test_idx])
    if mask_train_b.sum() >= 50 and mask_test_b.sum() >= 50:
        y_train_bin = (y_attn[train_idx][mask_train_b] > 0.5).astype(np.int8)
        y_test_bin = (y_attn[test_idx][mask_test_b] > 0.5).astype(np.int8)
        if y_train_bin.sum() > 5 and y_train_bin.sum() < y_train_bin.size - 5:
            clf = LogisticRegression(max_iter=5000, n_jobs=1).fit(
                X_train[mask_train_b], y_train_bin)
            y_score = clf.predict_proba(X_test[mask_test_b])[:, 1]
            auroc_full = roc_auc_score(y_test_bin, y_score)

            def _auroc(idx):
                return roc_auc_score(y_test_bin[idx], y_score[idx])
            out["attention_binary_auroc"] = {
                "point": float(auroc_full),
                "n_pos_test": int(y_test_bin.sum()),
                "n_neg_test": int(y_test_bin.size - y_test_bin.sum()),
                **bootstrap_ci(_auroc, n_bootstrap=n_bootstrap, seed=seed,
                               n=mask_test_b.sum()),
            }
        else:
            out["attention_binary_auroc"] = {"reason": "class imbalance after split"}
    else:
        out["attention_binary_auroc"] = {"reason": "insufficient labeled samples"}

    # ---------- A.2 — 6-task classification ----------
    y_task = extracted.task_label
    if len(set(y_task[train_idx].tolist())) >= 2:
        # sklearn 1.8 deprecated `multi_class`; lbfgs auto-detects multinomial.
        clf = LogisticRegression(max_iter=5000, n_jobs=1,
                                 solver="lbfgs").fit(X_train, y_task[train_idx])
        y_pred = clf.predict(X_test)
        bac_full = balanced_accuracy_score(y_task[test_idx], y_pred)
        wf1_full = f1_score(y_task[test_idx], y_pred, average="weighted")

        def _bac(idx):
            return balanced_accuracy_score(y_task[test_idx][idx], y_pred[idx])
        def _wf1(idx):
            return f1_score(y_task[test_idx][idx], y_pred[idx], average="weighted")
        out["task6_bac"] = {"point": float(bac_full),
                            **bootstrap_ci(_bac, n_bootstrap=n_bootstrap, seed=seed,
                                           n=test_idx.size)}
        out["task6_wf1"] = {"point": float(wf1_full),
                            **bootstrap_ci(_wf1, n_bootstrap=n_bootstrap, seed=seed,
                                           n=test_idx.size)}
    else:
        out["task6_bac"] = {"reason": "only one task class in train split"}

    # ---------- A.3 — k-NN top-1 on a 10k subset, 6-task labels ----------
    knn_n = min(knn_subset, train_idx.size + test_idx.size)
    if knn_n > 200:
        idx_subset = np.concatenate([train_idx, test_idx])
        if idx_subset.size > knn_n:
            rng = np.random.default_rng(seed)
            idx_subset = rng.choice(idx_subset, size=knn_n, replace=False)
        # Build a fresh 70/30 LNSO inside the subset (subject-disjoint)
        sub_subset = [extracted.subject_ids[i] for i in idx_subset]
        knn_train_idx, knn_test_idx = lnso_split(sub_subset, test_frac=0.30, seed=seed)
        X_knn = scaler.transform(X[idx_subset])
        knn = KNeighborsClassifier(n_neighbors=knn_k, metric="cosine", n_jobs=1)
        knn.fit(X_knn[knn_train_idx], y_task[idx_subset][knn_train_idx])
        y_pred = knn.predict(X_knn[knn_test_idx])
        top1 = float((y_pred == y_task[idx_subset][knn_test_idx]).mean())

        def _top1(idx):
            return float((y_pred[idx] == y_task[idx_subset][knn_test_idx][idx]).mean())
        out["knn_top1_task6"] = {"point": top1, "k": knn_k,
                                  **bootstrap_ci(_top1, n_bootstrap=n_bootstrap, seed=seed,
                                                 n=knn_test_idx.size)}
    else:
        out["knn_top1_task6"] = {"reason": "subset too small"}

    return out


# =============================================================================
# Check D entrypoint
# =============================================================================


def run_random_init_probe(
    cfg: model.ModelConfig,
    *,
    derived_root: Path,
    max_subjects: int = 50,
    max_windows_per_shard: int = 20,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CheckResult:
    """Build a random-init `EEGSSLModel`, extract frozen features, run the eval.

    The result table this writes is the *ablation floor* for every later
    pretrained encoder in this experiment. Per
    `01_sanity_baselines/README.md`, any future SSL run whose linear-probe
    metrics fall within 1% of this floor is broken — either SSL didn't
    transfer, or the feature extraction is wrong (wrong layer, wrong
    pooling, wrong split).
    """
    print(f"\n{'='*70}\nCheck D — Random-init linear-probe floor\n{'='*70}")
    print(f"derived_root={derived_root}")
    print(f"max_subjects={max_subjects} max_windows_per_shard={max_windows_per_shard}")

    torch.manual_seed(seed)
    encoder = model.build_model(cfg).to(device).eval()
    p = model.count_params(encoder)
    print(f"random-init model: total params={p['total']:,}")

    extracted = extract_features(
        encoder, derived_root,
        max_subjects=max_subjects,
        max_windows_per_shard=max_windows_per_shard,
        seed=seed, device=device,
    )

    # Quick feature health check — flat features mean a broken encoder
    # or pooling. Standard exp-floor sanity.
    feat_std = extracted.features.std(axis=0)
    feat_dim = extracted.features.shape[1]
    print(f"feature stats: per-dim mean abs = {np.abs(extracted.features.mean(0)).mean():.4f}")
    print(f"               per-dim std mean = {feat_std.mean():.4f}")
    print(f"               per-dim std min  = {feat_std.min():.4f}")
    print(f"               per-dim std max  = {feat_std.max():.4f}")

    metrics = run_protocol_a(extracted, seed=seed)

    # Pretty-print the headline numbers
    print(f"\n{'metric':<26}{'point':>10}{'95% CI low':>14}{'95% CI high':>14}")
    print("-" * 64)
    for name in ["externalizing_r2", "externalizing_mae",
                 "attention_r2", "attention_mae", "attention_binary_auroc",
                 "task6_bac", "task6_wf1", "knn_top1_task6"]:
        d = metrics.get(name, {})
        if "point" in d:
            print(f"{name:<26}{d['point']:>10.4f}"
                  f"{d.get('ci_low_95', float('nan')):>14.4f}"
                  f"{d.get('ci_high_95', float('nan')):>14.4f}")
        else:
            reason = d.get("reason", "MISSING")
            print(f"{name:<26}  {reason}")

    return CheckResult(
        name="D_random_init_probe",
        status="GREEN",                       # always GREEN — Check D records, doesn't gate
        details={
            "n_features": int(extracted.features.shape[0]),
            "feat_dim": int(extracted.features.shape[1]),
            "feat_per_dim_std_mean": float(feat_std.mean()),
            "feat_per_dim_std_min": float(feat_std.min()),
            "feat_per_dim_std_max": float(feat_std.max()),
            "n_subjects_used": len(set(extracted.subject_ids)),
            "metrics": metrics,
        },
        notes=("This is the ablation floor — every later pretrained encoder "
               "must clearly beat these numbers. Any future encoder whose "
               "linear-probe metrics fall within 1% of this floor is broken."),
    )
