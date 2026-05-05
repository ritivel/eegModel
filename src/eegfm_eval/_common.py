"""Shared helpers: metrics, splits, bootstrap CIs, parquet readers.

Tiny module — every helper is a function, not a class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Splits (subject-disjoint)
# ---------------------------------------------------------------------------


def lnso_split(
    subject_ids: list[str], *, test_frac: float = 0.30, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-N-Subjects-Out: returns (train_idx, test_idx). No subject in both."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(subject_ids))
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_frac)))
    test_subs = set(unique[:n_test])
    sub_arr = np.asarray(subject_ids)
    test_idx = np.where(np.isin(sub_arr, list(test_subs)))[0]
    train_idx = np.where(~np.isin(sub_arr, list(test_subs)))[0]
    return train_idx, test_idx


def kfold_subject(
    subject_ids: list[str], *, k: int = 5, seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """k-fold CV with subject-disjoint folds. Returns list of (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(subject_ids))
    rng.shuffle(unique)
    folds = np.array_split(unique, k)
    sub_arr = np.asarray(subject_ids)
    out = []
    for i in range(k):
        test_subs = set(folds[i].tolist())
        test_idx = np.where(np.isin(sub_arr, list(test_subs)))[0]
        train_idx = np.where(~np.isin(sub_arr, list(test_subs)))[0]
        out.append((train_idx, test_idx))
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    fn: Callable[[np.ndarray], float],
    *,
    n: int,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Resample-with-replacement CI on a scalar metric.

    `fn(idx)` takes a length-`n` index array and returns one float.
    Returns `{point optional, mean, std, ci_low_95, ci_high_95, n_bootstrap}`.
    """
    rng = np.random.default_rng(seed)
    base = np.arange(n)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        bi = rng.choice(base, size=n, replace=True)
        try:
            samples.append(float(fn(bi)))
        except Exception:                                       # noqa: BLE001
            continue
    if not samples:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_low_95": float("nan"), "ci_high_95": float("nan"),
                "n_bootstrap": 0}
    arr = np.asarray(samples)
    alpha = (1 - ci) / 2
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_low_95": float(np.quantile(arr, alpha)),
        "ci_high_95": float(np.quantile(arr, 1 - alpha)),
        "n_bootstrap": len(samples),
    }


# ---------------------------------------------------------------------------
# Metrics — task_type-keyed dispatch
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    y_score: np.ndarray | None = None,        # required for AUROC / AUPRC
    task_type: Literal["binary", "multiclass", "regression"],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Return one dict of {metric_name: {point, ci_low_95, ci_high_95, ...}}.

    For classification, `y_pred` (hard labels) and `y_score` (probabilities,
    shape `(N,)` for binary or `(N, C)` for multiclass) may both be passed.
    For regression, only `y_pred` is used.
    """
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                                  f1_score, mean_absolute_error,
                                  mean_squared_error, r2_score, roc_auc_score)
    out: dict[str, dict[str, float]] = {}
    n = y_true.size

    if task_type == "binary":
        if y_pred is not None:
            point = float(balanced_accuracy_score(y_true, y_pred))
            ci = bootstrap_ci(lambda i: balanced_accuracy_score(y_true[i], y_pred[i]),
                              n=n, n_bootstrap=n_bootstrap, seed=seed)
            out["bac"] = {"point": point, **ci}
            point = float(f1_score(y_true, y_pred, average="binary"))
            ci = bootstrap_ci(lambda i: f1_score(y_true[i], y_pred[i], average="binary"),
                              n=n, n_bootstrap=n_bootstrap, seed=seed)
            out["f1"] = {"point": point, **ci}
        if y_score is not None:
            point = float(roc_auc_score(y_true, y_score))
            ci = bootstrap_ci(lambda i: roc_auc_score(y_true[i], y_score[i]),
                              n=n, n_bootstrap=n_bootstrap, seed=seed)
            out["auroc"] = {"point": point, **ci}

    elif task_type == "multiclass":
        if y_pred is not None:
            point = float(balanced_accuracy_score(y_true, y_pred))
            ci = bootstrap_ci(lambda i: balanced_accuracy_score(y_true[i], y_pred[i]),
                              n=n, n_bootstrap=n_bootstrap, seed=seed)
            out["bac"] = {"point": point, **ci}
            point = float(f1_score(y_true, y_pred, average="weighted"))
            ci = bootstrap_ci(lambda i: f1_score(y_true[i], y_pred[i], average="weighted"),
                              n=n, n_bootstrap=n_bootstrap, seed=seed)
            out["weighted_f1"] = {"point": point, **ci}
            point = float(cohen_kappa_score(y_true, y_pred))
            ci = bootstrap_ci(lambda i: cohen_kappa_score(y_true[i], y_pred[i]),
                              n=n, n_bootstrap=n_bootstrap, seed=seed)
            out["cohen_kappa"] = {"point": point, **ci}

    elif task_type == "regression":
        if y_pred is None:
            raise ValueError("regression requires y_pred")
        std_y = float(np.std(y_true)) or 1.0
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        out["nrmse"] = {"point": rmse / std_y,
                        **bootstrap_ci(lambda i: float(np.sqrt(
                            mean_squared_error(y_true[i], y_pred[i]))) / std_y,
                            n=n, n_bootstrap=n_bootstrap, seed=seed)}
        out["mae"] = {"point": float(mean_absolute_error(y_true, y_pred)),
                      **bootstrap_ci(lambda i: mean_absolute_error(y_true[i], y_pred[i]),
                                     n=n, n_bootstrap=n_bootstrap, seed=seed)}
        out["r2"] = {"point": float(r2_score(y_true, y_pred)),
                     **bootstrap_ci(lambda i: r2_score(y_true[i], y_pred[i]),
                                    n=n, n_bootstrap=n_bootstrap, seed=seed)}
        # Pearson and Spearman
        from scipy.stats import pearsonr, spearmanr
        out["pearson_r"] = {"point": float(pearsonr(y_true, y_pred).statistic)}
        out["spearman_r"] = {"point": float(spearmanr(y_true, y_pred).statistic)}

    else:
        raise ValueError(f"unknown task_type {task_type!r}")
    return out


# ---------------------------------------------------------------------------
# Parquet reader — small helper for tasks that read our derived shards
# ---------------------------------------------------------------------------


def read_parquet_columns(path: Path, columns: Iterable[str]) -> dict[str, Any]:
    """Read selected columns from a parquet shard into a dict of numpy arrays.

    The `signal` column (if requested) is returned as `(n_rows, n_samples)`
    `float32` after f16 → f32 cast and reshape from the underlying ListArray.
    """
    import pyarrow.parquet as pq
    cols = list(columns)
    table = pq.read_table(path, columns=cols)
    out: dict[str, Any] = {}
    for c in cols:
        arr = table.column(c)
        if c == "signal":
            ca = arr.combine_chunks() if hasattr(arr, "combine_chunks") else arr
            if hasattr(ca, "chunks"):
                ca = ca.chunks[0]
            flat = ca.values.to_numpy(zero_copy_only=False)
            n = table.num_rows
            T = flat.size // n
            out[c] = flat.astype(np.float32).reshape(n, T)
        else:
            out[c] = np.asarray(arr.to_pylist())
    return out


def list_subject_shards(derived_root: Path, subject_filter: set[str] | None = None) -> list[Path]:
    """All `derived/*/sub-<id>/*.parquet` shards (filtered to `subject_filter` if given)."""
    out: list[Path] = []
    for sub_dir in sorted(Path(derived_root).glob("sub-*")):
        sub_id = sub_dir.name[len("sub-"):]
        if subject_filter is not None and sub_id not in subject_filter:
            continue
        for shard in sorted(sub_dir.glob("*.parquet")):
            out.append(shard)
    return out
