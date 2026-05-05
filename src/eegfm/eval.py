"""Encoder-health diagnostics for the pretrained `EEGSSLModel`.

This module is **not** the downstream-task eval — that lives in `eegfm_eval`.
This module only checks whether a frozen encoder learned anything useful, or
whether it just memorised subject identity / collapsed dimensionally.

Three cheap probes, each a few lines:

- `lnso_split`           subject-disjoint train/test indices (Leave-N-Subjects-Out).
- `bootstrap_ci`         95 % bootstrap CI on any scalar metric.
- `diagnostics`          encoder-health probes (subject-ID, site-ID, eff-rank).

References:

- Subject-ID fingerprint: ContentVec ICML 2022 — HuBERT hits ~81 % speaker-ID
  linear-probe accuracy, which signals that the encoder dominantly encodes
  speaker identity rather than content. Threshold > 0.60 in our suite.
- Effective rank / dim collapse: Roy & Vetterli 2007 entropy rank;
  Jing et al. ICLR 2022 (`https://openreview.net/forum?id=YevsQ05DEN7`).
  Threshold `rank_ratio < 0.5` flags collapse.
- Site fingerprint: CRCC arXiv:2602.19138 — EEG models often memorise
  site / amplifier hardware rather than neural signal.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------


def lnso_split(
    subject_ids: list[str], *, test_frac: float = 0.30, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-N-Subjects-Out: returns (train_idx, test_idx); no subject in both."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(subject_ids))
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_frac)))
    test_subs = set(unique[:n_test])
    sub_arr = np.asarray(subject_ids)
    test_idx = np.where(np.isin(sub_arr, list(test_subs)))[0]
    train_idx = np.where(~np.isin(sub_arr, list(test_subs)))[0]
    return train_idx, test_idx


# ---------------------------------------------------------------------------


def bootstrap_ci(
    fn: Callable[[np.ndarray], float],
    *,
    n: int,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap-CI on a scalar metric.

    `fn` takes a resampled-with-replacement index array of shape `(n,)`
    and returns one float. We resample `n_bootstrap` times and report the
    central `ci` of the empirical distribution.
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


def diagnostics(
    features: np.ndarray,
    subject_ids: list[str],
    *,
    sites: list[str] | None = None,
    seed: int = 0,
    subject_id_max: int = 50,
) -> dict[str, Any]:
    """Cheap encoder-health probes on a `(N, D)` feature matrix.

    Returns three groups, all under one dict so a caller can JSON-dump:

    - **effective_rank** / **rank_ratio** / **top1_eigenvalue_share**:
      entropy rank of the feature covariance. `rank_ratio < 0.5` ⇒ collapse.
    - **subject_id_acc**: a logreg trained to predict subject ID from frozen
      features (in-distribution split, NOT LNSO — we explicitly want to know
      if subjects are linearly separable). `> 0.60` ⇒ fingerprint dominance.
    - **site_id_acc**: same for the recording site if multi-site data is
      passed via `sites`. `> 0.60` ⇒ site confound.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    out: dict[str, Any] = {}
    X = features
    n, d = X.shape

    # 1. Effective rank of feature covariance --------------------------------
    Xz = StandardScaler(with_mean=True, with_std=False).fit_transform(X)
    try:
        s = np.linalg.svd(Xz / max(np.sqrt(n - 1), 1.0), compute_uv=False)
        eigvals = s ** 2
        eigvals = eigvals[eigvals > 1e-12]
        if eigvals.size > 0:
            p = eigvals / eigvals.sum()
            entropy = float(-(p * np.log(p)).sum())
            eff_rank = float(np.exp(entropy))
            out["effective_rank"] = eff_rank
            out["rank_ratio"] = float(eff_rank / d)
            out["top1_eigenvalue_share"] = float(p[0])
            out["top5_eigenvalue_share"] = float(p[: min(5, p.size)].sum())
        else:
            out["effective_rank"] = 0.0
            out["rank_ratio"] = 0.0
    except np.linalg.LinAlgError as e:                          # noqa: BLE001
        out["effective_rank"] = float("nan")
        out["error_svd"] = str(e)

    # 2. Subject-ID linear probe --------------------------------------------
    rng = np.random.default_rng(seed)
    sub_arr = np.asarray(subject_ids)
    unique_subs, counts = np.unique(sub_arr, return_counts=True)
    keep_n = min(subject_id_max, unique_subs.size)
    if keep_n >= 2:
        top_subs_idx = np.argsort(counts)[-keep_n:]
        keep_mask = np.isin(sub_arr, unique_subs[top_subs_idx])
        Xs = X[keep_mask]
        ys_str = sub_arr[keep_mask]
        sub_to_label = {s: i for i, s in enumerate(sorted(set(ys_str.tolist())))}
        ys = np.asarray([sub_to_label[s] for s in ys_str])
        idx = np.arange(Xs.shape[0])
        rng.shuffle(idx)
        n_test = max(int(idx.size * 0.30), keep_n)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        try:
            scaler = StandardScaler()
            clf = LogisticRegression(max_iter=2000, n_jobs=1)
            clf.fit(scaler.fit_transform(Xs[train_idx]), ys[train_idx])
            acc = float(accuracy_score(
                ys[test_idx], clf.predict(scaler.transform(Xs[test_idx]))
            ))
            out["subject_id_probe"] = {
                "accuracy": acc,
                "chance": 1.0 / keep_n,
                "n_subjects": int(keep_n),
            }
        except Exception as e:                                  # noqa: BLE001
            out["subject_id_probe"] = {"error": f"{type(e).__name__}: {e}"}

    # 3. Site-ID linear probe (multi-site only) ------------------------------
    if sites is not None:
        site_arr = np.asarray(sites)
        unique_sites = np.unique(site_arr)
        if unique_sites.size >= 2:
            site_to_label = {s: i for i, s in enumerate(sorted(unique_sites.tolist()))}
            ys = np.asarray([site_to_label[s] for s in site_arr])
            idx = np.arange(X.shape[0])
            rng.shuffle(idx)
            n_test = max(int(idx.size * 0.30), unique_sites.size)
            test_idx, train_idx = idx[:n_test], idx[n_test:]
            try:
                scaler = StandardScaler()
                clf = LogisticRegression(max_iter=2000, n_jobs=1)
                clf.fit(scaler.fit_transform(X[train_idx]), ys[train_idx])
                acc = float(accuracy_score(
                    ys[test_idx], clf.predict(scaler.transform(X[test_idx]))
                ))
                out["site_id_probe"] = {
                    "accuracy": acc,
                    "chance": 1.0 / unique_sites.size,
                    "n_sites": int(unique_sites.size),
                }
            except Exception as e:                              # noqa: BLE001
                out["site_id_probe"] = {"error": f"{type(e).__name__}: {e}"}
        else:
            out["site_id_probe"] = {"reason": "single-site data"}

    return out
