"""End-to-end smoke test of the eval harness.

Uses synthetic random Gaussian data + a random-init encoder. Validates that:
- The runner dispatch works
- An encoder can be loaded from `random_init`
- A task module can be imported and registered
- Linear probe runs end-to-end and emits well-formed metrics
- Bootstrap CIs are sensible (low ≤ point ≤ high)

These tests need torch + sklearn. Skipped automatically if not available.
Run from the repo root:

    python -m pytest src/eegfm_eval/tests/ -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
sklearn = pytest.importorskip("sklearn")


def test_helpers_lnso_no_subject_overlap():
    """Subject IDs in train and test must be fully disjoint."""
    from eegfm_eval._common import lnso_split
    sub_ids = ["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5 + ["e"] * 5 + ["f"] * 5
    train, test = lnso_split(sub_ids, test_frac=0.33, seed=0)
    train_subs = {sub_ids[i] for i in train}
    test_subs = {sub_ids[i] for i in test}
    assert not (train_subs & test_subs), f"overlap: {train_subs & test_subs}"
    assert train_subs | test_subs == set(sub_ids), "missing subjects"


def test_helpers_bootstrap_ci_sane():
    """Bootstrap CI on a constant-mean distribution should bracket the mean."""
    import numpy as np
    from eegfm_eval._common import bootstrap_ci
    rng = np.random.default_rng(0)
    samples = rng.normal(0.5, 0.1, 200)
    ci = bootstrap_ci(lambda i: float(samples[i].mean()), n=200, n_bootstrap=300, seed=0)
    assert ci["ci_low_95"] <= ci["mean"] <= ci["ci_high_95"]
    assert abs(ci["mean"] - 0.5) < 0.05


def test_helpers_compute_metrics_binary_shape():
    """Binary metrics should include auroc, bac, f1 with point + CI fields."""
    import numpy as np
    from eegfm_eval._common import compute_metrics
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    s = np.array([.1, .9, .2, .8, .1, .9, .2, .8, .4, .4])
    m = compute_metrics(y, y_pred=p, y_score=s, task_type="binary", n_bootstrap=100)
    for k in ("auroc", "bac", "f1"):
        assert k in m, f"missing metric {k}"
        assert "point" in m[k] and "ci_low_95" in m[k] and "ci_high_95" in m[k]
        assert m[k]["ci_low_95"] <= m[k]["point"] <= m[k]["ci_high_95"]


@pytest.mark.skipif(
    not pytest.importorskip("eegfm.model", reason="eegfm not importable"),
    reason="needs the eegfm pretraining package")
def test_runner_smoke_lp_random_init():
    """End-to-end: random-init encoder + smoke task + linear probe → ~0.5 BAC."""
    pytest.importorskip("mamba_ssm", reason="mamba-ssm needed for random_init mamba2 encoder")
    from eegfm_eval import run
    result = run(
        task="smoke", strategy="lp", encoder_kind="random_init",
        derived_root=None, device="cpu", seed=0,
    )
    assert result["task"] == "smoke"
    assert result["strategy"] == "lp"
    assert "metrics" in result and "bac" in result["metrics"]
    bac = result["metrics"]["bac"]["point"]
    # Random labels ⇒ BAC should hover around 0.50 (chance). Allow generous slack.
    assert 0.30 < bac < 0.70, f"BAC {bac} suspiciously far from chance"
