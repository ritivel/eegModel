"""Reproduction: validate the harness by reproducing LaBraM-Base's published
TUAB AUROC of 0.9022 ± 0.0009 (LaBraM ICLR 2024, Table 1).

This is the keystone correctness test: if it passes, the entire eval pipeline
(data loading, preprocessing, channel mapping, training loop, optimiser,
metrics, splits) is validated on the most-cited EEG-FM benchmark cell.

Cost: ~$1.50 on Lambda 1×A100 (~45 min full FT). Default-skipped to avoid
accidental GPU spend; opt in via:

    EEGFM_EVAL_REPRODUCE=1 pytest src/eegfm_eval/tests/test_reproduce_labram.py -v -s

Required env on the GPU box:
    EEG_DATA_ROOT          /opt/dlami/nvme/eeg (default)
    + raw/tuab/v3.0.1/edf/{train,eval}/{normal,abnormal}/01_tcp_ar/*.edf  (NEDC sync)
    + braindecode>=1.2 + mamba-ssm  (for the eegfm encoder, not strictly for this test)

Pass criterion: |measured - 0.9022| < 0.010   (1% absolute slack on AUROC).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EEGFM_EVAL_REPRODUCE", "0") != "1",
    reason="set EEGFM_EVAL_REPRODUCE=1 to run reproduction tests (~45 min on 1×A100, ~$1.50)",
)

torch = pytest.importorskip("torch")
braindecode = pytest.importorskip("braindecode")


# ---------------------------------------------------------------------------


def test_reproduce_labram_base_tuab_ft():
    """LaBraM-Base + our tuab/ft → AUROC 0.9022 ± 0.010 (LaBraM ICLR 2024)."""
    from eegfm_eval import run

    derived_root = os.environ.get("EEG_DATA_ROOT", "/opt/dlami/nvme/eeg") + "/derived"
    result = run(
        task="tuab",
        strategy="ft",
        encoder_kind="labram_base",
        checkpoint=None,                          # auto-download from HF
        derived_root=derived_root,
        device="cuda",
        seed=0,
        epochs=50, batch_size=64, lr=5e-4,
    )

    auroc = result["metrics"]["auroc"]["point"]
    target = 0.9022
    delta = abs(auroc - target)
    print(f"\n  measured AUROC: {auroc:.4f}")
    print(f"  target  AUROC:  {target:.4f} (LaBraM ICLR 2024, Table 1)")
    print(f"  |delta|:        {delta:.4f}")
    print(f"  CI:             [{result['metrics']['auroc']['ci_low_95']:.4f}, "
          f"{result['metrics']['auroc']['ci_high_95']:.4f}]")
    assert delta < 0.010, (
        f"LaBraM-Base TUAB AUROC = {auroc:.4f}, target = {target:.4f}, "
        f"|delta| = {delta:.4f} > 0.010 tolerance. Harness likely has a bug "
        f"in: preprocessing, channel mapping, splits, or training recipe."
    )


def test_reproduce_labram_base_tuev_ft():
    """LaBraM-Base + our tuev/ft → Cohen's κ 0.6637 ± 0.010 (LaBraM ICLR 2024)."""
    from eegfm_eval import run

    derived_root = os.environ.get("EEG_DATA_ROOT", "/opt/dlami/nvme/eeg") + "/derived"
    result = run(
        task="tuev",
        strategy="ft",
        encoder_kind="labram_base",
        checkpoint=None,
        derived_root=derived_root,
        device="cuda",
        seed=0,
        epochs=50, batch_size=64, lr=5e-4,
    )

    kappa = result["metrics"]["cohen_kappa"]["point"]
    target = 0.6637
    delta = abs(kappa - target)
    print(f"\n  measured κ: {kappa:.4f}")
    print(f"  target  κ:  {target:.4f}")
    print(f"  |delta|:    {delta:.4f}")
    assert delta < 0.010, (
        f"LaBraM-Base TUEV κ = {kappa:.4f}, target = {target:.4f}, "
        f"|delta| = {delta:.4f} > 0.010."
    )


def test_reproduce_cbramod_physionet_mi_ft():
    """CBraMod + our physionet_mi/ft → Cohen's κ 0.5222 ± 0.010 (CBraMod ICLR 2025)."""
    from eegfm_eval import run

    derived_root = os.environ.get("EEG_DATA_ROOT", "/opt/dlami/nvme/eeg") + "/derived"
    result = run(
        task="physionet_mi",
        strategy="ft",
        encoder_kind="cbramod",
        checkpoint=None,
        derived_root=derived_root,
        device="cuda",
        seed=0,
        epochs=50, batch_size=64, lr=5e-4,
    )

    kappa = result["metrics"]["cohen_kappa"]["point"]
    target = 0.5222
    delta = abs(kappa - target)
    print(f"\n  measured κ: {kappa:.4f}")
    print(f"  target  κ:  {target:.4f}")
    print(f"  |delta|:    {delta:.4f}")
    assert delta < 0.010, (
        f"CBraMod PhysioNet-MI κ = {kappa:.4f}, target = {target:.4f}, "
        f"|delta| = {delta:.4f} > 0.010."
    )
