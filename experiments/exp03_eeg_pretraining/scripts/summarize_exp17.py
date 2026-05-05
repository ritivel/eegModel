"""Aggregate per-cell summary.json files into a single results table.

Usage:
    python scripts/summarize_exp17.py [/path/to/runs/exp17]

For each cell in `runs/exp17/<paradigm>_<control>_seed<N>/summary.json`,
emits a markdown table with:

* train end-step + step/s
* train final loss + composite components (l1_raw, mrstft_logmag)
* eval Protocol A metrics: HBN 6-task BAC, k-NN top-1, CBCL externalising R²,
  attention R², attention-binary AUROC

Plus per-paradigm × per-control aggregates (mean ± std across 5 seeds).

Designed to run on Mac, the GPU box, or anywhere with the runs dir mounted.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PARADIGMS = ("mae", "ar", "mar")
CONTROLS = ("eeg", "noise")
SEEDS = tuple(range(5))


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None
    except Exception as e:                                      # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _row(cell_id: str, summary: dict | None) -> dict:
    if summary is None:
        return {"cell": cell_id, "status": "missing"}
    if "error" in summary:
        return {"cell": cell_id, "status": "error", "msg": summary["error"]}
    cfg = summary.get("config", {})
    eval_a = summary.get("eval_protocol_a", {})
    final_step = summary.get("step", 0)
    elapsed = summary.get("elapsed_s", 0)
    sps = summary.get("step_per_s", 0)
    out = {
        "cell": cell_id,
        "status": "ok" if final_step >= cfg.get("max_steps", 0) - 1 else "partial",
        "step": final_step,
        "elapsed_s": elapsed,
        "step_per_s": sps,
    }
    for k in ("task6_bac", "task6_wf1", "knn_top1_task6",
              "externalizing_r2", "externalizing_mae",
              "attention_r2", "attention_mae", "attention_binary_auroc"):
        sub = eval_a.get(k)
        if isinstance(sub, dict) and "point" in sub:
            out[k] = sub["point"]
            out[f"{k}_ci"] = (sub.get("ci_low_95"), sub.get("ci_high_95"))
    return out


def _agg(rows: list[dict], key: str) -> tuple[float | None, float | None]:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))
            and not (isinstance(r[key], float) and math.isnan(r[key]))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def main(root: Path) -> None:
    rows: list[dict] = []
    for paradigm in PARADIGMS:
        for control in CONTROLS:
            for seed in SEEDS:
                cell = f"{paradigm}_{control}_seed{seed}"
                rows.append(_row(cell, _load(root / cell / "summary.json")))

    # ---- per-cell table -------------------------------------------------
    print(f"# exp17 results — {root}\n")
    print(f"_{sum(r['status'] == 'ok' for r in rows)} of {len(rows)} cells complete_\n")

    print("## Per-cell\n")
    print("| cell | status | step | step/s | task6_bac | knn_top1 | ext_r2 | att_auroc |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        bac = r.get("task6_bac")
        knn = r.get("knn_top1_task6")
        ext = r.get("externalizing_r2")
        att = r.get("attention_binary_auroc")
        print(f"| {r['cell']} | {r['status']} | "
              f"{r.get('step', '-')} | "
              f"{r.get('step_per_s', 0):.2f} | "
              f"{f'{bac:.3f}' if bac is not None else '-'} | "
              f"{f'{knn:.3f}' if knn is not None else '-'} | "
              f"{f'{ext:+.3f}' if ext is not None else '-'} | "
              f"{f'{att:.3f}' if att is not None else '-'} |")

    # ---- per-paradigm × per-control aggregate (mean ± std across 5 seeds) -
    print("\n## Aggregate by paradigm × control (mean ± std across seeds)\n")
    print("| paradigm | control | n | task6_bac | task6_wf1 | knn_top1 | ext_r2 | att_r2 | att_auroc |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for paradigm in PARADIGMS:
        for control in CONTROLS:
            cells = [r for r in rows
                     if r["cell"].startswith(f"{paradigm}_{control}_seed")
                     and r["status"] == "ok"]
            n = len(cells)
            row = f"| {paradigm.upper()} | {control} | {n} |"
            for k in ("task6_bac", "task6_wf1", "knn_top1_task6",
                      "externalizing_r2", "attention_r2", "attention_binary_auroc"):
                m, s = _agg(cells, k)
                if m is None:
                    row += " - |"
                else:
                    row += f" {m:+.3f} ± {s:.3f} |"
            print(row)

    # ---- decision rule check (G0 MAE-EEG vs G1 AR-EEG / G2 MAR-EEG) -----
    print("\n## Decision rule (vs G0 MAE-EEG baseline, primary metric = HBN 6-task BAC)\n")
    g0_eeg = [r for r in rows if r["cell"].startswith("mae_eeg") and r["status"] == "ok"]
    g0_bac, g0_std = _agg(g0_eeg, "task6_bac")
    if g0_bac is None:
        print("_G0 EEG cells not yet complete — decision rule pending_")
        return
    print(f"- **G0 MAE-EEG baseline**: BAC = {g0_bac:.3f} ± {g0_std:.3f} (n={len(g0_eeg)})")
    for paradigm, label in (("ar", "G1 AR-EEG"), ("mar", "G2 MAR-EEG")):
        cells = [r for r in rows if r["cell"].startswith(f"{paradigm}_eeg") and r["status"] == "ok"]
        m, s = _agg(cells, "task6_bac")
        if m is None:
            print(f"- **{label}**: not yet complete")
            continue
        delta = m - g0_bac
        flag = "STRICT-WIN" if delta >= 0.02 else ("WEAK-WIN" if delta >= 0.01 else
              ("LOSS" if delta <= -0.01 else "TIE"))
        print(f"- **{label}**: BAC = {m:.3f} ± {s:.3f} (n={len(cells)}) — Δ = {delta:+.3f} ⇒ **{flag}**")

    # ---- noise-twin sanity (each paradigm's noise-twin should NOT win) --
    print("\n## Noise-twin sanity (each paradigm's noise cells should be ≤ EEG cells)\n")
    for paradigm in PARADIGMS:
        eeg_cells = [r for r in rows if r["cell"].startswith(f"{paradigm}_eeg") and r["status"] == "ok"]
        noise_cells = [r for r in rows if r["cell"].startswith(f"{paradigm}_noise") and r["status"] == "ok"]
        e, _ = _agg(eeg_cells, "task6_bac")
        n, _ = _agg(noise_cells, "task6_bac")
        if e is None or n is None:
            continue
        delta = e - n
        flag = "OK" if delta > 0 else "WARN: noise ≥ EEG, signal hypothesis under threat"
        print(f"- **{paradigm.upper()}**: EEG {e:.3f} vs noise {n:.3f} (Δ = {delta:+.3f}) — {flag}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/dlami/nvme/eeg/runs/exp17")
    main(root)
