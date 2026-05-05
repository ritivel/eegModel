# Agent guide

You are working in `eegModel`, a research codebase for **self-supervised pretraining of EEG foundation models**. This file is your starting orientation.

## What lives where

- **Code**: this repo. `src/eegfm/` is the core training/eval package. `packages/eeg_common` holds shared building blocks. `packages/eeg_ops` is the operational CLI (cluster lifecycle, Notion writes, S3 sync).
- **Run metrics, configs, plots**: WandB project `eegfm`. Each experiment runs as a WandB group named like `02_frontend_ablation`.
- **Run artifacts (checkpoints, summaries)**: S3, `s3://eegmodel-warehouse/runs/<group>/<run_id>/`.
- **Knowledge** (methodology, hypotheses, results, decisions, insights, daily standups): **Notion**, under the page **"EEG Foundation Models — Operations Hub"** (`357939fbcda4813fab4bc5fe0d84ea2c`). Five databases: Events, Sessions, Experiments, Runs, Findings, Standups.

## Repo layout

```
.
├── AGENTS.md                 ← you are here
├── BEST_PRACTICES.md         ← durable research/engineering discipline; READ THIS before designing or reviewing experiments
├── README.md                 ← orientation + links
├── pyproject.toml            ← root `eegfm` package
├── .cursor/rules/            ← Cursor-specific auto-attached guidance
├── src/eegfm/                ← the training package (`eegfm` CLI, model, data, train, eval, paradigms, …)
├── packages/
│   ├── eeg_common/           ← shared building blocks
│   └── eeg_ops/              ← cluster + Notion CLI; writes Events/Sessions/Runs to Notion automatically
├── experiments/NN_<name>/    ← per-experiment configs, launch scripts, slim README pointing to Notion
└── scripts/                  ← cross-experiment shell (full_hbn_pipeline.sh, launch_*.sh, watchdogs)
```

## Conventions

- **Never write progress/results/methodology markdown into the repo.** Those go in Notion. The repo holds only code, configs, and durable engineering docs (`BEST_PRACTICES.md`, this file, `.cursor/rules/`).
- **Every training run logs to WandB** with `wandb_project="eegfm"`, group equal to the experiment folder name, run name `{group}__{config}__seed{n}`, and tags including the experiment ID and paradigm.
- **Every consequential decision, finding, or run goes to Notion** via the `eeg-ops` CLI (auto-written for cluster events) or the Notion MCP (manual entries).
- **All paths and S3 prefixes resolve from `$EEG_DATA_ROOT`** (default `/opt/dlami/nvme/eeg`).
- **Experiment IDs (01–20) are stable identifiers**, not ordering — never renumber.

## When you need context that isn't in the repo

Use the Notion MCP. Do not ask the user. Examples:
- "What's the hypothesis for experiment 14?" → `notion-search` for "14 — Context-length scaling" or fetch the Experiments DB.
- "What was decided about masking?" → search Findings DB.
- "What runs are currently active?" → query Runs DB filtered by status.

## Don't

- Don't add new top-level prose markdown files for narrative content.
- Don't create a `progress.md`, `results.md`, `findings.md`, etc. anywhere in the repo. All of those go in Notion.
- Don't hardcode S3 paths; use `eegfm.storage.from_env()`.
- Don't bypass `eeg-ops events log` for important manual decisions — it writes to Notion automatically.
