# 21 — LeJEPA on EEG (Balestriero & LeCun 2025)

- **Notion:** [21 — LeJEPA on EEG](https://www.notion.so/357939fbcda481219926f3258cf6154b)
- **WandB:** project `eegfm`, group `21_lejepa_eeg`
- **Status:** in_progress

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Setup (one-time, on the GPU box)

The vendored LeJEPA package is a git submodule at `vendor/lejepa/`
(upstream: [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa),
CC BY-NC 4.0 — non-commercial use only). After cloning the eegfm repo
or pulling new code, init the submodule and install it editable into
the eegfm venv (it is *not* pulled in via `pyproject.toml` because
local-path editable installs don't compose well with the published
PyPI extras pattern):

```bash
git submodule update --init --recursive
uv pip install -e vendor/lejepa
```

This makes `import lejepa` available alongside the eegfm package.

## Run

```bash
# Smoke test (single GPU, 100 steps, wandb disabled) — verify wiring
eegfm train --paradigm lejepa --steps 100 --batch-size 16 \
    --wandb-mode disabled

# Headline cell — single seed, ~500h of HBN, lambda=0.02 per LeJEPA MINIMAL.md
eegfm train --paradigm lejepa --steps 25000 --batch-size 64 \
    --blr 5e-4 --weight-decay 5e-2 \
    --lejepa-lambda-sigreg 0.02 \
    --lejepa-n-views-global 2 --lejepa-n-views-local 4 \
    --lejepa-mask-frac-global 0.30 --lejepa-mask-frac-local 0.60 \
    --wandb-project eegfm --wandb-tags 21_lejepa_eeg,lejepa,headline \
    --wandb-run-name 21_lejepa_eeg__hbn_500h__seed0
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting
protocol every experiment inherits.
