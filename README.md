# eegModel

Self-supervised pretraining of EEG foundation models. Single-channel iid recipe, frozen-probing eval.

## Where things live

- **Code**: this repo. Core training package is `src/eegfm/`. Operational tooling is `packages/eeg_ops`. Shared building blocks are `packages/eeg_common`.
- **Knowledge** (designs, results, findings, daily standups, decisions): [Notion — EEG Foundation Models — Operations Hub](https://www.notion.so/357939fbcda4813fab4bc5fe0d84ea2c).
- **Run metrics, configs, plots**: WandB project `eegfm`.
- **Run artifacts** (checkpoints, summaries): S3, `s3://eegmodel-warehouse/runs/<group>/<run_id>/`.

For agents: read [`AGENTS.md`](./AGENTS.md). For research/engineering discipline: read [`BEST_PRACTICES.md`](./BEST_PRACTICES.md).

## Layout

```
.
├── AGENTS.md                 ← agent guide; read at session start
├── BEST_PRACTICES.md         ← research / engineering playbook
├── pyproject.toml            ← root `eegfm` package
├── src/eegfm/                ← training, eval, paradigms, CLI
├── packages/
│   ├── eeg_common/           ← preprocessing, storage, pretrained-encoder loaders
│   └── eeg_ops/              ← cluster + Notion CLI; auto-writes Events/Sessions/Runs to Notion
├── experiments/NN_<name>/    ← per-experiment configs + slim README pointing to Notion (01–20)
└── scripts/                  ← cross-experiment shell pipelines and launchers
```

## Quick start

```bash
# Install (CPU dev)
uv pip install -e .

# Install (GPU box)
uv pip install -e ".[gpu]"
uv pip install --no-build-isolation mamba-ssm causal-conv1d

# Discover the CLI
eegfm --help

# Sanity check the pipeline (mini-experiment 01)
eegfm sanity check-all

# Launch one training cell
eegfm train --paradigm mae --steps 17500 --batch-size 32 \
  --wandb-project eegfm --wandb-tags 02_frontend_ablation
```

## Pretrained EEG encoders we depend on

| Model   | arXiv                                          | GitHub                                                                          | Open weights                                                              |
| ------- | ---------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| REVE    | [2510.21585](https://arxiv.org/abs/2510.21585) | [elouayas/reve_eeg](https://github.com/elouayas/reve_eeg)                       | [brain-bzh/reve](https://huggingface.co/collections/brain-bzh/reve)         |
| DIVER-1 | [2512.19097](https://arxiv.org/abs/2512.19097) | [DIVER-1](https://anonymous.4open.science/r/DIVER-1/README.md)                  | [DIVER-1](https://anonymous.4open.science/r/DIVER-1/README.md)              |
| TFM     | [2502.16060](https://arxiv.org/abs/2502.16060) | [Jathurshan0330/TFM-Tokenizer](https://github.com/Jathurshan0330/TFM-Tokenizer) | [Jathurshan/TFM-Tokenizer](https://huggingface.co/Jathurshan/TFM-Tokenizer) |
