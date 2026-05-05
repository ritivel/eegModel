# 13 — Adversarial source-dataset probe

- **Notion:** [13 — Adversarial source-dataset probe](https://www.notion.so/357939fbcda4812cb884cf19d9814954)
- **WandB:** project `eegfm`, group `13_adversarial_dataset_probe`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 13_adversarial_dataset_probe --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
