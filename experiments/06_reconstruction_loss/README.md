# 06 — Reconstruction loss

- **Notion:** [06 — Reconstruction loss](https://www.notion.so/357939fbcda4813bab2ccc9cd7e200b3)
- **WandB:** project `eegfm`, group `06_reconstruction_loss`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 06_reconstruction_loss --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
