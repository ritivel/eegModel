# 15 — Loss-weight sensitivity & curriculum

- **Notion:** [15 — Loss-weight sensitivity & curriculum](https://www.notion.so/357939fbcda48114bb1bd65005d9732b)
- **WandB:** project `eegfm`, group `15_loss_weight_curriculum`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 15_loss_weight_curriculum --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
