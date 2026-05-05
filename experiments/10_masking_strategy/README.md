# 10 — Masking strategy × ratio

- **Notion:** [10 — Masking strategy × ratio](https://www.notion.so/357939fbcda48122af6be8854de4398e)
- **WandB:** project `eegfm`, group `10_masking_strategy`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 10_masking_strategy --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
