# 12 — Quick-wins consolidation

- **Notion:** [12 — Quick-wins consolidation](https://www.notion.so/357939fbcda481959c0ace1339351a53)
- **WandB:** project `eegfm`, group `12_quick_wins_consolidation`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 12_quick_wins_consolidation --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
