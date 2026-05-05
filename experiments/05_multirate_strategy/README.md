# 05 — Multi-rate strategy

- **Notion:** [05 — Multi-rate strategy](https://www.notion.so/357939fbcda48117a005cae77090ad8f)
- **WandB:** project `eegfm`, group `05_multirate_strategy`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 05_multirate_strategy --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
