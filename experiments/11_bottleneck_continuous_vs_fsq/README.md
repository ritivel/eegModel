# 11 — Continuous bottleneck vs FSQ

- **Notion:** [11 — Continuous bottleneck vs FSQ](https://www.notion.so/357939fbcda48193ad00fc7774ce59a9)
- **WandB:** project `eegfm`, group `11_bottleneck_continuous_vs_fsq`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 11_bottleneck_continuous_vs_fsq --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
