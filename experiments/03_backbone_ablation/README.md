# 03 — Backbone ablation

- **Notion:** [03 — Backbone ablation](https://www.notion.so/357939fbcda48118929be940d84e9f7a)
- **WandB:** project `eegfm`, group `03_backbone_ablation`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 03_backbone_ablation --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
