# 02 — Frontend ablation

- **Notion:** [02 — Frontend ablation](https://www.notion.so/357939fbcda481c09c2eccb8f4f420e1)
- **WandB:** project `eegfm`, group `02_frontend_ablation`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 02_frontend_ablation --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
