# 04 — SSL framework ablation

- **Notion:** [04 — SSL framework ablation](https://www.notion.so/357939fbcda4818bb1b8e777825ccf74)
- **WandB:** project `eegfm`, group `04_ssl_framework_ablation`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 04_ssl_framework_ablation --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
