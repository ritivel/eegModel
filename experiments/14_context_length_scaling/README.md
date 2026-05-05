# 14 — Context-length scaling

- **Notion:** [14 — Context-length scaling](https://www.notion.so/357939fbcda4813aa6d6ecb77dce02f5)
- **WandB:** project `eegfm`, group `14_context_length_scaling`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 14_context_length_scaling --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
