# 07 — Phase handling

- **Notion:** [07 — Phase handling](https://www.notion.so/357939fbcda48195ab79eaec88935e3d)
- **WandB:** project `eegfm`, group `07_phase_handling`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 07_phase_handling --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
