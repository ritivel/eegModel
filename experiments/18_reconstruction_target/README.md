# 18 — Reconstruction target

- **Notion:** [18 — Reconstruction target](https://www.notion.so/357939fbcda4812a96dad30190aea508)
- **WandB:** project `eegfm`, group `18_reconstruction_target`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 18_reconstruction_target --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
