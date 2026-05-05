# 08 — Denoising target

- **Notion:** [08 — Denoising target](https://www.notion.so/357939fbcda48151bb9dd4b2b3041988)
- **WandB:** project `eegfm`, group `08_denoising_target`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 08_denoising_target --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
