# 19 — Decoder design (depth × type)

- **Notion:** [19 — Decoder design (depth × type)](https://www.notion.so/357939fbcda481c4ac01e7f2972978e6)
- **WandB:** project `eegfm`, group `19_decoder_design`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 19_decoder_design --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
