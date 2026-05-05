# 20 — Position embedding

- **Notion:** [20 — Position embedding](https://www.notion.so/357939fbcda48162a020ee88473da42b)
- **WandB:** project `eegfm`, group `20_position_embedding`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 20_position_embedding --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
