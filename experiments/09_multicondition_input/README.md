# 09 — Multi-condition input (WavLM-style)

- **Notion:** [09 — Multi-condition input (WavLM-style)](https://www.notion.so/357939fbcda4815491e6d91079e64b23)
- **WandB:** project `eegfm`, group `09_multicondition_input`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 09_multicondition_input --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
