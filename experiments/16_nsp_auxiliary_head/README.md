# 16 — NSP auxiliary head

- **Notion:** [16 — NSP auxiliary head](https://www.notion.so/357939fbcda481c0a2f7dee64e813f47)
- **WandB:** project `eegfm`, group `16_nsp_auxiliary_head`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 16_nsp_auxiliary_head --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
