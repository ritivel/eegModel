# 17 — Generative paradigm (G0/G1/G2/G3)

- **Notion:** [17 — Generative paradigm (G0/G1/G2/G3)](https://www.notion.so/357939fbcda481e89639f3635c218199)
- **WandB:** project `eegfm`, group `17_generative_paradigm`
- **Status:** see Notion

Design, hypothesis, results, and decisions live in Notion (see link above).
This folder holds per-experiment configs and launch scripts only.

## Run

```bash
eegfm train --wandb-project eegfm --wandb-tags 17_generative_paradigm --paradigm <mae|ar|mar|jepa> --steps <N>
```

See [`BEST_PRACTICES.md`](../../BEST_PRACTICES.md) for the cross-cutting protocol every experiment inherits.
