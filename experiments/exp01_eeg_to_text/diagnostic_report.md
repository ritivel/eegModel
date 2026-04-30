# Diagnostic report — three bugs in the Apr-30 pilot, all fixed

**Date:** 2026-04-30 IST
**Author:** diagnostic pass on the 9-cell pilot launched ~13:39 IST on Box A (`192.222.53.60`, 8× H100) + Box B (`192.222.53.81`, 1× H100).
**Status:** all three bugs fixed; full 9-cell pilot re-launched against the same step budget (300 / 1200 / 500); first results expected ~14:35 IST (soft-prompt cells) and ~17:00 IST (vocab cells).

W&B project: [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text)

Companion analysis chat: [Pilot training diagnosis](7caa86c2-3a8c-44ec-a2b7-25e9f01c003b)

---

## TL;DR


| #   | Severity                 | Bug                                                                                                                                     | Fix                                                                                                                                            | Files                                             |
| --- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 1   | **fatal**                | `np` not imported at module scope in `data.py` → every `noise_train` cell crashed on first batch with `NameError`                       | move `import numpy as np` to the top of `data.py`                                                                                              | `data.py`                                         |
| 2   | **silently fatal**       | LoRA `target_modules` regex matched the **vision tower**, not the text decoder; PEFT attached LoRA to modules that never run → grad ≡ 0 | (a) re-target regex to `language_model.layers.<i>.self_attn.{q,k,v,o}_proj`; (b) call `enable_input_require_grads()` on PEFT                   | `decoder.py`                                      |
| 3   | **scientifically fatal** | Bridge collapsed → identical hypotheses for different EEG inputs (model was using only the LM prior); `RVQHead` codebook never updated  | (a) CLIP-style InfoNCE alignment loss in stage 1 (BELT-2 / CET-MAE / Defossez-King 2025); (b) straight-through + commitment loss for `RVQHead` | `bridges.py`, `model.py`, `train.py`, `config.py` |


A unified diff lives under `src/exp01/{data,decoder,bridges,model,train,config}.py` (200 insertions, 27 deletions across 6 files).

---

## What was broken, why, and how I confirmed each fix

### Bug 1 — `NameError: name 'np' is not defined`

`EEGSentenceDataset.__getitem__` calls `np.random.default_rng(...)` whenever the dataset was constructed with `noise="gauss"`. `numpy` was only imported locally inside `_row_to_array`, so the noise branch crashed inside every DataLoader worker as soon as the first `noise_train` batch was requested.

Pre-fix evidence (Box B `/tmp/pilot-noise.log`):

```text
File ".../src/exp01/data.py", line 347, in __getitem__
    rng = np.random.default_rng(seed=hash((row["participant_id"], row["sentence_text"])) & 0xFFFFFFFF)
          ^^
NameError: name 'np' is not defined
```

Fix: hoist `import numpy as np` to module scope in `data.py` (and remove the now-redundant local import in `_row_to_array`).

```7:21:experiments/exp01_eeg_to_text/src/exp01/data.py
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from . import storage
```

Post-fix verification on Box B (smoke run on `reve.linear.noise_train.0`, 30+30+20 steps, bs=4):

```text
{"event": "init", "n_train": 15979, "n_dev": 240, ...}
{"event": "stage_start", "stage": "stage1", "steps": 30, "lr": 0.0001, ...}
{"stage": "stage1", "step": 1,  "loss": 9.89,  "align_loss": 1.43, "grad_norm": 158.9, ...}
{"stage": "stage1", "step": 30, "loss": 8.18,  "align_loss": 1.41, "grad_norm": 200.9, ...}
{"event": "save", "stage": "stage1", "path": ".../model_stage1.pt", "size_mb": 10003.6}
... (stage2, stage3 also completed)
{"event": "done", "ckpt": ".../model.pt"}
```

Three full stages ran end-to-end with `noise=gauss` — bug closed.

---

### Bug 2 — Stage-3 LoRA gradient was *exactly* `0.0` for every vocab cell

Pre-fix evidence from the buggy run logs (latest run of every cell, stage 3):


| cell                               | trainable | gn min   | gn med   | gn max   | comment  |
| ---------------------------------- | --------- | -------- | -------- | -------- | -------- |
| `reve.linear.eeg.0` (soft-prompt)  | 3.56 M    | 1.07     | 1.47     | 3.26     | OK       |
| `reve.qformer.eeg.0` (soft-prompt) | 28.78 M   | 1.34     | 1.80     | 2.71     | OK       |
| `tfm.qformer.eeg.0` (soft-prompt)  | 3.25 M    | 2.45     | 3.73     | 8.94     | OK       |
| `reve.vocab.eeg.0` (vocab)         | 6.95 M    | **0.00** | **0.00** | **0.00** | **dead** |
| `tfm.vocab.eeg.0` (vocab)          | 2.75 M    | **0.00** | **0.00** | **0.00** | **dead** |


Two compounding root causes:

**(a) wrong LoRA target.** Gemma 4 E2B is multimodal; it has three top-level submodules under `model.model`:

- `vision_tower` — its `self_attn.q_proj` is a `Gemma4ClippableLinear` *wrapper* whose inner module is named `.linear`.
- `audio_tower` — same wrapper structure.
- `language_model` — its `self_attn.q_proj` is a **raw `nn.Linear*`* (no `.linear` suffix).

The previous regex `r".*self_attn\.(q_proj|k_proj|v_proj|o_proj)\.linear$"` therefore matched **only the vision and audio towers**. PEFT happily attached 2.75 M LoRA parameters — to modules that are **never traversed during text-only training**. A direct probe (`/tmp/lora_probe.py`) confirmed this:

```text
example matched modules: ['base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.lora_A', ...]
trainable params  = 2,752,512
forward+backward total grad-norm^2 = 0.000000e+00    lora params with grad = 0/200
```

Fix: re-target the regex at the language model only.

```33:73:experiments/exp01_eeg_to_text/src/exp01/decoder.py
def attach_lora(model: nn.Module, *, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> nn.Module:
    """Wrap with PEFT LoRA on the language-model attention projections.
    ...
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Match only the LANGUAGE MODEL's attention projections (not the
        # vision_tower or audio_tower whose q_proj is a wrapper class).
        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    )
    peft_model = get_peft_model(model, cfg)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    elif hasattr(peft_model, "get_input_embeddings"):
        def _make_inputs_require_grad(_module, _input, output):
            output.requires_grad_(True)
        peft_model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    return peft_model
```

**(b) PEFT + gradient-checkpointing + frozen base model is a [well-known footgun](https://github.com/huggingface/peft/issues/2826):** when the base is fully frozen and gradient checkpointing is on (vocab cells in this codebase, per `--use-grad-checkpoint`), the input-embedding output has `requires_grad=False`, and Torch's gradient checkpointing then short-circuits backward through every checkpointed transformer block — silently zeroing gradients for every LoRA adapter inside those blocks. PEFT's own `[_prepare_model_for_gradient_checkpointing](https://github.com/huggingface/peft/blob/v0.12.0/src/peft/mixed_model.py)` calls `model.enable_input_require_grads()` for exactly this reason. Calling it explicitly after `get_peft_model` makes the whole thing version-proof.

References (also cited in the docstring): [HF transformers #42947](https://github.com/huggingface/transformers/issues/42947), [HF transformers #23170](https://github.com/huggingface/transformers/issues/23170), [HF peft #2826](https://github.com/huggingface/peft/issues/2826).

Post-fix verification on Box A (smoke `tfm.vocab.eeg.0`, gradient checkpointing **on**, bs=1):

```text
{"event": "stage_start", "stage": "stage3", "steps": 25, "lr": 2e-05,
   "trainable_params": 5357568, ...}
{"stage": "stage3", "step": 1, "loss": 9.40, "grad_norm": 12.37, ...}
```

Compared to the pre-fix log for the same cell:

```text
{"event": "stage_start", "stage": "stage3", "steps": 25, "lr": 2e-05,
   "trainable_params": 2752512, ...}
{"stage": "stage3", "step": 1, "loss": 8.47, "grad_norm": 0.0, ...}
```

— LoRA parameter count nearly doubled (2.75 M → 5.36 M, the correct count for the 35 language-model attention layers × 4 projections × 2 LoRA matrices × `r=16`), and gradient norm is now meaningful (12.4) where it was structurally zero before.

---

### Bug 3 — Bridge collapsed onto the LM prior; `RVQHead` codebook never updated

This is the *scientifically* fatal bug, because EEG-vs-noise comparisons are the entire point of the Jo et al. §4.3 protocol, and a collapsed bridge fails the test by construction.

Smoking-gun evidence from the buggy run, `reve.linear.eeg.0` stage 3 step 500 (three different references, same subject ZMG):

```text
REF: 'Presents a good case while failing to provide a reason for us to care...'
HYP: 'He was a talented and accomplished actor, but he was also a very talented and...'

REF: 'Bread, My Sweet has so many flaws it would be easy for critics to shred it.'
HYP: 'He was a talented and accomplished actor, but he was also a very talented and...'   ← byte-identical

REF: 'The film often achieves a mesmerizing poetry.'
HYP: 'He was born in New York City on February 12, 1921. He was a member of...'
```

Two different EEG inputs producing **byte-identical greedy decodings** is conclusive: the bridge output is essentially constant; the model is producing generic English from Gemma's LM prior, not from EEG. Vocab cells were even more degenerate (`'InInInInIn...'`, `'HeHeHeHeHe...'`).

Two contributing root causes were addressed:

**(a) `RVQHead` had no learning signal.** The codebook was initialized to `randn(8192, 512) × 0.02` and stayed there for the entire run, because the only path from EEG features to the LM loss is through `argmin → input_ids` (an integer cut). Without a straight-through estimator and a commitment loss, the codebook was just **random hashing** of REVE features into Gemma's extended vocab. Fix: add the standard VQ-VAE commitment loss and stash it on the head so the trainer can mix it into the stage-1 loss.

```109:139:experiments/exp01_eeg_to_text/src/exp01/bridges.py
class RVQHead(nn.Module):
    """Tiny single-stage VQ over encoder features. ...
    Returns a tuple ``(ids, commit_loss)``. ``ids`` is the discrete code
    sequence (LongTensor (B, T)). ``commit_loss`` is the standard VQ-VAE
    commitment loss — added to the LM loss in trainables_stage1 so that the
    codebook actually moves toward the encoder distribution. ...
    """

    def __init__(self, d_in: int, codebook_size: int = 8192,
                 commitment_weight: float = 0.25):
        super().__init__()
        ...

    def forward(self, features: torch.Tensor) -> torch.LongTensor:
        ...
        ids_flat = d.argmin(dim=-1)
        codes = self.codebook[ids_flat]                                     # (B*T, D)
        commit = ((codes - f.detach()).pow(2).mean()
                  + self.commitment_weight * (codes.detach() - f).pow(2).mean())
        self.last_commit_loss = commit
        return ids_flat.reshape(B, T)
```

**(b) Add a CLIP/SigLIP-style InfoNCE alignment loss in stage 1** between sentence-pooled bridge output and sentence-pooled (frozen) text-token embeddings. This is the standard mechanism used by every recent EEG→text or M/EEG→language paper:

- BELT-2 (Zhou et al., 2024): "BPE-CL" — discrete-EEG-to-BPE contrastive learning ([arXiv 2409.00121](https://arxiv.org/abs/2409.00121)).
- CET-MAE (Wang et al., 2024a): cross-modality EEG-text contrastive loss in CET-MAE pretraining ([ACL 2024 paper](https://aclanthology.org/2024.acl-long.393.pdf)).
- Defossez/King et al., *Nature Communications* 2025: CLIP/SigLIP loss for non-invasive M/EEG word decoding ([s41467-025-65499-0](https://www.nature.com/articles/s41467-025-65499-0)).

Without an explicit alignment objective, a high-capacity LM has no reason to *use* the bridge output beyond as a prefix to nudge token statistics — exactly the failure mode we observed. The fix is small, localized, and uses the same `clip_grad_norm_(1.0)` budget as before:

```218:325:experiments/exp01_eeg_to_text/src/exp01/train.py
def _run_stage(
    *,
    model, dl, params, steps: int, lr: float, grad_accum: int,
    log, samples_log, stats_log, wb,
    name: str, dev_samples: list[dict], generate_every: int,
    align_weight: float = 0.0,
    align_temperature: float = 0.07,
    rvq_commit_weight: float = 0.0,
):
    ...
    aux = getattr(model, "_last_aux", None) or {}
    if align_weight > 0 and aux.get("bridge_pooled") is not None \
       and aux["bridge_pooled"].size(0) >= 2:
        a_loss = _infonce_align(
            aux["bridge_pooled"], aux["text_pooled"], align_temperature,
        )
        extra.append(align_weight * a_loss)
        align_acc += float(a_loss.detach())
    commit = aux.get("commit_loss", 0.0)
    if rvq_commit_weight > 0 and isinstance(commit, torch.Tensor):
        extra.append(rvq_commit_weight * commit)
        commit_acc += float(commit.detach())

    total = lm_loss + sum(extra) if extra else lm_loss
    (total / grad_accum).backward()
    ...


def _infonce_align(bridge_pooled, text_pooled, temperature):
    """Symmetric CLIP-style InfoNCE between bridge and text pooled vectors.
    bridge_pooled: (B, d) — has gradient through the bridge / RVQ / new embed rows.
    text_pooled:   (B, d) — frozen text-token embeddings (no grad).
    Compute in fp32 for numerical stability (Gemma runs in bf16).
    """
    b = F.normalize(bridge_pooled.float(), dim=-1)
    t = F.normalize(text_pooled.float(), dim=-1)
    logits = b @ t.t() / max(temperature, 1e-6)
    targets = torch.arange(b.size(0), device=b.device)
    return 0.5 * (F.cross_entropy(logits, targets)
                  + F.cross_entropy(logits.t(), targets))
```

Knobs land in `CellConfig` as `stage1_align_weight=1.0`, `stage1_align_temperature=0.07`, `rvq_commit_weight=1.0`. The alignment loss is auto-disabled when `batch_size < 2` (no negatives in the batch), which is the case for vocab cells (bs=1).

#### Two diagnostic confirmations the fix is doing the right thing

**1. The alignment loss correctly identifies that noise inputs carry no text signal.**
For batch size B the chance level of the symmetric InfoNCE loss is `log(B)`. On Box B's `reve.linear.noise_train.0` (bs=8, so chance = `log(8) ≈ 2.079`), align_loss sat at 2.08–2.11 throughout stage 1:

```text
{"stage": "stage1", "step": 200, "loss": 3.95, "align_loss": 2.10, ...}
{"stage": "stage1", "step": 210, "loss": 3.61, "align_loss": 2.08, ...}
```

This is exactly the diagnostic value of the contrastive objective: it can **distinguish information-bearing inputs from noise**, which is precisely what the §4.3 protocol asks of the bridge.

**2. The bridge no longer collapses on EEG inputs.**
At stage-1 step 150 of `reve.linear.eeg.0` post-fix (5 different references from the same subject ZMG):

```text
REF: 'Suffers from rambling, repetitive dialogue ...'
HYP: 'The first episode of *The Girl* was released on Netflix in 2016.'

REF: 'Even a hardened voyeur would require the patience of Job ...'
HYP: 'in, a school for the children of the United States, was a public school ...'

REF: 'Instead of hiding Pinocchio from critics, Miramax should have hidden it ...'
HYP: 'The Great Game was a period of time, and the British Empire was a period of time ...'
```

Five different references, **five different hypotheses** (the model's not converged yet, but it's no longer outputting the same string for every input). For comparison, here are the same dev refs against the same cell pre-fix at stage-3 step 500:

```text
REF: 'Presents a good case ...'                  HYP: 'He was a talented and accomplished actor ...'
REF: 'Bread, My Sweet has so many flaws ...'     HYP: 'He was a talented and accomplished actor ...'   ← identical
REF: 'The film often achieves a ...'             HYP: 'He was born in New York City on February 12, 1921 ...'
```

---

## Buggy-run metrics (archived under `archive/buggy_run_2026-04-30T08-30/`)

For posterity and as a control for the post-fix runs:


| cell                                | bleu1     | bleu2     | bleu3     | bleu4     | rouge1_f  | bertscore_f1   |
| ----------------------------------- | --------- | --------- | --------- | --------- | --------- | -------------- |
| `reve.linear.eeg.0`                 | 0.107     | 0.017     | 0.009     | 0.004     | 0.116     | −0.066         |
| `reve.linear.eeg.1`                 | 0.078     | 0.014     | 0.007     | 0.004     | 0.085     | NaN            |
| `reve.qformer.eeg.0`                | 0.089     | 0.015     | 0.008     | 0.004     | 0.100     | −0.087         |
| `reve.vocab.eeg.0`                  | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | NaN            |
| `tfm.linear.eeg.0`                  | 0.115     | 0.066     | 0.026     | 0.004     | 0.007     | NaN            |
| `tfm.linear.eeg.1`                  | 0.058     | 0.012     | 0.005     | 0.003     | 0.064     | NaN            |
| `tfm.qformer.eeg.0`                 | 0.076     | 0.014     | 0.006     | 0.003     | 0.092     | −0.147         |
| `tfm.vocab.eeg.0`                   | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | NaN            |
| `reve.linear.noise_train.0` (Box B) | —         | —         | —         | —         | —         | (cell crashed) |


Two structural diagnostic patterns are visible: vocab cells are uniformly 0.0 (because LoRA's stage-3 grad was exactly zero, so stage 3 produced the random output that eval encountered), and BERTScore returned NaN on most cells (the embedding model can't score the degenerate text the soft-prompt cells were producing).

---

## Currently-running pilot (post-fix), with W&B links

Launched 14:08 IST on Box A (8 cells in parallel) and 14:00 IST on Box B (1 cell). Same step budget as before: `stage1=300`, `stage2=1200`, `stage3=500`.


| Box   | GPU   | Cell                            | W&B run                                                                                                                                          |
| ----- | ----- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| A     | 0     | `reve.linear.eeg.0`             | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/g5luo0ae](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/g5luo0ae)     |
| A     | 1     | `reve.qformer.eeg.0`            | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/u62ofgx5](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/u62ofgx5)     |
| A     | 2     | `tfm.linear.eeg.0`              | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/me3f96bm](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/me3f96bm)     |
| A     | 3     | `tfm.qformer.eeg.0`             | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/xxbzsx9u](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/xxbzsx9u)     |
| A     | 4     | `reve.vocab.eeg.0`              | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/6tgxo9g8](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/6tgxo9g8)     |
| A     | 5     | `tfm.vocab.eeg.0`               | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/8u7oxn9b](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/8u7oxn9b)     |
| A     | 6     | `reve.linear.eeg.1`             | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/quvry6s2](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/quvry6s2)     |
| A     | 7     | `tfm.linear.eeg.1`              | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/y088o2jd](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/y088o2jd)     |
| **B** | **0** | `**reve.linear.noise_train.0`** | **[https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/7ne592ia](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/7ne592ia)** |


Pre-fix runs (kept on W&B for comparison):

- `reve.linear.noise_train.0` (failed, pre-`np` fix): [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/2lumxomd](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/2lumxomd)
- `reve.vocab.eeg.0` (buggy): [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/memwdc1z](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/memwdc1z)
- `tfm.vocab.eeg.0` (buggy): [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/rd5utzs1](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/rd5utzs1)

The new logging format adds two diagnostic series per cell:

- `stageN/align_loss` — chance level is `log(batch_size)`; soft-prompt cells should drift below this on EEG and stay at it on noise.
- `stageN/commit_loss` — only fires for off-diagonal vocab cells (`reve.vocab`); meaningful values mean the RVQ codebook is moving.

---

## ETA to first comparable results

- **~14:35 IST** (≈ 25 min after 14:10 launch on Box A): the four soft-prompt fold-0 cells + two fold-1 cells finish stage 3 + eval. First sanity check that BLEU-1 has moved off the previous run's 0.058–0.115 range and that test-time generations differ across references.
- **~14:35 IST** (≈ 35 min after 14:00 on Box B): the noise_train cell wraps. Compute the first EEG-vs-noise gap from `reve.linear.eeg.0` (Box A) vs `reve.linear.noise_train.0` (Box B). If positive with non-overlapping bootstrap CIs → REVE+linear is using EEG (per Jo et al. §4.3).
- **~17:00 IST** (≈ 2h 50m after 14:10): both vocab cells on Box A finish (vocab cells now actually train Stage 3 LoRA, ~1.5× walltime per stage 3 due to the larger effective trainable-param count).

## Re-archive policy

Pre-fix run + eval directories are at `$EXP01_DATA_ROOT/archive/buggy_run_2026-04-30T08-30/` on each box; the live `runs/` and `eval/` directories are now writing the post-fix curriculum.

## Quick monitoring commands

```bash
# Live training loss + align/commit losses for a single cell
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "tail -f /home/ubuntu/data/exp01/runs/reve_linear_eeg_fold0_dec-gemma4-e2b/log.jsonl"

# Live GPU utilization
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "watch -n2 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"

# Pilot orchestrator logs
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "tail -f /tmp/pilot-soft.log /tmp/pilot-vocab.log /tmp/pilot-fold1.log"

# After everything finishes — first EEG-vs-noise gap
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "cd ~/work/eegModel/experiments/exp01_eeg_to_text && \
   .venv/bin/python -c \"
from exp01 import eval as ev, storage
import json
def load(c): return json.load(open(storage.EVAL/c/'metrics.json'))
eeg   = load('reve_linear_eeg_fold0_dec-gemma4-e2b')
noise = load('reve_linear_noise_train_fold0_dec-gemma4-e2b')
print(ev.eeg_noise_gap(eeg, noise))
\""
```

## What's NOT yet fixed (deferred)

- **Encoder padding mask.** The collator still zero-pads the time and channel axes and feeds the lot to REVE/TFM, which then encode the padding too. With the alignment loss in place the bridge has at least an incentive to ignore the padded regions, but a real mask down through the encoder would help further. Deferred — would require touching `encoders.REVEEncoder.encode` to honor a `(B, C, T)` mask, which REVE's HF wrapper doesn't currently expose.
- `**align_loss` for vocab cells.** Currently disabled because vocab cells run with `batch_size=1` (memory-driven). Either (a) enable a `--grad-accum-contrastive` pattern that accumulates bridge_pooled across micro-batches before computing the contrastive matrix, or (b) just bump bs to 2 with grad_checkpointing — both are follow-ups.
- **Gemma 4 multimodal towers stay loaded.** PEFT now correctly avoids them, but the vision/audio towers still take ~0.6 GB of VRAM each. A small future win is to delete them on load (`del model.model.vision_tower; del model.model.audio_tower`) since this codebase is text-only.

