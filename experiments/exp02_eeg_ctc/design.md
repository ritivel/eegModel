# Exp02 design — single-stage CTC EEG-to-text

## Why exp02

`exp01`'s 3-stage curriculum (`alignment via InfoNCE → frozen-LM SFT → LoRA SFT`)
was designed for the *soft-prompt + frozen-Gemma bridge*. Stage 1 brings a
new modality into the LM's embedding space; stages 2-3 fine-tune the bridge
and then the LM. None of those stages do anything sensible for a CTC head
that has no language model in the loop:

- Stage 1's `align_loss` is computed against frozen text-token embeddings the
  CTC head literally never sees.
- Stage 2's "frozen-LM SFT" reduces to "more bridge training" because there's
  no LM to freeze.
- Stage 3 has no LoRA target. `exp01` worked around this by hacking stage 3
  to reuse stage 1/2's params at a lower LR, which is just learning-rate
  decay.

Worse, the CTC code lives entangled with the soft-prompt + PLE-aware decoder
hooks (`is_ctc` branches throughout `model.py`, `train.py`, `bridges.py`,
`eval.py`), making it harder to iterate on CTC-specific tricks.

`exp02` is the architecturally-clean counterfactual: **strip everything that
exists for the LM bridge, run a single end-to-end CTC training loop, and
add the standard ASR fixes that `results_track_b_ctc.md` §5 identified as
missing**.

## What modern CTC training actually looks like

Verified against the canonical recipes:

| recipe | training shape | citation |
| --- | --- | --- |
| HF Wav2Vec2 fine-tune (TIMIT, LibriSpeech) | **single end-to-end CTC**, ≥30 epochs, one optimizer, one loss | [HF blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) |
| Facebook Omnilingual ASR (300M / 1B / 3B Wav2Vec2 + CTC) | single end-to-end loop; **`freeze_encoder_for_n_steps: 10_000`** as a brief warmup, then unfreeze. Fine-tune from CTC checkpoint sets `freeze_encoder_for_n_steps: 0`. | [`ctc-from-encoder.yaml`](https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/recipes/wav2vec2/asr/configs/ctc-from-encoder.yaml) |
| Hybrid CTC+AED ([Watanabe 2017](https://arxiv.org/abs/1609.06773)) | **joint multi-objective** `λ·L_CTC + (1−λ)·L_attn` from step 0; not staged | MERL paper |
| Willett 2023 (Nature handwriting BCI), Card 2024 (NEJM speech BCI), Metzger 2023 (ECoG-to-text) | RNN + CTC, **single-stage end-to-end**, language model only at decode time | [Willett 2021 PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8163299/), [Berkeley thesis EECS-2025-147](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-147.pdf) |
| CR-CTC (ICLR 2025) | single-stage CTC + consistency reg between two augmented views | [arXiv 2410.05101](https://arxiv.org/abs/2410.05101) |
| d'Ascoli & King 2025 (non-invasive *Nat. Commun.* 16, 10521) | single-stage D-SigLIP word alignment; *explicitly* mentions CTC as the natural next step | [doi:10.1038/s41467-025-65499-0](https://doi.org/10.1038/s41467-025-65499-0) |

**No major modern CTC system uses an `align (InfoNCE) → frozen-LM SFT → LoRA SFT`
curriculum**, because the curriculum's job is to bring a new modality into a
frozen LM. CTC has no LM in the loop.

The closest thing to "stages" that *is* used:

- **Warmup-freeze** (10k steps frozen encoder, then unfreeze) — Omnilingual
  ASR pattern. Single optimizer, two parameter groups.
- **Sequence-length curriculum** (short utterances first) — addresses the
  well-known "predict-only-blanks" plateau ([Bluche & Kermorvant](http://www.tbluche.com/ctc_and_blank.html)).
  Often unnecessary with Adam / RMSProp.
- **Two-step pretrain → fine-tune** for distilled / lightweight models
  ([arXiv 2505.16991](https://arxiv.org/html/2505.16991v1)) — different
  setting from ours.

## Why CTC needs more than just CTC

`results_track_b_ctc.md` §5 identified partial blank-collapse + length-precision
gaming as the failure mode. The literature gives this a name and explains
why it *must* happen:

1. **Peaky behavior is a property of the CTC objective, not a bug.** Zeyer,
   Schlüter & Ney (ICML 2021): *"a uniformly-initialized FFNN trained with
   CTC from gradient descent provably converges to peaky behavior with 100%
   error rate"* on a trivially-learnable example. ([arXiv 2105.14849](https://arxiv.org/abs/2105.14849)).
   Why: blank is the dominant label in every alignment, so its gradient is
   over-represented — the network first learns to predict only blank, then
   character peaks emerge later if there's enough signal and enough training.
   Track-B stopped during the "predict-only-blanks" plateau.

2. **Standard fixes** (each is a knob in `CTCConfig`):
   - **Label-prior CTC loss** (Zeyer 2021 §7): subtract a learned label prior
     from the logits. No peaky behavior, faster convergence.
   - **CR-CTC** ([Yao et al. ICLR 2025](https://arxiv.org/abs/2410.05101)):
     two SpecAugmented views + KL between their CTC distributions. Achieves
     SOTA *with CTC alone*, comparable to transducer / CTC+AED.
   - **Self-conditioned / Intermediate CTC** ([Komatsu 2022](https://www.isca-archive.org/interspeech_2022/komatsu22_interspeech.html)):
     auxiliary CTC losses at intermediate encoder layers, each conditioning
     the next layer. Relaxes the conditional-independence assumption.
   - **Hybrid CTC + AED** ([Watanabe 2017](https://arxiv.org/abs/1609.06773)):
     joint loss with attention decoder. CTC's monotonic alignment guides
     the attention decoder; cross-entropy from the decoder breaks the
     blank-collapse trap.

3. **Beam search + n-gram LM rescore is essentially mandatory** for CTC ASR.
   Distill.pub: *"speech recognizers using CTC don't learn a language model
   over the output … However, a separate language model can be included
   and usually gives a good boost to accuracy."* ([Hannun, distill.pub/2017/ctc](https://distill.pub/2017/ctc)).
   Greedy CTC is a diagnostic, not a deployment decoder.

4. **Encoder fine-tune ≫ frozen encoder.** Wav2Vec2 paper, Brain-to-Text
   Decoder ([arXiv 2501.06326](https://arxiv.org/abs/2501.06326)). Frozen
   REVE is the wrong baseline.

## Default recipe (the "headline" cell)

`reve.bpe1k.crctc.eeg.0` is the cell that defines `exp02`'s headline:

| knob | default | citation |
| --- | --- | --- |
| training stages | **none** — single end-to-end loop | Wav2Vec2 (HF), Willett 2023, Card 2024 |
| encoder | REVE-base, **fully fine-tuned** | [arXiv 2501.09459](https://arxiv.org/abs/2501.09459) §3.2 Fig 5 |
| encoder warmup freeze | first 10% of total steps frozen | [Omnilingual `ctc-from-encoder.yaml`](https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/recipes/wav2vec2/asr/configs/ctc-from-encoder.yaml) |
| total steps | **12 000** | Wav2Vec2 fine-tunes ~30 epochs ≈ 12k–24k steps at our scale |
| LR schedule | linear-warmup → cosine decay | Wav2Vec2 paper |
| head LR | 1e-3 | standard |
| encoder LR | 1e-5 | standard |
| optimizer | AdamW(weight_decay=0.01) | standard |
| grad clip | 1.0 | standard |
| precision | bf16 mixed | Omnilingual ASR |
| vocab | **BPE-1k** (sentencepiece on ZuCo train refs + Wikipedia) | Wav2Vec2 / Whisper / Omnilingual |
| augmentation | **SpecAugment** (2 time masks ≤ 200 ms + 2 freq masks ≤ 8 chans) | [Park 2019](https://arxiv.org/abs/1904.08779) |
| CTC variant | **CR-CTC** with KL weight 1.0 | [Yao 2024 ICLR 2025](https://arxiv.org/abs/2410.05101) |
| label prior | enabled (running EMA, weight 0.3) | [Zeyer 2021 §7](https://arxiv.org/abs/2105.14849) |
| decode | greedy + beam (50) + beam + KenLM 4-gram rescore | [distill.pub/2017/ctc](https://distill.pub/2017/ctc) |
| §4.3 noise twin | **mandatory** for every cell | Jo et al. 2024 |

## Run matrix (full Track-C scope)

```
GROUP A — Headline + matched noise twin
  reve.bpe1k.crctc.eeg.0
  reve.bpe1k.crctc.noise_train.0

GROUP B — Encoder ablation
  tfm.bpe1k.crctc.eeg.0
  tfm.bpe1k.crctc.noise_train.0
  diver1.bpe1k.crctc.eeg.0          (skipped if weights absent)
  diver1.bpe1k.crctc.noise_train.0

GROUP C — Vocab ablation
  reve.char.crctc.eeg.0             (vs default BPE-1k)
  reve.char.crctc.noise_train.0

GROUP D — CTC variant ablation
  reve.bpe1k.ctc.eeg.0              (vanilla CTC, no CR-CTC, no label-prior)
  reve.bpe1k.ctc.noise_train.0
  reve.bpe1k.intctc.eeg.0           (intermediate-CTC, no CR-CTC)
  reve.bpe1k.intctc.noise_train.0
  reve.bpe1k.ctcaed.eeg.0           (CTC + attention decoder hybrid)
  reve.bpe1k.ctcaed.noise_train.0

GROUP E — Encoder freeze ablation
  reve.bpe1k.crctc.eeg.0_frozen     (head only, encoder permanently frozen)
  reve.bpe1k.crctc.noise_train.0_frozen

GROUP F — 5-fold extension on the survivor cell
  <survivor>.fold0..fold4 × {EEG, noise_train}              (10 cells)
```

GROUPS A+B+C+D+E ≈ 14 distinct training cells × ~1.5 h on H100 (smaller than
exp01 cells because no Gemma forward) ≈ 21 H100-hours; fits comfortably on
the 9-GPU box-A + box-B layout in 3-4 hours.

GROUP F adds 10 more cells once a survivor is identified.

## Decode-time ablation (free, eval-only)

For every trained cell we evaluate three decode modes:

1. **greedy** — argmax + collapse repeats + drop blanks. Diagnostic only.
2. **beam** — beam width 50, no LM. Tests whether the model has multi-hypothesis
   content.
3. **beam + KenLM** — same beam plus a 4-gram KenLM rescore. The headline.

`pyctcdecode` runs all three on CPU after the GPU forward, so these
ablations cost only minutes per cell.

## §4.3 verdict (per Jo et al. 2024)

For each EEG cell we report the matched-pair gap against its `noise_train`
twin:

- **PASS (clean)**: EEG CER < noise CER with 95% bootstrap CIs disjoint AND
  sign-flip permutation `p < 0.01`. → Confirms decoding from EEG content.
- **PASS (weak)**: EEG CER < noise CER, CIs overlap. → Suggestive; run the
  5-fold matrix to tighten.
- **TIE**: EEG CER ≈ noise CER, both far from 1. → CTC head is learning the
  marginal distribution; EEG content not used.
- **FAIL**: EEG CER ≈ noise CER ≈ 1. → Encoder features carry no
  text-relevant signal at this SR / channel layout.

CER is the primary metric; BLEU is reported for cross-comparison with
`exp01` but its precision-without-brevity-penalty form is misleading for
short CTC hyps (see `results_track_b_ctc.md` §4.3).

## Citations

- **Single-stage end-to-end CTC**: HF Wav2Vec2 fine-tune blog;
  facebookresearch/omnilingual-asr (`ctc-from-encoder.yaml`); Willett 2021
  *Nature*; Card 2024 *NEJM*.
- **Why CTC is peaky**: Zeyer, Schlüter & Ney, *Why does CTC result in peaky
  behavior?*, ICML 2021. [arXiv 2105.14849](https://arxiv.org/abs/2105.14849).
- **CR-CTC**: Yao et al., *CR-CTC: Consistency regularization on CTC for
  improved speech recognition*, ICLR 2025. [arXiv 2410.05101](https://arxiv.org/abs/2410.05101).
- **Intermediate / Self-conditioned CTC**: Komatsu et al., *Better
  Intermediates Improve CTC Inference*, INTERSPEECH 2022.
- **Hybrid CTC + AED**: Watanabe et al., *Hybrid CTC/Attention Architecture
  for End-to-End Speech Recognition*, IEEE JSTSP 2017. [arXiv 1609.06773](https://arxiv.org/abs/1609.06773).
- **Wav2Vec2 fine-tune-vs-frozen for brain decoding**:
  [arXiv 2501.09459](https://arxiv.org/abs/2501.09459) §3.2 Fig 5.
- **REVE preprocessing**: El Ouahidi et al., *REVE: A Foundation Model for
  EEG*, NeurIPS 2025. [arXiv 2510.21585](https://arxiv.org/abs/2510.21585) §3.1.1.
- **TFM-Tokenizer recipe**: Pradeepkumar et al., *Tokenizing Single-Channel
  EEG with Time-Frequency Motif Learning*, ICLR 2026.
  [arXiv 2502.16060](https://arxiv.org/abs/2502.16060) §B.2.
- **§4.3 noise baseline protocol**: Jo, Lee, et al., *Are EEG-to-Text Models
  Working?*, *Sci. Reports* 2025. [arXiv 2405.06459](https://arxiv.org/abs/2405.06459).
- **SpecAugment**: Park et al., INTERSPEECH 2019.
  [arXiv 1904.08779](https://arxiv.org/abs/1904.08779).
- **Distill.pub CTC tutorial** (greedy CTC needs LM rescore): Hannun, 2017.
  [distill.pub/2017/ctc](https://distill.pub/2017/ctc).
- **D-SigLIP word-level alignment**: d'Ascoli et al., *Towards decoding
  individual words from non-invasive brain recordings*,
  *Nat. Commun.* 16, 10521 (2025). [doi:10.1038/s41467-025-65499-0](https://doi.org/10.1038/s41467-025-65499-0).
