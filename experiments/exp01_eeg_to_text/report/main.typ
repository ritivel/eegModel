// Exp01 — EEG-to-Text: experimental design report
// Compile: typst compile main.typ

#set document(
  title: "Exp01 — EEG-to-Text Decoding: Experimental Design",
  author: "Pavan Kalyan Tankala",
)

#set page(
  paper: "a4",
  margin: (x: 1.8cm, y: 2.0cm),
  numbering: "1 / 1",
  number-align: center,
)

#set text(font: ("New Computer Modern", "Times New Roman"), size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#show heading.where(level: 1): it => block(below: 0.7em, above: 1.1em)[
  #set text(size: 13pt, weight: "bold")
  #it
]
#show heading.where(level: 2): it => block(below: 0.5em, above: 0.9em)[
  #set text(size: 11pt, weight: "bold")
  #it
]
#show heading.where(level: 3): it => block(below: 0.4em, above: 0.7em)[
  #set text(size: 10.5pt, weight: "bold", style: "italic")
  #it
]
#show link: it => underline(text(fill: rgb("#1a56db"), it))
#show raw.where(block: false): it => box(
  fill: rgb("#f3f4f6"),
  outset: (y: 0.18em),
  inset: (x: 0.3em),
  radius: 0.2em,
  text(size: 0.92em, it),
)

// ===== Title block =====
#align(center)[
  #text(size: 16pt, weight: "bold")[
    Exp01 — EEG-to-Text Decoding
  ]
  #v(0.2em)
  #text(size: 11pt)[Experimental Design Report]
  #v(0.3em)
  #text(size: 9.5pt, fill: rgb("#4b5563"))[
    Pavan Kalyan Tankala  ·  v1.0  ·  Apr 2026
  ]
]

#v(0.4em)
#line(length: 100%, stroke: 0.4pt + rgb("#d1d5db"))
#v(0.2em)

// ===== Abstract =====
#block(
  fill: rgb("#f9fafb"),
  inset: (x: 0.9em, y: 0.7em),
  radius: 0.3em,
  stroke: 0.4pt + rgb("#e5e7eb"),
)[
  #set text(size: 9.8pt)
  *Summary.* We fine-tune three EEG foundation models — REVE, DIVER-1, and the TFM-Tokenizer — into open-vocabulary
  EEG-to-text decoders, training on a unified multi-source sentence-level corpus (8 datasets, \~72 GB) and
  evaluating only on ZuCo v1+v2 under the non-leaky *unique-sentence* split of Yin et al. (2024) combined
  with a *leave-N-subjects-out* protocol. Per Jo et al. (2024), every reported result is paired with a
  Gaussian-noise baseline; a model that does not strictly outperform its noise twin on BLEU-1/2/3/4,
  ROUGE-1-F, and BERTScore-F1 (`roberta-large`) is reported as not using the EEG signal.
]

= 1. Task

Open-vocabulary EEG-to-text: given the multi-channel EEG recorded while a subject reads an English
sentence, reconstruct that sentence as free text. Inputs are sentence-level EEG segments
(channels-first, native sampling rate per source); outputs are tokenised English sentences.

= 2. Models

The design factorially crosses three EEG encoders (§2.1), three bridge architectures (§2.3),
and a fixed text-decoder family (§2.2). Holding the decoder constant across all cells means
any difference in metrics is attributable to the EEG side — encoder, bridge, or their
interaction — not to the language prior.

== 2.1 EEG encoders (three foundation models being benchmarked)

#table(
  columns: (auto, 1fr, auto, auto),
  align: (left, left, center, center),
  stroke: 0.4pt + rgb("#d1d5db"),
  inset: 6pt,
  table.header(
    [*Model*], [*Reference*], [*arXiv*], [*Weights*],
  ),
  [REVE],
  [Generalist EEG foundation model trained on a large multi-paradigm corpus.],
  [#link("https://arxiv.org/abs/2510.21585")[2510.21585]],
  [#link("https://huggingface.co/collections/brain-bzh/reve")[`brain-bzh/reve`]],

  [DIVER-1],
  [Diverse-paradigm EEG foundation model.],
  [#link("https://arxiv.org/abs/2512.19097")[2512.19097]],
  [#link("https://anonymous.4open.science/r/DIVER-1/README.md")[anon repo]],

  [TFM-Tokenizer],
  [Tokenised time-frequency EEG representation; pretrained tokenizer + transformer.],
  [#link("https://arxiv.org/abs/2502.16060")[2502.16060]],
  [#link("https://huggingface.co/Jathurshan/TFM-Tokenizer")[`Jathurshan/TFM-Tokenizer`]],
)

== 2.2 Text decoder

The text decoder is a *Gemma 4* language model
(#link("https://deepmind.google/models/gemma/gemma-4")[Google DeepMind, 02 Apr 2026]) — the
first frontier-quality open-weight family released under the Apache 2.0 licence, built from
the same research as Gemini 3. We use the *same decoder family across all three encoders*
so that differences are attributable to the EEG encoder, not to the language prior. Four
sizes are available; the specific variant for each encoder is fixed in the per-encoder
training spec and is not committed to here.

#table(
  columns: (auto, auto, 1fr, auto),
  align: (left, center, left, left),
  stroke: 0.4pt + rgb("#d1d5db"),
  inset: 6pt,
  table.header(
    [*Variant*], [*Effective params*], [*Architecture*], [*Hugging Face ID*],
  ),
  [Gemma 4 E2B-IT], [\~2.3B], [Dense + per-layer embeddings],
    [#link("https://huggingface.co/google/gemma-4-E2B-it")[`google/gemma-4-E2B-it`]],
  [Gemma 4 E4B-IT], [\~4.5B], [Dense + per-layer embeddings],
    [#link("https://huggingface.co/google/gemma-4-E4B-it")[`google/gemma-4-E4B-it`]],
  [Gemma 4 26B-A4B-IT], [\~3.8B active / 26B total], [MoE (128 experts, 8 active)],
    [#link("https://huggingface.co/google/gemma-4-26B-A4B-it")[`google/gemma-4-26B-A4B-it`]],
  [Gemma 4 31B-IT], [31B], [Dense],
    [#link("https://huggingface.co/google/gemma-4-31B-it")[`google/gemma-4-31B-it`]],
)

*Why a frontier decoder strengthens (not weakens) the noise test of §4.3.* A stronger
language prior raises the noise-only ceiling — Gemma 4 alone, with no EEG, will already
produce fluent, on-distribution English. Per Jo et al. (2024), this *raises the bar* a
model must clear to be reported as using the EEG signal: the EEG−noise gap (§4.4) must
remain strictly positive against the stronger reference, with non-overlapping bootstrap
95 % CIs. We treat this as a feature, not a bug.

== 2.3 Bridges (encoder → Gemma 4)

We promote the bridge from EEG encoder into Gemma 4 to a *third experimental factor*
crossed with the encoder. Three bridge families are evaluated, drawn from the
architectural traditions for connecting a non-text encoder to a frozen-or-LoRA'd LLM:
(i) *linear projector + soft prompt* (LLaVA / Gemma-native), (ii) *Q-Former* (BLIP-2 /
BELT-2), and (iii) *vocabulary extension + multi-channel autoregression* (NeuroLM). All
three share the same staged curriculum — Stage 1 modality alignment (encoder + Gemma
frozen) → Stage 2 frozen-LM SFT → Stage 3 LoRA-on-Gemma SFT — only the bridge differs.

The 3-encoder × 3-bridge factorial yields main effects of encoder and bridge plus the
interaction term needed to claim any single combination is best. We *pre-register the
diagonal* (REVE → Linear, DIVER-1 → Q-Former, TFM-Tokenizer → Vocab ext.) as the prior
hypothesis grounded in each encoder's architectural property; off-diagonal wins are
reported as honest falsifications.

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  align: (left, center, center, center),
  stroke: 0.4pt + rgb("#d1d5db"),
  inset: 6pt,
  table.header(
    [*Encoder*], [*Linear projector*], [*Q-Former* ($K{=}32$)], [*Vocab ext. + AR*],
  ),
  [REVE], [*pre-registered*], [variable-length fits], [+ small RVQ head on features],
  [DIVER-1], [any-variate fits], [*pre-registered*], [+ small RVQ head on features],
  [TFM-Tokenizer], [codebook lookup → Linear], [codebook lookup → Q-Former], [*pre-registered*],
)

=== 2.3.1 Diagonal cells (pre-registered)

*REVE → linear projector + soft prompt.* Attention-pool REVE's per-patch tokens over time
(preserving channel structure), apply `Linear(512 → d_Gemma) → RMSNorm`, prepend as soft
tokens before `<bos>`. *Why:* REVE's headline result is strong linear probing across 10
downstream tasks, so its embeddings are downstream-ready by design. The bridge is
byte-identical to Gemma 4's native multimodal embedder
(#link("https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma4_mm.py")[`Gemma4MultimodalEmbedder`]:
`RMSNorm → Linear`), so we inherit the official multimodal serving stack at zero
architectural cost. The low-capacity bridge also yields the cleanest §4.3 noise gap — the
projector has nowhere to hide a learned LM prior.

*DIVER-1 → Q-Former bridge (BLIP-2 / BELT-2 style).* 32 learnable queries cross-attend to
DIVER-1's frozen features every other block, project to `d_Gemma`, and prepend as a
fixed-size soft prompt. *Why:* DIVER-1's any-variate attention emits long, variable-channel
sequences — a fixed-$K$ bottleneck is the only way to keep Gemma's context bounded across
4–105 channels. DIVER-1's own scaling law is data-constrained (smaller backbone trained
longer beats larger backbone trained briefly under fixed compute), so we spend the
trainable budget in the bridge (\~100M-parameter Q-Former, Li et al., 2023) and pick a
sub-200M DIVER-1 variant rather than the 1.83B. The bottleneck is also the strongest
architectural §4.3 guard of the three: under matched noise the queries have less
EEG-specific structure to extract.

*TFM-Tokenizer → vocabulary extension + multi-channel autoregression (NeuroLM style).* Add
TFM's 8192 VQ codes as new rows in Gemma 4's embedding table and SFT Gemma to autoregress
over `<eeg_ch1_t1> … <eeg_chC_tT> <bos> <text>`. *Why:* TFM is already discrete (8192-code
VQ-VAE, 64-dim codes), so vocabulary extension is the structurally correct bridge — no
continuous projector to invent or tune. It directly inherits NeuroLM's recipe (Jiang et
al., 2024) — one of the strongest published EEG-LLM results — with TFM swapped in for
NeuroLM's neural tokenizer. TFM's single-channel design also unifies all eight sources
(4–105 channels) without resampling, matching §3's heterogeneity stance. *Tradeoff:*
highest §4.3 leakage risk — new embedding rows can absorb Gemma's LM prior — so Stage 1
alignment must include the train-noise / test-noise twin before Stage 2 introduces text.

=== 2.3.2 Off-diagonal cells (factorial completion)

The remaining six cells are constructed minimally so that the bridge is the only thing
that varies relative to the diagonal. *REVE × Vocab ext.* and *DIVER-1 × Vocab ext.* train
a small Residual VQ head (NeuroRVQ-style, Barmpas et al., 2025) on the encoder's
penultimate features to obtain a discrete codebook of size comparable to TFM's 8192, then
proceed exactly as in the TFM diagonal cell. *TFM × Linear* and *TFM × Q-Former* embed
each TFM code through TFM's existing 64-d codebook lookup and feed the resulting
per-(channel, time) embeddings into the same Linear / Q-Former modules used in the REVE /
DIVER-1 diagonal cells. No further architectural changes; the Stage 1/2/3 schedule is
unchanged.

= 3. Training data

We train on the unified sentence-level corpus
#link("https://huggingface.co/datasets/tankalapavankalyan/exp01-eeg-to-text-sentences")[`tankalapavankalyan/exp01-eeg-to-text-sentences`]
(\~72 GB, parquet, one row = one sentence read by one participant, raw-lossless). Eight source
datasets are bundled under a single schema; per-source licences are preserved verbatim:

#table(
  columns: (auto, 1fr, auto, auto, auto),
  align: (left, left, center, center, center),
  stroke: 0.4pt + rgb("#d1d5db"),
  inset: 5.5pt,
  table.header(
    [*`dataset`*], [*Source / paradigm*], [*Sr (Hz)*], [*Channels*], [*Licence*],
  ),
  [`zuco_v1_sr`],   [ZuCo 1.0 task1 — natural reading + sentiment], [500],  [\~105], [CC BY 4.0],
  [`zuco_v1_nr`],   [ZuCo 1.0 task2 — natural reading],             [500],  [\~105], [CC BY 4.0],
  [`zuco_v1_tsr`],  [ZuCo 1.0 task3 — relation-extraction reading], [500],  [\~105], [CC BY 4.0],
  [`zuco_v2_nr`],   [ZuCo 2.0 task1 — natural reading],             [500],  [\~105], [CC BY 4.0],
  [`zuco_v2_tsr`],  [ZuCo 2.0 task2 — relation-extraction reading], [500],  [\~105], [CC BY 4.0],
  [`derco_preprocessed`], [DERCo — RSVP word-by-word + flanker],    [1000], [32],    [CC BY 4.0],
  [`emmt`],         [EMMT — natural reading + eye-tracking (Muse)], [256],  [4],     [MIT / CC BY 4.0],
  [`eeg_sem_relev`], [Quoron — RSVP + topic-relevance judgement],   [\~2858],[32],   [Apache-2.0],
)

*Heterogeneity is intentional.* Foundation models are expected to absorb 4–105 channels and
256 Hz–2.86 kHz sampling rates without resampling or channel selection; the unified schema preserves
each source's native EEG layout and per-word/per-sentence segments.

= 4. Evaluation set, splits, and protocol

Although training uses all eight sources, *all reported numbers are computed only on ZuCo v1+v2
held-out splits.* This isolates a clean reading paradigm with rich per-subject metadata and lets us
compare directly against the EEG-to-text literature.

== 4.1 Unique-sentence split (Yin et al., 2024)

We adopt the non-leaky split criterion of Yin et al., 2024 — _Rethinking Cross-Subject Data Splitting
for Brain-to-Text Decoding_, #link("https://arxiv.org/abs/2312.10987")[arXiv:2312.10987] — in
explicit preference to the Wang & Ji (2022) split, which leaks both text and signal. The two
non-negotiable rules are:

+ *No text-stimuli leakage.* If a sentence appears in dev or test, no occurrence of that sentence
  text — by any subject — appears in train.
+ *No subject leakage.* If subject $S$ contributes any sample to dev or test, no EEG from $S$
  appears in train (this is the subject-independent rule, formalised in §4.2).

Concretely, we (i) deduplicate ZuCo v1+v2 sentences by `sentence_text_normalized`, (ii) randomly
partition the unique-sentence pool 80 / 10 / 10 with a fixed seed, then (iii) apply the
subject-independent constraint of §4.2 on top.

== 4.2 Subject-independent split (leave-N-subjects-out)

Let $cal(S) = cal(S)_("v1") union cal(S)_("v2")$ be the union of ZuCo v1 (12 subjects) and ZuCo v2
(18 subjects), $|cal(S)| = 30$. We hold out $N{=}3$ subjects for test and $N{=}3$ for dev, sampled
without replacement and stratified to keep both ZuCo versions represented in each fold. The
evaluation is repeated as 5-fold cross-validation; we report mean ± std across folds. The intersection
of §4.1 and §4.2 means: *no test sentence and no test subject appears in train*.

== 4.3 Mandatory noise baseline (Jo et al., 2024)

Per Jo et al., 2024 — _Are EEG-to-Text Models Working?_,
#link("https://arxiv.org/abs/2405.06459")[arXiv:2405.06459] — every model configuration is
re-trained and re-evaluated with EEG inputs replaced by Gaussian noise of matched per-channel mean
and variance. We run *two* noise variants, both required:

- *Train-noise / test-noise* — encoder never sees real EEG; isolates the language-prior contribution.
- *Train-EEG / test-noise* — encoder sees real EEG at train time but is fed noise at evaluation;
  isolates whether the decoder has memorised label statistics.

A model passes only if it strictly beats *both* noise variants on every metric below, with
non-overlapping bootstrap 95 % CIs (1 000 resamples). Otherwise it is reported as *not using EEG*.

== 4.4 Metrics

All metrics are computed sentence-by-sentence and averaged uniformly over the test set, with
*teacher forcing disabled* at evaluation (also per Jo et al.):

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, center),
  stroke: 0.4pt + rgb("#d1d5db"),
  inset: 6pt,
  table.header([*Metric*], [*What it measures*], [*Implementation*]),
  [BLEU-1/2/3/4], [n-gram precision against the reference (n = 1…4).],
    [`sacrebleu`, lowercase, default tokenizer.],
  [ROUGE-1-F], [Unigram F-measure between hypothesis and reference.],
    [`rouge-score` package, `rouge1`, `use_stemmer=True`.],
  [BERTScore-F1], [Contextual-embedding similarity, F1 of greedy token alignment.],
    [`bert-score` with `roberta-large`, baseline-rescaled.],
)

We additionally report bootstrap 95 % confidence intervals on every metric, the *EEG−Noise gap*
($Delta = "score"_"EEG" - "score"_"noise"$) with sign and CI, and a paired permutation test
(10 000 permutations) for the EEG-vs-noise comparison.

= 5. Run matrix and execution protocol

§2.3 promotes the bridge to a third experimental factor, giving a *3 encoders × 3 bridges
× 3 input conditions (EEG / noise-train / noise-test) × 5 subject-folds* = 135 evaluation
cells, with Gemma 4 as the fixed text decoder. Each cell reports six metrics
(BLEU-1/2/3/4, ROUGE-1-F, BERTScore-F1) with bootstrap 95 % CIs (1 000 resamples). The
train-EEG/test-noise condition reuses the train-EEG checkpoint and only re-evaluates, so
the 135 cells reduce to *90 unique training runs*; within each (encoder, input) group the
three bridges share cached Stage 1 encoder forward passes.

== 5.1 Staged execution

To avoid committing 90 trainings up-front, the matrix is executed in three phases:

+ *Pilot.* Fold 1 only, EEG input only, Gemma 4 E2B-IT, all 9 encoder × bridge cells
  (\~9 trainings). Reports BLEU-1 and ROUGE-1-F per cell without CIs; purpose is to rank
  the bridge effect per encoder.
+ *Prune.* Keep the top bridge per encoder; if two bridges for the same encoder are within
  2 BLEU-1 points, keep both. Yields 3–6 surviving (encoder, bridge) cells.
+ *Fan-out.* Run the surviving cells through the full §4 protocol — 3 input conditions ×
  5 LNSO folds × the per-encoder decoder size — with the Jo et al. noise baselines and
  bootstrap CIs.

== 5.2 Pre-registered hypothesis and reproducibility

The §2.3 diagonal (REVE → Linear, DIVER-1 → Q-Former, TFM → Vocab ext.) is the prior
hypothesis. Two outcomes are flagged in advance: (a) the diagonal wins on every encoder
(bridge–encoder coupling confirmed; §2.3 becomes the recommended design); (b) one bridge
column dominates across encoders (bridge choice universal; encoder ablation is the only
axis that matters). Anything else is interpreted as an interaction effect — joint
encoder × bridge optimisation matters — and reported as such, without re-pre-registering
after the fact.

Configurations, seeds, ZuCo unique-sentence assignments, the 5 LNSO fold IDs, and the RVQ
codebook checkpoints for the off-diagonal cells are pinned in the experiment config so
that any single cell is reproducible from a single command.

= 6. References

#set par(hanging-indent: 1.2em, first-line-indent: 0em)
#set text(size: 9.6pt)

- Hollenstein et al., 2018. *ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence
  reading.* Sci. Data 5:180291.
- Hollenstein et al., 2020. *ZuCo 2.0: A dataset of physiological recordings during natural reading
  and annotation.* LREC.
- Wang & Ji, 2022. *Open vocabulary electroencephalography-to-text decoding and zero-shot sentiment
  classification.* AAAI. #link("https://arxiv.org/abs/2112.02690")[arXiv:2112.02690].
- Yin et al., 2024. *Rethinking cross-subject data splitting for brain-to-text decoding.*
  #link("https://arxiv.org/abs/2312.10987")[arXiv:2312.10987].
- Jo et al., 2024. *Are EEG-to-text models working?*
  #link("https://arxiv.org/abs/2405.06459")[arXiv:2405.06459].
- Gemma Team (Google DeepMind), 2026. *Gemma 4: Byte for byte, the most capable open models.*
  #link("https://deepmind.google/models/gemma/gemma-4")[deepmind.google/models/gemma/gemma-4].
- REVE — #link("https://arxiv.org/abs/2510.21585")[arXiv:2510.21585].
- DIVER-1 — #link("https://arxiv.org/abs/2512.19097")[arXiv:2512.19097].
- TFM-Tokenizer — #link("https://arxiv.org/abs/2502.16060")[arXiv:2502.16060].
- Li, J., Li, D., Savarese, S., Hoi, S., 2023. *BLIP-2: Bootstrapping language-image
  pre-training with frozen image encoders and large language models.*
  #link("https://arxiv.org/abs/2301.12597")[arXiv:2301.12597].
- Zhou, J., et al., 2024. *BELT-2: Bootstrapping EEG-to-language representation alignment
  for multi-task brain decoding.*
  #link("https://arxiv.org/abs/2409.00121")[arXiv:2409.00121].
- Jiang, W.-B., et al., 2024. *NeuroLM: A universal multi-task foundation model for
  bridging the gap between language and EEG signals.* ICLR 2025.
  #link("https://proceedings.iclr.cc/paper_files/paper/2025/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html")[proceedings.iclr.cc].
- Barmpas, K., et al., 2025. *NeuroRVQ: Multi-scale EEG tokenization for generative
  large brainwave models.* arXiv preprint.
