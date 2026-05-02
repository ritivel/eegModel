# exp03 — small-scale → large-scale pretraining methodology

> **Purpose.** A self-contained playbook for designing self-supervised
> pretraining experiments on EEG. Written so future-me / future-agent can
> open it and answer: *"How do people responsibly take a pretraining idea
> from a 30-min run on 1 GPU to a multi-day run on a fleet?"* The answer is
> synthesised from the consensus across mature modalities (text, image,
> speech) plus the recent EEG-FM literature. It is a working document and
> should be edited as exp03 progresses.

This is the same kind of document as the [Smol Training Playbook
(HF, 2025)](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
and Karpathy's [A Recipe for Training Neural
Networks](https://karpathy.github.io/2019/04/25/recipe/), but specialised
to **self-supervised pretraining of biosignal foundation models on a small
academic compute budget**.

---

## 0. TL;DR — the five rules

Hard rules, in order of importance. Violate at your peril.

1. **Always run a small ablation before a big run, never the reverse.**
  Every modern foundation-model technical report (BERT, wav2vec 2.0,
   HuBERT, MAE, DINOv2, I-JEPA, LaBraM, REVE, SmolLM3, Llama 3) explicitly
   uses a fixed-cost small-scale ablation engine and only then commits
   compute. SmolLM3 spent **>50% of its total compute on ablations and
   debugging, not on the main run**
   ([HF, 2025](https://huggingfacetb-smol-training-playbook.hf.space/)).
2. **Two valid ablation shapes: (a) target-size model on fewer tokens, or
  (b) proxy-size smaller model on the full mixture**. SmolLM3 picked (a) —
   3B model on 100B tokens vs the final 11.2T. FineWeb picked (b) —
   1.82B model on 28B tokens. Both work; pick whichever fits your hardware
   ([SmolLM3](https://huggingface.co/blog/smollm3),
   [FineWeb](http://huggingfacefw-blogpost-fineweb-v1.static.hf.space/)).
3. **Trust your eval suite, not your loss curve**. A lower SSL/CTC/MLM
  loss does not mean a better representation — it can mean overfitting
   to a peaky head, codebook collapse, or shortcut features. Multiple eval
   tasks, run cheaply on each ablation, are non-negotiable
   ([Smol Playbook §"Every Big Model Starts with a Small Ablation"](https://huggingfacetb-smol-training-playbook.hf.space/);
   [SSL Risk Decomposition, Dubois et al. 2023](https://export.arxiv.org/pdf/2302.03068v2.pdf)).
4. **Hyperparameters do not transfer for free**. They transfer reliably
  across **width** under µP / µTransfer
   ([Yang et al. 2022](https://arxiv.org/abs/2203.03466),
   [MS Research blog](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/)),
   they often transfer empirically across **depth, batch size, sequence
   length** within reasonable ranges, but **not across a different optimiser
   or dataset**. Re-tune the LR for each new data recipe — the
   [DataDecide paper](https://arxiv.org/html/2504.11393v1) and the
   [Dec 2025 "Can Small Training Runs Reliably Guide Data Curation?"
   paper](https://www.arxiv.org/abs/2512.24503) both show that data-recipe
   rankings flip when you fix hyperparameters across them.
5. **In SSL specifically, monitor for collapse, not just for loss**. Look
  at: representation eigenvalue rank, linear-probe accuracy on a tiny
   labelled set, k-NN classification, embedding entropy. SSL methods can
   minimise their loss perfectly while learning nothing useful (codebook
   collapse, dimensional collapse, shortcut features). Concrete monitors
   below in §6.

---

## 1. The mental model: pretraining is *capability-engineering*, not luck

The single conceptual shift you need is borrowed from Karpathy's recipe:
*"a 'fast and furious' approach to training neural networks does not work
and only leads to suffering"*
([Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)).

Pretraining at scale fails *silently*. You will not see a stack trace; you
will see a beautiful loss curve that drops monotonically into the
gibberish floor. The Smol Training Playbook describes their **1
trillion token restart**: a subtle bug where every GPU in a tensor-parallel
group was initialised with the same random seed; loss curves were perfect;
only the downstream eval suite caught it
([HF, 2025](https://huggingfacetb-smol-training-playbook.hf.space/);
[The Neuron summary](https://www.theneuron.ai/explainer-articles/hugging-faces-new-playbook-reveals-the-messy-bug-filled-secrets-to-training-world-class-llms)).
That happened to a Tier-1 lab who *had* derisked everything else.

The job of small-scale experiments is to **derisk every component before
they compose**, so that when (not if) the full run goes sideways, you can
isolate the one new component you didn't validate. This is exactly what
modded-nanoGPT speedrunners do: they ablate every change individually
([leaderboard](https://github.com/KellerJordan/modded-nanogpt)), and
that is why the WR went from 45 min (May 2024) to 17.35 min (Dec 2025) on
the same hardware.

**Heuristic budget.** Plan for ablations + debugging to be ≥ the cost of
your headline run. SmolLM3: 161,280 GPU-hours on ablations vs ~~277,000 on
the main run
([HF, 2025](https://huggingfacetb-smol-training-playbook.hf.space/)).
Your equivalent: if you're planning a 3-day, 8-GPU pretraining run
(~~576 GPU-hours), expect ~500 GPU-hours of ablations + debugging *first*.

---

## 2. The five-phase methodology

This synthesises Karpathy's recipe with the modern HF Smol Playbook
framework, adapted to SSL on biosignals. Each phase has a hard exit
criterion — do not advance until it is met.

### Phase 0 — Become one with the data (½ day, no GPU)

Borrowed verbatim from
[Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/) §1: *"the
first step to training a neural net is to not touch any neural net code
at all and instead begin by thoroughly inspecting your data"*.

For an EEG SSL pretraining setup, this means a dataset audit *before*
writing any trainer. Adopt the audit format we already use in
`[../exp01_eeg_to_text/next_experiments.md` §2](../exp01_eeg_to_text/next_experiments.md):


| What to record                                                                                                                                                                                                                                             | Why                                                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Per-source: sampling rate, #channels, channel-naming convention (10-20 / EGI HydroCel / Muse / BioSemi), recording duration distribution, std (µV), abs_max, fraction of NaN / all-zero rows, fraction of rows with usable text labels for downstream eval | Picks the input shape your model has to handle                                 |
| Per-source: subject ID, recording session, paradigm (resting / reading / motor / sleep)                                                                                                                                                                    | Predicts subject-leakage failure modes during eval                             |
| Per-source: presence of canonical preprocessing (notch, bandpass, ICA-cleaned)                                                                                                                                                                             | Determines what your model will actually be fed                                |
| Histogram of EEG magnitude (µV scale) per source after bandpass + per-recording z-score                                                                                                                                                                    | If sources are not in a unified scale, you'll learn the source ID, not the EEG |
| Histogram of recording length, label length (where applicable)                                                                                                                                                                                             | Drives sequence-length design, sets cap on `max_seconds` filters               |


This is exactly what saved exp02: the audit in
`[../exp02_eeg_ctc/findings.md` §2.3](../exp02_eeg_ctc/findings.md) found
**19 hours of EEG silently chopped off**, **NaN noise in the §4.3 twin**,
**channel-mismatch zero-padding > 80% of rows**, and **8 garbage labels** —
all data bugs that would have invalidated weeks of pretraining if not
caught upfront. Hold any future pretraining run to the same standard.

**Exit criterion:** you can write down, on one A4 page, exactly how many
unique subjects, sessions, hours, and channel-configurations your
pretraining corpus has, and what fraction of rows survive your filter
chain. If you can't, you do not yet understand your data.

### Phase 1 — Sanity baselines (½ day, 1 GPU)

Before any SSL objective is wired up, validate the *infrastructure* on a
toy version of the task. Karpathy's eight-step list still applies; all
are O(seconds–minutes) on one GPU.

For **SSL pretraining specifically**, the canonical sanity tests are:

1. **Verify loss-at-init.** For a masked-reconstruction objective with
  d-dim continuous targets, MSE at init should be ≈ Var(target) — no
   surprises. For contrastive InfoNCE with batch size B, chance loss is
   `log(B)` (e.g., 2.08 for B=8, 3.47 for B=32).
   [wav2vec 2.0 §3.2](http://arxiv.org/abs/2006.11477) makes this
   explicit. Plot it on day 1.
2. **Input-independent baseline.** Train the same SSL objective with the
  inputs zeroed out. If the loss decreases, you have a leakage bug
   (Karpathy 2019). Fix it before doing anything else.
3. **Overfit one batch.** Take 4 EEG snippets, mask-and-reconstruct or
  contrast them. The model should drive the SSL loss to zero in
   <1000 steps. If it doesn't, your encoder, decoder, masking, or loss is
   broken — and you'd never figure that out in a 100k-step run with 60k
   hours of EEG.
4. **Linear probe on a frozen randomly-initialised encoder.** Run your
  downstream eval (in our case the HBN 6-task classification + HBN
   ADHD-vs-no-diagnosis binary; in the broader EEG-FM literature this is
   typically TUEV event classification + TUAB binary normal/abnormal) with
   a frozen random encoder. This is your **ablation floor**
   — anything below it is broken; anything within ~1% of it is also
   broken (see
   [EEG-FM-Bench finding (1)](https://arxiv.org/html/2508.17742v1)
   showing many EEG-FMs barely beat random under linear probing).
5. **Read the tensor shapes through the whole pipeline.** Print
  `eeg.shape, mask.shape, loss.item()` at each module boundary on a
   tiny batch. *"A relatively common bug I've come across a few times is
   that people get this wrong (e.g. they use `view` instead of
   `transpose/permute`) and inadvertently mix information across the
   batch dimension. The network will typically still train okay because
   it will learn to ignore data from the other examples"* (Karpathy 2019).

**Exit criterion:** one-batch overfit succeeds, loss-at-init matches
theory, input-independent baseline is at chance, and the random-init
linear probe gives a number you can write down. You now have a
**trustworthy training+eval pipeline**.

### Phase 2 — Ablation engine (1–3 days, 1–8 GPUs)

This is the single most important phase. You're picking your model
architecture, masking strategy, augmentations, hyperparameters, and data
mix — with enough statistical power to predict what happens at full
scale.

Both shapes are valid; a survey of how labs/projects pick:


| Approach                                        | Model                                                              | Tokens / hours          | When to use                                                                                 | Reference                                                                                                                                                                                    |
| ----------------------------------------------- | ------------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Target model, fewer tokens**                  | Final architecture                                                 | ~1% of final tokens     | When you have ≥1 node and your final model fits                                             | SmolLM3 (3B / 100B vs 11.2T) — [HF, 2025](https://huggingface.co/blog/smollm3)                                                                                                               |
| **Proxy model, full data**                      | Scaled-down (~1B for 70B target)                                   | Full or near-full mix   | When the final model doesn't fit on one node, or you want lots of architectural sweeps      | FineWeb (1.82B / 28B) — [HF, 2024](http://huggingfacefw-blogpost-fineweb-v1.static.hf.space/); DataDecide (150M proxy → 1B target) — [arXiv 2504.11393](https://arxiv.org/html/2504.11393v1) |
| **Hybrid: 1B baseline → final-arch validation** | 1B for architectural ablations, 3B for the final-arch sanity sweep | 45B / 100B respectively | When you can afford one node-day for many ablations and one node-day for a final-arch sweep | SmolLM3 actually used both — [HF blog](https://github.com/huggingface/blog/blob/main/smollm3.md)                                                                                             |


Three rules of thumb for the ablation setup, all from the
[Smol Playbook §"Every Big Model Starts with a Small Ablation"](https://huggingfacetb-smol-training-playbook.hf.space/):

1. **Hold the architecture constant across all data ablations**, and the
  data constant across all architecture ablations. The
   [Dec 2025 "Can Small Training Runs Reliably Guide Data Curation?"
   paper](https://www.arxiv.org/abs/2512.24503) shows data-recipe
   rankings *flip* if you don't separately tune the LR per recipe — but
   the practical fix is to *test multiple LRs per recipe* (a 3-LR sweep
   is enough), not to cross-tune everything.
2. **Test one change at a time.** Don't bundle "new optimizer" + "new
  masking" + "new architecture" into one ablation; you cannot
   attribute the effect.
3. **Use a small enough setup that you can iterate in <1 day.** SmolLM3:
  3B model + 100B tokens = ~1.5 days on 1× 8×H100 node = 12 H100-days
   per ablation. Too slow if you're testing 20 ideas. They also kept a
   **1B + 45B token (≈1.5 days on 8×H100) ablation tier**
   ([Smol Playbook §Ablation Setup](https://huggingfacetb-smol-training-playbook.hf.space/))
   for fast iteration.

**Eval suite for ablations.** This is where most projects fail. The
playbook is sharp on this:

> *"a lower loss when training on Wikipedia does not mean the model has
> stronger capabilities; changing the tokenizer also makes the loss
> values not directly comparable. Therefore, more fine-grained downstream
> evaluations must be used."*
> ([HF, 2025](https://huggingfacetb-smol-training-playbook.hf.space/))

A reliable eval task must satisfy four properties:

1. **Monotonicity.** Score improves with training compute (no plateau in
  the regime you care about).
2. **Low noise.** Run-to-run variance from random seed alone is small
  relative to the effect you want to detect.
3. **Above random.** The model must be able to beat random by enough at
  small scale that you're not measuring noise. *FineWeb ablations
   intentionally drop SIQA because it's "too noisy"
   ([HF, 2024](http://huggingfacefw-blogpost-fineweb-v1.static.hf.space/))*.
4. **Ranking consistency with the target scale.** This is the hard one —
  it's the *whole question* asked by
   [DataDecide](https://arxiv.org/html/2504.11393v1).

For SSL on EEG specifically, your eval suite during ablations should
include:


| Task                                                                                                                                                   | Why                                                                                   | Prior precedent                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Linear probe on HBN 6-task classification** (which of the 6 cognitive tasks the subject is performing — primary)                                     | 6-class symmetry with TUEV's role; uses HBN's natural task labels                     | NeurIPS 2025 EEG Foundation Challenge winners (e.g. ST-EEGFormer); in TUH-using literature this slot is TUEV                 |
| **Linear probe on HBN ADHD-vs-no-diagnosis binary** (primary)                                                                                          | Cheap, monotone, good early signal; ~40 % positive (well-balanced)                    | Same role as TUAB binary normal/abnormal in the LaBraM / BIOT / EEGPT eval recipe                                            |
| **Linear probe on TUAB + TUEV** (secondary, when TUH NEDC access lands)                                                                                | Direct apples-to-apples vs LaBraM / BIOT / EEGPT / CBraMod / REVE                     | The canonical EEG-FM literature benchmark; reported alongside HBN but not used for the §4.4 winner-picker                    |
| **k-NN classification** on a small labelled set                                                                                                        | Architecture-independent representation quality                                       | ([Wightman et al. notes;](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) DINO uses this)                    |
| **Embedding rank / silhouette** (label-free)                                                                                                           | Detects collapse without needing labels                                               | [Lourenço & Storkey 2024 (label-free SSL monitoring)](https://arxiv.org/html/2409.06612v1)                                   |
| **The downstream task you actually care about** (in our case, EEG-to-text §4.3 noise gap from exp01/exp02) — at *least* one fine-tune run per ablation | Reality check on whether your representation is doing anything useful for *your* task | Best reference: [Brain4FMs (arXiv 2602.11558)](https://arxiv.org/pdf/2602.11558) — they fine-tune all 15 BFMs on 18 datasets |


The **EEG-FM-Bench** finding is brutal but instructive: *"linear probing
is frequently insufficient, specialist models trained from scratch
remain competitive across many tasks, and larger foundation models do
not necessarily yield better generalization performance"*
([arXiv 2601.17883](https://arxiv.org/pdf/2601.17883)). Plan your eval
suite assuming this — your pretraining objective will look like it's
working before fine-tuning reveals it isn't.

**Exit criterion:** you have at least one ablation-tier setup that
finishes in <1 day and gives a noticeable, reproducible separation between
two known-good and known-bad recipes (e.g. "with band-pass" vs
"without band-pass" should give a clear win for the former). You are now
ready to spend a week of compute.

### Phase 3 — Intermediate-scale validation (~1–3 days, ~1 node)

Before the headline run, do a *single* run at intermediate scale (~5–10×
the ablation scale, ~10× less than the headline) using the recipe that
won the ablations. Purpose: catch failure modes that only emerge at scale.

Failure modes that consistently emerge between 1B/45B and 3B/300B+:


| Failure                                              | First seen at scale ~                               | Fix                                                                                | Reference                                                                                                                                                                           |
| ---------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Codebook collapse (VQ-VAE / LaBraM-style)            | >100B tokens, sometimes <50B                        | Re-init dead codes; restart with EMA codebook; add codebook diversity loss         | LaBraM Appendix I & symmetric masking discussion ([ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)) |
| Dimensional collapse (joint-embedding / contrastive) | Variable; sometimes from step 0                     | Variance/covariance reg (VICReg), feature whitening, orthogonality reg             | [Preventing Dimensional Collapse via Orthogonality Regularization (NeurIPS 2024)](https://arxiv.org/html/2411.00392v1)                                                              |
| Loss spike / divergence                              | Anywhere; correlated with bad data shards & high LR | Skip-batch + re-warmup; lower peak LR; gradient clipping                           | Llama 3, GPT-3 spike-recovery protocol                                                                                                                                              |
| Throughput collapse / IO starvation                  | Multi-node                                          | Move data to local NVMe; pre-load shards; "spare node" pattern                     | Smol Playbook §"Vanishing Throughput"                                                                                                                                               |
| Subject leakage on dev (specific to EEG)             | Any scale, but only visible after a long run        | Strict subject-disjoint splits; verify with a join on subject_id between train/dev | exp01/exp02 §"Subject splits" — already in our pipeline; do not regress                                                                                                             |


**Exit criterion:** the intermediate run reproduces (within statistical
noise) the ablation winner's lead over the runner-up. If it doesn't, your
ablation didn't transfer to scale and you need to back up to Phase 2 with
a longer/larger ablation, *not* push forward.

### Phase 4 — Headline run (the actual pretraining)

By now everything is derisked: data, architecture, optimiser, masking
strategy, eval suite, infrastructure. The headline run should feel
*boring*. The Smol Playbook explicitly says: *"the real value of ablation
experiments lies not only in building good models but also in providing
confidence for future debugging: when the main training inevitably goes
wrong, the systematic experimental results can help the team quickly
locate problems"*
([HF, 2025](https://huggingfacetb-smol-training-playbook.hf.space/)).

Things that still go wrong at this stage and how to handle them:

- Loss spike → resume from checkpoint *before* the spike, skip the
offending data shard, optionally lower LR transiently.
- Eval regression mid-run → save more checkpoints (every ~5% of total
steps), pick the best post hoc.
- Annealing / decay phase contributes most of the final-eval gain — Llama
3 and SmolLM3 both confirm this. Don't truncate it
([Llama 3 §3.4.3](https://arxiv.org/abs/2407.21783),
[SmolLM3 stage 3](https://huggingface.co/blog/smollm3)).

---

## 3. How to design *one* ablation: the matrix shape

Each ablation is a **matrix**, not a single run. The two axes:

- **Rows = the variants you're testing** (e.g. 4 masking strategies).
- **Columns = the controls** (often: target model size × { with intervention, without intervention } and ideally a "noise" control matched to your study, like our §4.3 twin from exp01/exp02).

Concretely, the same shape we used in exp02 wave-3 for ablating
`{ data cleaning, LM bridge, augmentation }`
([findings.md §3.2](../exp02_eeg_ctc/findings.md)) is the right shape for
SSL pretraining ablations:

```
                  | EEG | matched-noise
clean (baseline)  |  ✓  |      ✓
+ intervention 1  |  ✓  |      ✓
+ intervention 2  |  ✓  |      ✓
both interventions|  ✓  |      ✓
```

Each cell shares a fixed seed; all hyperparameters except the one being
tested are held constant. The value of the matched-noise control is
that it tells you whether your intervention is moving the *EEG signal*
or just lifting the floor.

Every cell ends with a verdict — strict win / weak win / tie / loss —
based on disjoint 95% bootstrap CIs and a paired sign-flip permutation
test, exactly the matched-pair §4.3 test from
[Jo et al. 2024 / Sci. Reports 2025](https://arxiv.org/abs/2405.06459)
that we already use in exp01/exp02.

---

## 4. Hyperparameter transfer: when small-scale tuning is trustworthy

The full µP / µTransfer story is in
[Yang et al. 2022 (Tensor Programs V)](https://arxiv.org/abs/2203.03466)
and the [MS Research blog](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/).
The minimum operational summary:


| Hyperparameter                                     | Transfers across width?                                                                         | Transfers across depth? | Transfers across batch size?                                             | Transfers across data?           |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------ | -------------------------------- |
| Peak learning rate                                 | **Yes**, via µP                                                                                 | Yes, empirically        | Yes, with linear-LR-with-batch-size scaling, up to a critical batch size | **No** — re-tune per data recipe |
| Weight decay                                       | No (must re-tune)                                                                               | No                      | No                                                                       | No                               |
| Dropout / drop-path                                | No                                                                                              | No                      | No                                                                       | No                               |
| LR schedule shape (warmup fraction, cosine vs WSD) | Yes                                                                                             | Yes                     | Yes                                                                      | Yes                              |
| Batch size                                         | Yes (up to critical batch size, see [McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)) | Yes                     | n/a                                                                      | Yes                              |


What this means in practice:

1. **Use µP if you can afford the engineering cost.**
  `pip install mup` ([microsoft/mup](https://github.com/microsoft/mup)).
   You write your model in their parametrization, tune LR on a
   width-256 model, and zero-shot transfer to width-2048+. **Caveat**:
   the EleutherAI / Cerebras port slightly differs from the MS official
   on whether you scale by `m_d` or `√m_d`; verify against your own coord-check
   plot before trusting transfer
   ([Stevens muP writeup](https://samuelstevens.me/writing/mup)).
2. **Otherwise, tune at the proxy scale and verify at the
  intermediate scale.** A 3-LR sweep at proxy is sufficient if you have
   no µP; the
   [DataDecide](https://arxiv.org/html/2504.11393v1) paper explicitly
   recommends this and the
   [Dec 2025 follow-up](https://www.arxiv.org/abs/2512.24503) shows
   that "use a smaller LR than you think for the proxy" is the cheapest
   single fix that makes data-recipe rankings transfer faithfully.
3. **Do not re-use the LR found on a different data recipe.** Re-do the
  LR sweep when you change the pretraining mixture — even a partial
   one. This is the most common cause of "ablation that worked at 1B
   stops working at 3B".

---

## 5. Compute allocation: scaling laws (the ones that survive 2024)

The Chinchilla-style relation
([Hoffmann et al. 2022](https://www.educatingsilicon.com/2024/04/29/revised-chinchilla-scaling-laws-impact-on-llm-compute-and-token-requirements/),
[arXiv 2203.15556](https://arxiv.org/abs/2203.15556))
gives ~20 tokens / parameter as the compute-optimal training-token
budget for raw cross-entropy, derived from a parametric loss fit to ~400
runs from 70M to 16B parameters. For text. As of 2024–2025 the literature
has converged on a few revisions worth knowing:

1. **Inference-aware scaling: train smaller, train longer.**
  [Sardana et al. 2024 (MosaicML/Databricks)](https://databricks.com/blog/how-long-should-you-train-your-language-model)
   show that if you'll do >>1× the training-token volume of inference,
   you should train models *much smaller* than Chinchilla-optimal on
   *much more* data — Llama 3 8B does ~75 tokens/parameter, Phi-3 does
   ~870. The improvement is monotone up to ~10,000 tokens/parameter at
   the extremes.
2. **DataDecide's empirical finding for *small-scale to large-scale*
  prediction**: a single 150M-parameter run gets ~80% of pairwise
   data-recipe rankings right at 1B
   ([arXiv 2504.11393](https://arxiv.org/html/2504.11393v1)). No fitted
   scaling law beats single-scale prediction in their study, with the
   notable exception of the
   [Bhagia et al. 2024](https://arxiv.org/abs/2412.04403) two-step
   loss→accuracy fit on multi-scale data. So: a single small-scale
   ablation is a strong baseline; you only need scaling-law fits if
   you're trying to predict *absolute* numbers (e.g. for a budget
   decision) rather than *ranking* (which architecture / data is best).

For SSL on EEG specifically, scaling laws are **not yet established**.
[LaBraM](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)
§3.6 plots TUAB / TUEV performance for {Base 5.8M, Large 46M, Huge 369M}
× {1, 10, 100, 500, 1000, 1500, 2000, 2500} pretraining hours. Two
findings:

- Performance improves with both data and parameters, but **gains
saturate around 1000 hours** for TUAB.
- The Huge model is only marginally better than Base on TUAB (binary
classification); the gain on TUEV (multi-class event classification)
is larger.

The [EEG-FM-Bench (arXiv 2601.17883, 2026)](https://arxiv.org/pdf/2601.17883)
finding **"larger foundation models do not necessarily yield better
generalization performance under current data regimes and training
practices"** is sobering. Provisional planning rule for EEG SSL: do not
budget for >100M-parameter models on <2000 hours of pretraining EEG
without an ablation showing the scale is justified.

---

## 6. What to monitor *during* SSL pretraining (the diagnostic dashboard)

Beyond the standard `loss / lr / grad_norm / wallclock`, an SSL run needs
these additional plots, logged ≥ every 1–5% of total training steps:

### 6.1 Encoder-feature health

Borrowed from the exp02 `stats.jsonl` we already log:

```python
# Per evaluation step:
feat_mean   = encoder_features.mean().item()
feat_std    = encoder_features.std().item()
feat_absmax = encoder_features.abs().max().item()
feat_rank   = torch.linalg.matrix_rank(features.cov(), tol=1e-3)
```

Hard rules:

- `feat_std` should be **stable across training** (drift > 5× between
steps is a warning sign).
- `feat_absmax / feat_std` should be **bounded** (a ratio > 50 means
outliers are dominating; this happened to TFM in exp02).
- `feat_rank / feature_dim` should be **> 0.5** — anything lower is
dimensional collapse
([He et al. NeurIPS 2024](https://arxiv.org/html/2411.00392v1)).
- For VQ codebooks: **codebook usage rate** (fraction of codes used
≥1 time per 1000 steps). LaBraM logs this; <5% means dead codes.

### 6.2 Linear probe trajectory on tiny labelled set

Run a *nightly* (or every-5%-of-steps) frozen-encoder linear probe on
the HBN 6-task classification (small labelled subset, ~10k samples; same
labels we use in §4.3 Protocol A.2). When TUH access lands, also run the
TUEV probe in parallel — both should track the same trajectory shape if
the representation is universal across the pediatric → adult clinical
distribution shift. Expected shape:

- Probe accuracy should rise monotonically until ~50% of pretraining
steps, then plateau or slightly decrease (this is the "critical
period" effect from
[Yu et al. NeurIPS 2025 / OpenReview UxIRc97ecL](https://openreview.net/pdf?id=UxIRc97ecL),
which is well-documented in vision SSL).
- A probe that *peaks then crashes* is a sign of overspecialization to
the pretext task — checkpoint averaging and CP-guided checkpoint
selection (CPCS) help.
- A probe that *never crosses random + 5%* is a sign of representation
collapse, dataset-of-recording-id leakage, or that your masking is
too easy / too hard.

### 6.3 Label-free monitors

If you have unlabelled validation EEG (you do), the
[Lourenço & Storkey 2024 paper](https://arxiv.org/html/2409.06612v1)
proposes:

- **k-means cluster silhouette score** on a held-out batch's embeddings.
Should rise then plateau.
- **Cluster agreement** with a reference clustering — a coarse one (e.g.
by source dataset or paradigm) tests how much of your representation is
spurious source-ID; a fine one (by subject) tests how much is subject
identity (often *too high* in EEG SSL).
- **Embedding entropy** — should rise during early training (the model
is learning to spread out the manifold), plateau in the middle, and
may decrease in the late annealing phase. A monotone decrease from
step 0 is a collapse warning.

### 6.4 Stepwise SSL learning

[The Berkeley AI Research blog post (Simon et al. 2023)](https://bairblog.github.io/2023/07/10/stepwise-ssl/)
showed that joint-embedding SSL learns *eigenmodes one at a time*, in
discrete steps. Plotting the rank of the embedding covariance is a
robust qualitative diagnostic — if the rank jumps in steps, your SSL is
working as designed; if it sits flat at 1 (or D), something is wrong.

---

## 7. What works, what doesn't — synthesis from text / image / speech FMs

A consolidated summary of the recipes that have worked in adjacent
modalities. You can read this in 5 minutes and use it as your menu of
priors.

### 7.1 Text (causal LM, BERT)

- **Architecture**: vanilla decoder-only transformer with
RoPE/NoPE/RNoPE; GQA over MHA for KV-cache efficiency; Llama-style
norm and MLP. Tested by SmolLM3 ablations and confirmed across labs.
- **Tokenizer**: BPE / SentencePiece with vocab size set by
fertility tradeoff — Llama-3 tokenizer has 128k vocab and is cited as
the best general-purpose default by SmolLM3. A bigger vocab compresses
the same text into fewer tokens (Llama 3 fertility ≈ 1.4 vs Llama 2's
≈ 1.6).
- **Data mixture**: multi-stage. SmolLM3 used 3 stages over 11.2T
tokens: stage 1 = web-heavy, stage 2 = introduce code+math, stage 3 =
decay phase upsampling high-quality content.
- **Annealing data quality probe**: Llama 3 §3.1.3 — anneal a half-
trained 8B model on 40B tokens with `30% candidate dataset / 70% baseline`. Performance delta on GSM8k tells you if a small dataset is
worth keeping at scale. **More efficient than fitting a per-dataset
scaling law.**
- **Eval suite for ablations**: HellaSwag, ARC-easy, OBQA, PIQA, MMLU,
WinoGrande, CommonsenseQA, Stack-Edu (code), GSM8k (math), RULER
(long context). Evaluated on cloze formulation (CF) for ablations
because multiple-choice (MCF) is too noisy at small scale
([Smol Playbook](https://huggingfacetb-smol-training-playbook.hf.space/),
[Du et al. 2025](https://arxiv.org/abs/2506.04391)).
- **Headline lesson**: every architectural choice was independently
ablated against a fixed 1B/45B baseline. SmolLM3 ran ~hundreds of such
ablations; the WSD scheduler, GQA-4, NoPE-on-every-4th-layer, document
masking, intra-document attention all came from these.

### 7.2 Image (joint-embedding, masked-image-modeling, JEPA)

- **Joint-embedding** (DINO/DINOv2/iBOT): EMA teacher, multi-crop,
centring + sharpening, optional photometric augmentations. Modern
finding: **DINOv2 + crop-only (no resize, no photometric augs) reaches
SOTA at ImageNet22k scale**
([arXiv 2406.09294](https://arxiv.org/html/2406.09294v1)) — i.e.
augmentations were a sample-efficiency hack, not a representation-
learning necessity.
- **Masked image modeling** (BEiT, MAE): random masking of 75% patches,
ViT encoder over visible patches only, lightweight decoder
reconstructs pixels. Robust at small dataset sizes.
- **Joint-embedding predictive architecture** (I-JEPA): predict in
*representation space*, not pixel space. Multi-block masking is the
key ablation — predicting 4 large target blocks from 1 large context
block. Linear probe on ImageNet1k jumps from ~50% (block masking) to
~70% (multi-block), before any other tweaks
([Assran et al. 2023](https://arxiv.org/abs/2301.08243)).
- **Headline lesson**: representation-space prediction beats
pixel-space prediction; this generalises well to spectrogram /
channel-patch settings (LaBraM and EEGPT both adopt latent-space
prediction; raw-signal MAE often underperforms).

### 7.3 Speech (wav2vec 2.0, HuBERT, BEATs)

- **wav2vec 2.0**: 1D-CNN waveform encoder → quantize → mask spans of
the latent → contrastive InfoNCE over masked positions with
in-utterance negatives. Trained on 60k hours of LibriLight on 64 GPUs
for ~1 week.
  - **Small-scale ablation**: §3.2 of the
  [paper](https://www.huggingface.co/papers/2006.11477) — reduced LS-960,
  250k updates, 1× GPU, three seeds. Tested mask probability,
  quantizer hyperparameters. The 250k-update reduced setup is the
  canonical wav2vec 2.0 ablation cell.
- **HuBERT**: replaces the quantizer with offline k-means on MFCCs / on
intermediate transformer features, makes the loss CE over discrete
cluster labels of masked frames. Iterative — re-cluster after first
iteration on transformer-layer features.
  - **Small-scale ablation**: §IV.D of the
  [paper](http://arxiv.org/pdf/2106.07447v1) — 100k-step pretrain on
  LS-960, fine-tune on 10h. Tested mask predict-only vs predict-all,
  cluster ensemble, hyperparameters.
  - **Academic-budget reproduction**: [SLT 2024 paper
  "Reproducing HuBERT in academic constraints"](http://arxiv.org/pdf/2306.06672v1)
  — 8 GPUs (vs original 32–256), gradient accumulation, optional
  semi-SSL initialisation from an ASR model. Useful template if you
  want to do a HuBERT-style approach on a single node.
- **BEATs**, **EAT**, **MMS** etc. follow the same general pattern.
- **Headline lesson**: discrete-target objectives (HuBERT, LaBraM, BEATs)
generally outperform continuous-target objectives (wav2vec 2.0, MAE
raw-pixel) for downstream classification. This is well-documented in
speech and now in EEG. See
[EEG-FM-Bench finding (5)](https://arxiv.org/html/2508.17742v1) for
the EEG-side confirmation.

---

## 8. EEG-specific guidance

Now the application layer. Synthesised from
[Brain4FMs (2026)](https://arxiv.org/pdf/2602.11558),
[EEG-FM-Bench (2025)](https://arxiv.org/html/2508.17742v1),
[Critical Review (Mantilla-Ramos et al. 2025)](https://arxiv.org/abs/2507.11783),
and individual EEG-FM papers.

### 8.1 The current EEG-FM landscape (cheat-sheet)


| Model                    | Pretraining data                               | SSL objective                                               | Params            | Key design choice                                                                 | Reference                                                                                                                  |
| ------------------------ | ---------------------------------------------- | ----------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **BENDR** (2021)         | 1.5 TB TUEG (≥10k subjects)                    | wav2vec2-style contrastive                                  | 4M                | Adapted from speech; CNN encoder + transformer                                    | [Frontiers 2021](https://www.frontiersin.org/articles/10.3389/fnhum.2021.653659/full)                                      |
| **BrainBERT** (2023)     | 43.7h iEEG (Wang et al.)                       | Masked spectrogram                                          | 43M               | STFT + masked recon; iEEG only                                                    | [arXiv 2302.14367](https://arxiv.org/abs/2302.14367)                                                                       |
| **BIOT** (2023)          | 58k h EEG+ECG                                  | Contrastive (MoCo-style)                                    | 3.2M              | Channel-independent token; handles any montage                                    | [NeurIPS 2023](https://openreview.net/forum?id=9w8fF1A1nC)                                                                 |
| **LaBraM** (ICLR 2024)   | 2500h EEG (~20 datasets)                       | VQ neural-spectrum prediction → masked code prediction      | 5.8M / 46M / 369M | 2-stage: VQ tokenizer first, then transformer; channel-patch tokens               | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf) |
| **EEGPT** (NeurIPS 2024) | mixed multi-task                               | Dual SSL: spatio-temporal alignment + masked reconstruction | 4.7M / 25M        | Predicts in *representation space* (I-JEPA-style)                                 | [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/93793)                                                               |
| **CBraMod** (2025)       | EEG                                            | Masked reconstruction with criss-cross attention            | —                 | Architectural specialisation for spatial+temporal                                 | [Wang et al. 2024b](https://arxiv.org/abs/2412.07236)                                                                      |
| **REVE** (NeurIPS 2025)  | **60,000 h** from 92 datasets, 25,000 subjects | Masked autoencoding                                         | —                 | 4D positional encoding for arbitrary montage; **largest pretraining to date**     | [arXiv 2510.21585](https://arxiv.org/abs/2510.21585)                                                                       |
| **MTDP** (2026)          | 25% of REVE-scale data                         | Multi-teacher distillation from DINOv3 + Chronos            | —                 | **Bootstraps EEG-FM by distilling vision/time-series FMs** — bypasses scratch SSL | [arXiv 2603.04478](https://arxiv.org/html/2603.04478v1)                                                                    |


Three non-obvious findings from the surveys:

1. **Channel-independent tokenisation is a key design choice.** Both BIOT
  and LaBraM tokenise *one EEG channel at a time* and let the
   transformer attention handle inter-channel structure. This is what
   makes them work across heterogeneous montages.
2. **Codebook objectives beat reconstruction objectives.** LaBraM,
  NeuroLM, EEGFormer (with codebooks) outperform pure raw-signal
   reconstruction across most downstream tasks
   ([EEG-FM-Bench finding (5)](https://arxiv.org/html/2508.17742v1)).
3. **Distillation from frozen vision/time-series FMs is becoming
  competitive with EEG-from-scratch.**
   [MTDP (Li et al. 2026)](https://arxiv.org/html/2603.04478v1) get
   parity with LaBraM-Base using **25% of the data** by distilling
   DINOv3 + Chronos. This is a strong signal that scratch-SSL on EEG is
   *not* the only or even necessarily the best way forward.

### 8.2 EEG-specific gotchas (synthesised from exp01/exp02 + literature)

- **Subject leakage**: EEG is so subject-specific that any train/dev
split that doesn't enforce subject disjointness is essentially
measuring how well your model memorises individuals. Always enforce
leave-N-subjects-out (LNSO).
- **Source-ID leakage**: similarly, the model can learn to recognise the
*recording rig* (sampling rate fingerprint, notch filter signature,
channel-naming convention) and use that as a shortcut. The exp01/exp02
defense was *aggressive* canonical offline preprocessing (resample +
notch + bandpass + per-recording z-score + 15-σ clip). For exp03 SSL
pretraining we deliberately depart from this orthodoxy: offline
preprocessing is reduced to the minimum (NaN sanitation + per-channel
z-score + ±5σ clip + 4-second windowing per
[`mini_experiments.md` §4.1](./mini_experiments.md#41-pretraining-corpus)),
because notch / bandpass / resampling are themselves the questions
asked by exp02 (frontend), exp05 (multi-rate), and exp14 (context
length). Doing them offline pre-decides those experiments. The
anti-shortcut defenses become **architectural and adversarial** rather
than preprocessing-based:

  * The "predict recording site" probe (HBN's 4 CMI sites are the
    closest analogue of source-dataset in our HBN-only corpus) and
    "predict subject ID" k-NN, both run every checkpoint and required
    to **decrease** over training (§4.3 monitor).
  * Optionally, the gradient-reversal adversarial head from exp13 if
    the passive monitors show the rig fingerprint isn't fading on its
    own.

  The orthodox "pre-process to a canonical pipeline" approach remains
  appropriate for *fine-tuning* a frozen pretrained encoder (where the
  encoder's input expectation is fixed and you're trying to match it),
  which is why exp01 / exp02 still use the V2 pipeline. The departure
  is specific to the SSL-pretraining-from-scratch setting.
- **Channel-mismatch zero-padding**: when sources have different channel
counts, padding with zeros means most rows are mostly zero. LaBraM and
REVE both solve this with channel-as-token; do the same.
- **Sample-rate mismatch**: anything more aggressive than ~5× resample
ratio gives bad results with linear interpolation; use anti-aliased
decimation (`scipy.signal.resample_poly`).
- **Low SNR**: EEG SNR is much worse than speech or images. This means:
(a) you need much more data per parameter than text scaling laws
suggest; (b) augmentations can hurt
([BENDR's wav2vec-style CL works less well than masked
reconstruction](https://arxiv.org/html/2403.03222v1) Knowledge-guided
EEG paper); (c) latent-space objectives (EEGPT, MTDP) generally
outperform pixel/raw-signal objectives.
- **Heterogeneous label semantics**: a binary clinical label (HBN
ADHD-vs-no-diagnosis, or TUAB normal/abnormal) is a much weaker signal
than a per-class label (HBN 6-task classification, or TUEV per-event
labels). Running both a binary AUROC and a multi-class BAC/WF1 probe as
the ablation eval suite is cheap insurance.

### 8.3 Assessment of "treat each EEG channel as a different signal /

speech and pretrain"

This idea has serious precedent and merits a small ablation, but is
**not novel** as currently phrased — and the existing implementations
have known limitations. Triangulation:

1. **BENDR (2021)** is essentially this idea: take the wav2vec 2.0
  architecture (originally for 16 kHz mono speech), feed EEG into it
   channel-by-channel, contrastive-pretrain on TUEG. It works. It is
   also **reliably outperformed by latent-space EEG-FMs (LaBraM, EEGPT,
   REVE)** in modern benchmarks
   ([EEG-FM-Bench](https://arxiv.org/html/2508.17742v1) Table 1;
   EEGPT ablations show **+9.4% on BCIC-2A vs BENDR**).
2. **SPP-EEGNet (Li & Metsis 2022)** does single-channel SSL with a
  spatial-pyramid pooling layer to handle variable channel counts:
   [GitHub](https://github.com/imics-lab/eeg-transfer-learning).
3. **BIOT (NeurIPS 2023)** uses channel-independent tokens explicitly:
  "channel-independent methods for variable inputs"
   ([survey](https://arxiv.org/pdf/2602.11558)).
4. **LaBraM (ICLR 2024)** — segments each channel into 1-second
  patches, embeds them independently, then does cross-channel
   attention in the transformer. This is the strongest evidence that
   "treat channels independently at the token level, but jointly at the
   attention level" is the right shape.
5. **MAEEG (NeurIPS-W 2022)** — exactly a masked autoencoder on EEG,
  per-channel. Works but is ~order-of-magnitude smaller than current
   SOTA.

Implications for design choices:

- **Don't reinvent BENDR.** Reuse `Kostas et al. 2021`'s code from
[SPOClab-ca/BENDR](https://github.com/SPOClab-ca/BENDR) as the
baseline and ablation control.
- **Don't reinvent LaBraM either.** It's the current strongest
channel-independent codebook-prediction recipe; their checkpoints +
code are open ([935963004/LaBraM](https://github.com/935963004/labram)).
- **Plausible novelty surfaces** (where the literature is currently
thin, in rough order of cheapness):
  1. **Speech-pretrained encoder transfer**. Use a HuBERT-Base /
    wav2vec2 encoder pretrained on LibriSpeech, treat each EEG
     channel as a 16 kHz mono signal (after 200 Hz → 16 kHz upsampling
     or letting the CNN learn the rate adaptation), light fine-tune on
     EEG. There is one paper exploring exactly this:
     [Chien et al., "Leveraging Pretrained Speech Model for EEG Signal
     Recognition" (IEEE 2023)](https://ieeexplore.ieee.org/iel7/7333/4359219/10106018.pdf).
     But to my knowledge no one has done this *as a pretraining
     stage* for an EEG-FM.
  2. **HuBERT-style iterative clustering on EEG.** Cluster 1-second
    EEG patches with k-means on MFCC-equivalents (spectral features),
     train an EEG transformer to predict the cluster of each masked
     patch, re-cluster on transformer features, iterate. This is
     LaBraM's nearest neighbour but with HuBERT-style iterative
     refinement instead of a static VQ. Cheap ablation.
  3. **Cross-modal distillation à la MTDP**. Use a frozen wav2vec2 /
    HuBERT encoder as teacher and an EEG transformer as student. Take
     a pretraining EEG dataset, run wav2vec2 over each channel, and
     train the EEG model to match those features. MTDP showed this
     works for DINOv3 + Chronos teachers
     ([arXiv 2603.04478](https://arxiv.org/html/2603.04478v1)); the
     same recipe with a speech teacher would be novel in the EEG
     literature as far as I can tell.

The **honest summary**: "treat EEG channels as speech" is not a
single experiment, it's a family of three: (a) initialise EEG model
from speech weights and fine-tune; (b) pretrain EEG with a
HuBERT/wav2vec2-style objective from scratch; (c) distill from a
speech FM into an EEG model. (a) and (c) have novelty; (b) is BENDR.
A small ablation matrix testing all three against a LaBraM-style
codebook baseline would be a publishable design.

---

## 9. Common SSL failure modes & remedies

Cookbook of failure modes specific to SSL pretraining, drawn from the
literature and from our own exp01 / exp02 experience.


| Symptom                                                 | Probable cause                                               | Diagnostic                                                                                  | Remedy                                                                                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| SSL loss → 0 fast, downstream stays at random           | Shortcut learning (e.g. recording-ID, padding pattern)       | Train a "predict source dataset" probe on encoder features                                  | Add data normalisation; randomise position of zero-padded channels; channel-independent tokens                                      |
| SSL loss steady, eigenvalues collapse to top-k          | Dimensional collapse (joint-embedding)                       | Plot eigenvalue spectrum every 5% steps                                                     | Add VICReg variance/covariance terms or orthogonality reg; reduce projector width                                                   |
| VQ codebook usage <5%                                   | Codebook collapse                                            | Log codebook usage histogram                                                                | Re-init dead codes (codebook reset every N steps); EMA codebook (LaBraM); add diversity loss                                        |
| Loss curve looks great, downstream eval crashes mid-run | Critical-period overspecialization                           | Plot linear-probe trajectory                                                                | Save checkpoints every 5%; pick best-probe checkpoint, not last (CPCS — [Yu et al. 2025](https://openreview.net/pdf?id=UxIRc97ecL)) |
| Diverging loss + spike                                  | Bad data shard at high LR                                    | Log per-shard loss; check for NaN                                                           | Skip-batch, lower LR, resume from checkpoint pre-spike                                                                              |
| Throughput drops mid-run                                | IO starvation                                                | Monitor `dataloader_time / step_time`                                                       | Pre-load to local NVMe; fewer workers; smaller shards                                                                               |
| TFM-encoder-frozen bug we hit in exp02                  | `@torch.no_grad()` on tokenizer despite `requires_grad=True` | Compare encoder feature stats between two checkpoints — if bit-identical, encoder is frozen | Drop `@torch.no_grad()`, drop `.eval()` in `__init`__, verify on a 100-step toy run                                                 |
| Linear probe peaks at 30% then crashes                  | Pretext-task overspecialization (critical period)            | Track probe accuracy every checkpoint                                                       | Average last-K checkpoints; CPCS-style early stopping based on Fisher Information plateau                                           |
| Fine-tuning works, linear probe doesn't                 | Representation usable but not linearly separable             | Compare LP vs FT vs k-NN                                                                    | This is a known property of MAE; not a bug. EEGPT has a multi-layer head for this reason.                                           |


---

## 10. Operational pre-flight checklist

Before kicking off any SSL pretraining run at scale, this list must be
all-✓. Treat any unchecked item as a critical bug.

- Phase-0 dataset audit complete; per-source sr / channels / hours /
magnitude tables in repo.
- Phase-1 sanity baselines all passed: loss-at-init ≈ theory,
input-independent at chance, 1-batch overfit succeeds, random-init
linear probe number recorded.
- Encoder-feature stats logging (mean/std/abs_max/rank) in trainer.
- Codebook usage logging in trainer (if VQ).
- Linear probe + k-NN + label-free silhouette evaluator runs on
every checkpoint, takes <10 min on 1 GPU.
- A noise-twin control (per §3 matrix shape, §4.3 of exp01/exp02) is
wired in alongside every signal-bearing run.
- Subject-disjoint splits verified; "predict source dataset" probe
accuracy recorded as a control (should drift down, not rise).
- LR sweep at proxy scale (≥ 3 LRs × 3 seeds = 9 runs) before
committing to the headline LR — re-done whenever the data recipe
changes (§4).
- Checkpoints saved every ~5% of total steps so the best-probe
checkpoint can be picked post hoc.
- Throughput monitor: `dataloader_time / step_time < 0.2` after
warmup.
- Reproducibility: every run has a seed, a config hash, and a git
SHA in its `log.jsonl` header.
- All runs logged through a uniform storage layout (e.g. the
exp02-style `runs/<id>/`, `eval/<id>/` convention) so cross-run
comparisons are mechanical.

---

## 11. References (curated)

Group ordering: **methodology → scaling laws → SSL diagnostics → modality
recipes → EEG-specific**. Each item is annotated with what to actually
read it for.

### 11.1 Methodology / experimentation

- **Karpathy, A. (2019). *A Recipe for Training Neural Networks*.**
[karpathy.github.io/2019/04/25/recipe/](https://karpathy.github.io/2019/04/25/recipe/)
— *Read this start-to-finish. The 6-step recipe is foundational.*
- **HuggingFaceTB. (2025). *The Smol Training Playbook: The Secrets to
Building World-Class LLMs*.**
[huggingface.co/spaces/HuggingFaceTB/smol-training-playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
— *The closest thing to a 2025-era equivalent of Karpathy's recipe,
for pretraining specifically. Read §"Every Big Model Starts with a
Small Ablation".*
- **Penedo et al. (2024). *FineWeb*.**
[HF blog](http://huggingfacefw-blogpost-fineweb-v1.static.hf.space/)
— *The canonical reference for "test data recipes with proxy 1.82B /
28B-token ablation models on a fixed eval suite".*
- **Magnusson et al. (2025). *DataDecide: How to Predict Best
Pretraining Data with Small Experiments*.**
[arXiv 2504.11393](https://arxiv.org/html/2504.11393v1)
— *Empirical study showing single-scale 150M proxy models predict
~80% of recipe rankings at 1B. Companion HF collection has all 30k
checkpoints.*
- **Wang et al. (2025). *Can Small Training Runs Reliably Guide Data
Curation? Rethinking Proxy-Model Practice*.**
[arXiv 2512.24503](https://www.arxiv.org/abs/2512.24503)
— *The "use lower LR than you think for proxy models" paper. Critical
read if you're doing data ablations.*
- **Yang, Hu et al. (2022). Tensor Programs V: Tuning Large Neural
Networks via Zero-Shot Hyperparameter Transfer (µTransfer).**
[arXiv 2203.03466](https://arxiv.org/abs/2203.03466) /
[microsoft/mup](https://github.com/microsoft/mup) /
[MS Research blog](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/)
— *The standard reference for transferring LR across width.*

### 11.2 Scaling laws

- **Hoffmann et al. (2022). *Training Compute-Optimal Large Language
Models* (Chinchilla).**
[arXiv 2203.15556](https://arxiv.org/abs/2203.15556)
— *The 20-tokens-per-parameter result. Always cite alongside revisions.*
- **Sardana et al. (2024). *Beyond Chinchilla-Optimal: Accounting for
Inference in Language Model Scaling Laws*.**
[arXiv 2401.00448](http://arxiv.org/abs/2401.00448) /
[Databricks blog](https://databricks.com/blog/how-long-should-you-train-your-language-model)
— *Inference-aware revision. Validates models trained 100× past
Chinchilla-optimal.*
- **Romero, T. (2026). *Thinking about Scaling Laws*.**
[tylerromero.com/posts/2026-01-scaling-laws](https://www.tylerromero.com/posts/2026-01-scaling-laws/)
— *Practical engineer-facing distillation, with implementations.*

### 11.3 SSL diagnostics & failure modes

- **He, Du, Ma. (2024). *Preventing Dimensional Collapse in
Self-Supervised Learning via Orthogonality Regularization*.**
[arXiv 2411.00392](https://arxiv.org/html/2411.00392v1)
/ NeurIPS 2024
— *Dimensional collapse is the silent killer of joint-embedding SSL.
This paper has eigenspectrum plots showing the symptom and a clean
fix.*
- **Lourenço, Storkey. (2024). *Label-free Monitoring of Self-Supervised
Learning Progress*.**
[arXiv 2409.06612](https://arxiv.org/html/2409.06612v1)
— *k-means silhouette / cluster agreement / entropy as label-free
signals.*
- **Yu et al. (2025). *Critical Periods in Self-Supervised Learning*.**
[OpenReview UxIRc97ecL](https://openreview.net/pdf?id=UxIRc97ecL)
— *The "linear probe peaks then crashes" phenomenon, with
Fisher-Information-based checkpoint selection (CPCS).*
- **Simon, Bordelon, Pehlevan, Pennington. (2023). *On the Stepwise
Nature of Self-Supervised Learning*.**
[BAIR blog](https://bairblog.github.io/2023/07/10/stepwise-ssl/) /
ICML 2023
— *Eigenvalues of the embedding covariance learn one-at-a-time. Best
qualitative diagnostic plot for SSL pretraining.*
- **Dubois, Hashimoto, Liang. (2023). *Evaluating Self-Supervised
Learning via Risk Decomposition*.**
[arXiv 2302.03068](https://export.arxiv.org/pdf/2302.03068v2.pdf)
/ ICML 2023
— *Decomposes SSL error into approximation, usability, probe-gen,
encoder-gen. Useful framework for diagnosing what is failing.*

### 11.4 Modality-specific FMs (recipes & ablations)

- **wav2vec 2.0 (Baevski et al. 2020).**
[HF paper page](https://www.huggingface.co/papers/2006.11477) /
[arXiv 2006.11477](http://arxiv.org/pdf/2006.11477)
— *§3.2 of the paper has the reduced-LS-960 ablation setup. Use it
as your template for a wav2vec-style ablation.*
- **HuBERT (Hsu et al. 2021).**
[arXiv 2106.07447](http://arxiv.org/pdf/2106.07447v1)
— *§IV.D ablation studies. Iterative k-means → CE-on-cluster targets.*
- **Chen et al. (2023). *Reproducing HuBERT in academic constraints*.**
[arXiv 2306.06672](http://arxiv.org/pdf/2306.06672v1)
— *8-GPU HuBERT-Large. The template for academic-budget speech SSL.*
- **MAE (He et al. 2021).** [arXiv 2111.06377](https://arxiv.org/abs/2111.06377)
— *75% masking; pixel-space reconstruction; lightweight decoder.*
- **DINOv2 (Oquab et al. 2023).** [arXiv 2304.07193](https://arxiv.org/abs/2304.07193)
— *EMA teacher + multi-crop + iBOT loss; the strongest joint-embedding
recipe at scale.*
- **I-JEPA (Assran et al. 2023).** [arXiv 2301.08243](https://arxiv.org/abs/2301.08243)
— *Predict in representation space. Multi-block masking ablation.*
- **Llama 3 Herd of Models (Meta, 2024).**
[arXiv 2407.21783](https://arxiv.org/abs/2407.21783)
— *§3.1.3 (annealing data quality probe), §3.4 (training stages),
scaling-law construction details.*
- **SmolLM3 (HF, 2025).**
[HF blog](https://huggingface.co/blog/smollm3)
— *Most transparent open recipe at the 3B scale. Read alongside the
Smol Training Playbook.*
- **modded-nanoGPT speedrun (Jordan et al. 2024–).**
[GitHub](https://github.com/KellerJordan/modded-nanogpt)
— *Living example of single-change ablation culture; every record is
a documented one-line change with its own ablation log.*

### 11.5 EEG / biosignal SSL & FMs

- **BENDR (Kostas et al. 2021).**
[Frontiers Hum. Neurosci.](https://www.frontiersin.org/articles/10.3389/fnhum.2021.653659/full)
/ [SPOClab-ca/BENDR](https://github.com/SPOClab-ca/BENDR)
— *The original "treat EEG like wav2vec speech" paper. Reproducible
weights available.*
- **LaBraM (Jiang, Zhao, Lu 2024). ICLR 2024 spotlight.**
[PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)
/ [GitHub](https://github.com/935963004/labram)
— *Best-published channel-independent codebook-prediction recipe.
3 sizes (5.8M / 46M / 369M); Appendix has full hyperparameters.*
- **EEGPT (Wang et al. 2024). NeurIPS 2024.**
[NeurIPS poster](https://neurips.cc/virtual/2024/poster/93793) /
[GitHub](https://github.com/BINE022/EEGPT)
— *I-JEPA-inspired representation-space prediction; spatio-temporal
alignment loss; small (10M).*
- **REVE (El Ouahidi et al. 2025). NeurIPS 2025.**
[arXiv 2510.21585](https://arxiv.org/abs/2510.21585)
— *60k-hour pretraining, 25k subjects, 4D positional encoding for
arbitrary montage. We're already using its checkpoints; the paper has
the largest-scale published EEG-FM ablation.*
- **MTDP — Multi-Teacher Distillation Pretraining for EEG (Li et al.
2026).** [arXiv 2603.04478](https://arxiv.org/html/2603.04478v1)
— *Distill DINOv3 + Chronos into an EEG-FM. 25% of the data, parity
with LaBraM-Base. Strong evidence that scratch-SSL on EEG is not the
only path.*
- **EEG-FM-Bench (2025).**
[arXiv 2508.17742](https://arxiv.org/html/2508.17742v1)
— *Comprehensive evaluation of BIOT, BENDR, LaBraM, EEGPT, CBraMod
across paradigms. Five sobering findings (linear probing weak,
scaling doesn't always work, etc.).*
- **Brain4FMs (2026).**
[arXiv 2602.11558](https://arxiv.org/pdf/2602.11558)
— *Survey of 50 BFMs + benchmark of 12 across 18 datasets. The
"linear probing is frequently insufficient" finding is from here.*
- **Mantilla-Ramos et al. (2025). *EEG Foundation Models: A Critical
Review*.**
[arXiv 2507.11783](https://arxiv.org/abs/2507.11783)
— *Methodological review of 10 EEG-FMs: input representation,
pretraining objective, evaluation strategy.*
- **Knowledge-guided EEG Representation Learning (Kommineni et al.
2024).** [arXiv 2403.03222](https://arxiv.org/html/2403.03222v1)
— *Argues that wav2vec/MAE objectives don't translate well to EEG due
to low SNR; proposes a knowledge-guided alternative. Read for the
argument, not necessarily the method.*

### 11.6 Notable people / accounts to follow

- **Andrej Karpathy** — [@karpathy](https://twitter.com/karpathy) on X,
[karpathy.github.io](https://karpathy.github.io). Recipe + nanoGPT +
llm.c.
- **Keller Jordan** — [@kellerjordan0](https://x.com/kellerjordan0).
Modded-nanoGPT, Muon, speedrun culture.
- **Loubna Ben Allal** ([HuggingFace](https://huggingface.co/loubnabnl)),
**Anton Lozhkov**, and the SmolLM/FineWeb team on HF.
- **Greg Yang** — Tensor Programs / µP author. Important if you're
considering µTransfer.
- **Sebastian Raschka** ([@rasbt](https://x.com/rasbt)) — pragmatic
pretraining commentary on Substack.
- **Lilian Weng** ([@lilianweng](https://lilianweng.github.io)) —
highest-quality summary blog posts (her [2019 SSL post](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)
is still useful as a taxonomy reference).
- **Jean-Rémi King**, **Alexandre Défossez** (Meta FAIR brain-decoding
group). Their D-SigLIP / contrastive recipe is the only one to clear
§4.3 on EEG-to-text; they're worth following for biosignal-specific
insights.

---

## 12. Maintenance — when to update this document

Update this file when:

- A new ablation (ours or in the literature) reveals a result that
contradicts §8 (e.g. "k-NN beats linear probe as an early-signal eval
on EEG").
- A new EEG-FM lands that changes the cheat-sheet in §8.1 — at the
current pace of the field (2024 → 2026), expect this every ~3 months.
- A new failure mode is discovered in our own runs that isn't in §9.
- A future experiment establishes a new baseline strong enough that
the §0 rules or the §2 phase exit criteria need to be updated.

Each update should add a one-line entry to the changelog below, with
date and (optional) commit SHA.

### Changelog

- 2026-05-01 — Initial draft. Synthesised from a broad sweep of
methodology, scaling-laws, SSL-diagnostics, and EEG-FM literature
(Karpathy 2019, Smol Playbook 2025, FineWeb 2024, DataDecide 2025,
µTransfer 2022, Llama 3 herd 2024, BENDR 2021, LaBraM 2024, EEGPT
2024, REVE 2025, MTDP 2026, EEG-FM-Bench 2025, Brain4FMs 2026,
critical-period-SSL 2025).
- 2026-05-01 — Stripped the project-specific pilot plan (former §10:
hypothesis tree, pilot setup, C0–C5 ablation matrix, decision rule)
to keep this document a pure insights/methodology playbook. Renumbered
former §11/§12/§13 → §10/§11/§12; the operational checklist (now §10)
was rewritten to be generic across SSL pretraining runs.

