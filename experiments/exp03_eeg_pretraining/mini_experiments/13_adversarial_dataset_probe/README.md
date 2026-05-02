# exp03 / mini-experiment 13 — Adversarial source-dataset probe

> **Status:** planned
>
> **Cross-reference:** [`brain/cortico-ssl-hypothesis.typ` §7.7](../../../../../brain/cortico-ssl-hypothesis.typ),
> [`methodology.md` §6.4](../../methodology.md#64-pretraining-objectives--what-works-and-what-doesnt),
> [DANN, Ganin et al. JMLR 2016](https://jmlr.org/papers/v17/15-239.html)
>
> **Compute budget:** 12 H100-hours (4 variants × 2 control columns × 3 seeds =
> 24 cells × 30 min average)
>
> **Gates:** none downstream — exp13 is a hardening experiment that reports a
> single configuration choice for the headline run

## Question

Does adding a *gradient-reversal* adversarial head that explicitly penalises
the encoder for being able to predict the source dataset of each window
materially reduce subject/rig/montage fingerprinting (P6) without hurting
downstream task performance, and at what loss weight?

## Why it matters

The cortico-ssl-hypothesis lists $\mathcal{L}_\text{adv}^\text{dataset}$ as
one of the seven heads in the composite loss (recipe weight $\lambda_\text{adv}
= 0.1$) and as the cheapest defence against P6 — the pathology where the
encoder finds it easier to learn rig identity than neural content. Hypothesis
H6 demands that source-dataset linear-probe accuracy on the encoder pool be
below 50 % (random = 25 % on a 4-dataset mix) by 100 k steps.

The earlier mini-experiments 02–12 only *monitor* this probe as a checkpoint
sanity. None of them directly compares "with adversarial head" to "without
adversarial head". This is a real omission because:

- The probe could trend down on its own as the encoder learns better
  features; the adversarial head might be redundant.
- The probe could be too weak to push the encoder; a stronger adversary
  (multi-layer head, higher $\lambda_\text{adv}$) might be required.
- The adversarial head might **hurt** downstream performance by destroying
  generally useful features that happen to correlate with dataset identity
  (sampling-rate bias, montage convention, age distribution).

Domain-adversarial neural networks (DANN,
[Ganin et al. JMLR 2016](https://jmlr.org/papers/v17/15-239.html)) are a
mature technique with a known pathology: too-aggressive adversarial weight
collapses representation quality, too-weak adversarial weight does nothing.
The right $\lambda_\text{adv}$ is empirical and dataset-specific. NeuroLM
([Jiang et al. ICLR 2025](https://openreview.net/forum?id=Io9yFt7XH7)) is the
only published EEG foundation model that uses an explicit adversarial domain
classifier; their setup is multi-task, not single-channel SSL.

## Variants

The variants differ in the *strength* and *form* of the adversarial probe.
All sit on top of the W4 (or best-of-Phase-A) configuration from exp12.

| Code | Variant | Adversarial head | Adversarial weight $\lambda_\text{adv}$ |
| ---- | ------- | ---------------- | ---------------------------------------- |
| A0 | None (the W4 baseline from exp12) | — | 0 |
| A1 | Single linear head over `K=4` source datasets, GRL with $\alpha=1$ | linear | 0.05 |
| A2 | Same head, higher weight | linear | 0.20 |
| A3 | 2-layer MLP head with hidden $D/2$, GRL with $\alpha=1$ | 2-layer MLP | 0.10 |
| A4 | A3 with subject-ID auxiliary head added (joint dataset + subject adversary) | 2-layer MLP × 2 heads | 0.10 each |

The $\alpha$ parameter of the gradient-reversal layer is a multiplier
applied to the negated gradient flowing back into the encoder; we keep it
fixed at 1 and use $\lambda_\text{adv}$ for strength control, per the
DANN paper convention.

A4 (subject-ID adversary) is included because the *real* worry is subject
fingerprinting, and dataset ID is a coarse proxy. If A4 strict-wins, the
recipe carries forward both heads.

## Controls (the §3 matrix)

|                                 | EEG signal | matched-noise twin |
| ------------------------------- | ---------- | ------------------ |
| A0 baseline (no adversary)      | ✓          | ✓                  |
| A1 linear head, $\lambda=0.05$  | ✓          | ✓                  |
| A2 linear head, $\lambda=0.20$  | ✓          | ✓                  |
| A3 MLP head, $\lambda=0.10$     | ✓          | ✓                  |
| A4 dataset+subject MLP heads    | ✓          | ✓                  |

The matched-noise twin here is informative as a sanity check rather than a
win/loss criterion: a model trained with an adversary on Gaussian noise
should not improve its HBN 6-task / HBN ADHD-binary metrics (there is no signal to learn).
If it does, the adversarial head is somehow stabilising the optimisation
rather than enforcing invariance.

## Held constant

Everything from exp12's headline configuration:

- Frontend: exp02 winner.
- Backbone: exp03 winner.
- SSL framework: exp04 winner.
- Multi-rate strategy: exp05 winner.
- Reconstruction loss: exp06 winner.
- Phase loss: exp07 winner.
- Target: exp08 winner.
- Multi-condition input: exp09 winner.
- Mask: exp10 winner.
- Bottleneck: exp11 winner.
- Quick-wins stack: exp12 W4 (or the best Phase-A subset).
- Optimiser: AdamW, LR carried forward (no re-tune; the adversary's effect
  on the LR landscape is small in our experience).
- Pretraining duration: 8 epochs.

The new ingredients are the GRL + classification head configuration and
the loss weight $\lambda_\text{adv}$.

## Decision rule

**Output A — chosen adversarial configuration** (HBN 6-task BAC, the
standard metric per §4.3 Protocol A.2):

- Strict win = ≥ 0.5 pp HBN 6-task BAC over A0, non-overlapping CIs,
  noise-twin flat. (Lower bar than other experiments because the
  adversary's primary purpose is invariance, not representation quality.)
- Tie = TOST equivalence within $\varepsilon = 0.5$ pp.
- Loss = ≥ 0.5 pp below baseline with $p < 0.05$ — disqualifies the
  variant.

**Output B — invariance achieved** (the H6 prediction). For each variant,
report end-of-training source-dataset linear-probe accuracy:

- Strict pass: < 50 % (matches H6's bar).
- Soft pass: 50–70 %.
- Fail: > 70 %.

A variant must (i) not lose on HBN 6-task BAC *and* (ii) reach at least the soft
pass on invariance. If no variant passes both, the recipe ships without
the adversary and §11 H6 is downgraded from "prediction" to "open
question".

For A4 specifically, also report the **subject-ID** linear-probe accuracy
on a 5-subject held-out set (random = 20 %). The bar is: < 35 % accuracy
to count as subject-invariance achieved.

## Pre-registered predictions

| Variant | Predicted HBN 6-task BAC | Predicted site-probe acc | Subject-probe acc |
| ------- | ------------------ | ---------------------------- | ------------------ |
| A0 baseline | reference | 70–85 % (the H6 worry) | 60–80 % |
| A1 linear $\lambda=0.05$ | tied | 55–70 % | 55–70 % |
| A2 linear $\lambda=0.20$ | weak loss, ~–0.5 pp | 35–50 % (passes H6) | 50–65 % |
| A3 MLP $\lambda=0.10$ | tied | 35–55 % (probably passes) | 50–65 % |
| A4 dataset+subject MLPs | tied or weak loss | 35–50 % | 30–45 % (best on subject) |

The honest expected outcome: **A3 wins on the site-probe target and
ties HBN 6-task; A4 additionally suppresses subject identity at the cost
of a small HBN 6-task regression**. The decision then depends on whether
the
deployment scenario emphasises clinical multi-rig generalisation
(A3 / A4) or peak within-rig accuracy (A0).

## Implementation pointers

- Gradient-reversal layer: trivial 5-line PyTorch implementation
  ```python
  class GRL(torch.autograd.Function):
      @staticmethod
      def forward(ctx, x, alpha):
          ctx.alpha = alpha
          return x.view_as(x)
      @staticmethod
      def backward(ctx, grad_output):
          return -ctx.alpha * grad_output, None
  def grl(x, alpha=1.0):
      return GRL.apply(x, alpha)
  ```
- Adversarial heads: standard `nn.Linear` (A1, A2) or `nn.Sequential(Linear, GeLU, Linear)`
  (A3, A4) over the encoder's mean-pooled output. Loss is plain
  cross-entropy over the dataset/subject labels.
- Source-dataset metadata: the iid-channel pretraining corpus is HBN-EEG
  collected at `K=4` Child Mind Institute sites (RU, CBIC, CUNY, SI); each
  `(channel, recording, window)` example carries a scalar site ID. Subject
  ID is similarly available. (When the broader multi-corpus mix lands —
  HBN + Sleep-EDF + THINGS-EEG2 + TUEG — the site label is replaced by a
  joint site×corpus label.)
- Reference DANN training loop: standard, see
  [`fungtion/DANN`](https://github.com/fungtion/DANN) or the
  `domainbed` benchmark.
- Loss schedule: the GRL's $\alpha$ is *ramped from 0 to 1 over the first
  10 % of training* per Ganin et al. — without this the adversary
  dominates early gradients and destabilises the encoder.

## Output

`mini_experiments/13_adversarial_dataset_probe/results.md` containing:

1. 5 × 2 (× 3 seed) results table with HBN 6-task BAC, HBN ADHD-binary AUROC, k-NN, site-probe acc, subject-probe acc (and TUAB/TUEV when TUH access lands).
2. Source-dataset probe trajectory per variant (must trend down for the
   adversarial variants; A0 is the control).
3. Subject-ID probe results for A4.
4. Encoder feature health per variant (the variance term should not
   degrade — adversarial training can suppress dimensions that were doing
   real work).
5. The chosen adversarial configuration with decision-rule justification,
   or a documented null result if no variant strict-wins on both axes.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| The adversary destabilises training (loss spikes during GRL ramp-up) | Smooth the ramp from 0 → 1 over the first 10 % of steps; reduce $\lambda_\text{adv}$ if instability persists. |
| The dataset labels are too coarse (4 datasets, easy to predict from rate alone) | Add a finer-grained label: subject ID *within* dataset (≥ 50 unique IDs in the mix). This is what A4 does. |
| The adversary suppresses real signal that happens to differ across sites (e.g. age-related $\alpha$ peak shifts that correlate with site demographics) | Compare variant A0 and the best A* on a *single-site* held-out subset (HBN/RU only) — if A* loses there, the adversary is destroying signal, not just nuisance. |
| The MLP head is too strong and the encoder cannot recover | Cap the head capacity; A3's MLP is intentionally small (hidden $D/2$). |
| Source-probe accuracy of A0 is already < 50 % (no problem to solve) | Then no variant can strict-win on Output B. The conclusion is *adversary is unnecessary* — record this and ship A0. |

## What gets carried forward

The single chosen adversarial configuration, including the GRL ramp
schedule, head architecture, and $\lambda_\text{adv}$. If A0 wins (no
adversary needed), the headline run uses no adversarial head and §11 H6
is downgraded to a passive monitor.
