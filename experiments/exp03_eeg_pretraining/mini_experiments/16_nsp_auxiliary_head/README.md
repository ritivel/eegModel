# exp03 / mini-experiment 16 — Neurodynamics Statistics Prediction auxiliary head

> **Status:** planned
>
> **Cross-reference:**
> [DeeperBrain (Wang et al., arXiv 2601.06134)](https://arxiv.org/abs/2601.06134) §III-G + Fig. 3 + Table IV;
> [`brain/cortico-ssl-hypothesis.typ` §6 (Axis D), §7](../../../../../brain/cortico-ssl-hypothesis.typ);
> sister experiments [04 SSL framework](../04_ssl_framework_ablation/),
> [07 phase handling](../07_phase_handling/).
>
> **Compute budget:** 12 H100-hours (4 variants × 2 control columns × 3 seeds =
> 24 cells × 30 min average).
>
> **Gates:** none downstream — exp16 is a hardening experiment that reports
> a single configuration choice for the headline run.

## Question

Does adding a **Neurodynamics Statistics Prediction (NSP)** auxiliary head —
a small MLP that, from the encoder representation of a partly-masked input,
directly predicts macroscopic dynamical order parameters of the *full*
window (relative spectral power per band, cross-frequency coupling, sample
entropy) — improve the encoder's frozen-probing performance, and how does it
interact with the existing reconstruction-side losses (Barron, IS, MR-STFT,
bispectral) that already operate on those quantities indirectly?

## Why it matters

DeeperBrain (arXiv 2601.06134, Dec 2025) demonstrates that a "predict the
statistic" auxiliary head produces dramatically better frozen-probing
representations than a masked-reconstruction-only objective on the same
backbone, despite using a much smaller pretraining corpus (17 200 hours
vs LaBraM / CBraMod / REVE's 60 000+ hours). Their Table IV is striking:
on FACED 9-class emotion under frozen probing, DeeperBrain reaches 50.96
% balanced accuracy while the next-best (REVE) reaches 37.76 % and CBraMod
reaches 25.84 %. On BCIC-IV-2a 4-class motor imagery, the gap is
51.01 % (DeeperBrain) vs 42.73 % (REVE) vs 33.06 % (CBraMod). Their
ablation in Fig. 3 specifically attributes most of this gap to the NSP
head — without NSP, performance drops by 4–7 pp on frozen probing.

The mechanism is *structural rather than statistical*. Existing
reconstruction-based losses (our exp06 / exp07 / hypothesis §7) compute
a statistic on the predicted waveform and compare to the same statistic
on the ground-truth waveform — the encoder is constrained only
*indirectly* (through the requirement that its decoded waveform produce
the right statistic). The NSP head, by contrast, reads the statistic
*directly* from the encoder representation through a small linear or MLP
classifier:

```
encoder representation H ──► linear head ──► predicted 19-dim statistic vector
                            (no decoder)              ▲
                                                      │
                                target = same statistic computed on full ground-truth signal
```

This forces the encoder representation itself to *encode* the statistic
— a much stronger constraint than asking the decoded signal to *exhibit*
the statistic. Frozen probing then works because the linear probe at
fine-tune time can read off the same kind of statistic the encoder has
already been compelled to make linearly accessible.

The honest empirical question is whether this still works once we have
already paid for the four spectral / phase loss heads in our hypothesis
recipe. The hypothesis already enforces alignment with spectral power
distribution (via Itakura–Saito), with phase relationships (via the
unit-circle MR-STFT loss), and with cross-frequency coupling (via the
bispectral loss on $theta times gamma$). NSP could be:

- *Fully redundant*: the existing losses already linearise the same
  statistics, NSP adds nothing.
- *Partially complementary*: NSP picks up an order parameter the existing
  losses miss (sample entropy, broadband coupling).
- *Critically additive*: the framing matters — predicting the statistic
  directly produces qualitatively different representations from
  matching the statistic on the decoded signal.

Only the third would justify adding a sixth loss head; the experiment
discriminates the three.

## Variants

All variants sit on top of the W4 (or best-of-Phase-A) configuration from
exp12 — i.e. with the front-end, backbone, bottleneck, masking, target,
multi-condition input, reconstruction loss, phase loss, and quick-wins
already settled. The only thing that varies is the NSP head and the
loss weight $lambda_"NSP"$.

| Code | Variant | NSP head | $lambda_"NSP"$ | Statistics predicted |
| ---- | ------- | -------- | --------------- | --------------------- |
| N0 | No NSP head (the W4 baseline) | none | 0 | none |
| N1 | NSP head on per-channel statistics, weak weight | linear | 0.3 | 5 spectral + 3 CFC + 1 entropy = 9D per (channel, second) |
| N2 | NSP head on per-channel statistics, full weight | linear | 1.0 | 9D as above (matches DeeperBrain's reported $lambda_"NSP" = 1.0$) |
| N3 | N2 + an MLP head with hidden $D/2$ | 2-layer MLP | 1.0 | 9D as above |
| N4 | N2 with PLV pseudo-target via random-channel pairing | linear | 1.0 | 9D + 2D PLV = 11D (see §implementation) |

Note: the original DeeperBrain NSP target is 19D and includes a 10D
phase-locking-value (PLV) component computed across pairs of channels.
PLV is genuinely multi-channel and does not directly fit the iid-channel
framing. N4 includes a workaround in which, at training time only, each
window is paired with a randomly-sampled second channel from the
*same recording* (which exists in the original multi-channel data
before iid expansion) to compute a 2D PLV summary. This breaks strict
iid expansion but preserves the spirit of single-channel-per-token
pretraining.

If N4 strict-wins over N2, the iid-channel decision is partially
re-opened and the pretraining-experiment folder needs to know.

## Controls (the §3 matrix)

|                                | EEG signal | matched-noise twin |
| ------------------------------ | ---------- | ------------------ |
| N0 baseline (no NSP)           | ✓          | ✓                  |
| N1 linear head, $lambda=0.3$   | ✓          | ✓                  |
| N2 linear head, $lambda=1.0$   | ✓          | ✓                  |
| N3 MLP head, $lambda=1.0$      | ✓          | ✓                  |
| N4 with PLV pseudo-target      | ✓          | ✓                  |

The matched-noise twin is informative as a sanity. Spectral power and
sample entropy of pure Gaussian noise are well-defined and meaningful
quantities (a noise window has a flat-ish spectrum and a high entropy
score). The encoder trained with NSP on a noise twin should still
predict those statistics accurately on its own training distribution
without that translating to *EEG-task* improvements. If TUEV / TUAB
metrics rise on the noise twin under NSP, the gain is being attributed
to representations that are sensitive to broadband statistics rather
than to neural structure, and we discount it.

## Held constant

Everything from the exp02–exp15 winners. Specifically:

- Frontend, backbone, bottleneck, multi-rate strategy, reconstruction
  loss, phase loss, target signal, multi-condition input, mask,
  quick-wins stack: all set to their respective experiment winners.
- Window length: exp14 winner.
- Adversarial head: exp13 winner.
- Loss weights and curriculum: exp15 winner — we add $lambda_"NSP"$ as
  a new weight on top.
- Pretraining duration: 8 epochs over the 100 h TUEG subset.
- Optimiser: AdamW, LR carried forward.

## Decision rule

Two parallel decisions:

**Output A — frozen-probing performance** (the primary metric, given the
DeeperBrain motivation):

- *Strict win*: ≥ 1 pp TUEV BAC under frozen-probing (linear probe on
  frozen encoder pool), non-overlapping CIs, noise-twin flat.
- *Weak win*: ≥ 0.5 pp with paired permutation $p < 0.05$.
- *Loss*: ≥ 0.5 pp below baseline with $p < 0.05$ — disqualifies the
  variant.

**Output B — fine-tune performance** (the safety check):

- *Pass*: TUEV BAC under end-to-end fine-tuning within 0.5 pp of the
  baseline. The intent is that NSP not destroy fine-tune quality even
  if it does not help frozen-probing.
- *Fail*: ≥ 0.5 pp below baseline under fine-tuning. NSP is hurting
  rather than helping; reject the variant.

A variant is adopted if it passes Output B *and* either strict-wins or
weak-wins on Output A. If no variant strict-wins on Output A, the
recipe ships without NSP and the existing exp06 / exp07 spectral / phase
losses carry the load.

A diagnostic to log throughout: per-batch correlation between predicted
and target statistic, per dimension. DeeperBrain's Fig. 6 reports
$r approx 0.82$ for alpha power, $r approx 0.75$ for sample entropy,
$r approx 0.6$ for PLV, and $r approx 0.01$ for CFC under high masking.
We expect a similar pattern.

## Pre-registered predictions

| Variant | Frozen-probe TUEV BAC | Fine-tune TUEV BAC | $r$ on alpha power | $r$ on entropy | $r$ on CFC |
| ------- | ---------------------- | ------------------- | ------------------- | --------------- | ----------- |
| N0 baseline | reference | reference | n/a | n/a | n/a |
| N1 linear $lambda=0.3$ | weak win, ~+0.5 pp | tied | $approx 0.6$ | $approx 0.6$ | $approx 0.05$ |
| N2 linear $lambda=1.0$ | strict win, ~+1–2 pp | tied | $approx 0.8$ | $approx 0.7$ | $approx 0.05$ |
| N3 MLP $lambda=1.0$ | tied with N2 | tied or weak loss | $approx 0.8$ | $approx 0.75$ | $approx 0.05$ |
| N4 with PLV pseudo-target | strict win, ~+2–3 pp on emotion / motor imagery | tied | $approx 0.8$ | $approx 0.75$ | $approx 0.05$ |

The honest expected outcome:
**N2 strict-wins on frozen-probing, N3 ties (the linear head is enough
because the encoder representation is already rich), N4 wins specifically
on the tasks DeeperBrain reported the largest gaps on (emotion, motor
imagery, vigilance) but at the cost of breaking strict iid-channel
pretraining**. The decision becomes whether the iid-channel constraint
is worth more than the 1–2 pp incremental gain from PLV.

## Implementation pointers

The four NSP statistics, with PyTorch-level definitions, in the
DeeperBrain numbering:

- *Relative spectral power (5D)*: take the FFT of the patch, integrate
  squared magnitude over the canonical bands $delta$ (1–4 Hz), $theta$
  (4–8), $alpha$ (8–12), $beta$ (12–30), $gamma$ (30–80), normalise to
  a probability simplex. 5 numbers per (channel, second) patch.

- *Cross-frequency coupling (3D, per-channel)*: for each canonical pair
  in {(theta-low, gamma), (theta-high, gamma), (alpha, gamma)},
  compute the modulation strength
  $"CFC" = (1/P) sum_t cos(phi_"low" (t)) dot A_"high" (t)$
  via the Hilbert transform of the bandpassed signal. 3 numbers.

- *Sample entropy (1D)*: with embedding dim $m=2$ and tolerance
  $r = 0.2 dot "std"(x)$, $"SampEn" = -log(C^(m+1)(r) / C^m (r))$.
  1 number. Use [neurokit2](https://github.com/neuropsychology/NeuroKit)
  or a hand-port; the Hilbert and FFT pieces are standard PyTorch.

- *PLV summary (2D, multi-channel only)*: for the per-band analytic
  phase $phi_c (t)$, compute
  $"PLV"_(i j) = |(1/P) sum_t e^(j (phi_i (t) - phi_j (t)))|$
  for the iid-channel-paired second channel (N4 only). Take a single
  per-band scalar (averaged over bands) plus its log-variance — 2
  numbers in N4.

DeeperBrain's reference implementation uses Smooth L1 loss with $beta=1$
on the NSP target. Z-score the target across the corpus once
(per-statistic) before computing the loss; this matters because raw
spectral power and entropy have very different magnitudes. Cache the
NSP target alongside the cleaned target signal from exp08; computing
the FFT and Hilbert per training step is unnecessary overhead.

NSP head architecture (from DeeperBrain §III-G):
```python
class NSPHead(nn.Module):
    def __init__(self, d_model, n_stat=9, mlp=False):
        super().__init__()
        self.proj = (
            nn.Sequential(nn.Linear(d_model, d_model // 2),
                          nn.GELU(),
                          nn.Linear(d_model // 2, n_stat))
            if mlp else nn.Linear(d_model, n_stat)
        )
    def forward(self, h):
        return self.proj(h)  # (B, T', n_stat)

# loss against precomputed target Y_NS:
loss_nsp = F.smooth_l1_loss(nsp_head(H), Y_NS, beta=1.0)
```

## Output

`mini_experiments/16_nsp_auxiliary_head/results.md` containing:

1. 5 × 2 (× 3 seed) results table — TUEV BAC, TUAB AUROC, k-NN — under
   *both* frozen-probing and end-to-end fine-tuning.
2. NSP target reconstruction quality per statistic (correlations as in
   DeeperBrain Fig. 6) — alpha / theta / beta / gamma power, sample
   entropy, three CFC components, PLV (N4 only).
3. Per-task frozen-probing comparison: emotion (FACED proxy), motor
   imagery (BCIC-IV-2a proxy if licensable), seizure detection
   (CHB-MIT proxy). The DeeperBrain pattern says NSP helps state-modelling
   tasks more than event-detection tasks; we want to verify.
4. Pre-fine-tune-vs-post-fine-tune gap. A small gap means the
   representation was already useful, which is the universality argument.
   A large gap means the encoder needed task-specific re-optimisation,
   which is the failure mode DeeperBrain calls out.
5. The chosen NSP configuration (or a documented null result if no
   variant strict-wins on frozen-probing).

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| The 9D NSP target is too coarse to differentiate variants in our 100 h corpus | Re-run on the 500 h corpus before declaring null; DeeperBrain's pattern emerges most clearly at scale. |
| The Smooth L1 NSP loss is dominated by the largest-magnitude statistic (spectral power) | Z-score each NSP dimension globally before computing the loss; report per-dimension loss separately. |
| The NSP target is computed with an artifact-contaminated signal (eye blinks bias the spectrum) | Use the exp08-cleaned signal as the NSP target source, not the raw input. |
| The encoder reaches the NSP target via a shortcut (predicting the per-recording mean) | The "predict source dataset" probe from exp13 should already catch this — re-use that diagnostic here. |
| N3 (MLP head) overfits the NSP target and loses generality | The hidden $D/2$ choice is intentionally narrow; if N3 strict-loses to N2, drop the MLP variant. |
| N4 (PLV pseudo-target) breaks the strict iid-channel guarantee | If N4 wins, document explicitly and decide jointly with the channel-aggregation work in `brain/experiments/pretraining-experiment/`. |
| The NSP head and the existing exp06/exp07 spectral / phase losses interact destructively | Run a 2 × 2 lesion: with/without NSP × with/without exp06+07 spectral heads. Reports which subsets compose. |

## What gets carried forward

The single chosen NSP configuration (N0 if none strict-wins, otherwise
the best of N1–N4). The 9-dim or 11-dim NSP target schema, the head
architecture, and $lambda_"NSP"$ are all frozen for the headline run.
The per-statistic correlation diagnostic is added to the §11 falsifiable
predictions of the cortico-ssl-hypothesis as a new entry, alongside the
existing H4 (phase) and H7 (bispectral) predictions.
