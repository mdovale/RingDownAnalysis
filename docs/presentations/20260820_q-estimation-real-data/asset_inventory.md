# Asset inventory (Phase 4)

Every canvas is laid out at slide proportions and drawn at `FIGSCALE = 0.72` of that
size, so on-figure text lands close to Beamer body size once the PDF is placed on a frame,
with room left for descriptive bullets.

Generated 2026-08-21 (revised).

## Two-stage pipeline

```
raw phasemeter exports  --> build_cache.py -->  assets/_cache/{edu_cache.npz, edu_cache.json}
assets/_cache           --> make_figures.py -->  assets/*.pdf
```

`build_cache.py` is the only script that touches the multi-GB records. It loads both EDU
ring downs through `mokutools`, runs `RingDownAnalyzer.analyze_array` and
`SegmentedDemodEstimator` over the canonical windows, the window-sweep grid and the
offset-9000 specimen, and writes decimated traces plus per-segment arrays (`.npz`) and all
scalars (`.json`). `make_figures.py` reads only that cache, so a figure iteration is a few
seconds rather than several minutes.

Rebuild:

```bash
make cache     # expensive; needs the raw records
make figures   # cheap; cache only
```

## Data sources

| Source | Used for |
|--------|----------|
| EDU Pre-Vibe and Post-Vibe phasemeter exports (`mokutools`) | all real-data figures |
| `ringdownanalysis.RingDownAnalyzer.analyze_array` | before/after estimator behaviour, adequacy checks, crops |
| `ringdownanalysis.demod.SegmentedDemodEstimator` | reference $Q$, drift, plateau, $Q(A)$ bands |
| `ringdownanalysis.q_envelope.q_envelope_diagnostic` | measured envelope, candidate-agreement mismatch |
| Executed companion notebook (controlled experiments) | injection ladder, $\dft$ scan, crop-cascade $\tau$ values |

The injection ladder and the coherence-budget scan are Monte-Carlo studies on synthetic
records. Their measured values are carried as literal tables in `make_figures.py` with the
provenance in a comment, because regenerating them would re-run the whole study for no change
in the plotted numbers.

## Figures

| File | Slide | Content | Key parameters |
|------|-------|---------|----------------|
| `title-strip.pdf` | 1 | Decorative: measured envelope (blue) against the fitted one (red dashed), first 0.9 h | Pre-Vibe canonical window |
| `symptom-synth-vs-real.pdf` | 2 | Time series $\pm$ envelopes + log amplitude; synthetic vs real | matched SNR synthetic; Pre-Vibe 3 h |
| `overlay-real.pdf` | 3 | Measured envelope, incoherent slope, coherent fit; log-y | Pre-Vibe canonical 3 h |
| `pre-post-traces.pdf` | 5 | Pre/Post time series with coherent and reference $\pm$ envelopes | canonical windows |
| `window-sweep-before.pdf` | 6 | Coherent $Q$ / reference per window, log-y ratio | start 0–100 s $\times$ duration 1–6 h |
| `assumptions-card.pdf` | 4 | Model equation + three assumptions as questions | early card |
| `assumptions-card-late.pdf` | 10 | Same card with measured violations and physics/nuisance tags | matched protocol |
| `drift-pull.pdf` | 7 | Ideal synthetic + EDU $f$ vs $A$ and vs time | 120 s segments |
| `q-vs-amplitude.pdf` | 8 | Synthetic vs EDU log-envelope residual; Pre $Q(A)$ | bands 60–500 cycles |
| `plateau.pdf` | 9 | Synthetic vs EDU amplitude; plateau spectrum | |
| `injection-ladder.pdf` | 11 | Coherent and segmented $Q$ / truth for nine injected cases | literal table |
| `coherence-budget.pdf` | 12 | $Q/Q_{\rm true}$ vs $\dft$ | literal scan |
| `crop-cascade.pdf` | 13 | Retained-extent bars; round-off $\tau$ pair annotated | offset 9000 s |
| `demod-method.pdf` | 14 | Segmentation; floor correction | Pre-Vibe |
| `gates-flow.pdf` | 15 | Adequacy-check chain | diagram |
| `window-sweep-after.pdf` | 16, B4 | Same early-start sweep after the method change | as before |
| `pre-post.pdf` | 17 | Local $Q$ per matched band; pull coefficients | matched protocol |
| `backup-spectra.pdf` | B2 | Decay and plateau spectra | Pre-Vibe |
| `lasso-logo.pdf` | 1 | Institutional mark | |

## Style notes

- No in-figure titles; no gridlines heavier than `#DDDDDD`; legends frameless.
- Log axes carry plain numeric ticks with minor-tick labels suppressed (`_log_axis_plain`).
- Amplitude axes are inverted where amplitude decreases with time, and the direction is
  stated in the axis label rather than as a floating arrow.
- Red marks a wrong or refused result, green a correct or accepted one, grey a neutral
  reference. This mapping is used consistently across every figure and in the slide text.
- Annotations are placed in measured empty regions of each panel; where a panel had no room,
  the statement moved to the frame's bullet instead of shrinking the type.

## Cached quantities used in slide text

| Quantity | Pre-Vibe | Post-Vibe |
|----------|----------|-----------|
| Coherent $Q$ (ungated, canonical window) | 58 141, CI $\pm0.12\,\%$ | 27 082, CI $\pm0.08\,\%$ |
| Segmented reference $Q$ (same window) | 92 584 | 102 972 |
| Envelope $Q$ (same window) | 93 676 | 107 797 |
| Drift over the matched window | $+1.19$ mHz | $+0.22$ mHz |
| Plateau amplitude | 16.3 cycles | 17.1 cycles |
| Local $Q$ across matched bands | 73.7k $\to$ 182.6k | 78.9k $\to$ 190.9k |
| Pull coefficient | $-0.40$ mHz/e-fold | $-0.08$ mHz/e-fold |
| Window-sweep spread, before | $\times50.2$ | $\times1.38$ (centred $0.30\times$) |
| Window-sweep spread, after | $\times1.31$ | $\times1.55$ |
| Frequency shift, Post $-$ Pre | $-77$ ppm at high $A$ | $-140$ ppm at low $A$ |
