# Q-estimation on real resonators — Presentation brief (Phase 1)

> **Superseded numbers.** The evidence map below cites the investigation report. The built deck
> recomputes every real-data value from the shipped pipeline; where they differ, the deck's value
> is authoritative. See `presentation_review.md` § "Numbers as built".

> Math uses `$…$` / `$$…$$` (Markdown + KaTeX).

- **Mode:** create
- **Title (working):** When a tight error bar lies — quality-factor estimation on real ring-down data
- **Subtitle:** Root cause, fix, and validation in `ringdownanalysis`
- **Audience:** instrument / metrology researchers; center of gravity on the estimator
  failure and the fix
- **Duration:** ~20 min (slide budget: ~15 content slides + title + closing; ≈ 75 s/slide)
- **Anchor sources:**
  `docs/investigations/20260818_q_estimation_failure_investigation.md` (evidence chain),
  `notebooks/20260818_EDU_PreVibe_vs_PostVibe_RingDown.ipynb` (measurements, controlled
  experiments, figures `docs/investigations/figures/fig01…fig10`),
  `docs/handoffs/20260818_q-estimation-evolution-implementation.md` (what shipped and the
  two measured deviations), `README.md` §"Which Q should I trust?",
  `ringdownanalysis/demod.py`, `nonlinear.py`, `selection.py`,
  `tests/test_real_data_regression.py`

---

## Audience

Experimentalists who measure mechanical or optical resonator $Q$ from ring-down records,
plus the estimation-minded people who write or maintain their fitting code. Assume:

- Fluent with $Q = \pi f \tau$, exponential decay fits, and least squares; they do **not**
  need a derivation of nonlinear least squares or of a profile likelihood.
- Comfortable with spectra, demodulation, and SNR in the working-practice sense.
- Used to trusting a fit's reported confidence interval. **This is the assumption the talk
  attacks**, so it must be surfaced explicitly rather than assumed shared.
- Care about "what do I do differently on Monday", not about software architecture.

Prior knowledge to bridge, not assume:

- The distinction between **phase-coherent** and **incoherent** amplitude estimation, and
  why it decides whether frequency drift matters at all.
- Why a $\chi^2$ / curvature confidence interval says nothing about **model adequacy**:
  it bounds variance under an assumed model, not bias from the wrong model.
- The **coherence budget** $\delta\!f \cdot \tau$ as the single number that decides whether
  a coherent fit is admissible.

## Goal

Primary: **teach a transferable diagnostic lesson**. Anyone fitting ring-downs on a real
resonator is exposed to the same failure — a confidently wrong $Q$ with a per-mille error
bar — and the talk should leave the audience able to detect it in their own data with two
cheap diagnostics (measured drift, envelope agreement).

Secondary: **inform** — announce the shipped estimator and its validation, so the deck also
functions as the public record of what changed in `ringdownanalysis` and why.

Tertiary: **solicit** — invite others to check whether their records share the drift /
plateau phenomenology, since the survey beyond the EDU pair is not done.

This is a measurement-and-methods talk, not a software tutorial and not a resonator-physics
talk.

## Key messages (5)

1. **A tight confidence interval is not evidence of correctness.** The library's
   *recommended* estimator returned $Q$ biased $1.5\text{--}4\times$ on real records while
   reporting `status="valid"` with intervals of relative width $10^{-4}$. The validity test
   checked interval closure, i.e. that the fit converged — never that the model was right.
2. **The primary cause is phase decoherence, and its tolerance is brutally tight.** The
   resonance frequency moves by $0.3\text{--}1.1$ mHz during the decay (an
   amplitude-dependent anharmonic pull). Every coherent estimator needs
   $\delta\!f\cdot\tau \lesssim 0.01$, here $\approx 3\ \mu$Hz — violated by two to three
   orders of magnitude. Beyond that the least-squares objective has many near-degenerate
   minima and the returned $Q$ is a **lottery over inputs that should be irrelevant**:
   $0.51\times$, $1.77\times$, $3.80\times$ of truth across two seeds and two
   initializations, each `valid`.
3. **Real resonators break three model assumptions at once, and two of them are physics.**
   Constant $f$ (violated: pull), single exponential (violated: local $Q$ rises
   $6.5\times10^{4} \to 1.8\times10^{5}$ with falling amplitude), decay to zero (violated:
   an ambient-driven plateau at $\approx 17$ cycles that is *coherent signal*, not noise).
   Amplitude-dependent damping and the pull coefficient should be **reported as
   measurements**, not averaged away into one number.
4. **Diagnose by injection, not by argument.** Start from a synthetic record the estimator
   handles exactly at the measured SNR, then inject one *measured* pathology at a time.
   Injecting only the measured $f(t)$ reproduced the entire field failure class; injecting
   noise at the measured level reproduced nothing. That experiment is what turned a list of
   suspicions into a ranked, falsifiable cause.
5. **The fix is incoherent segmented demodulation, gated by measured diagnostics.** Segment
   the record, fit a tone per segment (coherent *within* a segment, incoherent across),
   model the plateau as an explicit amplitude floor, fit the floor-corrected log envelope
   robustly, and report $Q$ versus amplitude with bootstrap uncertainties. Window-to-window
   spread on the same record fell from $12\times$ to $1.3\times$, and no coherent estimator
   can now report `valid` through a drift or envelope-mismatch violation.

Payoff to state out loud (the reason any of this mattered): the EDU vibration-qualification
verdict **flipped**. The single-number coherent estimates said $-30\ \%$ $Q$ (damage);
amplitude-matched demodulation says $Q$ is equal or **higher** after vibration
(whole-decay $+13\ \%$), with the real change being a $-66$ to $-157$ ppm frequency shift
and a $\approx 5\times$ weaker frequency-amplitude pull.

## Evidence map (source → slide idea)

| Source | Evidence | Slide idea |
|--------|----------|------------|
| Investigation §2, prior handoffs `20260513_*` | Estimators excellent on synthetics, visibly wrong on real data; three rounds of gating did not fix it | 2. The symptom: it works on synthetics and fails in the lab |
| `fig02_overlay_canonical.png`, investigation §4 table | Overlay: red `Q_profile` envelope decays visibly faster than the data; only the green envelope diagnostic tracks it | 3. What "confidently wrong" looks like |
| Investigation §4 table | `Q_profile` = 58 141 `valid` CI (58 103, 58 245) vs reference $8.9\times10^{4}$; Post 25 078 vs $1.0\times10^{5}$ | 4. The numbers, and the meaningless interval |
| `fig10_window_sensitivity.png`, notebook §5.4 | Pre-Vibe `Q_profile` spans $6.5\times10^{3}$–$7.5\times10^{4}$ (12×) over start 0/1800 s × duration 1–6 h; round-off-level input change moved one window from 0.65× to 0.45× | 5. Not a bias — a lottery (window sweep + irreproducibility) |
| `fig03_demod_amplitude_frequency.png`, `fig05` right panel, investigation §5 | $f$ rises $+1.09$ mHz (Pre) / $+0.29$ mHz (Post) monotonically with falling amplitude; linear in $\log A$ | 7. Violation 1: the frequency moves |
| `fig05` left/middle panels, notebook §3.2 band table | Log-envelope residual curvature; local $Q$ $6.5\times10^{4}\to1.8\times10^{5}$, local $\tau$ 2696 → 7325 s (Pre) | 8. Violation 2: not a single exponential |
| `fig01_overview.png`, `fig04_spectra.png`, investigation §5, §13 | Plateau at $\approx 16.5$–17.6 cycles with nulls and revivals; tone still strong in plateau spectra; broadband noise only 1.3–1.9 cycles RMS (SNR ≈ 400) | 9. Violation 3: it never decays to zero — and that is signal |
| Investigation §9 table (E1–E5) | E1 exact at matched SNR; E2c (measured $f(t)$ only) → 0.51×/1.77×/1.77×/3.80×, all `valid`; E3 plateau biases envelope only | 10. Diagnose by injection (the decisive experiment) |
| `fig08_df_tau_scan.png`, investigation §9 scan | $Q/Q_{\rm true}$ = 1.000, 0.998, 0.979, 0.838, 0.395, 0.142 for $\delta\!f\cdot\tau$ = 0.0037 … 1.11; all `valid` | 11. The coherence budget $\delta\!f\cdot\tau \lesssim 0.01$ |
| `fig07_specimen_offset9000.png`, investigation §4, §8-D, §12 | Crop cascade: 12 600 s window → 107 s (0.8 %); two round-off-level variants give $\tau_{\rm est}$ = 35.6 s vs 1051 s | 12. The amplifier: a collapsed $\tau$ poisons the crop |
| `ringdownanalysis/demod.py`, investigation §6, §14, §19-P1 | Segment tone fit, plateau floor, contiguous $A > 3A_{\rm floor}$ region, floor-corrected robust log-linear fit; $< 2\ \%$ on all controlled experiments | 13. The fix: segmented demodulation |
| `analyzer.py` gates, `q_profile.py`, `selection.py`, README §"Which Q should I trust?" | `coherence_ratio` drift gate, envelope-mismatch demotion, `Q_selected` regime classifier, honest overlay annotation, bootstrap CIs | 14. Gates: make the estimator refuse to answer |
| `tests/test_real_data_regression.py`, handoff §Outcome | Window spread 12 → 1.31 (bound 1.5); shipped recipe pinned ±3 %; Theil-Sen vs LS gap 5–10 % from genuine curvature | 15. Validation, and two honest deviations |
| `fig06_pre_post_comparison.png`, notebook §4, investigation §1 | Post-Vibe $Q$ equal or higher ($+13\ \%$ whole-decay, $+4$ to $+51\ \%$ band-resolved); $-66$ to $-157$ ppm frequency shift; pull $-0.42 \to -0.09$ mHz per amplitude e-fold | 16. The payoff: the qualification verdict flips |
| Investigation §17, §18, handoff §Outcome | Physical origin of pull/nonlinear damping out of scope; plateau source not formally identified; SN1/SN2 records not surveyed | 17. Takeaways + what we still do not know |

## Anticipated Q&A (one-slide answers, mostly backup)

- **Was it not just low SNR?** No. SNR at release $\approx 400$; the matched-SNR ideal
  synthetic (E1) is exact for every estimator. Noise is exonerated by construction.
- **Multiple modes / beating?** Ruled out: one dominant peak at 7.669–7.672 Hz, nearest
  features $\sim 10^{5}\times$ weaker and consistent with leakage sidebands of the drifting
  tone (`fig04`). Backup slide.
- **Could a better constant-frequency estimate fix it?** No, and this is the crux: the true
  frequency moves by far more than the tolerance *within* the window, so **no** constant $f$
  keeps the model coherent. Frequency estimation itself is healthy (DFT, NLS, and the
  segment-demodulated mean agree sub-mHz).
- **Why not fit a polynomial or spline phase and stay coherent?** Possible in principle, but
  it buys nothing over demodulation at SNR 400 and adds optimizer fragility. Prony /
  matrix-pencil / linewidth methods inherit the same constant-frequency pole model.
- **Which $Q$ do you quote if $Q$ depends on amplitude?** Either a stated amplitude band, or
  the fitted law $1/\tau(A) = 1/\tau_0 + \beta A$ with the zero-amplitude $Q_0$. Refuse the
  single unqualified number.
- **Why is window stability 1.3 and not the 1.2 you targeted?** Because re-estimating per
  window exposes the *physical* $Q(A)$: short early windows genuinely sample the low-$Q$
  high-amplitude decay. Every window stays inside the measured band
  $6.5\times10^{4}$–$1.8\times10^{5}$. Backup slide.
- **Why Theil-Sen instead of least squares on the log envelope?** Robustness against
  plateau revivals. The cost is 5–10 % on genuinely curved decays; both are pinned in the
  regression tests so the gap is visible rather than hidden.
- **Is the plateau not just noise you could filter?** No — it is narrowband, centered on the
  analysis frequency, and $\approx 10\times$ the broadband noise. Filtering cannot separate
  it from the signal; only an explicit floor model handles it.
- **What about the profile likelihood — is it wrong?** It is correct on its own model, and
  still the right choice for short coherent records. It was wrong as a *default* and wrong
  to report `valid` without a model-adequacy test.

## Out of scope (must NOT appear)

- The physical origin of the frequency pull and nonlinear damping (resonator intrinsic vs
  mount vs readout). Only the signal-level consequences are established.
- Formal identification of the ambient excitation source behind the plateau.
- Generalization beyond the EDU pair: the SN1/SN2 records were not surveyed, so no claim
  that these defaults suit all resonators.
- An analytical prediction of coherent-fit bias direction for arbitrary drift trajectories.
  Bounded empirically only.
- Software walkthroughs: module layout, class diagrams, line-by-line code, or API signature
  tours as main content. One or two short public-API lines on the fix slide, no more.
- Any internal development-workflow references in slide-visible text or exported speaker
  notes.
