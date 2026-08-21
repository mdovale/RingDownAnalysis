# Q-estimation on real resonators — Slide specification (Phase 3)

> **Superseded numbers.** Slides 4, 5, 15 and 16 quote the investigation report. The built deck
> recomputes every real-data value from the shipped pipeline, so the Post-Vibe coherent $Q$ is
> 27 082 (not 25 078), the before-fix window spread is $\times50$ on the raw coherent estimate
> (not $12\times$ on the gated one), and the Pre/Post claim is band-resolved. See
> `presentation_review.md` § "Numbers as built".

> Math uses `$…$` / `$$…$$`. Every content slide carries a visual; on-slide text
> budget is **≤ 40 visible words** excluding the frame title. Everything else lives in
> `\note{}`.
>
> Asset paths are relative to `assets/`. Vector PDFs are produced by
> `assets/make_figures.py`, which reads only `assets/_cache/` (built once from the raw
> records by `assets/build_cache.py`).

Deck totals: 17 frames in the main line (title + 16 content) + 6 backup frames.

---

## 1. Title — When a tight error bar lies

- **Key message:** A confident $Q$ can be four times wrong.
- **Visual type:** title block (Beamer) + decorative strip
- **Assets:** `title-strip.pdf`, `lasso-logo.pdf`
- **Word budget:** title, subtitle, author, institute, date only
- **Notes stub:** Ring-down $Q$ from a real mechanical resonator. The recommended estimator
  in our own library returned values biased by up to a factor of four while reporting a
  confidence interval of relative width $10^{-4}$ and a status of `valid`. This talk is the
  root-cause analysis, the fix, and the general lesson.

## 2. The symptom — flawless on synthetics, wrong in the lab

- **Key message:** Passing every synthetic test predicted nothing about real records.
- **Visual type:** map/plot, two panels
- **Assets:** `symptom-synth-vs-real.pdf` — left: matched-SNR synthetic record, fitted
  envelope lying on the data; right: the real Pre-Vibe record, fitted envelope peeling away.
  Identical axes styling so the contrast is the only difference.
- **Word budget:** ≤ 24. Two bullets: "Synthetic: exact, every estimator" /
  "Real record: same code, visibly wrong".
- **Notes stub:** Concepts: matched SNR (the synthetic is built at the measured
  SNR $\approx 400$, so it is not an easy case); Monte-Carlo validation. Three prior rounds
  of validity hardening plus a profile-likelihood estimator had already shipped, and the
  overlay still exposed wrong-but-`valid` values. State the honest position: at this point
  we had a library that was excellent on its own model and untrustworthy on the instrument
  it was written for.

## 3. What "confidently wrong" looks like

- **Key message:** The fitted envelope decays visibly faster than the data.
- **Visual type:** map/plot, single large panel
- **Assets:** `overlay-real.pdf` — Pre-Vibe canonical 3 h window: grey min/max data band,
  measured peak-to-peak envelope points, envelope-slope fit tracking the data, and the
  profile-$Q$ envelope diverging; annotated with the slope-mismatch factor.
- **Word budget:** ≤ 20. One line naming each of the three curves.
- **Notes stub:** Concepts: robust windowed peak-to-peak amplitude
  $\tfrac12(P_{95}-P_{5})$ over 3-cycle windows (phase-insensitive, spike-robust; reads
  $0.987A$ for a pure sinusoid); the candidate-agreement diagnostic. The library computed
  the mismatch — `candidate_agrees=False`, slope mismatch $1.4$–$4\times$ — and threw it
  away: it never reached any validity field, and the overlay still labelled the mismatching
  value "best". The information needed to catch the failure was already being computed.

## 4. The numbers, and the interval that meant nothing

- **Key message:** $1.5\times$ and $4.0\times$ low, both `valid`, intervals of width $10^{-4}$.
- **Visual type:** table (native Beamer `booktabs`)
- **Assets:** none (table); numbers from investigation §4
- **Word budget:** ≤ 34 including table cells; one closing line:
  "`valid` tested interval closure, never model adequacy".
- **Notes stub:** Concepts: profile likelihood (profile $\log\tau$ by variable projection,
  95 % interval where $\nu\log(\mathrm{RSS}/\mathrm{RSS}_{\min}) \le \chi^2_{0.95,1}$);
  what a $\chi^2$ curvature interval does and does not bound. With $N\sim10^{6}$ samples the
  interval shrinks to nothing whether or not the model is right, because it measures
  curvature of the residual surface under an assumed white-noise model — it bounds variance,
  never bias from misspecification. The `valid` flag tested that the interval closed on both
  sides, i.e. that the optimizer found an interior optimum. That is a convergence test
  wearing the clothes of a validity test.

## 5. Not a bias — a lottery

- **Key message:** Ordinary window choices scatter $Q$ by $12\times$.
- **Visual type:** map/plot, two panels (Pre, Post)
- **Assets:** `window-sweep-before.pdf` — ungated profile $Q$ per window (start 0/1800 s ×
  duration 1–6 h) against the drift-immune reference and the physical $Q(A)$ band; the
  $12\times$ span annotated.
- **Word budget:** ≤ 28. Two bullets: the $12\times$ span; the round-off irreproducibility.
- **Notes stub:** Concepts: near-degenerate least-squares minima; ULP-level input
  differences. The same Pre-Vibe window returned $0.65\times$ truth in one session and
  $0.45\times$ in another differing only at round-off; the offset-9000 specimen gave
  $\tau_{\rm est}$ = 35.6 s vs 1051 s for two selections of the same window with the same
  $N$ and span. Why this matters more than bias: a bias is a calibration you can measure and
  remove, and it would have been caught by any repeat measurement. A lottery passes every
  repeatability check you would think to run, because rerunning the same input gives the same
  wrong answer.

## 6. So look at the model, not the noise

- **Key message:** Three assumptions, all violated; SNR is 400.
- **Visual type:** diagram (equation + annotation card)
- **Assets:** `assumptions-card.pdf` — the fitted model
  $A_0 e^{-t/\tau}\cos(2\pi f t + \phi) + c$ with braces to its three assumptions
  (constant $f$; single $\tau$; decays to zero), each carrying the measured violation.
- **Word budget:** ≤ 30 (mostly inside the figure; keep frame text to one line).
- **Notes stub:** The pivot of the talk. Every estimator in the pipeline — the $\tau$ seed,
  the full-record NLS, the DFT plus fixed-frequency NLS, and the profile likelihood — fits
  this one model and is phase-coherent: one constant $f$ and $\phi$ across a multi-hour
  window. The envelope diagnostic is the only incoherent member and it has no plateau model.
  That split is the crux. Say explicitly that broadband noise is $1.3$–$1.9$ cycles RMS
  against a $600$-cycle release amplitude, so low SNR is not available as an explanation.

## 7. Violation 1 — the frequency moves while you fit

- **Key message:** $f$ climbs $1.1$ mHz as the amplitude falls: an anharmonic pull.
- **Visual type:** map/plot with inset
- **Assets:** `drift-pull.pdf` — per-segment $f$ vs amplitude (log $x$) for both records with
  the fitted pull line; inset $f$ vs time.
- **Word budget:** ≤ 26. Drift magnitudes; "linear in $\log A$"; "amplitude-dependent, not
  thermal".
- **Notes stub:** Concepts: segmented demodulation as an estimator-independent diagnostic
  (120 s segments, tone fit per segment, so a drifting frequency is measured rather than
  assumed away); anharmonic (Duffing-like) frequency pull, where the resonance frequency
  depends on amplitude. Measured: $+1.09$ mHz (Pre), $+0.29$ mHz (Post), monotonic with
  falling amplitude and approximately linear in $\log A$. Scale intuition to say out loud:
  a 1 mHz frequency error accumulates a full cycle of phase slip in about 1000 s, and these
  windows are 3–6 hours long.

## 8. Violation 2 — not a single exponential

- **Key message:** Local $Q$ rises $6.5\times10^{4} \to 1.8\times10^{5}$ as amplitude falls.
- **Visual type:** map/plot, two panels
- **Assets:** `q-vs-amplitude.pdf` — left: log-envelope residual of a single-exponential fit
  showing systematic curvature far above segment scatter; right: local $Q$ per amplitude band
  for both records.
- **Word budget:** ≤ 26. The band range; "no single true $Q$ exists"; "window choice picks
  an average".
- **Notes stub:** Concepts: amplitude-dependent damping; local $\tau$ (2696 → 7325 s Pre,
  3439 → 7628 s Post). Any single-$\tau$ estimator returns an average weighted by its window
  *and* by its implicit amplitude weighting — NLS weights the early high-amplitude data
  heavily, a log-envelope fit weights all used windows equally, so the two legitimately
  disagree with nothing broken. This reframes the most common complaint about the tool
  ("changing `tau_init` or the duration changes my answer") as a correct response to real
  physics. The lowest band carries the largest floor-correction uncertainty; say so.

## 9. Violation 3 — it never decays to zero, and that is signal

- **Key message:** An ambient-driven plateau at $\approx 17$ cycles, not a noise floor.
- **Visual type:** map/plot (log $y$) with inset spectrum
- **Assets:** `plateau.pdf` — full-record segment amplitude on a log axis: decay into a
  plateau with nulls and revivals; inset amplitude spectrum of a plateau-only segment with
  the tone still dominant.
- **Word budget:** ≤ 28. Plateau level; "$\approx 10\times$ the broadband noise";
  "narrowband, on the analysis frequency".
- **Notes stub:** Concepts: driven equilibrium amplitude; narrowband vs broadband residual.
  Both records settle at $16.5$–$17.6$ cycles with slow wandering, deep nulls, and revivals
  (Post-Vibe up to $\approx 50$ cycles near $+48\,000$ s); the tone remains at high power in
  plateau-only spectra, so the resonator is being continuously re-excited by ambient drive.
  Consequences: no filter can separate it from the signal, because it *is* the signal at the
  analysis frequency; the envelope fit's 5 %-of-maximum amplitude floor lets it into the fit
  for long windows and flattens the slope; and treating it as white noise is a second reason
  the profile interval is meaningless. An explicit amplitude floor handles it; nothing else
  does.

## 10. The decisive experiment — inject one measured pathology at a time

- **Key message:** Injecting only the measured $f(t)$ reproduced the whole failure.
- **Visual type:** map/plot (grouped ratio bars)
- **Assets:** `injection-ladder.pdf` — for each case (ideal / amplitude-proportional pull /
  measured $f(t)$ × 2 seeds × 2 inits / plateau / amplitude-dependent damping), the coherent
  $Q$ and the incoherent reference as ratios to truth, with a line at $1.0$ and every bar
  labelled `valid`.
- **Word budget:** ≤ 26. "Start from a synthetic that is exact"; the $0.51$–$3.80\times$
  span; "noise injected: no failure".
- **Notes stub:** Concepts: controlled reproduction as the standard of proof for a root
  cause. Base record matched to Pre-Vibe: $f_s = 149.012$ Hz, $f_0 = 7.6699$ Hz,
  $\tau = 3700$ s ($Q_{\rm true} = 8.92\times10^{4}$), $A_0 = 600$, white
  $\sigma = 1.5$ cycles, 3 h. E1 (nothing injected) is exact for every estimator, which
  exonerates noise by construction rather than by argument. E2c injects *only* the measured
  Pre-Vibe frequency trajectory and returns $0.51\times$, $1.77\times$, $1.77\times$,
  $3.80\times$ across two seeds and two initializations — every one `valid` with a per-mille
  interval, and note that `tau_init` mattered more than the noise seed. E3 shows the plateau
  leaves coherent fits alone (random phase averages out) and biases the envelope high. E5
  shows every estimator returning a different legitimate window average. This is the slide
  that turns four suspicions into one ranked cause.

## 11. The coherence budget — $\delta\!f\cdot\tau \lesssim 0.01$

- **Key message:** You need the frequency stable to $3\ \mu$Hz; it moves by $1000$.
- **Visual type:** map/plot
- **Assets:** `coherence-budget.pdf` — measured $Q/Q_{\rm true}$ vs $\delta\!f\cdot\tau$
  with the $\pm 2\ \%$ tolerance band, the admissibility edge at $0.01$, and this
  resonator's operating point marked far to the right.
- **Word budget:** ≤ 24. The dimensionless number; the tolerance; the measured value.
- **Notes stub:** Concepts: the dimensionless coherence budget $\delta\!f\cdot\tau$, the
  product of the frequency error and the decay time — how much phase the model loses over
  the interval that carries the information. Measured degradation on the ideal record:
  ratio $1.000$, $0.998$, $0.979$, $0.838$, $0.395$, $0.142$ at $\delta\!f\cdot\tau$ =
  $0.0037$, $0.011$, $0.037$, $0.111$, $0.37$, $1.11$ — and every point reports `valid`.
  Bias under 2 % needs $\lesssim 0.01$, which is $\approx 3\ \mu$Hz at $\tau \approx 3700$ s.
  This resonator drifts by $300$–$1100\ \mu$Hz. The conclusion to state plainly: coherent
  estimation is infeasible on these records at *any* implementation quality, so this is not a
  bug to fix but a method to retire. This is also the one number to take home — it is cheap
  to compute from per-segment frequencies before choosing an estimator.

## 12. The amplifier — a collapsed $\tau$ silently chose the window

- **Key message:** A 12 600 s record was cropped to 107 s, and nothing complained.
- **Visual type:** map/plot
- **Assets:** `crop-cascade.pdf` — the offset-9000 specimen envelope with the two
  round-off-level crop outcomes (107 s and 3154 s) shaded against the 5698 s envelope-based
  seed the pipeline had already computed and discarded.
- **Word budget:** ≤ 26. The crop collapse; "the sanity check it passed was `min_samples`".
- **Notes stub:** Concepts: crop cascade; guard design. The pipeline cropped to
  $3\tau_{\rm est}$ with a floor of 1000 samples — which a 107 s crop at 149 Hz clears
  easily, so the guard was real but measured the wrong thing. Two selections of the same
  window differing only by float round-off gave $\tau_{\rm est}$ = 35.6 s (crop 107 s, 0.8 %
  of the data, $Q$ reported as an upper limit) and 1051 s (crop 3154 s, $Q = 2.79\times10^4$
  reported **`valid`** against an envelope value near $1.3\times10^{5}$). The design rule to
  state: a fitted parameter must never silently select the analysis window, and if a
  pipeline already computes an independent estimate of that parameter, cross-check against it
  rather than discarding it.

## 13. The fix — segmented demodulation

- **Key message:** Coherent inside a segment, incoherent across, with an explicit floor.
- **Visual type:** workflow diagram over real data
- **Assets:** `demod-method.pdf` — record split into segments; per-segment
  $(t_i, f_i, A_i, \sigma_i)$; plateau floor drawn on the amplitude axis; the
  floor-corrected $\log\sqrt{A^2 - A_{\rm floor}^2}$ fit over the contiguous
  $A > 3A_{\rm floor}$ region.
- **Word budget:** ≤ 32. Four design choices, one line each.
- **Notes stub:** Concepts: incoherent averaging; floor correction; robust regression
  (Theil-Sen median-of-slopes, chosen for immunity to plateau revivals); block bootstrap.
  Per segment: remove a linear baseline, locate the tone by zero-padded FFT within a band
  hint, refine with a two-stage linear-least-squares scan over frequency, emit
  $(t_i, f_i, A_i, \sigma_i)$. Then estimate the floor from the late-record amplitude
  distribution, select the contiguous decay region $A > k A_{\rm floor}$ with $k = 3$, and
  fit $\log\sqrt{A^2 - A_{\rm floor}^2}$ against $t$. Each choice answers one measured
  violation: segmenting answers the drift, the floor answers the plateau, amplitude bands
  answer the nonlinear damping, and the bootstrap answers the fake interval. Accuracy under
   2 % on every controlled experiment, including measured drift and plateau. Frame it as
  "log-decrement done right", not a new algorithm.

## 14. Gates — teach the estimator to refuse

- **Key message:** Disagreement now invalidates instead of warning quietly.
- **Visual type:** decision-flow diagram
- **Assets:** `gates-flow.pdf` — measured diagnostics (drift, envelope agreement, dynamic
  range) routing to one of three outcomes: coherent $Q$, incoherent $Q$, or limit / no
  finite $Q$.
- **Word budget:** ≤ 30. The two gates and the three outcomes.
- **Notes stub:** Concepts: model-adequacy gating vs convergence testing; regime
  classification. What shipped: a drift gate that demotes coherent estimators when the
  measured coherence ratio is significant (using a $2\sigma$ lower bound so
  drift-measurement noise cannot fire it on a genuinely coherent record); demotion on
  envelope mismatch; a regime classifier that returns the segmented-demodulation $Q$ for
  long or drifting records, the profile $Q$ for short coherent ones, and *no finite $Q$* for
  plateau-dominated windows; cross-estimator agreement within a factor 1.5 as a hard
  condition, so disagreement yields no number at all rather than a quiet warning; and an
  overlay that draws a mismatching candidate dashed and labelled. Say the general lesson:
  the useful output of an estimator on bad data is a refusal, and a refusal is only possible
  if some check can fail.

## 15. Validation — the spread that was $12\times$ is now $1.3\times$

- **Key message:** Window-to-window spread fell from $12\times$ to $1.3\times$.
- **Visual type:** map/plot, before/after on a shared axis
- **Assets:** `window-sweep-after.pdf` — the same window sweep as slide 5 with the shipped
  default, the physical $Q(A)$ band drawn behind, and the before/after spread annotated.
- **Word budget:** ≤ 30. Spread before/after; "no coherent `valid` under injected drift";
  one line flagging the two deviations.
- **Notes stub:** Concepts: window-stability as an acceptance metric. What is pinned in the
  regression tests: whole-decay $Q$ to $\pm 3\ \%$ of the shipped recipe, plateau 16–18
  cycles, drift $+1.1$/$+0.3$ mHz, per-band $Q$ within 10 % of the reference table.
  Be explicit about the two honest deviations. First, the original target was a spread under
  $1.2$ and the measured value is $1.31$, because re-estimating per window (rather than
  evaluating one whole-decay fit per window, which is stable by construction) exposes the
  genuine $Q(A)$: short early windows sample the low-$Q$ high-amplitude decay. Every window
  stays inside the measured band. Second, the shipped Theil-Sen recipe lands 5–10 % below
  the notebook's unweighted least-squares reference on these curved decays; both are pinned,
  tightly for the shipped recipe and loosely for the reference, so the gap stays visible
  instead of being tuned away.

## 16. The payoff — the qualification verdict flips

- **Key message:** Vibration did not degrade $Q$; the frequency shifted instead.
- **Visual type:** map/plot, two panels
- **Assets:** `pre-post.pdf` — left: local $Q$ vs amplitude, Pre vs Post, showing Post equal
  or higher at matched amplitude; right: the frequency-pull coefficient weakening
  $\approx 5\times$. Superseded single-number claims shown struck through.
- **Word budget:** ≤ 30. The Q verdict; the ppm shift; the pull change.
- **Notes stub:** Why any of this mattered. This pair of records is a vibration
  qualification test, and the single-number coherent estimates had said $-30\ \%$ $Q$ from
  the profile estimator and $+27\ \%$ from raw NLS on the same data — a contradiction that
  should itself have been a red flag. Amplitude-matched demodulation says the Post-Vibe $Q$
  is equal or higher: $+13\ \%$ whole-decay average, $+4$ to $+51\ \%$ band-resolved, i.e.
  no damage signature. What did change: the resonance frequency dropped by $\approx 66$ ppm
  at high amplitude to $\approx 157$ ppm at low amplitude, and the frequency-amplitude pull
  coefficient weakened from $\approx -0.42$ to $\approx -0.09$ mHz per amplitude e-fold,
  roughly $5\times$. Emphasize the comparison protocol: compare $Q$ only at matched
  amplitude when $Q$ depends on amplitude, otherwise the comparison is between two different
  window averages.

## 17. Takeaways, and what we still do not know

- **Key message:** Test model adequacy, budget your coherence, report $Q(A)$.
- **Visual type:** takeaway card + open-questions strip (native Beamer)
- **Assets:** none
- **Word budget:** ≤ 40. Three rules plus three open items.
- **Notes stub:** The three portable rules: (1) validity must test model adequacy, not
  convergence — a tight interval under a wrong model is worse than no interval, because it
  suppresses doubt; (2) compute $\delta\!f\cdot\tau$ from per-segment frequencies before you
  choose an estimator, and if it exceeds $\approx 0.01$ do not fit coherently; (3) if $Q$
  depends on amplitude, report the amplitude band or the fitted law, never a bare number.
  Still open, and say so: the physical origin of the pull and the nonlinear damping
  (resonator intrinsic versus mount versus readout) is out of scope here; the plateau is
  attributed to ambient drive from its spectral and temporal signature but the source was
  not formally identified; and we have not surveyed our other records, so the defaults are
  validated on one resonator pair. Invite the audience to compute the drift diagnostic on
  their own records and tell us whether they see the same phenomenology.

---

## Backup slides

| # | Title | Visual | Assets |
|---|-------|--------|--------|
| B1 | Why not fit a drifting phase? | bullets + one equation | none |
| B2 | Multi-mode and beating ruled out | high-resolution spectra, decay vs plateau | `backup-spectra.pdf` |
| B3 | Why the profile interval collapses | equation card | none |
| B4 | Why window stability is $1.3$, not $1.2$ | $Q(A)$ band with per-window points | reuses `window-sweep-after.pdf` |
| B5 | Theil-Sen vs least squares on the log envelope | bullets + numbers | none |
| B6 | Conventions and record parameters | reference card | none |

---

## Compliance check

- Every content slide has a visual: slides 4 and 17 use native Beamer table / card
  constructions, all others use a generated PDF.
- No slide is bullets-only: slides 4 and 17 are structured tables/cards, which the workflow
  permits as visual types.
- No `minted`, no live code highlighting, no `-shell-escape`. Public API appears only as two
  short inline names on slide 13.
- External-blind: no development-workflow references in slide text or speaker notes.
