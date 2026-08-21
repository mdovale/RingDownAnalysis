# Quality factor estimation on real resonators — Slide outline

> Math uses `$…$` / `$$…$$`. One idea per slide.
> Main line ≈ 18 content frames + title. Speaking budget ≈ 20 min.

Narrative arc: **symptom → model assumptions → numbers → window sensitivity → three
measured violations → model revisited → injection → coherence budget → crop cascade →
segmented demodulation → adequacy checks → validation → Pre/Post payoff.**

Library-agnostic throughout: physics, model assumptions, and estimators only.

---

## 1. Title — Bias in coherent $Q$ fits on drifting resonators
- **Subtitle:** Quality factor estimation on ring down data from real, drifting resonators
- **Footer line:** Ring down measurements from the ODIN resonators

## 2. Matched-SNR synthetic versus ODIN EDU Pre-Vibe
- Stacked time series ($\pm Q$ envelopes) + log amplitude for synthetic and real
- Same coherent estimator; exact on the model, wrong on the instrument

## 3. Envelope mismatch on the Pre-Vibe 3 h window
- Measured envelope, incoherent slope, coherent single-frequency fit
- No library status strings or “reported $Q$ / envelope says” labels on the figure

## 4. The fitted model and its three assumptions (early)
- Equation + constant $f$, single $\tau$, decays to zero (questions only)
- No measured violation magnitudes yet

## 5. Pre-Vibe and Post-Vibe: numbers and envelopes
- Table of coherent $Q$, interval, incoherent reference
- Time series with wrong vs reference $\pm$ envelopes for both records

## 6. Window start and duration dependence
- Starts 0, 20, 40, 60, 80, 100 s; durations 1–6 h
- Round-off lottery deferred to crop-cascade slide (where it is plotted)

## 7–9. Three assumption failures (each with matched-SNR synthetic reference)
1. Frequency pull vs amplitude
2. Non-single-exponential / $Q(A)$
3. Driven plateau vs noise floor

## 10. Same model, measured violations (late assumptions card)
- Pull and $Q(A)$: physics to report
- Plateau: fitting nuisance (amplitude floor)

## 11. Controlled injection ladder
## 12. Coherence budget $\delta\!f\cdot\tau$
## 13. Crop cascade / float round-off on $\tau$ seed
## 14. Segmented demodulation
## 15. Model-adequacy checks
## 16. Validation window sweep
## 17. Pre/Post at matched amplitude
## 18. Takeaways

### Backup
- Drifting phase; multi-mode spectra; curvature interval; residual spread; Theil–Sen vs LS;
  record table
