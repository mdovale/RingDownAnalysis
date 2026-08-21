# Q-estimation on real resonators — Slide outline (Phase 2)

> **Superseded numbers.** Slides 5, 15 and 16 quote figures from the investigation report.
> The built deck recomputes them from the shipped pipeline: the before-fix window spread is
> $\times50$ (raw coherent estimate) rather than $12\times$ (gated), and the Pre/Post $Q$ claim
> is band-resolved ($+5$ to $+54\ \%$) rather than a single $+13\ \%$. See
> `presentation_review.md` § "Numbers as built".

> Math uses `$…$` / `$$…$$`. One idea per slide.
> 16 content slides + title. Speaking budget ≈ 20 min = 1200 s; sum below ≈ 1270 s,
> inside the ~10 % buffer (1320 s).

Narrative arc: **symptom → it is not noise → the three broken assumptions → the decisive
experiment → the quantitative law → the fix → the gates → validation → the payoff.**

Open on the failure as the audience would meet it in their own lab, not on the package.

One slide over the default 15-slide cap: the crop-cascade slide (12) is kept in the main
line rather than pushed to backup because it is the one failure whose lesson
("never let a fitted parameter silently choose your analysis window") is a design rule the
audience can apply without adopting any of our code. Time budget still fits.

---

## 1. Title — When a tight error bar lies
- **Takeaway:** Ring-down $Q$ on a real resonator can be wrong by $4\times$ while reporting a
  per-mille confidence interval.
- **Purpose:** Set topic and stakes in one line; frame as measurement + method, not software.
- **Visual:** Title block with LASSO logo; faint background strip of the real decay envelope
  with the wrong fitted envelope crossing it.
- **Time:** 30 s

## 2. The symptom — flawless on synthetics, wrong in the lab
- **Takeaway:** The estimator passed every synthetic test and still contradicted the visible
  decay of real records.
- **Purpose:** Establish the problem exactly as the audience would encounter it, and note
  that three prior rounds of validity hardening did not fix it.
- **Visual:** Two-panel contrast — synthetic record with estimate on truth (left) vs real
  record with estimate visibly off the envelope (right). No equations.
- **Time:** 70 s

## 3. What "confidently wrong" looks like
- **Takeaway:** The fitted envelope decays visibly faster than the data, and only the
  diagnostic nobody was reading disagreed.
- **Purpose:** Make the failure undeniable by eye before any numbers; this is the image the
  audience should remember.
- **Visual:** Real Pre-Vibe decay with the measured envelope points, the envelope-slope fit
  tracking the data, and the recommended-estimator envelope peeling away from it.
- **Time:** 80 s

## 4. The numbers, and the interval that meant nothing
- **Takeaway:** $Q$ low by $1.5\times$ (Pre) and $4.0\times$ (Post), each `valid`, each with
  a $10^{-4}$ relative interval.
- **Purpose:** Convert the picture into the quantitative indictment, and name the bug in the
  validity test: it checked convergence, not model adequacy.
- **Visual:** Compact table — reported $Q$ + interval vs drift-immune reference, both
  records; the reported intervals drawn as error bars too small to see.
- **Time:** 75 s

## 5. Not a bias — a lottery
- **Takeaway:** Ordinary window choices scatter $Q$ by $12\times$, and a round-off-level
  input change moved the *same* window from $0.65\times$ to $0.45\times$ of truth.
- **Purpose:** Kill the natural hope that this is a fixable calibration offset. A biased
  estimator can be corrected; a lottery cannot.
- **Visual:** Window-sweep scatter (start $\times$ duration grid) with the reference line
  and the physical $Q(A)$ band; annotate the $12\times$ span.
- **Time:** 80 s

## 6. So look at the model, not the noise
- **Takeaway:** Every estimator assumed constant $f$, a single exponential, and decay to
  zero; the real record violates all three.
- **Purpose:** The pivot of the talk. Signpost the three violations that the next three
  slides measure, and state up front that the SNR is 400 so noise is not the story.
- **Visual:** Assumption card: the fitted model equation with its three assumptions called
  out, each tagged with the measured violation and a checkmark/cross.
- **Time:** 60 s

## 7. Violation 1 — the frequency moves while you fit
- **Takeaway:** $f$ rises $+1.1$ mHz (Pre) / $+0.3$ mHz (Post) during the decay,
  monotonically with falling amplitude: an anharmonic pull, linear in $\log A$.
- **Purpose:** Establish the primary mechanism as a *measurement*, using an
  estimator-independent diagnostic (per-segment demodulation).
- **Visual:** Per-segment $f$ vs amplitude for both records, with the fitted pull line;
  inset: $f$ vs time.
- **Time:** 85 s

## 8. Violation 2 — not a single exponential
- **Takeaway:** Local $Q$ rises from $6.5\times10^{4}$ to $1.8\times10^{5}$ as amplitude
  falls, so no single $Q$ exists to estimate.
- **Purpose:** Reframe the "tuning changes my answer" complaint as physics: different windows
  genuinely have different mean decay rates.
- **Visual:** Local $Q$ vs amplitude band for both records, plus the log-envelope residual
  curvature that reveals it.
- **Time:** 80 s

## 9. Violation 3 — it never decays to zero, and that is signal
- **Takeaway:** Both records settle at $\approx 17$ cycles of ambient-driven oscillation —
  narrowband, on the analysis frequency, $\approx 10\times$ the broadband noise.
- **Purpose:** Distinguish a driven plateau from a noise floor, and show why no filter can
  remove it while an explicit floor model can.
- **Visual:** Full-record envelope on log axis showing decay $\to$ plateau with revivals;
  inset spectrum of a plateau-only segment with the tone still dominant.
- **Time:** 80 s

## 10. The decisive experiment — inject one measured pathology at a time
- **Takeaway:** Injecting only the measured $f(t)$ into an ideal synthetic reproduced the
  entire field failure: $0.51\times$, $1.77\times$, $3.80\times$ of truth, every one `valid`.
- **Purpose:** The methodological heart of the talk. Show how to convert suspicion into a
  ranked cause, and exonerate noise by construction.
- **Visual:** Experiment ladder — ideal / pull / measured $f(t)$ (×2 seeds, ×2 inits) /
  plateau / amplitude-dependent damping, with coherent-$Q$ ratio bars against $1.0$ and the
  incoherent reference pinned at $1.0$.
- **Time:** 90 s

## 11. The coherence budget — $\delta\!f\cdot\tau \lesssim 0.01$
- **Takeaway:** Coherent fitting needs the frequency stable to $\approx 3\ \mu$Hz here; the
  resonator drifts by $300\text{--}1100\ \mu$Hz, so it is infeasible at any implementation
  quality.
- **Purpose:** Give the audience the one number to compute on their own data before choosing
  an estimator.
- **Visual:** $Q/Q_{\rm true}$ vs $\delta\!f\cdot\tau$ (measured scan) with the 2 % tolerance
  band, the tolerance edge at $0.01$, and this resonator's operating point far to the right.
- **Time:** 80 s

## 12. The amplifier — a collapsed $\tau$ silently chose the window
- **Takeaway:** A decoherence-collapsed $\tau$ cropped a 12 600 s record to 107 s, and
  round-off-level input changes changed the crop by $30\times$.
- **Purpose:** Show how one bad estimate cascades through a pipeline, and state the design
  rule: never let a fitted parameter silently select the analysis window.
- **Visual:** The record with the two crop windows overlaid (107 s sliver vs 3154 s) against
  the envelope-based seed that the pipeline already had and discarded.
- **Time:** 65 s

## 13. The fix — segmented demodulation
- **Takeaway:** Be coherent inside a short segment and incoherent across segments, model the
  plateau as an explicit floor, and fit the corrected log envelope robustly.
- **Purpose:** Present the estimator as four design choices, each answering one measured
  violation — not as a new algorithm to admire.
- **Visual:** Method diagram: record $\to$ segments $\to$ per-segment $(t_i,f_i,A_i)$ $\to$
  floor-corrected log-envelope fit, with the floor subtraction shown on the amplitude axis.
- **Time:** 90 s

## 14. Gates — teach the estimator to refuse
- **Takeaway:** A finite $Q$ is now reported only when the measured drift is small and the
  coherent and incoherent estimates agree; otherwise the answer is a limit or nothing.
- **Purpose:** The transferable software lesson: validity must test model adequacy, and
  cross-estimator disagreement must invalidate rather than warn.
- **Visual:** Decision flow from the measured diagnostics to the reported answer
  (coherent $Q$ / incoherent $Q$ / limit only), with the two gates labeled.
- **Time:** 80 s

## 15. Validation — the spread that was $12\times$ is now $1.3\times$
- **Takeaway:** Window-to-window spread fell from $12\times$ to $1.3\times$, no coherent
  estimator can report `valid` under injected drift, and the residual $1.3\times$ is the
  physical $Q(A)$.
- **Purpose:** Close the loop on the symptom slide with a measured before/after, and be
  explicit about the two places the shipped recipe fell short of the original target.
- **Visual:** Before/after window-sweep panels on a shared axis, with the physical $Q(A)$
  band drawn behind and the two deviations noted in small type.
- **Time:** 75 s

## 16. The payoff — the qualification verdict flips
- **Takeaway:** Vibration did not degrade $Q$; at matched amplitude it is equal or higher
  ($+13\ \%$ whole-decay), and the real change is a $-66$ to $-157$ ppm frequency shift with
  a $\approx 5\times$ weaker pull.
- **Purpose:** Show that this was never a software-quality exercise: the estimator artifact
  had produced a false "damage" verdict on a flight-qualification test.
- **Visual:** Pre vs Post local $Q$ and pull coefficient at matched amplitude, with the
  superseded single-number claims ($-30\ \%$, $+27\ \%$) struck through.
- **Time:** 80 s

## 17. Takeaways, and what we still do not know
- **Takeaway:** Test model adequacy, not convergence; compute $\delta\!f\cdot\tau$ before
  choosing an estimator; report $Q(A)$ rather than one number.
- **Purpose:** Land the three portable rules, then name the open questions honestly and
  invite others to check their own records.
- **Visual:** Three-rule takeaway card plus a short "still open" strip (origin of the pull,
  plateau excitation source, no survey beyond this resonator pair).
- **Time:** 70 s

---

### Backup slides (after `\appendix`, not in the 20 min)

- B1. Why not just fit a drifting phase? Polynomial/spline phase, Prony, matrix pencil,
  spectral linewidth — and why each inherits the same pole model.
- B2. Multi-mode and beating ruled out: high-resolution spectra of decay and plateau
  segments, neighbours $10^{5}\times$ down.
- B3. Why the profile-likelihood interval shrinks under misspecification
  ($\chi^2$ curvature on $10^{6}$ samples).
- B4. Why window stability is $1.3\times$, not the $1.2\times$ target: re-estimation per
  window exposes the physical $Q(A)$ band.
- B5. Theil-Sen vs least squares on the log envelope: robustness against plateau revivals
  costs 5–10 % on genuinely curved decays; both pinned in regression tests.
- B6. Conventions and record parameters: $f_s$, $f_0$, segment duration, floor factor $k$,
  amplitude units (cycles), and the `start_time` offset semantics hazard.
