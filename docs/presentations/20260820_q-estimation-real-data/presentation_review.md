# Presentation review

Deck: `slides.tex` (17 main frames + 6 backup). Build: `make pdf`; page images: `make slides-png`.

---

## Numbers as built (read this before comparing to the investigation report)

Every real-data number on a slide was **recomputed** for the deck by running the shipped
`RingDownAnalyzer` and `SegmentedDemodEstimator` over the canonical windows, so the deck is
internally consistent and reproducible from `assets/_cache`. Three values therefore differ from
the investigation report, which used a slightly different seeding of the coherent estimator:

| Quantity | Report | Deck (recomputed) | Why |
|----------|--------|-------------------|-----|
| Post-Vibe coherent $Q$, canonical window | 25 078 | 27 082 | different frequency seed; same failure, different draw of the same lottery |
| Pre-Vibe window-sweep spread, before | $12\times$ (gated result) | $\times50.2$ (ungated `Q_profile_raw`) | the deck plots the raw coherent estimate per window, which is the quantity that was actually reported before the fix |
| Post $-$ Pre $Q$, whole decay | $+13\ \%$ | $+3\ \%$ | the deck uses the matched protocol (identical segment length and window on both records); the band-resolved statement, $+5$ to $+54\ \%$, is the one the slide makes |

The frequency shift is quoted from the deck's own matched-band computation, $-77$ ppm at high
amplitude to $-140$ ppm at low amplitude, in place of the report's $-66$ to $-157$ ppm.

The direction and character of every conclusion is unchanged. Where the recomputed value is
larger (the $50\times$ sweep), the slide states the measured value rather than the report's.

---

## Phase 6 — static check

| # | Frame | Message clear without notes | Overflow | Type size | One focal element | Words |
|---|-------|------------------------------|----------|-----------|-------------------|-------|
| 1 | Title | n/a | no | ok | strip | title block |
| 2 | Flawless on synthetics | yes | no | ok | two-panel contrast | 13 |
| 3 | Confidently wrong | yes | no | ok | one large panel | 10 |
| 4 | The numbers | yes | no | large table | table | 30 |
| 5 | Not a bias, a lottery | yes | no | ok | sweep | 22 |
| 6 | Look at the model | yes | no | ok | equation card | 11 |
| 7 | Violation 1, drift | yes | no | ok | left panel | 21 |
| 8 | Violation 2, $Q(A)$ | yes | no | ok | right panel | 21 |
| 9 | Violation 3, plateau | yes | no | ok | left panel | 21 |
| 10 | Injection ladder | yes | no | ok | bar ladder | 22 |
| 11 | Coherence budget | yes | no | ok | knee curve | 21 |
| 12 | Crop cascade | yes | no | ok | extent bars | 22 |
| 13 | The fix | yes | no | ok | method panels | 20 |
| 14 | Gates | yes | no | small in boxes | flow chain | 9 |
| 15 | Validation | yes | no | ok | sweep | 20 |
| 16 | The payoff | yes | no | ok | left panel | 24 |
| 17 | Takeaways | yes | no | ok | boxed statement | 34 |

No frame exceeds the 40-visible-word budget. No `\note{}` content appears on a slide. No
external-blind violations: no development-workflow references in slide text or speaker notes.
LaTeX reports no overfull or underfull boxes.

### Strengths

- The narrative lands the failure before the fix, and every claim on screen is a measured
  number from the same two records.
- Colour is used as a code rather than decoration: red is a wrong or refused answer, green a
  correct or accepted one, grey a neutral reference, and this holds in figures and text alike.
- Slides 5 and 15 are the same axes before and after, so the validation slide needs no
  explanation.
- The two honest deviations (window stability $1.31$ not $1.2$; Theil--Sen vs least squares)
  are stated on the slide's speaker notes and given backup frames rather than buried.

### Weaknesses

- Slide 4 is the only frame without a generated visual; it relies on a table.
- The gates diagram carries the smallest type in the deck.
- Slide 8's right panel shows a single record, deliberately, so the Pre/Post comparison stays
  new on slide 16; a viewer expecting both curves may notice the asymmetry with the left panel.

---

## Phase 7 — Iteration 1

Reviewed all 23 page images. Fixes applied, in the order they mattered:

1. **On-figure type was ~25 % smaller than Beamer body text**, because canvases were 12.4 in
   wide against a 5.9 in text block. Introduced `FIGSCALE = 0.82` in `make_figures.py`, applied
   through a single `panels()` helper, so all sixteen figures shrink their canvas while keeping
   absolute point sizes. On-figure annotations now read at roughly body size.
2. **Annotation collisions after the rescale.** The plateau figure's three in-axes labels became
   two legend entries plus nothing; the drift figure's pull note and the plateau figure's
   "nulls and revivals" moved into the frame bullets. In every case the statement moved to
   slide text rather than the type getting smaller.
3. **Gate boxes overflowed their outlines.** Reduced in-box type, widened boxes, and split the
   longest labels over three lines; the caption moved below the diagram baseline.
4. **Window-sweep refusal markers sat at $0.30\times$**, where they could be misread as $Q$
   values. Parked them on the axis floor and labelled them in the legend.
5. **Legend and figure placement.** The sweep legend moved below both panels (all four corners
   of each panel carry annotations); the injection legend moved to the empty upper right; the
   crop-cascade note shortened to one line clear of the bar labels.
6. **Slide 4 was sparse and slide 17 top-heavy.** The table is now `\large` and vertically
   centred under a single closing statement; the takeaway frame uses `[c]` alignment with a
   boxed punch line.
7. **Title strip cropped to the first 0.9 h**, where the fitted envelope visibly separates from
   the measured one. Over the full window both curves collapse onto a flat tail.
8. **Fit line hidden by data** on the symptom and method figures: fits are now dashed and drawn
   last, so they read on top of the record instead of disappearing into it.

### Remaining known issues

- Gate-diagram type is the smallest on the deck. It is legible at 150 dpi page render but is
  the first thing to check on the actual projector.
- Slide 4 has no figure. Acceptable: the table *is* the evidence, and it is the only frame
  where four numbers side by side beat a plot.

## Phase 7 — Iteration 2 (verification pass)

Re-rendered all 23 pages after Iteration 1 and re-read each image. No major issues found; the
work in this pass was consistency rather than layout:

1. **Earlier phase documents contradicted the built deck.** The brief, outline and specification
   quote the investigation report, whose numbers the deck deliberately recomputes. Each now
   carries a superseded-numbers note pointing here instead of silently disagreeing.
2. **Figure PDFs were excluded by the repository-wide `*.pdf` ignore rule**, so a clone without
   the raw phasemeter records could not rebuild the deck. Added a scoped `.gitignore` that tracks
   `assets/*.pdf` while keeping `slides.pdf`, `_review/`, `assets/_cache/` and `assets/_preview/`
   out of the index.
3. **Repository CI checks** (`ruff format`, `ruff check`) now pass on both asset scripts.

### Remaining known issues

- Gate-diagram type is the smallest on the deck. It is legible at 150 dpi page render but is
  the first thing to check on the actual projector.
- Slide 4 has no figure. Acceptable: the table *is* the evidence, and it is the only frame
  where four numbers side by side beat a plot.
- The deck's Post-Vibe coherent $Q$ (27 082) and window spread ($\times50$) differ from the
  investigation report. Both are correct for what they measure; see "Numbers as built" above.
  If the report is being circulated alongside the talk, reconcile the two or say so on slide 4.

### Stop condition

Two iterations completed. Every frame passes the five-second test on its page image, no type is
cropped or overlapping, and the notation matches the deck's own reference card. Further
iteration needs user feedback on content rather than layout.
