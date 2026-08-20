# Data Analysis Pipeline Audit

Date: 2026-04-28

## Scope

This audit reviewed the real-data analysis path in `ringdownanalysis`:

1. `RingDownDataLoader` loads Moku CSV/MAT files and detrends the phase channel.
2. `RingDownAnalyzer` validates the time base, estimates a decay constant, crops the record, runs NLS and DFT estimators, estimates residual noise, and reports a plug-in CRLB-style uncertainty.
3. `BatchRingDownAnalyzer` runs the single-file analyzer across directories/files and builds summary, consistency, Q-factor, and CRLB-comparison tables.
4. The notebooks in `notebooks/` demonstrate the same workflows interactively.

Verification performed:

- `pytest` passed: 130 tests.
- Focused probes reproduced two untested boundary issues:
  - a CSV with `nan` phase data reaches the optimizer and fails with `ValueError: Initial guess is outside of provided bounds`;
  - a MAT file with fewer than four columns fails with `IndexError` rather than the documented `ValueError`.

## Findings

### 1. File-based analysis does not validate finite signal values before fitting

Severity: high

`analyze_array()` rejects non-finite data through `_parse_array_input()`, but `analyze_file()` only validates timestamps after loading. Loaded `data` can contain `NaN` or `Inf` and then flows into `_estimate_initial_parameters_from_dft()`, `least_squares()`, and CRLB/noise calculations. The resulting failures are optimizer-specific and misleading rather than input-validation errors.

Evidence:

- `ringdownanalysis/analyzer.py`: `_parse_array_input()` checks `np.isfinite(data_arr)`, but `analyze_file()` calls `RingDownDataLoader.load()` and then only `_validate_uniform_timebase(t)`.
- `ringdownanalysis/data_loader.py`: `load_csv()` and `load_mat()` detrend and return arrays without checking finite values.

Observed behavior:

```text
nan_csv: ValueError Initial guess is outside of provided bounds
```

Impact:

- Bad rows in files are reported as fitting failures, not data problems.
- Batch processing records these as generic per-file failures.
- A single non-finite value can contaminate the FFT initializer and later uncertainty calculations.

Recommendation:

- Add finite checks for both time and signal channels in `RingDownDataLoader` or immediately after load in `RingDownAnalyzer.analyze_file()`.
- Raise a clear `ValueError` that names the affected channel and file.
- Add tests for `NaN`/`Inf` in CSV and MAT phase/time data.

### 2. Parallel batch processing ignores the injected analyzer

Severity: high

`BatchRingDownAnalyzer` accepts a custom `RingDownAnalyzer`, and sequential processing uses `self.analyzer.analyze_file()`. In parallel mode, however, `_process_single_file()` constructs a fresh default `RingDownAnalyzer()` for every file.

Evidence:

- `ringdownanalysis/batch_analyzer.py`: sequential branch calls `self.analyzer.analyze_file(filepath)`.
- `ringdownanalysis/batch_analyzer.py`: parallel branch submits `_process_single_file(filepath)`, and `_process_single_file()` always creates `RingDownAnalyzer()`.

Impact:

- Custom NLS/DFT estimators, `f_min`, windowing, or other analyzer settings silently disappear when `n_jobs > 1` or `n_jobs=-1`.
- Sequential and parallel runs can produce different scientific results from the same `BatchRingDownAnalyzer` instance.
- This is especially risky because the README encourages `n_jobs=-1` for batch analysis.

Recommendation:

- Either serialize the analyzer configuration into the worker call, or disallow parallel mode when a non-default analyzer is injected.
- Add a regression test showing that a custom estimator setting is preserved in both sequential and parallel paths.

### 3. Notebooks and docs still describe a 3x-tau crop, but code defaults to 1x-tau

Severity: medium

The current package default is `max_tau_multiplier=1.0` in `RingDownAnalyzer.analyze_file()` and `analyze_array()`. Several notebooks describe the pipeline as cropping to `3*tau` or plot a `3×tau` marker, even though the actual call uses the default 1x crop unless explicitly overridden.

Evidence:

- `ringdownanalysis/analyzer.py`: `analyze_file(..., max_tau_multiplier=1.0)` and `analyze_array(..., max_tau_multiplier=1.0)`.
- `notebooks/0.2_drifting-resonators.ipynb`: markdown says “Crop data to 3*tau” while code calls `analyzer.analyze_file(filepath)`.
- `notebooks/0.1_batch-analysis.ipynb`: markdown says “Crop data to 3×tau” while the helper calls `analyzer.analyze_file(filepath, max_tau_multiplier=1.0)`.
- Both notebooks plot a `3×tau` boundary, not necessarily the actual `T_crop`.

Impact:

- Users may believe the analysis uses a longer observation window than it actually does.
- Reported frequency estimates and plug-in uncertainties can change materially with crop duration.
- Visualizations may mark a scientifically meaningful boundary that is not the actual crop boundary.

Recommendation:

- Decide whether the canonical default should be 1x or 3x.
- Update notebooks and README wording to match the actual default.
- In plots, mark `result["T_crop"]` as the actual crop boundary and optionally mark `3*tau` separately.

### 4. MAT column validation is incomplete

Severity: medium

The data-format document promises that MAT files with fewer than four columns raise `ValueError`. In practice, `load_mat()` indexes `moku_data[:, 3]` without first checking dimensionality or column count.

Observed behavior for a 3-column `moku.data`:

```text
short_mat: IndexError index 3 is out of bounds for axis 1 with size 3
```

Impact:

- Error handling differs from the public specification.
- Batch failures are less understandable.
- Similar malformed structures could fail with `IndexError` or shape-related errors instead of a controlled validation message.

Recommendation:

- Validate `moku_data.ndim == 2` and `moku_data.shape[1] >= 4` before indexing.
- Validate optional V2 only when `shape[1] > 8`.
- Add tests for 1D, empty, and under-columned MAT payloads.

### 5. CSV header handling is inconsistent with loader comments and helper code

Severity: medium

`RingDownDataLoader._is_data_line()` can distinguish numeric rows from comments/headers, but it is unused. `load_csv()` relies on `pandas.read_csv(..., header=None, dtype=float)`, so a non-comment text header such as `time,a,b,phase` causes the entire load to fail. The inline comment says header rows are skipped, but the implementation only skips `%` comments.

Impact:

- Plain CSV headers are not accepted, even though the loader code suggests they might be.
- Users get “No valid data lines” even if valid data follows the header.

Recommendation:

- Either document that only `%` comments are accepted and remove the stale helper/comment, or implement a pre-filter that uses `_is_data_line()`.

### 6. Plug-in CRLB output is easy to overinterpret

Severity: medium

The result fields `crlb_std_f`, `plugin_crlb_std_f`, and `uncertainty_std_f` all refer to a plug-in bound calculated after estimating frequency, tau, amplitude, and noise from the same cropped data. The reported uncertainty method says `plugin_crlb_known_tau_with_residual_dof_correction`, but tau is generally estimated, not known a priori.

Impact:

- Users may read `crlb_std_f` as a fundamental bound for the original experiment rather than a model-dependent post-fit diagnostic.
- `crlb_comparison_analysis()` compares `|f_NLS - f_DFT|` with that bound, but the two estimates are correlated and the DFT/NLS difference also includes method bias. The ratio is useful as a heuristic diagnostic, not a formal consistency test.

Recommendation:

- Rename or document these fields as plug-in diagnostics.
- Preserve `crlb_std_f` only as a backward-compatible alias if needed.
- Make reports/notebooks call out that the bound assumes the fitted model and the selected crop.

### 7. DFT tau/Q estimation can continue after a failed frequency stage

Severity: medium

`DFTFrequencyEstimator.estimate_full()` records DFT frequency-stage failures through `frequency_result.success`, but still runs the fixed-frequency tau fit using the fallback frequency. It can therefore return tau/Q values associated with a failed or fallback frequency stage.

Impact:

- Downstream code may see non-null `tau_dft` or `Q_dft` and treat them as meaningful even when `dft_success` is false.
- Summary/reporting code only partially surfaces success/fallback flags.

Recommendation:

- If the DFT frequency stage fails, either skip tau/Q estimation or mark tau/Q as fallback-derived in separate metadata.
- Include `dft_success` and `dft_used_fallback` in summary tables.

### 8. Summary tables stringify numeric data

Severity: low

`BatchRingDownAnalyzer.get_summary_table()` and `get_consistency_table()` format numeric values as strings. This is convenient for display but makes downstream sorting, filtering, aggregation, and plotting error-prone.

Impact:

- A user who builds a `pandas.DataFrame(summary["data"])` gets object/string columns rather than numeric columns.
- The README and examples encourage this pattern.

Recommendation:

- Return raw numeric values from API methods and leave formatting to presentation code.
- If formatted output is useful, expose a separate `get_formatted_summary_table()`.

### 9. Public type hints and documentation drift from behavior

Severity: low

Examples:

- `BatchRingDownAnalyzer.process_directory()` is annotated as returning `list[dict]`, but returns `ProcessResult`.
- The `DFTFrequencyEstimator.__init__()` docstring says `use_zeropad` defaults to false, but the actual default is true.
- The top-level README says examples include `examples/usage_example.py`, but that file is not present in this checkout.

Impact:

- Static type users and notebook users get misleading guidance.
- API discovery is harder because docstrings and behavior disagree.

Recommendation:

- Update type hints and docstrings as part of the same cleanup as the notebook refresh.

### 10. Test coverage favors smoke tests over scientific regression tests

Severity: low

The suite is healthy and fast, but the real-data pipeline tests mostly assert that estimates are finite and metadata exists. They do not yet lock down scientific behavior across realistic files, crop multipliers, estimator settings, or malformed inputs.

Missing or thinly covered cases:

- finite-value validation for file input;
- malformed MAT shape/column count;
- parallel batch behavior with custom analyzer configuration;
- actual crop duration in notebooks/examples;
- accuracy/regression bounds for full `RingDownAnalyzer` outputs on known synthetic files;
- uncertainty semantics when NLS/DFT fits fall back.

Recommendation:

- Add focused tests around the high- and medium-severity findings above.
- Add a small deterministic synthetic fixture with expected frequency/tau/Q tolerances for the full file pipeline.

## Deeper Tau/Q Diagnosis

This section focuses on two user-observed behaviors:

- large discrepancies between the pre-crop `tau_est` and the cropped-stage `tau_nls` / `tau_dft`;
- large changes in Q for the same data when `max_tau_multiplier` changes.

### The pipeline currently has three distinct tau concepts

The names make this easy to misread:

1. Envelope/user seed: `_estimate_initial_tau_from_envelope()` or caller-provided `tau_init`.
2. Pre-crop full-record fit: `RingDownAnalyzer.estimate_tau()`, returned as `result["tau_est"]`.
3. Cropped-stage fits: `result["tau_nls"]` and `result["tau_dft"]`, which are used to compute `Q_nls` and `Q_dft`.

These are not equivalent. `tau_est` is fit on the full record before cropping. `tau_nls` and `tau_dft` are refit on the cropped record, with fit bounds derived from the cropped time array.

### Slow decay plus noisy data can make the automatic tau seed far too small

`_estimate_initial_tau_from_envelope()` looks for the first RMS window below `1/e` of the peak RMS. If no such point is found, it returns `t[-1] / 2`. That fallback is reasonable for some records, but it is badly biased for high-Q / slow-decay data where the observed record spans much less than one decay constant.

Controlled reproduction:

- true frequency: 7.67 Hz
- true Q: 60000
- true tau: 2490 s
- record length: 336 s, so `T / tau = 0.135`
- initial SNR: 20 dB

The envelope fallback returned about 168 s (`T/2`). With no explicit `tau_init`, the full-record NLS `tau_est` also stayed at about 168 s. That is not primarily a final-estimator failure; the record simply contains little decay information relative to the noise, and the automatic seed is poor.

When the same synthetic data was analyzed with a reasonable explicit `tau_init` (400 s or larger), the full-record fit recovered tau near 2444 s and Q near the true 60000.

### Crop multiplier changes the tau bounds, so Q can scale with the crop

The cropped-stage estimators call `_sanitize_tau_guess()` on the cropped time array:

- `tau_lower = t_crop[1]`
- `tau_upper = max(10 * T_crop, 1.1 * tau_guess)`

Because `Q = pi * f * tau`, any crop-driven tau instability appears directly as Q instability.

In the controlled high-Q noisy case above, with automatic tau initialization:

| `max_tau_multiplier` | `T_crop` | `tau_est` | `tau_nls` | `tau_dft` | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0.5 | 84 s | 168 s | 839 s | 839 s | both cropped fits hit the `10*T_crop` upper bound |
| 1.0 | 168 s | 168 s | 1678 s | 1678 s | both cropped fits hit the `10*T_crop` upper bound |
| 2.0 | 336 s | 168 s | 168 s | 2444 s | NLS sticks near the bad seed; DFT tau recovers near truth |

This explains why the same data can produce very different Q values depending on crop: the crop does not just choose samples, it also changes the feasible tau range and the local optimizer landscape.

### A user-provided tau_init has two roles

`tau_init` is used to seed the full-record `estimate_tau()` fit. Later, `_run_analysis_pipeline()` passes the original `tau_init` to the cropped NLS/DFT fits if it was provided; otherwise it passes `tau_est`.

That means:

- a good `tau_init` can rescue the full-record fit and prevent over-cropping;
- a bad `tau_init` can still drive crop and Q artifacts;
- the cropped-stage fit may be initialized from the original caller value instead of the improved full-record `tau_est`.

Controlled reproduction on the same high-Q noisy data:

| `tau_init` | `tau_est` | `T_crop` | `tau_nls` | `tau_dft` | Q behavior |
|---:|---:|---:|---:|---:|---|
| 10 s | 10 s | 10 s | 100 s | 100 s | Q follows the artificial `10*T_crop` bound |
| 100 s | 100 s | 100 s | 1000 s | 1000 s | Q again follows the crop-derived bound |
| 400 s | 2444 s | full record | 2444 s | 2444 s | Q stabilizes near the true value |
| 4000 s | 2444 s | full record | 2444 s | 2444 s | Q also stabilizes |

### Why DFT and NLS tau can disagree

`tau_nls` comes from a joint fit of frequency and tau on the cropped data. `tau_dft` comes from a fixed-frequency tau fit where the frequency is first estimated by DFT. In clean, well-observed decays these agree. In noisy partial-decay data they can diverge because:

- the joint NLS fit can keep tau near the seed while nudging frequency, amplitude, phase, and DC;
- the DFT path fixes frequency first, which can either stabilize tau or force tau to compensate for a slight frequency/model mismatch;
- the DFT tau fit can produce finite `Q_dft` even when the DFT frequency stage reports fallback metadata.

The EDU notebook saved output shows a severe example: one channel has `Q_nls` around `6.07e4` while `Q_dft` is about `1.67e-1`. That implies `tau_dft` collapsed near the lower bound. The current summary output does not make lower/upper-bound hits visible, so this kind of failure can look like a plausible numeric result unless users inspect `tau_dft`, `dft_success`, and `dft_used_fallback`.

### Recommendations for tau/Q stability

- Add tau diagnostics to every result: `tau_seed`, `tau_seed_method`, full-stage tau bounds, cropped-stage tau bounds, and flags for lower/upper-bound hits.
- Treat `tau_est` as low-confidence when the envelope seed falls back to `T/2`, when `T / tau_est` is small, or when a fitted tau lands near a bound.
- Do not crop to `tau_est` when the decay is not clearly observed. For high-Q partial-decay records, prefer full-record analysis or a user-specified physical crop window.
- Consider computing and reporting a `Q_pre_crop = pi * f_nls * tau_est` alongside `Q_nls` / `Q_dft`, so users can distinguish full-record decay estimates from cropped-stage estimates.
- Consider seeding cropped-stage fits from the full-record `tau_est` after `estimate_tau()` succeeds, even when the user supplied `tau_init`, or expose both behaviors explicitly.
- Add a crop-sensitivity diagnostic helper that runs several `max_tau_multiplier` values and marks Q as unstable if `Q_nls` / `Q_dft` vary beyond a chosen tolerance.
- For batch summaries, include `tau_est`, `tau_nls`, `tau_dft`, `Q_nls`, `Q_dft`, success/fallback flags, and bound-hit flags. Do not reduce Q reporting to a single `Q` value without provenance.

## Additional Observations

- The split between array input and file input is mostly clean, but validation should be shared so the two paths cannot diverge.
- `V2` is loaded from MAT files and preserved in the result, but it is not analyzed or summarized. If V2 is scientifically important, it should either be part of the analysis pipeline or explicitly documented as pass-through metadata.
- `max_tau_multiplier` is not validated. Non-positive values currently produce an empty crop and then usually fall back to the original data because of the minimum-sample guard, silently ignoring the bad setting.

## Recommended Fix Order

1. Add shared finite-value and shape validation for loaded data.
2. Fix parallel batch processing so custom analyzer configuration is honored.
3. Align crop defaults, notebook text, and crop visualizations.
4. Harden MAT/CSV loader error messages against malformed files.
5. Clarify plug-in CRLB/uncertainty semantics in field names, docs, notebooks, and tables.
6. Add tau/Q diagnostics and crop-sensitivity tests for high-Q partial-decay records.
7. Add regression tests for the above before refactoring summary-table formatting.

