# Frequency and quality factor estimation of exponentially decaying sinusoids

This repository contains theoretical analysis, numerical simulations, and experimental data analysis for **frequency and quality-factor (Q) estimation** of ring-down signals. Ring-down signals are exponentially decaying sinusoids that arise from measurements of harmonic oscillators with quality factor Q, where the amplitude decays exponentially due to energy dissipation.

## Quickstart

### Installation

From PyPI:

```bash
pip install ringdownanalysis
```

For development (testing, linting) or examples and notebooks, install from source:

```bash
git clone https://github.com/mdovale/RingDownAnalysis.git
cd RingDownAnalysis
pip install -e ".[dev]"      # testing, ruff, mypy, sphinx
pip install -e ".[examples]" # jupyter, pandas for notebooks
pip install -e ".[all]"     # everything
```

### Basic Usage

#### Generate and Analyze a Ring-Down Signal

```python
from ringdownanalysis import RingDownSignal, NLSFrequencyEstimator, DFTFrequencyEstimator
import numpy as np

# Generate a ring-down signal
signal = RingDownSignal(
    f0=5.0,      # Frequency (Hz)
    fs=100.0,    # Sampling frequency (Hz)
    N=10000,     # Number of samples
    A0=1.0,      # Initial amplitude
    snr_db=60.0, # Initial SNR (dB)
    Q=10000.0,   # Quality factor
)

rng = np.random.default_rng(42)
t, x, phi0 = signal.generate(rng=rng)

# Estimate frequency using NLS method
nls_estimator = NLSFrequencyEstimator(tau_known=None)
f_nls = nls_estimator.estimate(x, signal.fs)

# Estimate frequency using DFT method
dft_estimator = DFTFrequencyEstimator(window="rect")
f_dft = dft_estimator.estimate(x, signal.fs)

print(f"True frequency: {signal.f0:.6f} Hz")
print(f"NLS estimate:    {f_nls:.6f} Hz")
print(f"DFT estimate:    {f_dft:.6f} Hz")

# Or estimate frequency, tau, and Q together
result_nls = nls_estimator.estimate_full(x, signal.fs)
result_dft = dft_estimator.estimate_full(x, signal.fs)

print(f"\nNLS full result: f={result_nls.f:.6f} Hz, tau={result_nls.tau:.2f} s, Q={result_nls.Q:.2e}")
print(f"DFT full result: f={result_dft.f:.6f} Hz, tau={result_dft.tau:.2f} s, Q={result_dft.Q:.2e}")
```

#### Analyze Experimental Data

```python
from ringdownanalysis import RingDownAnalyzer, BatchRingDownAnalyzer
import pandas as pd

# Single-file analysis: Q_selected is the recommended per-record Q
analyzer = RingDownAnalyzer()
result = analyzer.analyze_file("data/example.csv")
if result["Q_selected_valid"]:
    print(
        f"Q = {result['Q_selected']:.3e} "
        f"({result['Q_selected_source']}, {result['Q_selected_regime']})"
    )

# Batch processing (aggregate Q defaults to valid Q_demod; see migration note)
batch_analyzer = BatchRingDownAnalyzer()
batch_result = batch_analyzer.process_directory("data", verbose=True)

# Get summary table (includes Q_demod, Q_profile, coherence_ratio, etc.)
summary = batch_analyzer.get_summary_table()
df_summary = pd.DataFrame(summary["data"])
print(df_summary)

# For notebook/console display strings:
df_display = pd.DataFrame(batch_analyzer.get_formatted_summary_table()["data"])
```

See `examples/usage_example.py` and `examples/batch_analysis_example.py` for more complete examples.
Notebook walkthroughs: `notebooks/0.4_frequency-estimation.ipynb` for the standalone
estimators, `notebooks/0.0_quick-start.ipynb` for the full pipeline.

#### Which Q should I trust?

The pipeline reports several Q estimates because they make different
assumptions, and real records violate some of them. In order of preference:

1. **`Q_selected`** — the recommended single answer. The pipeline classifies
   each record from its own diagnostics (`Q_selected_regime`) and picks the
   estimator whose assumptions hold: the segmented-demodulation Q for long or
   drifting records, the profile Q for short coherent ones, and *no finite Q*
   for plateau-dominated windows. Cross-estimator agreement within a factor
   1.5 is a hard condition: if the estimators disagree, `Q_selected` is
   `None` with status `warning` and reason `cross_estimator_disagreement`.
2. **`Q_demod`** — incoherent segmented-demodulation estimate
   (`SegmentedDemodEstimator`). Immune to the frequency drift that biases
   every phase-coherent fit, models the ambient-driven plateau, and reports
   amplitude-resolved local Q (`Q_demod_vs_amplitude`), the frequency drift
   (`Q_demod_drift_hz`), and the plateau level. Use this for long real-data
   records. `Q_demod_ci95` comes from a residual block bootstrap over
   segments, not from a white-noise model.
3. **`Q_profile`** — profile-likelihood coherent fit. Exact on short,
   drift-free records; on drifting resonators it can be biased by factors of
   2–4 *while reporting a tight CI*, so it is gated: it is demoted to
   non-valid when the measured drift is significant
   (`coherence_gate_fired=True`, reason `coherence_gate_drift`) or when it
   disagrees with the measured envelope slope (reason `envelope_mismatch`).
4. **`Q_envelope`** — incoherent envelope-slope diagnostic. Robust, but
   biased high on windows that extend past ≈ 3× the plateau level.
5. **`Q_nls_raw` / `Q_dft_raw`** — raw coherent optimizer products; keep for
   diagnostics only.

The dimensionless `coherence_ratio` = |measured drift| × τ quantifies drift:
coherent estimators are trustworthy only for `coherence_ratio` ≲ 0.01.

Always check status before quoting Q:

```python
from ringdownanalysis import RingDownAnalyzer

analyzer = RingDownAnalyzer()
result = analyzer.analyze_array(t=t, data=x)

if result["Q_selected_valid"]:
    print(f"Q = {result['Q_selected']:.3e} ({result['Q_selected_source']})")
elif result["Q_profile_valid"]:
    print(f"Profile Q = {result['Q_profile']:.3e} with 95% interval {result['Q_profile_ci95']}")
elif result["Q_profile_status"] == "lower_limit":
    print(f"Q > {result['Q_profile_lower_limit_95']:.3e} at approximately 95% confidence")
else:
    print(f"No finite Q: {result['Q_selected_status']} {result['Q_selected_reasons']}")
```

For amplitude-dependent damping (no single "true Q" exists), fit the
amplitude laws explicitly:

```python
from ringdownanalysis import SegmentedDemodEstimator, fit_nonlinear_damping

demod = SegmentedDemodEstimator().estimate(t, x, fs)
nl = fit_nonlinear_damping(demod)     # 1/tau(A) = 1/tau0 + beta*A
print(nl.tau0, nl.beta, nl.Q0)        # zero-amplitude decay time and Q
print(nl.f_zero, nl.f_pull)           # f(A) = f_zero + pull*A
print(nl.q_at(100.0))                 # amplitude-matched local Q
```

#### Profile Q, limits, and raw coherent diagnostics

Beyond `Q_selected`, each record exposes detailed estimator metadata:

- `Q_profile` is a finite profile-likelihood Q estimate when the log-tau
  profile closes on both sides of the optimum.
- `Q_profile_valid`, `Q_profile_status`, `Q_profile_reasons`,
  `Q_profile_ci95`, `Q_profile_lower_limit_95`, and
  `Q_profile_upper_limit_95` explain whether the data support a finite Q,
  a one-sided limit, or no useful bound.
- `Q_nls_raw` and `Q_dft_raw` preserve the direct fitted products from the
  legacy optimizers.
- `Q_nls` and `Q_dft` are retained as compatibility diagnostics and are only
  populated when the corresponding estimate is cleanly valid.
- `Q_nls_valid`, `Q_dft_valid`, `Q_nls_status`, `Q_dft_status`, and the
  `Q_*_reasons` lists explain invalid or warning-status estimates.

Common `Q_profile_status` values are:

- `valid`: `Q_profile` and `Q_profile_ci95` are finite and may be quoted.
- `lower_limit`: the record supports a lower bound, but not a finite upper
  interval endpoint. Quote `Q_profile_lower_limit_95`, not `Q_profile`.
- `upper_limit`: the record supports an upper bound only.
- `unbounded`, `invalid`, or `failed`: the record does not support a finite Q.

#### Batch aggregate Q

`BatchRingDownAnalyzer.calculate_q_factors()` fills the per-record `Q`,
`Q_valid`, and `Q_status` fields used in summary tables. By default
(`q_preference="demod"`, since 1.2.0) it prefers a valid `Q_demod` and falls
back to valid profile Q when demod is unavailable. Invalid or limit-only
estimates are skipped unless you pass `include_invalid=True` for diagnostic
work. Per-file `Q_selected` remains available in each result dict from
`process_directory()` / `process_files()` regardless of batch preference.

For array workflows, pass `detrend="constant"` to `analyze_array()` when you
want the same constant-offset removal used by file loading. To inspect
window/crop sensitivity, use `RingDownAnalyzer.q_sensitivity(...)`; it returns
DataFrame-ready records with valid Q, raw Q, status, reasons, tau bound flags,
and crop metadata for each requested start/duration/multiplier combination.

#### Data-quality notes for real records

- **Driven plateau.** Ambient excitation keeps the resonator oscillating at
  an equilibrium amplitude (≈ 17 cycles on the EDU records); the record never
  decays to zero. Windows past the point where the decay meets ≈ 3× the
  plateau bias envelope fits high, and windows containing only plateau are
  unusable for Q (`Q_demod_status="plateau_dominated"`).
- **Frequency drift.** Real resonance frequencies move by 0.3–1.1 mHz during
  a decay — orders of magnitude beyond the coherence tolerance of coherent
  fits. Check `coherence_ratio` before trusting `Q_profile`/`Q_nls`.
- **`start_time` offset semantics.** When loading Moku phasemeter data via
  `mokutools`, `start_time` is an *offset from the first sample of the file*,
  not an absolute timestamp. Windows selected with absolute times silently
  select the wrong data. Worked example:
  `notebooks/0.5_odin-phasemeter-data.ipynb`.

#### Migration note: batch Q preference

`BatchRingDownAnalyzer` defaults to `q_preference="demod"` as of 1.2.0,
preferring the drift-immune `Q_demod` and falling back to the profile logic
when demod has no valid estimate. Pass `q_preference="profile"` to preserve
the 1.1.x batch Q behavior.

#### Configure Logging

The package uses `NullHandler` by default (no log output). For easier debugging, enable console logging:

```python
import logging
from ringdownanalysis import configure_logging, BatchRingDownAnalyzer

# Quick setup: INFO-level console output
configure_logging(level=logging.INFO)

analyzer = BatchRingDownAnalyzer()
results = analyzer.process_directory("data")
```

For production (file + console, rotation, structured format):

```python
from examples.logging_config_example import setup_production_logging
import logging

setup_production_logging(log_dir='logs', log_level=logging.INFO)
```

See `examples/logging_config_example.py` for more logging configuration options.

## Overview

The project compares complementary approaches for **frequency and Q estimation**:

1. **Nonlinear Least Squares (NLS)** with explicit ring-down model
2. **Frequency-Domain Methods (DFT)** with Lorentzian peak fitting
3. **Profile-likelihood Q** with finite intervals and one-sided limits
4. **Segmented demodulation Q** — drift-immune, plateau-aware incoherent estimation
5. **Envelope and selection layers** — diagnostics plus a single recommended `Q_selected`

Frequency methods are evaluated against Cramér-Rao-style bounds derived from the explicit Fisher information matrix for ring-down signals. Real-data analyzer outputs such as `plugin_crlb_std_f`, `uncertainty_std_f`, and the backward-compatible `crlb_std_f` alias are plug-in diagnostics computed from the fitted model, residual noise, and selected crop. Treat batch ratios involving those values as heuristic consistency diagnostics rather than formal hypothesis tests.

## Features

### Object-Oriented API

The package provides a modern object-oriented API:

- **`RingDownSignal`**: Generate synthetic ring-down signals; pathology fixtures
  (`generate_driven_plateau`, `generate_pathological_ringdown`) for plateau and drift
- **`FrequencyEstimator`**: Base class for frequency estimation methods
  - **`NLSFrequencyEstimator`**: Nonlinear least squares estimation
    - `estimate()`: Returns frequency only
    - `estimate_full()`: Returns `EstimationResult` with frequency, tau, and Q
  - **`DFTFrequencyEstimator`**: DFT-based estimation with Lorentzian fitting
    - `estimate()`: Returns frequency only
    - `estimate_full()`: Returns `EstimationResult` with frequency, tau (via NLS with fixed frequency), and Q
- **`EstimationResult`**: Named tuple containing (f, tau, Q) estimates
- **`SegmentedDemodEstimator`**: Drift-immune segmented-demodulation Q with plateau modeling
- **`ProfileQEstimator`**: Standalone profile-likelihood Q with limits and CI fields
- **`QEnvelopeDiagnostic`**: Envelope-slope Q diagnostic (`q_envelope_diagnostic`)
- **`select_q_estimate` / `QSelection`**: Regime classifier behind `Q_selected` fields
- **`fit_nonlinear_damping`**: Amplitude-dependent decay and f(A) pull models
- **`CRLBCalculator`**: Calculate Cramér-Rao Lower Bound for frequency estimation
- **`RingDownDataLoader`**: Load CSV and MAT (Moku:Lab Phasemeter) ring-down files
- **`RingDownAnalyzer`**: Single-file analysis with multi-estimator Q outputs
- **`BatchRingDownAnalyzer`**: Batch process multiple data files (`q_preference` for aggregate Q)
- **`MonteCarloAnalyzer`**: Run Monte Carlo simulations to compare methods
- **Plotting helpers**: `plot_q_envelope_overlay`, `plot_q_individual_results`, and related performance plots

### Compatibility Layer

A function-based compatibility layer is also available for backward compatibility:

```python
from ringdownanalysis import (
    generate_ringdown,
    estimate_freq_nls_ringdown,
    estimate_freq_dft,
    estimate_freq_dft_optimized,
    db_to_lin,
    crlb_var_f_ringdown_explicit,
    monte_carlo_analysis,
)
```

## Usage Examples

### Monte Carlo Analysis

Compare estimation methods using Monte Carlo simulations:

```python
from ringdownanalysis import MonteCarloAnalyzer

analyzer = MonteCarloAnalyzer()

results = analyzer.run(
    f0=5.0,
    fs=100.0,
    N=1_000_000,
    A0=1.0,
    snr_db=60.0,
    Q=10000.0,
    n_mc=100,
    seed=42,
)

print(f"NLS std: {results['stats']['nls']['std']:.6e} Hz")
print(f"DFT std: {results['stats']['dft']['std']:.6e} Hz")
print(f"CRLB std: {results['crlb_std']:.6e} Hz")
```

Walkthrough with interpretation: `notebooks/0.6_monte-carlo-crlb.ipynb`.

### Calculate CRLB

```python
from ringdownanalysis import CRLBCalculator

crlb_calc = CRLBCalculator()
crlb_std = crlb_calc.standard_deviation(
    A0=1.0,
    sigma=0.001,
    fs=100.0,
    N=10000,
    tau=636.6,
)

print(f"CRLB standard deviation: {crlb_std:.6e} Hz")
```

### Batch Analysis

Process multiple experimental data files:

```python
from ringdownanalysis import BatchRingDownAnalyzer
import pandas as pd

# Default q_preference="demod" (since 1.2.0); pass q_preference="profile" for 1.1.x behavior
batch_analyzer = BatchRingDownAnalyzer()

# Process all files in data directory
batch_result = batch_analyzer.process_directory("data", verbose=True, n_jobs=-1)

# Per-file Q_selected is in each result dict; aggregate Q uses q_preference
batch_analyzer.calculate_q_factors()
q_stats = batch_analyzer.get_q_factor_statistics()

# Summary table includes Q_demod, Q_profile, coherence_ratio, and aggregate Q
summary = batch_analyzer.get_summary_table()
df_summary = pd.DataFrame(summary["data"])
# Use get_formatted_summary_table() or get_formatted_consistency_table()
# when you want preformatted display strings instead of numeric columns.

# Consistency analysis
consistency = batch_analyzer.consistency_analysis()

# CRLB comparison
crlb_analysis = batch_analyzer.crlb_comparison_analysis()
```

See `examples/batch_analysis_example.py` for a complete batch analysis example.

## Project Structure

### Core Package (`ringdownanalysis/`)

- **`signal.py`**: `RingDownSignal` class for signal generation, plus
  synthetic pathology generators (`generate_pathological_ringdown`,
  `generate_driven_plateau`)
- **`estimators.py`**: Frequency estimation classes (NLS, DFT)
- **`demod.py`**: `SegmentedDemodEstimator` — drift-immune incoherent Q
  estimation with plateau modeling and amplitude-resolved output
- **`nonlinear.py`**: nonlinear-damping and frequency-pull amplitude laws
  (`fit_nonlinear_damping`)
- **`selection.py`**: estimator-selection classifier (`select_q_estimate`)
  behind the `Q_selected` result fields
- **`q_profile.py`** / **`q_envelope.py`**: profile-likelihood Q and
  envelope-slope diagnostic
- **`plots.py`**: Monte Carlo and Q diagnostic plotting helpers
- **`_gridfit.py`**: Shared grid utilities for demod segment fitting
- **`crlb.py`**: CRLB calculation
- **`data_loader.py`**: Data loading utilities for CSV and MAT files
- **`analyzer.py`**: Single-file analysis
- **`batch_analyzer.py`**: Batch processing and analysis
- **`monte_carlo.py`**: Monte Carlo simulation framework
- **`compat.py`**: Compatibility layer (function-based API)

### Documentation (`docs/`)

- **`api/`**: Sphinx API documentation — build with ``make -C docs/api html``
- **`data_format.md`**: Data format specification for CSV and MAT files
- **`tn/main.tex`**: Comprehensive LaTeX document with theoretical foundation
- **`tn/main.pdf`**: Compiled technical note

### Examples (`examples/`)

- **`usage_example.py`**: Signal generation, frequency estimation, Monte Carlo, and plotting
- **`batch_analysis_example.py`**: Batch analysis workflow example
- **`benchmark.py`**: Simple performance benchmark comparing NLS and DFT methods
- **`crlb_scaling_figures.py`**: CRLB scaling figure generation
- **`logging_config_example.py`**: Examples for configuring logging in production and debugging

### Benchmarks (`benchmarks/`)

- **`benchmark_suite.py`**: Comprehensive pytest-benchmark test suite
- **`bench_qestimation.py`**: Before/after harness for the Q-estimation stack
- **`baseline_qestimation.py`**: Baseline timings for Q-estimation regressions
- **`run_benchmarks.py`**: Script to run benchmarks and generate reports
- **`run_profiling.py`**: Script to profile workloads and identify bottlenecks
- **`profile_utils.py`**: cProfile utilities for profiling workloads
- **`README.md`**: Detailed guide for benchmarking and profiling

See `benchmarks/README.md` for detailed information on performance benchmarking and profiling.

### Notebooks (`notebooks/`)

The `0.x` series is a tutorial ladder, in learning order:

| Notebook | What it covers |
| --- | --- |
| **`0.0_quick-start.ipynb`** | Synthetic quick start: `RingDownAnalyzer.analyze_array()`, input formats, a first look at `Q_selected` |
| **`0.1_drifting-resonators.ipynb`** | Real `.csv`/`.mat` records: drift and plateau diagnostics, why `Q_selected` prefers `Q_demod`, amplitude-dependent damping |
| **`0.2_batch-analysis.ipynb`** | Many files with `BatchRingDownAnalyzer`: default demod aggregate Q, consistency tables, plug-in uncertainty ratios |
| **`0.3_profile-likelihood-q.ipynb`** | Profile-likelihood Q on synthetics: finite estimates, one-sided limits, window sensitivity |
| **`0.4_frequency-estimation.ipynb`** | `NLSFrequencyEstimator` vs `DFTFrequencyEstimator` standalone: `estimate()`/`estimate_full()`, windows, `f_min`, accuracy vs cost |
| **`0.5_odin-phasemeter-data.ipynb`** | Zipped Moku:Pro exports via `mokutools`: the `start_time` offset pitfall, channel choice, one capped window end to end |
| **`0.6_monte-carlo-crlb.ipynb`** | Statistical foundations: `CRLBCalculator` bounds, `MonteCarloAnalyzer` efficiency, and how these differ from plug-in uncertainty on real records |

Dated notebooks (`20260819_EDU_SegmentedDemod_Estimator_Demo.ipynb`,
`20260818_EDU_PreVibe_vs_PostVibe_RingDown.ipynb`, and the other ODIN/EDU files) are
investigation records rather than tutorials, and are linked from the notebooks above
where relevant.

## Key Results

- **NLS Method**: Achieves statistical efficiency, approaching the CRLB for ring-down signals when using the explicit ring-down model
- **DFT Method**: Provides computationally efficient estimation with Lorentzian peak fitting, but suffers from statistical inefficiency due to discrete frequency sampling
- **Real-data Q**: Coherent profile Q is exact on short drift-free records but can be biased by factors of 2–4 under frequency drift; segmented demodulation (`Q_demod`) and the `Q_selected` classifier address plateau-dominated and drifting resonator records
- **Exponential Decay Impact**: The exponential amplitude decay reduces effective observation time and SNR, degrading estimation performance compared to constant-amplitude signals. The degradation depends on the ratio T/τ (observation time to decay time constant)
- **Scaling Relationships**: For slow decay (T ≪ τ), accuracy scales as T⁻³/². For rapid decay (T ≫ τ), accuracy is limited by τ and scales as τ⁻³/²

## Security

**File input**: Load only CSV and MAT files from trusted sources. MAT files use `struct_as_record=False` to reduce deserialization risks; for untrusted input, consider sandboxing or alternative loaders. CSV files via Pandas are generally safe for typical scientific data. By default, `RingDownDataLoader.load()` does not enforce a file-size cap; pass `max_file_size_bytes` to reject oversized files.

**Path handling**: `process_directory()` validates that the directory exists and rejects path traversal in the glob pattern (e.g., `../`). For production use with user-supplied paths (e.g., from a web form), validate and resolve paths to a trusted base directory before passing them to the API.

## Data Format

Experimental data files should be placed in the `data/` directory:

- **CSV files**: Moku:Lab Phasemeter format with time in column 1 and phase (cycles) in column 4. `%` comments and leading plain-text headers are skipped.
- **MAT files**: MATLAB format with `moku.data` as a non-empty 2D array containing time in column 1 and phase in column 4

See `docs/data_format.md` for the full specification (column indices, units, validation rules, optional file-size limit, edge cases).

## Dependencies

Core dependencies (automatically installed via `pip install ringdownanalysis`):
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- tqdm >= 4.60.0
- joblib >= 1.0.0
- pandas >= 1.3.0

For reproducible environments (examples, notebooks), install from pinned versions:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional dependencies:
- Jupyter >= 1.0.0 (for notebooks)
- pytest >= 7.0.0 (for testing)
- pytest-cov >= 4.0.0 (for coverage)
- pytest-benchmark >= 4.0.0 (for benchmarking)
- ruff >= 0.1.0 (for linting)

## API Documentation

Build the Sphinx API docs:

```bash
make -C docs/api html
```

Open `docs/api/_build/html/index.html` in a browser. Or install dev dependencies and run:

```bash
pip install -e ".[dev]"
cd docs/api && make html
```

## Testing

Run the test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=ringdownanalysis --cov-report=html
```

Real-data regression tests are opt-in (require local ODIN archives and `mokutools`):

```bash
RINGDOWN_REAL_DATA_TESTS=1 pytest -m real_data
```

## Benchmarking

The package includes a comprehensive benchmarking and profiling suite to measure performance and identify bottlenecks:

```bash
# Run benchmarks with medium workload
python benchmarks/run_benchmarks.py --size medium

# Before/after Q-estimation stack timings
python benchmarks/bench_qestimation.py

# Profile critical workloads
python benchmarks/run_profiling.py all --size medium
```

Or run benchmarks directly with pytest:

```bash
pytest benchmarks/benchmark_suite.py --benchmark-only
```

See `benchmarks/README.md` for detailed information on benchmarking and profiling workflows.

## Development

### CI/CD

GitHub Actions run on every push and pull request:

- **Lint**: Ruff check and format
- **Test**: pytest with coverage on Python 3.10, 3.11, 3.12
- **Typecheck**: mypy

Coverage is uploaded to [Codecov](https://codecov.io) (optional; add `CODECOV_TOKEN` secret for private repos).

### Releasing

To publish a new version to PyPI:

1. Update `version` in `pyproject.toml` and document the release in `CHANGELOG.md`
2. Create and push a tag (after merging to the default branch):

   ```bash
   git tag v1.2.0
   git push origin main
   git push origin v1.2.0
   ```

3. The release workflow builds and publishes to PyPI automatically.

**Setup**: Add `PYPI_API_TOKEN` as a repository secret (Settings → Secrets → Actions). Create a token at [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/).

## References

- S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*. Prentice Hall, 1993.
- D. C. Rife and R. R. Boorstyn, "Single tone parameter estimation from discrete-time observations," *IEEE Trans. Information Theory*, 1974.
