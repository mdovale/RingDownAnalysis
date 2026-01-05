# Frequency and Quality Factor Estimation of Exponentially Decaying Sinusoids

This repository contains theoretical analysis, numerical simulations, and experimental data analysis for frequency estimation of ring-down signals. Ring-down signals are exponentially decaying sinusoids that arise from measurements of harmonic oscillators with quality factor Q, where the amplitude decays exponentially due to energy dissipation.

## Quickstart

### Installation

Since this package is not yet available on PyPI, install it from source:

```bash
# Clone the repository
git clone https://github.com/mdovale/RingDownAnalysis.git
cd RingDownAnalysis

# Install in editable mode
pip install -e .
```

For development with testing and linting tools:

```bash
pip install -e ".[dev]"
```

For examples and notebooks:

```bash
pip install -e ".[examples]"
```

Or install everything:

```bash
pip install -e ".[all]"
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
from ringdownanalysis import BatchRingDownAnalyzer
import pandas as pd

# Initialize batch analyzer
batch_analyzer = BatchRingDownAnalyzer()

# Process all files in data directory
results = batch_analyzer.process_directory("data", verbose=True)

# Get summary table
summary = batch_analyzer.get_summary_table()
df_summary = pd.DataFrame(summary['data'])
print(df_summary)
```

See `examples/usage_example.py` and `examples/batch_analysis_example.py` for more complete examples.

#### Configure Logging

The package includes comprehensive logging support. Configure logging for production use or debugging:

```python
from examples.logging_config_example import setup_production_logging
import logging

# Set up production logging (file + console)
setup_production_logging(log_dir='logs', log_level=logging.INFO)

# Now use the package - logs will be written to files
from ringdownanalysis import BatchRingDownAnalyzer
analyzer = BatchRingDownAnalyzer()
results = analyzer.process_directory("data")
```

See `examples/logging_config_example.py` for more logging configuration options.

## Overview

The project compares two complementary approaches for frequency estimation:

1. **Nonlinear Least Squares (NLS)** with explicit ring-down model
2. **Frequency-Domain Methods (DFT)** with Lorentzian peak fitting

Both methods are evaluated against the Cramér-Rao Lower Bound (CRLB) derived from the explicit Fisher information matrix for ring-down signals.

## Features

### Object-Oriented API

The package provides a modern object-oriented API:

- **`RingDownSignal`**: Generate synthetic ring-down signals with specified parameters
- **`FrequencyEstimator`**: Base class for frequency estimation methods
  - **`NLSFrequencyEstimator`**: Nonlinear least squares estimation
    - `estimate()`: Returns frequency only
    - `estimate_full()`: Returns `EstimationResult` with frequency, tau, and Q
  - **`DFTFrequencyEstimator`**: DFT-based estimation with Lorentzian fitting
    - `estimate()`: Returns frequency only
    - `estimate_full()`: Returns `EstimationResult` with frequency, tau (via NLS with fixed frequency), and Q
- **`EstimationResult`**: Named tuple containing (f, tau, Q) estimates
- **`CRLBCalculator`**: Calculate Cramér-Rao Lower Bound for frequency estimation
- **`RingDownAnalyzer`**: Analyze individual ring-down data files
- **`BatchRingDownAnalyzer`**: Batch process multiple data files
- **`MonteCarloAnalyzer`**: Run Monte Carlo simulations to compare methods

### Compatibility Layer

A function-based compatibility layer is also available for backward compatibility:

```python
from ringdownanalysis import (
    generate_ringdown,
    estimate_freq_nls_ringdown,
    estimate_freq_dft,
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

batch_analyzer = BatchRingDownAnalyzer()

# Process all files in data directory
results = batch_analyzer.process_directory("data", verbose=True, n_jobs=-1)

# Q factors are automatically calculated during analysis (via estimate_full())
# Access them directly from results or use calculate_q_factors() for statistics
batch_analyzer.calculate_q_factors()  # Ensures Q is in results dict
q_stats = batch_analyzer.get_q_factor_statistics()

# Get summary table
summary = batch_analyzer.get_summary_table()
df_summary = pd.DataFrame(summary['data'])

# Consistency analysis
consistency = batch_analyzer.consistency_analysis()

# CRLB comparison
crlb_analysis = batch_analyzer.crlb_comparison_analysis()
```

See `examples/batch_analysis_example.py` for a complete batch analysis example.

## Project Structure

### Core Package (`ringdownanalysis/`)

- **`signal.py`**: `RingDownSignal` class for signal generation
- **`estimators.py`**: Frequency estimation classes (NLS, DFT)
- **`crlb.py`**: CRLB calculation
- **`data_loader.py`**: Data loading utilities for CSV and MAT files
- **`analyzer.py`**: Single-file analysis
- **`batch_analyzer.py`**: Batch processing and analysis
- **`monte_carlo.py`**: Monte Carlo simulation framework
- **`compat.py`**: Compatibility layer (function-based API)

### Documentation (`docs/`)

- **`tn/main.tex`**: Comprehensive LaTeX document with theoretical foundation
- **`tn/main.pdf`**: Compiled technical note

### Examples (`examples/`)

- **`usage_example.py`**: Comprehensive usage examples for all features
- **`batch_analysis_example.py`**: Batch analysis workflow example
- **`benchmark.py`**: Simple performance benchmark comparing NLS and DFT methods
- **`logging_config_example.py`**: Examples for configuring logging in production and debugging

### Benchmarks (`benchmarks/`)

- **`benchmark_suite.py`**: Comprehensive pytest-benchmark test suite
- **`run_benchmarks.py`**: Script to run benchmarks and generate reports
- **`run_profiling.py`**: Script to profile workloads and identify bottlenecks
- **`profile_utils.py`**: cProfile utilities for profiling workloads
- **`README.md`**: Detailed guide for benchmarking and profiling

See `benchmarks/README.md` for detailed information on performance benchmarking and profiling.

### Notebooks (`notebooks/`)

- **`analysis_example.ipynb`**: Interactive analysis examples
- **`batch_analysis_example.ipynb`**: Batch analysis in notebook format

## Key Results

- **NLS Method**: Achieves statistical efficiency, approaching the CRLB for ring-down signals when using the explicit ring-down model
- **DFT Method**: Provides computationally efficient estimation with Lorentzian peak fitting, but suffers from statistical inefficiency due to discrete frequency sampling
- **Exponential Decay Impact**: The exponential amplitude decay reduces effective observation time and SNR, degrading estimation performance compared to constant-amplitude signals. The degradation depends on the ratio T/τ (observation time to decay time constant)
- **Scaling Relationships**: For slow decay (T ≪ τ), accuracy scales as T⁻³/². For rapid decay (T ≫ τ), accuracy is limited by τ and scales as τ⁻³/²

## Data Format

Experimental data files should be placed in the `data/` directory:

- **CSV files**: Moku:Lab Phasemeter format with time in column 1 and phase (cycles) in column 4
- **MAT files**: MATLAB format with `moku.data` structure containing time in column 1 and phase in column 4

## Dependencies

Core dependencies (automatically installed):
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- tqdm >= 4.60.0
- joblib >= 1.0.0
- pandas >= 1.3.0

Optional dependencies:
- Jupyter >= 1.0.0 (for notebooks)
- pytest >= 7.0.0 (for testing)
- pytest-cov >= 4.0.0 (for coverage)
- pytest-benchmark >= 4.0.0 (for benchmarking)
- ruff >= 0.1.0 (for linting)

## Testing

Run the test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=ringdownanalysis --cov-report=html
```

## Benchmarking

The package includes a comprehensive benchmarking and profiling suite to measure performance and identify bottlenecks:

```bash
# Run benchmarks with medium workload
python benchmarks/run_benchmarks.py --size medium

# Profile critical workloads
python benchmarks/run_profiling.py all --size medium
```

Or run benchmarks directly with pytest:

```bash
pytest benchmarks/benchmark_suite.py --benchmark-only
```

See `benchmarks/README.md` for detailed information on benchmarking and profiling workflows.

## References

- S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*. Prentice Hall, 1993.
- D. C. Rife and R. R. Boorstyn, "Single tone parameter estimation from discrete-time observations," *IEEE Trans. Information Theory*, 1974.
