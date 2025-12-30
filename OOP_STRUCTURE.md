# Object-Oriented Structure for Ring-Down Analysis

This document describes the new object-oriented structure introduced to improve cohesion and encapsulation in the ring-down analysis codebase.

## Overview

The codebase has been refactored from a function-based approach to an object-oriented design with clear class responsibilities and invariants. The legacy function-based API is preserved through a compatibility layer.

## Class Structure

### Core Classes

#### `RingDownSignal`
**Location:** `ringdownanalysis/signal.py`

Encapsulates ring-down signal parameters and generation.

**Responsibilities:**
- Store signal parameters (f0, fs, N, A0, snr_db, Q)
- Compute derived parameters (tau, sigma)
- Generate noisy ring-down signals

**Key Methods:**
- `__init__(f0, fs, N, A0=1.0, snr_db=60.0, Q=10000.0)`: Initialize signal parameters
- `generate(phi0=None, rng=None)`: Generate signal with optional phase and RNG

**Invariants:**
- All parameters (f0, fs, N, A0, Q) must be positive
- tau = Q / (π * f0)
- sigma computed from SNR

#### `FrequencyEstimator` (Abstract Base Class)
**Location:** `ringdownanalysis/estimators.py`

Base class for frequency estimation methods.

**Subclasses:**
- `NLSFrequencyEstimator`: Nonlinear least squares with ring-down model
- `DFTFrequencyEstimator`: DFT peak fitting with Lorentzian function

**Key Methods:**
- `estimate(x, fs, **kwargs)`: Estimate frequency from signal

**Invariants:**
- Returns frequency in Hz
- Frequency must be in valid range [0, fs/2]

#### `NLSFrequencyEstimator`
**Location:** `ringdownanalysis/estimators.py`

Frequency estimation using nonlinear least squares.

**Key Methods:**
- `__init__(tau_known=None)`: Initialize with optional known tau
- `estimate(x, fs)`: Estimate frequency

**Invariants:**
- Uses ring-down model: A(t) * cos(2πft + φ)
- Handles both known and unknown tau cases

#### `DFTFrequencyEstimator`
**Location:** `ringdownanalysis/estimators.py`

Frequency estimation using DFT with Lorentzian fitting.

**Key Methods:**
- `__init__(window="kaiser", use_zeropad=False, pad_factor=4, lorentzian_points=7, kaiser_beta=9.0)`: Initialize with window options
- `estimate(x, fs)`: Estimate frequency

**Invariants:**
- Uses Lorentzian fitting (appropriate for ring-down signals)
- Supports multiple window types (kaiser, hann, rect, blackman)

#### `CRLBCalculator`
**Location:** `ringdownanalysis/crlb.py`

Calculates Cramér-Rao Lower Bound for frequency estimation variance.

**Key Methods:**
- `variance(A0, sigma, fs, N, tau)`: Calculate CRLB variance
- `standard_deviation(A0, sigma, fs, N, tau)`: Calculate CRLB standard deviation

**Invariants:**
- All input parameters must be positive
- Returns variance in Hz² or std in Hz

### Data Loading

#### `RingDownDataLoader`
**Location:** `ringdownanalysis/data_loader.py`

Handles loading ring-down measurement data from files.

**Key Methods:**
- `load_csv(filepath)`: Load CSV file (Moku:Lab format)
- `load_mat(filepath)`: Load MAT file (Moku:Lab format)
- `load(filepath)`: Auto-detect format and load

**Invariants:**
- Returns time array starting from 0
- Returns detrended phase data

### Analysis Classes

#### `RingDownAnalyzer`
**Location:** `ringdownanalysis/analyzer.py`

Orchestrates analysis pipeline for real measurement data.

**Responsibilities:**
- Load data from files
- Estimate tau from full data
- Crop data to 3*tau
- Estimate frequency using NLS and DFT
- Estimate noise parameters
- Calculate CRLB

**Key Methods:**
- `__init__(nls_estimator=None, dft_estimator=None)`: Initialize with optional custom estimators
- `analyze_file(filepath)`: Complete analysis pipeline for a file
- `estimate_tau(data, t, fs)`: Estimate decay time constant
- `crop_data_to_tau(t, data, tau_est, min_samples=100)`: Crop data to 3*tau
- `estimate_noise_parameters(data_cropped, t_crop, tau_est, fs)`: Estimate A0 and sigma

**Invariants:**
- Returns comprehensive results dictionary
- Handles both CSV and MAT file formats

#### `MonteCarloAnalyzer`
**Location:** `ringdownanalysis/monte_carlo.py`

Performs Monte Carlo analysis comparing estimation methods.

**Responsibilities:**
- Generate multiple signal realizations
- Estimate frequency for each realization
- Compute statistics (mean, std, RMSE)
- Compare against CRLB

**Key Methods:**
- `__init__(nls_estimator=None, dft_estimator=None)`: Initialize with optional custom estimators
- `run(f0, fs, N, A0, snr_db, Q, n_mc, seed, n_workers=None, timeout_per_trial=30.0)`: Run Monte Carlo analysis

**Invariants:**
- Supports parallel processing
- Returns statistics for both methods

## Compatibility Layer

**Location:** `ringdownanalysis/compat.py`

Provides function-based API wrappers for backward compatibility.

**Functions:**
- `db_to_lin(x_db)`: Convert dB to linear scale
- `crlb_var_f_ringdown_explicit(...)`: CRLB calculation
- `generate_ringdown(...)`: Signal generation
- `estimate_freq_nls_ringdown(...)`: NLS frequency estimation
- `estimate_freq_dft(...)`: DFT frequency estimation
- `monte_carlo_analysis(...)`: Monte Carlo analysis

All legacy functions are preserved and work exactly as before.

## Usage Examples

### OOP API

```python
from ringdownanalysis import (
    RingDownSignal,
    NLSFrequencyEstimator,
    DFTFrequencyEstimator,
    CRLBCalculator,
    RingDownAnalyzer,
    MonteCarloAnalyzer,
)

# Generate signal
signal = RingDownSignal(f0=5.0, fs=100.0, N=10000, A0=1.0, snr_db=60.0, Q=10000.0)
t, x, phi0 = signal.generate()

# Estimate frequency
nls_est = NLSFrequencyEstimator()
dft_est = DFTFrequencyEstimator(window="kaiser")
f_nls = nls_est.estimate(x, signal.fs)
f_dft = dft_est.estimate(x, signal.fs)

# Calculate CRLB
crlb_calc = CRLBCalculator()
crlb_std = crlb_calc.standard_deviation(signal.A0, signal.sigma, signal.fs, signal.N, signal.tau)

# Analyze real data
analyzer = RingDownAnalyzer()
results = analyzer.analyze_file("data/measurement.csv")

# Monte Carlo analysis
mc_analyzer = MonteCarloAnalyzer()
results = mc_analyzer.run(f0=5.0, fs=100.0, N=10000, n_mc=100)
```

### Compatibility API

```python
from ringdownanalysis import (
    generate_ringdown,
    estimate_freq_nls_ringdown,
    estimate_freq_dft,
    monte_carlo_analysis,
)

# Works exactly as before
t, x, sigma, phi0, tau = generate_ringdown(f0=5.0, fs=100.0, N=10000)
f_nls = estimate_freq_nls_ringdown(x, 100.0)
f_dft = estimate_freq_dft(x, 100.0)
```

## Unit Tests

Comprehensive unit tests are provided in the `tests/` directory:

- `test_signal.py`: Tests for `RingDownSignal`
- `test_estimators.py`: Tests for frequency estimators
- `test_crlb.py`: Tests for `CRLBCalculator`
- `test_compat.py`: Tests for compatibility layer

Run tests with:
```bash
pytest tests/
```

## Benefits of OOP Structure

1. **Encapsulation**: Related state and behavior are grouped together
2. **Single Responsibility**: Each class has a clear, focused purpose
3. **Reusability**: Classes can be easily composed and extended
4. **Testability**: Classes can be tested in isolation
5. **Maintainability**: Clear structure makes code easier to understand and modify
6. **Backward Compatibility**: Legacy API preserved through compatibility layer

## Migration Guide

### For New Code

Use the OOP API for better structure and flexibility:

```python
# Old way (still works)
from ringdownanalysis import generate_ringdown, estimate_freq_nls_ringdown
t, x, _, _, _ = generate_ringdown(...)
f = estimate_freq_nls_ringdown(x, fs)

# New way (recommended)
from ringdownanalysis import RingDownSignal, NLSFrequencyEstimator
signal = RingDownSignal(...)
t, x, _ = signal.generate()
estimator = NLSFrequencyEstimator()
f = estimator.estimate(x, signal.fs)
```

### For Existing Code

No changes needed! The compatibility layer ensures all existing code continues to work.

## File Organization

```
ringdownanalysis/
├── __init__.py          # Package exports
├── signal.py            # RingDownSignal class
├── estimators.py        # FrequencyEstimator classes
├── crlb.py              # CRLBCalculator class
├── data_loader.py       # RingDownDataLoader class
├── analyzer.py          # RingDownAnalyzer class
├── monte_carlo.py       # MonteCarloAnalyzer class
├── compat.py            # Compatibility layer
└── legacy_ring_down_mc.py  # Original file (untouched)

tests/
├── __init__.py
├── test_signal.py
├── test_estimators.py
├── test_crlb.py
└── test_compat.py

examples/
├── usage_example.py     # Usage examples
└── analysis_example.ipynb  # Notebook example
```

## Design Principles

1. **Single Responsibility**: Each class has one clear purpose
2. **Open/Closed**: Classes are open for extension, closed for modification
3. **Dependency Inversion**: High-level modules depend on abstractions (FrequencyEstimator)
4. **Interface Segregation**: Small, focused interfaces
5. **Don't Repeat Yourself**: Shared functionality extracted to base classes/utilities

## Future Enhancements

Potential extensions:
- Additional frequency estimation methods (e.g., MUSIC, ESPRIT)
- Batch processing capabilities
- Visualization methods as class methods
- Configuration management for analysis parameters
- Result serialization/deserialization

