# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-05-13

### Added

- First-stage Q reliability hardening for NLS and DFT estimates, including raw
  Q diagnostics, validity flags, status strings, reasons, tau-bound flags, crop
  diagnostics, and Q sensitivity records
- Profile-likelihood / variable-projection Q estimator with finite intervals,
  one-sided limits, and explicit validity/status fields
- `Q_profile` analyzer outputs, profile grids, profile confidence intervals, and
  profile limit fields
- Batch Q summaries that prefer valid profile Q values and count profile
  limit-only records separately
- Deterministic profile-Q example notebook and expanded batch notebook Q
  comparison cells
- EDU Day 2 ring-down analysis notebook
- Data analysis pipeline audit document
- Sphinx API documentation in `docs/api/`
- Data format specification in `docs/data_format.md`
- `requirements.txt` with pinned core dependencies for reproducibility
- Monte Carlo test coverage for failed-trial statistics

### Changed

- Documented profile-Q interpretation, raw NLS/DFT diagnostics, and one-sided
  limits in README
- Batch Q calculations now skip invalid or warning-status Q estimates by default
  and keep raw Q values available for diagnostic workflows
- Default analyzer crop policy now uses a three-tau crop to avoid treating a
  one-tau window as sufficient evidence for Q identifiability
- CI type checking now targets Python 3.10 compatibility, with related typing
  and lint cleanup across examples, benchmarks, compatibility helpers, data
  loading, Monte Carlo analysis, and tests
- Documented raised exceptions in `RingDownDataLoader`, estimators, and `analyze_file()`
- Added Security section to README (trusted-source assumption, path handling)
- Extended `.gitignore` for handoff documents, Jupyter exports, and LaTeX build
  outputs

### Fixed

- Bound-hit and crop-inflated NLS/DFT Q estimates are no longer exposed as
  valid user-facing Q values

## [0.1.0] - 2025-03-17

### Added

- **Signal generation**: `RingDownSignal` for synthetic ring-down signals
- **Frequency estimation**: `NLSFrequencyEstimator` and `DFTFrequencyEstimator` with `estimate()` and `estimate_full()`
- **CRLB calculation**: `CRLBCalculator` for Cramér-Rao lower bound
- **Data loading**: `RingDownDataLoader` for CSV and MAT (Moku:Lab Phasemeter format)
- **Analysis**: `RingDownAnalyzer` for single-file analysis, `BatchRingDownAnalyzer` for batch processing
- **ProcessResult**: Return type with `.results` and `.failed_files` for batch processing
- **Monte Carlo**: `MonteCarloAnalyzer` for method comparison
- **Compatibility layer**: Function-based API (`generate_ringdown`, `estimate_freq_nls_ringdown`, etc.)
- **Input validation**: Directory existence, path traversal rejection, file size limits, estimator input validation
- **Logging**: `configure_logging()` and structured logging support
- **CI/CD**: GitHub Actions for pytest, ruff, mypy, and release to PyPI
- **Tests**: Unit and integration tests for data loader, analyzer, batch processing, edge cases

### Security

- MAT files loaded with `struct_as_record=False` to reduce deserialization risks
- Path traversal rejected in `process_directory()` pattern

[Unreleased]: https://github.com/mdovale/RingDownAnalysis/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/mdovale/RingDownAnalysis/compare/v1.0.3...v1.1.0
[0.1.0]: https://github.com/mdovale/RingDownAnalysis/releases/tag/v0.1.0
