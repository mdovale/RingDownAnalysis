# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Sphinx API documentation in `docs/api/`
- Data format specification in `docs/data_format.md`
- `requirements.txt` with pinned core dependencies for reproducibility

### Changed

- Documented raised exceptions in `RingDownDataLoader`, estimators, and `analyze_file()`
- Added Security section to README (trusted-source assumption, path handling)
- Extended `.gitignore` for Jupyter exports and LaTeX build outputs

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

[Unreleased]: https://github.com/mdovale/RingDownAnalysis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mdovale/RingDownAnalysis/releases/tag/v0.1.0
