# Performance Benchmarking and Profiling Suite

This directory contains tools for establishing performance baselines, identifying bottlenecks, and measuring optimization improvements.

## Quick Start

### 1. Install Dependencies

```bash
pip install pytest-benchmark
```

Or install all dev dependencies:
```bash
pip install -e ".[dev]"
```

### 2. Run Benchmarks

```bash
# Run all benchmarks (medium size)
python benchmarks/run_benchmarks.py --size medium

# Run with custom output
python benchmarks/run_benchmarks.py --size large --output my_results.json --report my_report.txt
```

### 3. Profile Critical Workloads

```bash
# Profile single file analysis
python benchmarks/run_profiling.py single_file --size medium

# Profile all workloads
python benchmarks/run_profiling.py all --size medium
```

### 4. View Results

- Benchmark results: `benchmarks/results/benchmark_*.json`
- Benchmark reports: `benchmarks/results/report_*.txt`
- Profile stats: `benchmarks/results/profile_*.prof`
- Bottleneck summaries: `benchmarks/results/bottleneck_summary_*.txt`

## Files

- **`benchmark_suite.py`**: Comprehensive pytest-benchmark test suite
- **`run_benchmarks.py`**: Script to run benchmarks and generate reports
- **`profile_utils.py`**: cProfile utilities for profiling workloads
- **`run_profiling.py`**: Script to profile workloads and identify bottlenecks
- **`pyspy_guide.md`**: Guide for using py-spy for production profiling
- **`PERFORMANCE_BASELINE.md`**: Baseline analysis and optimization recommendations

## Workload Sizes

The benchmark suite defines four workload sizes:

- **small**: 10K samples, 5 files, 10 MC trials
- **medium**: 100K samples, 20 files, 50 MC trials
- **large**: 1M samples, 50 files, 100 MC trials
- **xlarge**: 10M samples, 100 files, 500 MC trials (stress test)

## Critical Workloads

1. **Signal Generation**: Creating ring-down signals
2. **Frequency Estimation**: NLS and DFT methods
3. **Initial Parameter Estimation**: DFT-based initial guesses
4. **CRLB Calculation**: Cramér-Rao lower bound computation
5. **Full Analysis Pipeline**: Complete single-file analysis
6. **Batch Operations**: Consistency analysis, CRLB comparison, Q factors
7. **Monte Carlo**: Multiple trials with statistics

## Running with pytest

You can also run benchmarks directly with pytest:

```bash
# Run all benchmarks
pytest benchmarks/benchmark_suite.py --benchmark-only

# Run specific test class
pytest benchmarks/benchmark_suite.py::TestFrequencyEstimation --benchmark-only

# Run with comparison (if you have baseline)
pytest benchmarks/benchmark_suite.py --benchmark-only --benchmark-compare
```

## Profiling Workflow

### 1. Establish Baseline

```bash
# Run benchmarks
python benchmarks/run_benchmarks.py --size medium

# Profile workloads
python benchmarks/run_profiling.py all --size medium
```

### 2. Analyze Bottlenecks

Review the bottleneck summaries in `benchmarks/results/bottleneck_summary_*.txt` to identify:
- Functions taking the most time
- Number of calls
- Time per call
- Categorization (scipy, numpy, ringdown, other)

### 3. Use py-spy for Production Profiling

See `pyspy_guide.md` for detailed instructions on using py-spy for profiling running processes.

### 4. Compare After Optimizations

```bash
# Re-run benchmarks
python benchmarks/run_benchmarks.py --size medium --output after_optimization.json

# Compare with baseline
pytest benchmarks/benchmark_suite.py --benchmark-only --benchmark-compare=baseline.json
```

## Interpreting Results

### Benchmark Output

- **Mean**: Average execution time
- **Std**: Standard deviation (variability)
- **Min/Max**: Best/worst case times
- **Rounds**: Number of benchmark iterations

### Profile Output

- **Cumulative time**: Total time including subcalls
- **Total time**: Time in function itself (excluding subcalls)
- **Calls**: Number of function invocations
- **Time per call**: Average time per invocation

### Bottleneck Analysis

Bottlenecks are ranked by cumulative time. Focus on:
1. Functions with high cumulative time (called many times or slow)
2. Functions with high total time (slow themselves)
3. Functions in our code (ringdownanalysis) vs. dependencies

## Best Practices

1. **Run benchmarks on dedicated machine**: Avoid background processes affecting results
2. **Warm up**: First run may be slower due to JIT/caching
3. **Multiple runs**: Run benchmarks multiple times to account for variability
4. **Consistent environment**: Use same Python version, dependencies, and system state
5. **Document changes**: Record system info, Python version, dependency versions

## Troubleshooting

### Benchmark fails to run

- Ensure pytest-benchmark is installed: `pip install pytest-benchmark`
- Check Python version compatibility
- Verify all dependencies are installed

### Profiling shows no bottlenecks

- Increase `--min-time` threshold
- Profile longer-running workloads (use `--size large`)
- Check that profiling actually ran (look for `.prof` files)

### Results are inconsistent

- Run multiple times and average
- Ensure system is idle
- Check for background processes
- Consider using `py-spy` for more stable sampling-based profiling

## Integration with CI/CD

You can integrate benchmarks into CI/CD:

```yaml
# Example GitHub Actions
- name: Run benchmarks
  run: |
    pip install pytest-benchmark
    python benchmarks/run_benchmarks.py --size small --output ci_benchmark.json
    # Compare with baseline or fail if regression
```

## Next Steps

1. Run baseline benchmarks: `python benchmarks/run_benchmarks.py --size medium`
2. Profile workloads: `python benchmarks/run_profiling.py all --size medium`
3. Review `PERFORMANCE_BASELINE.md` for optimization recommendations
4. Implement optimizations and re-measure

