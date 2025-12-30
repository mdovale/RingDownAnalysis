#!/usr/bin/env python3
"""
Run benchmark suite and generate baseline performance report.

Usage:
    python benchmarks/run_benchmarks.py [--size small|medium|large] [--output output.json]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess


def run_benchmarks(size: str = 'medium', output: str = None, verbose: bool = True):
    """
    Run benchmark suite and collect results.
    
    Parameters:
    -----------
    size : str
        Workload size to benchmark
    output : str, optional
        Path to save JSON results
    verbose : bool
        Print benchmark output
    """
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    
    # Run pytest-benchmark
    cmd = [
        sys.executable, '-m', 'pytest',
        'benchmarks/benchmark_suite.py',
        '--benchmark-only',
        '--benchmark-json=-',  # Output to stdout
        '-v',
    ]
    
    # Filter by size if needed (we can add markers later)
    if verbose:
        print(f"Running benchmarks (size: {size})...")
        print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        
        if verbose:
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr, file=sys.stderr)
        
        # Parse JSON output
        try:
            benchmark_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract from output
            print("Warning: Could not parse JSON output. Benchmark may have failed.")
            if verbose:
                print(result.stdout)
            return None
        
        # Save results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            print(f"\nResults saved to: {output_path}")
        
        return benchmark_data
        
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed with return code {e.returncode}", file=sys.stderr)
        print(f"STDOUT: {e.stdout}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        return None


def generate_report(benchmark_data: dict, output_path: str = None):
    """
    Generate human-readable performance report from benchmark data.
    
    Parameters:
    -----------
    benchmark_data : dict
        Benchmark results from pytest-benchmark
    output_path : str, optional
        Path to save report
    """
    if not benchmark_data or 'benchmarks' not in benchmark_data:
        print("No benchmark data to report")
        return
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PERFORMANCE BENCHMARK REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    benchmarks = benchmark_data['benchmarks']
    
    # Group by test class
    by_class = {}
    for bench in benchmarks:
        test_name = bench['name']
        # Extract class name (format: TestClass::test_method)
        if '::' in test_name:
            class_name, method_name = test_name.split('::', 1)
        else:
            class_name = 'Other'
            method_name = test_name
        
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append((method_name, bench))
    
    # Report by class
    for class_name in sorted(by_class.keys()):
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append(f"{class_name}")
        report_lines.append("-" * 80)
        
        for method_name, bench in sorted(by_class[class_name]):
            mean_time = bench['stats']['mean']
            std_time = bench['stats']['stddev']
            min_time = bench['stats']['min']
            max_time = bench['stats']['max']
            rounds = bench['stats']['rounds']
            
            report_lines.append(f"\n  {method_name}")
            report_lines.append(f"    Mean:   {mean_time*1000:.4f} ms")
            report_lines.append(f"    Std:    {std_time*1000:.4f} ms")
            report_lines.append(f"    Min:    {min_time*1000:.4f} ms")
            report_lines.append(f"    Max:    {max_time*1000:.4f} ms")
            report_lines.append(f"    Rounds: {rounds}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_path}")
    else:
        print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks and generate reports"
    )
    parser.add_argument(
        '--size',
        type=str,
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Workload size (default: medium)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save JSON results'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Path to save text report'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress benchmark output'
    )
    
    args = parser.parse_args()
    
    # Default output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.output:
        args.output = f"benchmarks/results/benchmark_{args.size}_{timestamp}.json"
    if not args.report:
        args.report = f"benchmarks/results/report_{args.size}_{timestamp}.txt"
    
    # Run benchmarks
    benchmark_data = run_benchmarks(
        size=args.size,
        output=args.output,
        verbose=not args.quiet,
    )
    
    if benchmark_data:
        # Generate report
        generate_report(benchmark_data, output_path=args.report)
    else:
        print("Benchmark run failed. Check output above for errors.")
        sys.exit(1)


if __name__ == '__main__':
    main()

