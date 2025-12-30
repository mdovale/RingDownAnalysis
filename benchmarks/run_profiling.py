#!/usr/bin/env python3
"""
Main profiling script for RingDownAnalysis.

Runs profiling on critical workloads and generates bottleneck reports.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import sys
from pathlib import Path

# Add benchmarks directory to path
benchmarks_dir = Path(__file__).parent
sys.path.insert(0, str(benchmarks_dir.parent))

from benchmarks.profile_utils import (
    profile_single_file_analysis,
    profile_frequency_estimation,
    profile_batch_analysis,
    profile_monte_carlo,
    analyze_bottlenecks,
    print_bottleneck_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Profile RingDownAnalysis workloads and identify bottlenecks"
    )
    parser.add_argument(
        'workload',
        choices=['single_file', 'frequency_est', 'batch', 'monte_carlo', 'all'],
        help='Workload to profile'
    )
    parser.add_argument(
        '--size',
        type=str,
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Workload size (default: medium)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='profiles',
        help='Directory to save profile stats (default: profiles)'
    )
    parser.add_argument(
        '--min-time',
        type=float,
        default=0.01,
        help='Minimum time (seconds) to consider a bottleneck (default: 0.01)'
    )
    parser.add_argument(
        '--filepath',
        type=str,
        help='Path to real data file (for single_file workload)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Size mappings
    size_map = {
        'small': {'N': 10_000, 'n_files': 5, 'n_mc': 10},
        'medium': {'N': 100_000, 'n_files': 20, 'n_mc': 50},
        'large': {'N': 1_000_000, 'n_files': 50, 'n_mc': 100},
    }
    sizes = size_map[args.size]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    workloads_to_run = []
    if args.workload == 'all':
        workloads_to_run = ['single_file', 'frequency_est', 'batch', 'monte_carlo']
    else:
        workloads_to_run = [args.workload]
    
    for workload in workloads_to_run:
        print(f"\n{'='*80}")
        print(f"Profiling: {workload} (size: {args.size})")
        print(f"{'='*80}\n")
        
        if workload == 'single_file':
            profiler = profile_single_file_analysis(
                filepath=args.filepath,
                N=sizes['N'],
                save_to=str(output_dir / f"profile_single_file_{args.size}_{timestamp}.prof"),
            )
        
        elif workload == 'frequency_est':
            profiler = profile_frequency_estimation(
                N=sizes['N'],
                method='both',
                save_to=str(output_dir / f"profile_frequency_est_{args.size}_{timestamp}.prof"),
            )
        
        elif workload == 'batch':
            profiler = profile_batch_analysis(
                n_files=sizes['n_files'],
                N_per_file=sizes['N'],
                save_to=str(output_dir / f"profile_batch_{args.size}_{timestamp}.prof"),
            )
        
        elif workload == 'monte_carlo':
            profiler = profile_monte_carlo(
                n_mc=sizes['n_mc'],
                N=sizes['N'],
                save_to=str(output_dir / f"profile_monte_carlo_{args.size}_{timestamp}.prof"),
            )
        
        # Analyze bottlenecks
        print("\nAnalyzing bottlenecks...")
        analysis = analyze_bottlenecks(profiler, min_time=args.min_time)
        print_bottleneck_report(analysis)
        
        # Save analysis summary
        summary_path = output_dir / f"bottleneck_summary_{workload}_{args.size}_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            print_bottleneck_report(analysis)
            sys.stdout = old_stdout
        print(f"\nSummary saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("Profiling complete!")
    print(f"Profile stats saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

