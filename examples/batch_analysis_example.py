"""
Example usage of BatchRingDownAnalyzer for analyzing multiple ring-down data files.

This example demonstrates the canonical analysis workflow from the legacy notebook,
now formalized as a reusable API.
"""

from pathlib import Path

import pandas as pd

from ringdownanalysis import BatchRingDownAnalyzer


def main():
    """Run batch analysis example."""
    # Initialize batch analyzer
    batch_analyzer = BatchRingDownAnalyzer()

    # Process all files in data directory
    data_dir = Path("data")
    if data_dir.exists():
        print("Processing files in data directory...")
        # Use all available CPU cores for parallel processing
        # Set n_jobs=1 for sequential processing, or specify a number
        results = batch_analyzer.process_directory(str(data_dir), verbose=True, n_jobs=-1)

        if len(results) == 0:
            print("No data files found. Skipping analysis.")
            return

        # Calculate Q factors
        print("\nCalculating Q factors...")
        q_factors = batch_analyzer.calculate_q_factors()
        q_stats = batch_analyzer.get_q_factor_statistics()

        print("\nQ Factor Statistics:")
        print(f"  Mean Q: {q_stats['mean']:.2e}")
        print(f"  Std Q: {q_stats['std']:.2e}")
        print(f"  Min Q: {q_stats['min']:.2e}")
        print(f"  Max Q: {q_stats['max']:.2e}")
        print(f"  Range: {q_stats['range']:.2e}")

        # Create summary table
        print("\nGenerating summary table...")
        summary = batch_analyzer.get_summary_table()
        df_summary = pd.DataFrame(summary["data"])
        print("\nSummary of Frequency Estimation Results:")
        print("=" * 120)
        print(df_summary.to_string(index=False))

        # Consistency analysis
        print("\nPerforming consistency analysis...")
        consistency = batch_analyzer.consistency_analysis()

        print("\nConsistency Analysis Summary:")
        print(f"  Number of realizations: {consistency['n_realizations']}")
        print(f"  Number of pairwise comparisons: {consistency['n_pairwise_comparisons']}")
        print("\nNLS Method:")
        print(f"  Mean frequency: {consistency['nls_mean']:.9f} Hz")
        print(f"  Std across realizations: {consistency['nls_std_across_realizations']:.6e} Hz")
        print(f"  Coefficient of variation: {consistency['nls_cv']:.2e}")
        print(f"  Range: [{consistency['nls_range'][0]:.9f}, {consistency['nls_range'][1]:.9f}] Hz")
        print(f"  Span: {consistency['nls_span']:.6e} Hz")
        print("\nDFT Method:")
        print(f"  Mean frequency: {consistency['dft_mean']:.9f} Hz")
        print(f"  Std across realizations: {consistency['dft_std_across_realizations']:.6e} Hz")
        print(f"  Coefficient of variation: {consistency['dft_cv']:.2e}")
        print(f"  Range: [{consistency['dft_range'][0]:.9f}, {consistency['dft_range'][1]:.9f}] Hz")
        print(f"  Span: {consistency['dft_span']:.6e} Hz")

        # CRLB comparison
        print("\nPerforming CRLB comparison analysis...")
        crlb_analysis = batch_analyzer.crlb_comparison_analysis()

        print("\nCRLB Statistics:")
        print(f"  Mean CRLB std: {crlb_analysis['crlb_statistics']['mean']:.6e} Hz")
        print(f"  Min CRLB std: {crlb_analysis['crlb_statistics']['min']:.6e} Hz")
        print(f"  Max CRLB std: {crlb_analysis['crlb_statistics']['max']:.6e} Hz")

        if len(crlb_analysis["valid_ratios"]) > 0:
            print("\nRatio Statistics (|Δf| / CRLB):")
            print(f"  Mean ratio: {crlb_analysis['ratio_statistics']['mean']:.4f}")
            print(f"  Median ratio: {crlb_analysis['ratio_statistics']['median']:.4f}")
            print(f"  Max ratio: {crlb_analysis['ratio_statistics']['max']:.4f}")
            print(f"  Min ratio: {crlb_analysis['ratio_statistics']['min']:.4f}")

        # Consistency table
        print("\nGenerating consistency table...")
        consistency_table = batch_analyzer.get_consistency_table()
        df_consistency = pd.DataFrame(consistency_table["data"])
        print("\nFrequency Estimates and Deviations from Mean:")
        print("=" * 120)
        print(df_consistency.to_string(index=False))

        print("\nAnalysis complete!")
    else:
        print(f"Data directory '{data_dir}' not found.")
        print("This example requires data files in the 'data' directory.")


if __name__ == "__main__":
    main()
