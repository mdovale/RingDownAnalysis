"""
Unit tests for BatchRingDownAnalyzer class.
"""

import numpy as np

from ringdownanalysis.analyzer import RingDownAnalyzer
from ringdownanalysis.batch_analyzer import BatchRingDownAnalyzer


class TestBatchRingDownAnalyzer:
    """Test BatchRingDownAnalyzer class."""

    def test_init_default(self):
        """Test initialization with default analyzer."""
        batch_analyzer = BatchRingDownAnalyzer()

        assert batch_analyzer.analyzer is not None
        assert isinstance(batch_analyzer.analyzer, RingDownAnalyzer)
        assert len(batch_analyzer.results) == 0

    def test_init_custom_analyzer(self):
        """Test initialization with custom analyzer."""
        analyzer = RingDownAnalyzer()
        batch_analyzer = BatchRingDownAnalyzer(analyzer=analyzer)

        assert batch_analyzer.analyzer is analyzer

    def test_calculate_q_factors(self):
        """Test Q factor calculation."""
        batch_analyzer = BatchRingDownAnalyzer()

        # Create mock results
        batch_analyzer.results = [
            {"f_nls": 5.0, "tau_est": 1000.0},
            {"f_nls": 5.0, "tau_est": 2000.0},
        ]

        q_factors = batch_analyzer.calculate_q_factors()

        assert len(q_factors) == 2
        assert abs(q_factors[0] - np.pi * 5.0 * 1000.0) < 1e-10
        assert abs(q_factors[1] - np.pi * 5.0 * 2000.0) < 1e-10
        assert "Q" in batch_analyzer.results[0]
        assert "Q" in batch_analyzer.results[1]

    def test_calculate_q_factors_empty(self):
        """Test Q factor calculation with empty results."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = []

        q_factors = batch_analyzer.calculate_q_factors()

        assert len(q_factors) == 0

    def test_get_summary_table(self):
        """Test summary table generation."""
        batch_analyzer = BatchRingDownAnalyzer()

        # Create mock results
        batch_analyzer.results = [
            {
                "filename": "test1.csv",
                "type": "CSV",
                "N": 1000,
                "N_crop": 800,
                "T": 10.0,
                "T_crop": 8.0,
                "fs": 100.0,
                "tau_est": 2.0,
                "f_nls": 5.0,
                "f_dft": 5.01,
                "A0_est": 1.0,
                "sigma_est": 0.1,
                "crlb_std_f": 1e-6,
            },
            {
                "filename": "test2.mat",
                "type": "MAT",
                "N": 2000,
                "N_crop": 1500,
                "T": 20.0,
                "T_crop": 15.0,
                "fs": 100.0,
                "tau_est": 3.0,
                "f_nls": 5.0,
                "f_dft": 5.02,
                "A0_est": 1.5,
                "sigma_est": 0.15,
                "crlb_std_f": 2e-6,
            },
        ]

        summary = batch_analyzer.get_summary_table()

        assert "data" in summary
        assert "columns" in summary
        assert len(summary["data"]) == 2
        assert len(summary["columns"]) > 0
        assert summary["data"][0]["Filename"] == "test1.csv"
        assert summary["data"][1]["Filename"] == "test2.mat"

    def test_get_summary_table_with_q(self):
        """Test summary table with Q factors."""
        batch_analyzer = BatchRingDownAnalyzer()

        batch_analyzer.results = [
            {
                "filename": "test1.csv",
                "type": "CSV",
                "N": 1000,
                "N_crop": 800,
                "T": 10.0,
                "T_crop": 8.0,
                "fs": 100.0,
                "tau_est": 2.0,
                "f_nls": 5.0,
                "f_dft": 5.01,
                "A0_est": 1.0,
                "sigma_est": 0.1,
                "crlb_std_f": 1e-6,
                "Q": 31415.93,
            },
        ]

        summary = batch_analyzer.get_summary_table()

        assert "Q" in summary["data"][0]

    def test_get_summary_table_empty(self):
        """Test summary table with empty results."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = []

        summary = batch_analyzer.get_summary_table()

        assert summary["data"] == []
        assert summary["columns"] == []

    def test_consistency_analysis(self):
        """Test consistency analysis."""
        batch_analyzer = BatchRingDownAnalyzer()

        # Create mock results with varying frequencies
        batch_analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.01},
            {"f_nls": 5.02, "f_dft": 5.03},
            {"f_nls": 5.01, "f_dft": 5.02},
        ]

        consistency = batch_analyzer.consistency_analysis()

        assert "n_realizations" in consistency
        assert consistency["n_realizations"] == 3
        assert "n_pairwise_comparisons" in consistency
        assert consistency["n_pairwise_comparisons"] == 3  # 3 choose 2 = 3
        assert "nls_pairwise_diffs" in consistency
        assert "dft_pairwise_diffs" in consistency
        assert len(consistency["nls_pairwise_diffs"]) == 3
        assert len(consistency["dft_pairwise_diffs"]) == 3
        assert "nls_statistics" in consistency
        assert "dft_statistics" in consistency
        assert "nls_mean" in consistency
        assert "dft_mean" in consistency
        assert "nls_std_across_realizations" in consistency
        assert "dft_std_across_realizations" in consistency

    def test_consistency_analysis_empty(self):
        """Test consistency analysis with empty results."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = []

        consistency = batch_analyzer.consistency_analysis()

        assert consistency == {}

    def test_consistency_analysis_single(self):
        """Test consistency analysis with single result."""
        batch_analyzer = BatchRingDownAnalyzer()

        batch_analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.01},
        ]

        consistency = batch_analyzer.consistency_analysis()

        assert consistency["n_realizations"] == 1
        assert consistency["n_pairwise_comparisons"] == 0
        assert len(consistency["nls_pairwise_diffs"]) == 0

    def test_crlb_comparison_analysis(self):
        """Test CRLB comparison analysis."""
        batch_analyzer = BatchRingDownAnalyzer()

        batch_analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.01, "crlb_std_f": 1e-6},
            {"f_nls": 5.0, "f_dft": 5.02, "crlb_std_f": 2e-6},
        ]

        crlb_analysis = batch_analyzer.crlb_comparison_analysis()

        assert "frequency_diffs" in crlb_analysis
        assert "crlb_stds" in crlb_analysis
        assert "ratios" in crlb_analysis
        assert len(crlb_analysis["frequency_diffs"]) == 2
        assert len(crlb_analysis["crlb_stds"]) == 2
        assert len(crlb_analysis["ratios"]) == 2
        assert "crlb_statistics" in crlb_analysis
        assert "ratio_statistics" in crlb_analysis

    def test_crlb_comparison_analysis_empty(self):
        """Test CRLB comparison with empty results."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = []

        crlb_analysis = batch_analyzer.crlb_comparison_analysis()

        assert crlb_analysis == {}

    def test_get_q_factor_statistics(self):
        """Test Q factor statistics."""
        batch_analyzer = BatchRingDownAnalyzer()

        batch_analyzer.results = [
            {"f_nls": 5.0, "tau_est": 1000.0},
            {"f_nls": 5.0, "tau_est": 2000.0},
            {"f_nls": 5.0, "tau_est": 1500.0},
        ]

        stats = batch_analyzer.get_q_factor_statistics()

        assert "values" in stats
        assert len(stats["values"]) == 3
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "range" in stats
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_get_q_factor_statistics_empty(self):
        """Test Q factor statistics with empty results."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = []

        stats = batch_analyzer.get_q_factor_statistics()

        assert stats == {}

    def test_get_consistency_table(self):
        """Test consistency table generation."""
        batch_analyzer = BatchRingDownAnalyzer()

        batch_analyzer.results = [
            {
                "filename": "test1.csv",
                "f_nls": 5.0,
                "f_dft": 5.01,
                "crlb_std_f": 1e-6,
            },
            {
                "filename": "test2.mat",
                "f_nls": 5.02,
                "f_dft": 5.03,
                "crlb_std_f": 2e-6,
            },
        ]

        table = batch_analyzer.get_consistency_table()

        assert "data" in table
        assert "columns" in table
        assert len(table["data"]) == 2
        assert "Index" in table["columns"]
        assert "Filename" in table["columns"]
        assert "f_NLS (Hz)" in table["columns"]
        assert "f_DFT (Hz)" in table["columns"]

    def test_get_consistency_table_empty(self):
        """Test consistency table with empty results."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = []

        table = batch_analyzer.get_consistency_table()

        assert table["data"] == []
        assert table["columns"] == []

    def test_process_files_empty_list(self):
        """Test processing empty file list."""
        batch_analyzer = BatchRingDownAnalyzer()

        results = batch_analyzer.process_files([], verbose=False)

        assert len(results) == 0
        assert len(batch_analyzer.results) == 0

    def test_consistency_analysis_statistics(self):
        """Test that consistency analysis computes correct statistics."""
        batch_analyzer = BatchRingDownAnalyzer()

        # Create results with known frequencies
        batch_analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.0},
            {"f_nls": 5.1, "f_dft": 5.1},
            {"f_nls": 5.2, "f_dft": 5.2},
        ]

        consistency = batch_analyzer.consistency_analysis()

        # Check pairwise differences
        nls_diffs = consistency["nls_pairwise_diffs"]
        assert len(nls_diffs) == 3  # 3 choose 2 = 3
        assert 0.1 in nls_diffs or abs(0.1 - nls_diffs[0]) < 1e-10
        assert 0.2 in nls_diffs or abs(0.2 - nls_diffs[1]) < 1e-10

        # Check statistics
        assert np.isclose(consistency["nls_mean"], 5.1)
        assert abs(consistency["nls_std_across_realizations"] - np.std([5.0, 5.1, 5.2])) < 1e-10

    def test_crlb_comparison_ratios(self):
        """Test CRLB comparison ratio calculation."""
        batch_analyzer = BatchRingDownAnalyzer()

        batch_analyzer.results = [
            {"f_nls": 5.0, "f_dft": 5.01, "crlb_std_f": 1e-6},  # ratio = 0.01 / 1e-6 = 10000
            {"f_nls": 5.0, "f_dft": 5.02, "crlb_std_f": 2e-6},  # ratio = 0.02 / 2e-6 = 10000
        ]

        crlb_analysis = batch_analyzer.crlb_comparison_analysis()

        ratios = crlb_analysis["ratios"]
        assert len(ratios) == 2
        assert np.isfinite(ratios[0])
        assert np.isfinite(ratios[1])
        assert ratios[0] > 0
        assert ratios[1] > 0
