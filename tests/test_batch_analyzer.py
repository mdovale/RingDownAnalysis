"""
Unit tests for BatchRingDownAnalyzer class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ringdownanalysis.analyzer import RingDownAnalyzer
from ringdownanalysis.batch_analyzer import BatchRingDownAnalyzer, ProcessResult
from ringdownanalysis.estimators import NLSFrequencyEstimator

# Path to committed fixture files (relative to project root)
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


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

    def test_calculate_q_factors_skips_invalid_q_by_default(self):
        """Test Q factor calculation excludes invalid analyzer Q estimates by default."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_nls": 123.0,
                "Q_nls_raw": 123.0,
                "Q_nls_valid": True,
                "Q_nls_status": "valid",
            },
            {
                "f_nls": 5.0,
                "tau_est": 200.0,
                "Q_nls": None,
                "Q_nls_raw": 999.0,
                "Q_nls_valid": False,
                "Q_nls_status": "invalid",
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors()

        assert q_factors == [123.0]
        assert batch_analyzer.results[0]["Q"] == 123.0
        assert batch_analyzer.results[1]["Q"] is None
        assert batch_analyzer.results[1]["Q_valid"] is False

    def test_calculate_q_factors_can_include_invalid_raw_q(self):
        """Test raw invalid Q can still be requested for diagnostics."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 200.0,
                "Q_nls": None,
                "Q_nls_raw": 999.0,
                "Q_nls_valid": False,
                "Q_nls_status": "invalid",
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors(include_invalid=True)

        assert q_factors == [999.0]
        assert batch_analyzer.results[0]["Q"] == 999.0
        assert batch_analyzer.results[0]["Q_valid"] is False

    def test_calculate_q_factors_prefers_valid_profile_q(self):
        """Profile Q is the preferred aggregate Q when it is valid."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_nls": 123.0,
                "Q_nls_raw": 123.0,
                "Q_nls_valid": True,
                "Q_nls_status": "valid",
                "Q_profile": 456.0,
                "Q_profile_valid": True,
                "Q_profile_status": "valid",
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors()

        assert q_factors == [456.0]
        assert batch_analyzer.results[0]["Q"] == 456.0
        assert batch_analyzer.results[0]["Q_valid"] is True
        assert batch_analyzer.results[0]["Q_status"] == "valid"

    def test_calculate_q_factors_skips_invalid_profile_without_nls_fallback(self):
        """Invalid profile Q prevents silent fallback to otherwise valid NLS Q."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_nls": 123.0,
                "Q_nls_raw": 123.0,
                "Q_nls_valid": True,
                "Q_nls_status": "valid",
                "Q_profile": None,
                "Q_profile_valid": False,
                "Q_profile_status": "lower_limit",
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors()

        assert q_factors == []
        assert batch_analyzer.results[0]["Q"] is None
        assert batch_analyzer.results[0]["Q_valid"] is False
        assert batch_analyzer.results[0]["Q_status"] == "lower_limit"

    def test_calculate_q_factors_skips_mismatch_demoted_profile(self):
        """A finite but non-valid profile Q (envelope mismatch) is not batch-preferred."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_nls": 123.0,
                "Q_nls_raw": 123.0,
                "Q_nls_valid": True,
                "Q_nls_status": "valid",
                "Q_profile": None,
                "Q_profile_raw": 456.0,
                "Q_profile_valid": False,
                "Q_profile_status": "warning",
                "Q_profile_reasons": ["envelope_mismatch"],
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors()

        assert q_factors == []
        assert batch_analyzer.results[0]["Q"] is None
        assert batch_analyzer.results[0]["Q_valid"] is False
        assert batch_analyzer.results[0]["Q_status"] == "warning"

    def test_calculate_q_factors_never_returns_stale_profile_value_as_valid(self):
        """Even a finite Q_profile field is skipped when Q_profile_valid is False."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_profile": 456.0,
                "Q_profile_valid": False,
                "Q_profile_status": "warning",
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors()

        assert q_factors == []
        assert batch_analyzer.results[0]["Q"] is None

    def test_calculate_q_factors_include_invalid_uses_profile_raw(self):
        """include_invalid falls back to the raw profile value when present."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_nls_raw": 123.0,
                "Q_profile": None,
                "Q_profile_raw": 456.0,
                "Q_profile_valid": False,
                "Q_profile_status": "warning",
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors(include_invalid=True)

        assert q_factors == [456.0]
        assert batch_analyzer.results[0]["Q_valid"] is False

    def test_q_preference_demod_prefers_valid_demod_q(self):
        """q_preference='demod' returns the valid demod Q over a valid profile Q."""
        batch_analyzer = BatchRingDownAnalyzer(q_preference="demod")
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_profile": 456.0,
                "Q_profile_valid": True,
                "Q_profile_status": "valid",
                "Q_demod": 789.0,
                "Q_demod_valid": True,
                "Q_demod_status": "valid",
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors()

        assert q_factors == [789.0]
        assert batch_analyzer.results[0]["Q"] == 789.0
        assert batch_analyzer.results[0]["Q_valid"] is True

    def test_q_preference_demod_falls_back_to_valid_profile(self):
        """When the demod Q is invalid, a valid profile Q is still used."""
        batch_analyzer = BatchRingDownAnalyzer(q_preference="demod")
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_profile": 456.0,
                "Q_profile_valid": True,
                "Q_profile_status": "valid",
                "Q_demod": None,
                "Q_demod_valid": False,
                "Q_demod_status": "invalid",
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors()

        assert q_factors == [456.0]
        assert batch_analyzer.results[0]["Q_valid"] is True

    def test_q_preference_demod_skips_warning_demod_by_default(self):
        """A finite but non-valid demod Q is not batch-preferred by default."""
        batch_analyzer = BatchRingDownAnalyzer(q_preference="demod")
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_profile": None,
                "Q_profile_valid": False,
                "Q_profile_status": "warning",
                "Q_demod": 789.0,
                "Q_demod_valid": False,
                "Q_demod_status": "warning",
            },
        ]

        q_factors = batch_analyzer.calculate_q_factors()

        assert q_factors == []
        assert batch_analyzer.results[0]["Q"] is None

        q_factors = batch_analyzer.calculate_q_factors(include_invalid=True)
        assert q_factors == [789.0]
        assert batch_analyzer.results[0]["Q_valid"] is False

    def test_q_preference_per_call_override(self):
        """calculate_q_factors(q_preference=...) overrides the constructor setting."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_profile": 456.0,
                "Q_profile_valid": True,
                "Q_profile_status": "valid",
                "Q_demod": 789.0,
                "Q_demod_valid": True,
                "Q_demod_status": "valid",
            },
        ]

        assert batch_analyzer.calculate_q_factors() == [456.0]
        assert batch_analyzer.calculate_q_factors(q_preference="demod") == [789.0]

    def test_q_preference_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="q_preference"):
            BatchRingDownAnalyzer(q_preference="banana")

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
        assert isinstance(summary["data"][0]["T (s)"], float)
        assert isinstance(summary["data"][0]["f_NLS (Hz)"], float)

        formatted = batch_analyzer.get_formatted_summary_table()
        assert isinstance(formatted["data"][0]["T (s)"], str)
        assert isinstance(formatted["data"][0]["f_NLS (Hz)"], str)

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
        assert stats["n_total"] == 3
        assert stats["n_valid"] == 3
        assert stats["n_skipped"] == 0

    def test_get_q_factor_statistics_empty(self):
        """Test Q factor statistics with empty results."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = []

        stats = batch_analyzer.get_q_factor_statistics()

        assert stats == {}

    def test_get_q_factor_statistics_reports_skipped_invalid_q(self):
        """Test Q statistics report invalid/skipped counts."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_nls": 123.0,
                "Q_nls_raw": 123.0,
                "Q_nls_valid": True,
                "Q_nls_status": "valid",
            },
            {
                "f_nls": 5.0,
                "tau_est": 200.0,
                "Q_nls": None,
                "Q_nls_raw": 999.0,
                "Q_nls_valid": False,
                "Q_nls_status": "invalid",
            },
        ]

        stats = batch_analyzer.get_q_factor_statistics()

        assert stats["values"].tolist() == [123.0]
        assert stats["n_total"] == 2
        assert stats["n_valid"] == 1
        assert stats["n_skipped"] == 1
        assert stats["n_invalid"] == 1
        assert stats["n_profile_limits"] == 0

    def test_get_q_factor_statistics_counts_profile_limits(self):
        """Q statistics report profile limit-only records separately."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [
            {
                "f_nls": 5.0,
                "tau_est": 100.0,
                "Q_nls": 123.0,
                "Q_nls_raw": 123.0,
                "Q_nls_valid": True,
                "Q_nls_status": "valid",
                "Q_profile": None,
                "Q_profile_valid": False,
                "Q_profile_status": "lower_limit",
            },
        ]

        stats = batch_analyzer.get_q_factor_statistics()

        assert stats["n_total"] == 1
        assert stats["n_valid"] == 0
        assert stats["n_skipped"] == 1
        assert stats["n_invalid"] == 0
        assert stats["n_profile_limits"] == 1

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
        assert isinstance(table["data"][0]["f_NLS (Hz)"], float)

        formatted = batch_analyzer.get_formatted_consistency_table()
        assert isinstance(formatted["data"][0]["f_NLS (Hz)"], str)

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

        result = batch_analyzer.process_files([], verbose=False)

        assert isinstance(result, ProcessResult)
        assert len(result) == 0
        assert len(result.failed_files) == 0
        assert len(batch_analyzer.results) == 0

    def test_process_directory_nonexistent_raises(self):
        """Test that process_directory raises FileNotFoundError for non-existent dir."""
        batch_analyzer = BatchRingDownAnalyzer()

        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            batch_analyzer.process_directory("/nonexistent/path/12345", verbose=False)

    def test_process_directory_file_path_raises(self):
        """Test that process_directory raises NotADirectoryError for file path."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            batch_analyzer = BatchRingDownAnalyzer()
            with pytest.raises(NotADirectoryError, match="not a directory"):
                batch_analyzer.process_directory(filepath, verbose=False)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_process_directory_invalid_pattern_raises(self):
        """Test that process_directory raises ValueError for path traversal in pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_analyzer = BatchRingDownAnalyzer()
            with pytest.raises(ValueError, match="path traversal"):
                batch_analyzer.process_directory(tmpdir, pattern="../*", verbose=False)

    def test_process_directory_valid_empty_dir(self):
        """Test process_directory on valid empty directory returns empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_analyzer = BatchRingDownAnalyzer()
            result = batch_analyzer.process_directory(tmpdir, verbose=False)
            assert isinstance(result, ProcessResult)
            assert len(result) == 0
            assert len(result.failed_files) == 0

    def test_process_files_returns_failed_files(self):
        """Test that process_files returns failed file info when files fail."""
        batch_analyzer = BatchRingDownAnalyzer()

        result = batch_analyzer.process_files(
            ["/nonexistent/file1.csv", "/nonexistent/file2.mat"],
            verbose=False,
        )

        assert isinstance(result, ProcessResult)
        assert len(result) == 0
        assert len(result.failed_files) == 2
        assert result.has_failures()
        for filepath, exc in result.failed_files:
            assert "nonexistent" in filepath
            assert isinstance(exc, FileNotFoundError)

    def test_process_result_list_like(self):
        """Test ProcessResult list-like behavior for backward compatibility."""
        batch_analyzer = BatchRingDownAnalyzer()
        batch_analyzer.results = [{"f_nls": 5.0}, {"f_nls": 5.1}]
        result = ProcessResult(results=batch_analyzer.results, failed_files=[])

        assert len(result) == 2
        assert result[0]["f_nls"] == 5.0
        assert list(result) == batch_analyzer.results

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


class TestProcessFilesWithRealFiles:
    """Tests for process_files and process_directory with real/mock files."""

    def test_process_files_with_valid_csv(self, tmp_csv_file):
        """Test process_files with valid CSV file."""
        batch_analyzer = BatchRingDownAnalyzer()
        result = batch_analyzer.process_files([tmp_csv_file], verbose=False)
        assert len(result) == 1
        assert len(result.failed_files) == 0
        assert result[0]["type"] == "CSV"
        assert np.isfinite(result[0]["f_nls"])

    def test_process_files_with_valid_mat(self, tmp_mat_file):
        """Test process_files with valid MAT file."""
        batch_analyzer = BatchRingDownAnalyzer()
        result = batch_analyzer.process_files([tmp_mat_file], verbose=False)
        assert len(result) == 1
        assert len(result.failed_files) == 0
        assert result[0]["type"] == "MAT"

    def test_process_files_mixed_success_and_failure(self, tmp_csv_file):
        """Test process_files with mix of valid and invalid files."""
        batch_analyzer = BatchRingDownAnalyzer()
        result = batch_analyzer.process_files(
            [tmp_csv_file, "/nonexistent/file.csv", tmp_csv_file],
            verbose=False,
        )
        assert len(result) == 2  # Two successful
        assert len(result.failed_files) == 1
        assert result.has_failures()

    def test_parallel_process_files_preserves_custom_analyzer(self, tmp_csv_file):
        """Test parallel workers use the injected analyzer configuration."""
        analyzer = RingDownAnalyzer(nls_estimator=NLSFrequencyEstimator(tau_known=0.3))
        batch_analyzer = BatchRingDownAnalyzer(analyzer=analyzer)

        result = batch_analyzer.process_files([tmp_csv_file], verbose=False, n_jobs=2)

        assert len(result) == 1
        assert len(result.failed_files) == 0
        assert result[0]["tau_nls"] == pytest.approx(0.3)

    def test_process_directory_with_pattern(self, tmp_path, tmp_csv_file):
        """Test process_directory pattern matching."""
        # Create second file with different name
        csv2 = tmp_path / "other_ringdown.csv"
        csv2.write_text(Path(tmp_csv_file).read_text())
        batch_analyzer = BatchRingDownAnalyzer()
        result = batch_analyzer.process_directory(str(tmp_path), pattern="*", verbose=False)
        assert len(result) >= 1
        # Pattern "other_*" should match only other_ringdown.csv
        result_other = batch_analyzer.process_directory(
            str(tmp_path), pattern="other_*", verbose=False
        )
        assert len(result_other) == 1
        assert "other_ringdown" in result_other[0]["filename"]


@pytest.mark.integration
class TestIntegrationWithFixtures:
    """Integration tests using committed fixture data."""

    def test_full_pipeline_csv_fixture(self):
        """Integration test: load fixture CSV and run full analysis pipeline."""
        csv_path = FIXTURES_DIR / "sample_ringdown.csv"
        if not csv_path.exists():
            pytest.skip("Fixture sample_ringdown.csv not found")
        batch_analyzer = BatchRingDownAnalyzer()
        result = batch_analyzer.process_files([str(csv_path)], verbose=False)
        assert len(result) == 1
        r = result[0]
        assert r["type"] == "CSV"
        assert np.isfinite(r["f_nls"])
        assert np.isfinite(r["f_dft"])
        assert r["N"] >= 1000
        assert r["tau_est"] > 0

    def test_full_pipeline_mat_fixture(self):
        """Integration test: load fixture MAT and run full analysis pipeline."""
        mat_path = FIXTURES_DIR / "sample_ringdown.mat"
        if not mat_path.exists():
            pytest.skip("Fixture sample_ringdown.mat not found")
        batch_analyzer = BatchRingDownAnalyzer()
        result = batch_analyzer.process_files([str(mat_path)], verbose=False)
        assert len(result) == 1
        r = result[0]
        assert r["type"] == "MAT"
        assert np.isfinite(r["f_nls"])
        assert r["N"] >= 1000
