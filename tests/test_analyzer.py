"""
Unit tests for RingDownAnalyzer.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ringdownanalysis.analyzer import RingDownAnalyzer


def _make_moku_mat(t: np.ndarray, phase: np.ndarray) -> dict:
    """Create moku.data structure for MAT file."""
    n = len(t)
    data = np.column_stack([t, np.zeros(n), np.zeros(n), phase])
    moku = np.empty((1, 1), dtype=[("data", object)])
    moku[0, 0]["data"] = data
    return {"moku": moku}


def _make_csv_content(t: np.ndarray, phase: np.ndarray) -> str:
    """Create Moku:Lab Phasemeter CSV format."""
    lines = ["% Comment\n"]
    for ti, pi in zip(t, phase):
        lines.append(f"{ti:.6f},0,0,{pi:.6f}\n")
    return "".join(lines)


class TestRingDownAnalyzer:
    """Test RingDownAnalyzer class."""

    def test_estimate_tau_returns_positive_finite(self, sample_ringdown_signal):
        """Test estimate_tau returns positive finite value."""
        t, data, fs = sample_ringdown_signal
        analyzer = RingDownAnalyzer()
        tau_est = analyzer.estimate_tau(data, t, fs)
        assert tau_est > 0
        assert np.isfinite(tau_est)

    def test_estimate_tau_with_initial_params(self, sample_ringdown_signal):
        """Test estimate_tau with provided initial_params."""
        t, data, fs = sample_ringdown_signal
        analyzer = RingDownAnalyzer()
        initial_params = (5.0, 0.0, 0.1, 0.0)  # f0, phi0, A0, c0
        tau_est = analyzer.estimate_tau(data, t, fs, initial_params=initial_params)
        assert tau_est > 0
        assert np.isfinite(tau_est)

    def test_crop_data_to_tau_crops_correctly(self, sample_ringdown_signal):
        """Test crop_data_to_tau crops to max_tau_multiplier * tau."""
        t, data, fs = sample_ringdown_signal
        analyzer = RingDownAnalyzer()
        tau_est = 0.2
        max_mult = 1.5
        t_crop, data_crop = analyzer.crop_data_to_tau(
            t, data, tau_est, min_samples=10, max_tau_multiplier=max_mult
        )
        t_max_expected = max_mult * tau_est
        assert np.all(t_crop <= t_max_expected + 1e-9)
        assert len(t_crop) <= len(t)
        assert len(data_crop) == len(t_crop)

    def test_crop_data_to_tau_returns_original_if_too_short(self):
        """Test crop_data_to_tau returns original when cropped length < min_samples."""
        t = np.linspace(0, 0.01, 50)
        data = np.cos(2 * np.pi * 5.0 * t)
        analyzer = RingDownAnalyzer()
        tau_est = 0.001  # Very small tau -> very short crop
        t_crop, data_crop = analyzer.crop_data_to_tau(
            t, data, tau_est, min_samples=100, max_tau_multiplier=1.0
        )
        assert len(t_crop) == len(t)
        assert np.allclose(t_crop, t)
        assert np.allclose(data_crop, data)

    def test_estimate_noise_parameters_returns_finite(self, sample_ringdown_signal):
        """Test estimate_noise_parameters returns finite A0 and sigma."""
        t, data, fs = sample_ringdown_signal
        analyzer = RingDownAnalyzer()
        tau_est = analyzer.estimate_tau(data, t, fs)
        t_crop, data_crop = analyzer.crop_data_to_tau(t, data, tau_est, min_samples=100)
        A0_est, sigma_est = analyzer.estimate_noise_parameters(data_crop, t_crop, tau_est, fs)
        assert np.isfinite(A0_est)
        assert np.isfinite(sigma_est)
        assert A0_est > 0
        assert sigma_est >= 0

    def test_analyze_file_csv_valid(self, tmp_csv_file):
        """Test analyze_file on valid CSV file."""
        analyzer = RingDownAnalyzer()
        result = analyzer.analyze_file(tmp_csv_file)
        assert "filename" in result
        assert "type" in result
        assert result["type"] == "CSV"
        assert "fs" in result
        assert "tau_est" in result
        assert "f_nls" in result
        assert "f_dft" in result
        assert "crlb_std_f" in result
        assert "N" in result
        assert "N_crop" in result
        assert np.isfinite(result["f_nls"])
        assert np.isfinite(result["f_dft"])

    def test_analyze_file_mat_valid(self, tmp_mat_file):
        """Test analyze_file on valid MAT file."""
        analyzer = RingDownAnalyzer()
        result = analyzer.analyze_file(tmp_mat_file)
        assert result["type"] == "MAT"
        assert np.isfinite(result["f_nls"])
        assert np.isfinite(result["f_dft"])

    def test_analyze_file_nonexistent_raises(self):
        """Test analyze_file raises FileNotFoundError for non-existent path."""
        analyzer = RingDownAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.analyze_file("/nonexistent/path/ringdown.csv")

    def test_analyze_file_unsupported_format_raises(self):
        """Test analyze_file raises ValueError for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("data")
            filepath = f.name
        try:
            analyzer = RingDownAnalyzer()
            with pytest.raises(ValueError, match="Unsupported"):
                analyzer.analyze_file(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_analyze_file_max_tau_multiplier(self, tmp_csv_file):
        """Test analyze_file accepts max_tau_multiplier and produces valid results."""
        analyzer = RingDownAnalyzer()
        result_1x = analyzer.analyze_file(tmp_csv_file, max_tau_multiplier=1.0)
        result_2x = analyzer.analyze_file(tmp_csv_file, max_tau_multiplier=2.0)
        # Both runs should produce valid results
        for r in (result_1x, result_2x):
            assert np.isfinite(r["f_nls"])
            assert np.isfinite(r["f_dft"])
            assert r["N_crop"] > 0
            assert r["tau_est"] > 0

    def test_analyze_array_numpy_t_data(self, sample_ringdown_signal):
        """Test analyze_array with numpy t and data arrays."""
        t, data, fs = sample_ringdown_signal
        analyzer = RingDownAnalyzer()
        result = analyzer.analyze_array(t=t, data=data)
        assert "fs" in result
        assert "tau_est" in result
        assert "f_nls" in result
        assert "f_dft" in result
        assert "crlb_std_f" in result
        assert "filename" not in result
        assert np.isfinite(result["f_nls"])
        assert np.isfinite(result["f_dft"])
        assert result["N"] == len(t)

    def test_analyze_array_data_fs(self, sample_ringdown_signal):
        """Test analyze_array with data and fs (t inferred)."""
        t, data, fs = sample_ringdown_signal
        analyzer = RingDownAnalyzer()
        result = analyzer.analyze_array(data=data, fs=fs)
        assert np.isfinite(result["f_nls"])
        assert result["N"] == len(data)

    def test_analyze_array_pandas_dataframe(self, sample_ringdown_signal):
        """Test analyze_array with pandas DataFrame."""
        t, data, fs = sample_ringdown_signal
        df = pd.DataFrame({"time": t, "phase": data})
        analyzer = RingDownAnalyzer()
        result = analyzer.analyze_array(data=df, time_col="time", data_col="phase")
        assert np.isfinite(result["f_nls"])
        assert result["N"] == len(df)

    def test_analyze_array_pandas_series(self, sample_ringdown_signal):
        """Test analyze_array with pandas Series for data, t as array."""
        t, data, fs = sample_ringdown_signal
        series = pd.Series(data)
        analyzer = RingDownAnalyzer()
        result = analyzer.analyze_array(t=t, data=series)
        assert np.isfinite(result["f_nls"])

    def test_analyze_array_no_data_raises(self):
        """Test analyze_array raises when data is None."""
        analyzer = RingDownAnalyzer()
        with pytest.raises(ValueError, match="data is required"):
            analyzer.analyze_array(t=np.linspace(0, 1, 100))

    def test_analyze_array_no_t_or_fs_raises(self):
        """Test analyze_array raises when neither t nor fs provided."""
        analyzer = RingDownAnalyzer()
        data = np.cos(np.linspace(0, 10 * np.pi, 1000))
        with pytest.raises(ValueError, match="Either t or fs must be provided"):
            analyzer.analyze_array(data=data)

    def test_analyze_array_mismatched_lengths_raises(self):
        """Test analyze_array raises when t and data lengths differ."""
        analyzer = RingDownAnalyzer()
        t = np.linspace(0, 1, 100)
        data = np.cos(np.linspace(0, 10 * np.pi, 50))
        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze_array(t=t, data=data)


class TestAnalyzerEdgeCases:
    """Edge case tests for RingDownAnalyzer."""

    def test_analyze_file_single_sample_raises(self):
        """Test that single-sample or very short signal raises or fails gracefully."""
        t = np.array([0.0])
        phase = np.array([0.1])
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write(_make_csv_content(t, phase))
            filepath = f.name
        try:
            analyzer = RingDownAnalyzer()
            with pytest.raises(
                (ValueError, ZeroDivisionError, IndexError, FloatingPointError, RuntimeWarning)
            ):
                analyzer.analyze_file(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_analyze_file_constant_time_raises(self):
        """Test that constant time array (np.diff zeros -> fs undefined) raises."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("0.0,0,0,0.1\n0.0,0,0,0.1\n0.0,0,0,0.1\n")
            filepath = f.name
        try:
            analyzer = RingDownAnalyzer()
            with pytest.raises(
                (
                    ValueError,
                    ZeroDivisionError,
                    FloatingPointError,
                    RuntimeError,
                    RuntimeWarning,
                )
            ):
                analyzer.analyze_file(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_analyze_file_very_short_signal_fails_or_raises(self):
        """Test very short signal (len < min_samples) completes without KeyError from logging."""
        t = np.linspace(0, 0.01, 50)
        phase = 0.1 * np.cos(2 * np.pi * 5.0 * t)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write(_make_csv_content(t, phase))
            filepath = f.name
        try:
            analyzer = RingDownAnalyzer()
            # Should not raise KeyError (logging reserves 'message'); may raise ValueError/IndexError or return
            result = analyzer.analyze_file(filepath)
            assert result is not None
        finally:
            Path(filepath).unlink(missing_ok=True)
