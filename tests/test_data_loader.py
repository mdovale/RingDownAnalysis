"""
Unit tests for RingDownDataLoader.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from ringdownanalysis.data_loader import RingDownDataLoader


def _make_moku_mat(t: np.ndarray, phase: np.ndarray) -> dict:
    """Create moku.data structure for MAT file."""
    n = len(t)
    data = np.column_stack([t, np.zeros(n), np.zeros(n), phase])
    moku = np.empty((1, 1), dtype=[("data", object)])
    moku[0, 0]["data"] = data
    return {"moku": moku}


class TestRingDownDataLoader:
    """Test RingDownDataLoader class."""

    def test_load_csv_valid(self):
        """Test loading valid CSV file."""
        csv_content = """% Comment line
0.0,0,0,0.0
0.01,0,0,0.1
0.02,0,0,0.2
0.03,0,0,0.3
"""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write(csv_content)
            filepath = f.name

        try:
            t, data = RingDownDataLoader.load_csv(filepath)
            assert len(t) == 4
            assert len(data) == 4
            assert t[0] == 0.0
            assert t[-1] == 0.03
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_csv_file_size_limit_raises(self):
        """Test that file exceeding size limit raises ValueError."""
        csv_content = """% Comment
0.0,0,0,0.0
0.01,0,0,0.1
"""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write(csv_content)
            filepath = f.name

        try:
            # Set limit smaller than file
            with pytest.raises(ValueError, match="exceeds maximum"):
                RingDownDataLoader.load_csv(filepath, max_file_size_bytes=1)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_csv_size_limit_disabled(self):
        """Test that max_file_size_bytes=None disables size check."""
        csv_content = """% Comment
0.0,0,0,0.0
0.01,0,0,0.1
"""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write(csv_content)
            filepath = f.name

        try:
            t, data = RingDownDataLoader.load_csv(filepath, max_file_size_bytes=None)
            assert len(t) == 2
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_file_size_limit_raises(self):
        """Test that load() enforces file size limit."""
        csv_content = """% Comment
0.0,0,0,0.0
0.01,0,0,0.1
"""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write(csv_content)
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="exceeds maximum"):
                RingDownDataLoader.load(filepath, max_file_size_bytes=1)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_unsupported_format_raises(self):
        """Test that unsupported format raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("some data")
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                RingDownDataLoader.load(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_csv_empty_file_raises(self):
        """Test that empty CSV file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("")
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="No valid data"):
                RingDownDataLoader.load_csv(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_csv_comments_only_raises(self):
        """Test that CSV with only comments raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("% Comment 1\n% Comment 2\n")
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="No valid data"):
                RingDownDataLoader.load_csv(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_csv_wrong_column_structure_raises(self):
        """Test that CSV with fewer than 4 columns raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("0.0,0.1\n0.01,0.2\n")  # Only 2 columns
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="No valid data|usecols"):
                RingDownDataLoader.load_csv(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_csv_malformed_numeric_raises(self):
        """Test that CSV with non-numeric data in first column raises."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("abc,0,0,0.0\n0.01,0,0,0.1\n")
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="No valid data|could not convert"):
                RingDownDataLoader.load_csv(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_mat_valid(self):
        """Test loading valid MAT file with moku.data structure."""
        t = np.linspace(0, 0.1, 100)
        phase = 0.1 * np.cos(2 * np.pi * 5.0 * t)
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            filepath = f.name
        try:
            savemat(filepath, _make_moku_mat(t, phase))
            t_out, data_out, V2_out = RingDownDataLoader.load_mat(filepath)
            assert len(t_out) == 100
            assert len(data_out) == 100
            assert t_out[0] == 0.0
            assert V2_out is None
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_mat_with_v2(self):
        """Test loading MAT file with V2 column."""
        t = np.linspace(0, 0.1, 100)
        phase = 0.1 * np.cos(2 * np.pi * 5.0 * t)
        v2 = phase * 0.5
        n = len(t)
        data = np.column_stack(
            [
                t,
                np.zeros(n),
                np.zeros(n),
                phase,
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                v2,
            ]
        )
        moku = np.empty((1, 1), dtype=[("data", object)])
        moku[0, 0]["data"] = data
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            filepath = f.name
        try:
            savemat(filepath, {"moku": moku})
            t_out, data_out, V2_out = RingDownDataLoader.load_mat(filepath)
            assert V2_out is not None
            assert len(V2_out) == 100
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_mat_invalid_structure_raises(self):
        """Test that MAT file with wrong structure raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            filepath = f.name
        try:
            savemat(filepath, {"other_key": np.array([1, 2, 3])})
            with pytest.raises(ValueError, match="Invalid MAT file structure"):
                RingDownDataLoader.load_mat(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_csv_via_load(self):
        """Test load() dispatches correctly to load_csv for .csv files."""
        csv_content = """% Comment
0.0,0,0,0.0
0.01,0,0,0.1
0.02,0,0,0.2
"""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write(csv_content)
            filepath = f.name
        try:
            t, data, V2, file_type = RingDownDataLoader.load(filepath, max_file_size_bytes=None)
            assert file_type == "CSV"
            assert V2 is None
            assert len(t) == 3
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_mat_via_load(self):
        """Test load() dispatches correctly to load_mat for .mat files."""
        t = np.linspace(0, 0.05, 50)
        phase = 0.1 * np.cos(2 * np.pi * 5.0 * t)
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            filepath = f.name
        try:
            savemat(filepath, _make_moku_mat(t, phase))
            t_out, data_out, V2_out, file_type = RingDownDataLoader.load(
                filepath, max_file_size_bytes=None
            )
            assert file_type == "MAT"
            assert len(t_out) == 50
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_nonexistent_file_raises(self):
        """Test that load() raises FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            RingDownDataLoader.load_csv("/nonexistent/path/file.csv")

    def test_load_mat_nonexistent_file_raises(self):
        """Test that load_mat() raises for non-existent path."""
        with pytest.raises(FileNotFoundError):
            RingDownDataLoader.load_mat("/nonexistent/path/file.mat")
