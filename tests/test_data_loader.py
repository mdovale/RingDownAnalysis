"""
Unit tests for RingDownDataLoader.
"""

import tempfile
from pathlib import Path

import pytest

from ringdownanalysis.data_loader import RingDownDataLoader


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
