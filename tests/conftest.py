"""
Pytest fixtures for ring-down analysis tests.
"""

import numpy as np
import pytest
from scipy.io import savemat


def _make_csv_content(t: np.ndarray, phase: np.ndarray) -> str:
    """Create Moku:Lab Phasemeter CSV format: time,col1,col2,phase."""
    lines = ["% Comment line\n"]
    for ti, pi in zip(t, phase):
        lines.append(f"{ti:.6f},0,0,{pi:.6f}\n")
    return "".join(lines)


def _make_moku_mat(t: np.ndarray, phase: np.ndarray, v2: np.ndarray | None = None) -> dict:
    """Create moku.data structure for MAT file. Columns: time, 0, 0, phase, ..., V2."""
    n = len(t)
    cols = [
        t,
        np.zeros(n),
        np.zeros(n),
        phase,
    ]
    if v2 is not None:
        cols.extend([np.zeros(n)] * 4 + [v2])
    data = np.column_stack(cols)
    moku = np.empty((1, 1), dtype=[("data", object)])
    moku[0, 0]["data"] = data
    return {"moku": moku}


@pytest.fixture
def fixture_csv_content():
    """Valid CSV content for Moku:Lab Phasemeter format."""
    t = np.linspace(0, 1.0, 2000)
    phase = 0.1 * np.exp(-t / 0.3) * np.cos(2 * np.pi * 5.0 * t)
    return _make_csv_content(t, phase)


@pytest.fixture
def fixture_mat_data():
    """Valid MAT data arrays (t, phase) for creating MAT files."""
    t = np.linspace(0, 1.0, 2000)
    phase = 0.1 * np.exp(-t / 0.3) * np.cos(2 * np.pi * 5.0 * t)
    return t, phase


@pytest.fixture
def tmp_csv_file(tmp_path, fixture_csv_content):
    """Create a temporary valid CSV file."""
    filepath = tmp_path / "ringdown.csv"
    filepath.write_text(fixture_csv_content)
    return str(filepath)


@pytest.fixture
def tmp_mat_file(tmp_path, fixture_mat_data):
    """Create a temporary valid MAT file."""
    t, phase = fixture_mat_data
    filepath = tmp_path / "ringdown.mat"
    savemat(str(filepath), _make_moku_mat(t, phase))
    return str(filepath)


@pytest.fixture
def tmp_mat_file_with_v2(tmp_path, fixture_mat_data):
    """Create a temporary MAT file with V2 column."""
    t, phase = fixture_mat_data
    v2 = phase * 0.5  # Second channel
    filepath = tmp_path / "ringdown_v2.mat"
    savemat(str(filepath), _make_moku_mat(t, phase, v2))
    return str(filepath)


@pytest.fixture
def sample_ringdown_signal():
    """Generate a minimal ring-down signal for analyzer tests (t, data, fs)."""
    fs = 1000.0
    N = 2000
    t = np.arange(N) / fs
    tau = 0.3
    f0 = 5.0
    A0 = 0.1
    data = A0 * np.exp(-t / tau) * np.cos(2 * np.pi * f0 * t)
    return t, data, fs
