"""
Tests for the Q envelope overlay plot annotations.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ringdownanalysis.plots import plot_q_envelope_overlay


def _result_dict(tau_signal: float = 2.0, **q_fields) -> dict:
    fs = 1000.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    data = np.exp(-t / tau_signal) * np.cos(2.0 * np.pi * 5.0 * t)
    result = {"t": t, "data": data, "f_profile": 5.0, "f_nls": 5.0, "f_dft": 5.0}
    result.update(q_fields)
    return result


class TestOverlayMismatchAnnotation:
    """Overlay must flag candidate lines that disagree with the envelope."""

    def test_mismatching_candidate_is_dashed_and_flagged(self):
        """A 5x-off candidate Q is drawn dashed with a MISMATCH label."""
        # Signal envelope tau = 2 s; candidate claims tau = 0.4 s.
        q_bad = float(np.pi * 5.0 * 0.4)
        result = _result_dict(Q_profile=None, Q_profile_raw=q_bad)
        fig, ax = plt.subplots()
        try:
            plot_q_envelope_overlay(ax, result, q_source="profile")
            labels = [line.get_label() for line in ax.get_lines()]
            mismatch_labels = [lab for lab in labels if "MISMATCH" in lab]
            assert len(mismatch_labels) == 1
            assert "×" in mismatch_labels[0]
            factor = float(mismatch_labels[0].split("×")[1].rstrip(")"))
            assert factor == pytest.approx(5.0, rel=0.2)
            mismatch_lines = [
                line for line in ax.get_lines() if "MISMATCH" in str(line.get_label())
            ]
            assert mismatch_lines[0].get_linestyle() == "--"
        finally:
            plt.close(fig)

    def test_agreeing_candidate_is_not_flagged(self):
        """A candidate Q matching the envelope keeps the solid endorsed style."""
        q_good = float(np.pi * 5.0 * 2.0)
        result = _result_dict(Q_profile=q_good, Q_profile_raw=q_good)
        fig, ax = plt.subplots()
        try:
            plot_q_envelope_overlay(ax, result, q_source="profile")
            labels = [str(line.get_label()) for line in ax.get_lines()]
            assert not any("MISMATCH" in lab for lab in labels)
            candidate_lines = [
                line for line in ax.get_lines() if str(line.get_label()) == "profile Q"
            ]
            assert candidate_lines
            assert candidate_lines[0].get_linestyle() == "-"
        finally:
            plt.close(fig)

    def test_best_source_skips_demoted_profile(self):
        """q_source='best' never selects a demoted (non-valid) profile value."""
        q_bad = float(np.pi * 5.0 * 0.4)
        q_env = float(np.pi * 5.0 * 2.0)
        result = _result_dict(
            Q_profile=None,
            Q_profile_raw=q_bad,
            Q_nls=None,
            Q_dft=None,
            Q_envelope=q_env,
        )
        fig, ax = plt.subplots()
        try:
            plot_q_envelope_overlay(ax, result, q_source="best")
            labels = [str(line.get_label()) for line in ax.get_lines()]
            assert any(lab.startswith("envelope Q") for lab in labels)
            assert not any("raw profile" in lab for lab in labels)
        finally:
            plt.close(fig)
