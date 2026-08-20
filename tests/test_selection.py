"""
Tests for the P3 estimator-selection architecture (ringdownanalysis.selection).

Unit tests exercise the classifier on fabricated result dictionaries;
integration tests run the full pipeline on time-scaled pathological records
(see tests/test_demod.py for the scaling rationale).
"""

import numpy as np
import pytest

from ringdownanalysis.analyzer import RingDownAnalyzer
from ringdownanalysis.selection import select_q_estimate
from ringdownanalysis.signal import generate_pathological_ringdown


def make_result(**overrides) -> dict:
    """A pipeline-result stub for a long, well-behaved record."""
    base = {
        "T": 1000.0,
        "tau_demod": 300.0,
        "coherence_gate_fired": False,
        "Q_demod": 9000.0,
        "Q_demod_valid": True,
        "Q_demod_status": "valid",
        "Q_profile": None,
        "Q_profile_valid": False,
        "Q_envelope": 9100.0,
        "Q_envelope_valid": True,
    }
    base.update(overrides)
    return base


class TestRegimeClassification:
    def test_long_record_prefers_demod(self):
        selection = select_q_estimate(make_result())
        assert selection.regime == "long_or_drifting"
        assert selection.source == "demod"
        assert selection.Q == 9000.0
        assert selection.valid

    def test_drifting_record_prefers_demod(self):
        selection = select_q_estimate(make_result(T=200.0, coherence_gate_fired=True))
        assert selection.regime == "long_or_drifting"
        assert selection.source == "demod"

    def test_short_coherent_record_prefers_profile(self):
        selection = select_q_estimate(
            make_result(
                T=200.0,
                Q_profile=9050.0,
                Q_profile_valid=True,
            )
        )
        assert selection.regime == "coherent_short"
        assert selection.source == "profile"
        assert selection.Q == 9050.0

    def test_short_record_without_profile_falls_back_to_demod(self):
        selection = select_q_estimate(make_result(T=200.0))
        assert selection.regime == "coherent_short"
        assert selection.source == "demod"

    def test_plateau_dominated_yields_limits_only(self):
        selection = select_q_estimate(
            make_result(
                Q_demod=None,
                Q_demod_valid=False,
                Q_demod_status="plateau_dominated",
            )
        )
        assert selection.regime == "plateau_dominated"
        assert selection.Q is None
        assert selection.source == "none"
        assert selection.status == "limit"
        assert not selection.valid
        assert "plateau_dominated_limits_only" in selection.reasons

    def test_nothing_valid_yields_invalid(self):
        selection = select_q_estimate(
            make_result(
                Q_demod=None,
                Q_demod_valid=False,
                Q_demod_status="invalid",
                Q_envelope_valid=False,
            )
        )
        assert selection.Q is None
        assert selection.status == "invalid"
        assert "no_trustworthy_estimator" in selection.reasons


class TestCrossEstimatorAgreement:
    def test_agreement_within_tolerance_stays_valid(self):
        selection = select_q_estimate(make_result(Q_envelope=9100.0))
        assert selection.valid
        assert selection.agreement_ratio == pytest.approx(9100.0 / 9000.0)

    def test_disagreement_is_hard_condition(self):
        selection = select_q_estimate(make_result(Q_envelope=20000.0))
        assert not selection.valid
        assert selection.Q is None
        assert selection.status == "warning"
        assert "cross_estimator_disagreement" in selection.reasons
        assert selection.agreement_ratio > 1.5

    def test_profile_selected_but_contradicted_by_demod(self):
        selection = select_q_estimate(
            make_result(
                T=200.0,
                Q_profile=30000.0,
                Q_profile_valid=True,
                Q_envelope_valid=False,
            )
        )
        assert selection.source == "profile"
        assert not selection.valid
        assert "cross_estimator_disagreement" in selection.reasons

    def test_no_comparators_means_no_agreement_ratio(self):
        selection = select_q_estimate(make_result(Q_envelope_valid=False))
        assert selection.valid
        assert selection.agreement_ratio is None


class TestPipelineIntegration:
    F0 = 7.6699
    FS = 30.0
    TAU = 370.0
    Q_TRUE = np.pi * F0 * TAU

    def test_drifting_record_selects_demod(self):
        t, data = generate_pathological_ringdown(
            f0=self.F0,
            fs=self.FS,
            duration=1080.0,
            a0=600.0,
            tau=self.TAU,
            sigma_white=1.5,
            linear_drift=1e-5,
            rng=np.random.default_rng(20260818),
        )
        result = RingDownAnalyzer().analyze_array(t=t, data=data)

        assert result["Q_selected_regime"] == "long_or_drifting"
        assert result["Q_selected_source"] == "demod"
        assert result["Q_selected_valid"] is True
        assert result["Q_selected"] == pytest.approx(self.Q_TRUE, rel=0.02)

    def test_selection_fields_always_present(self):
        t = np.arange(0.0, 2.0, 1.0 / 1000.0)
        data = np.exp(-t / 0.5) * np.cos(2.0 * np.pi * 50.0 * t)
        result = RingDownAnalyzer().analyze_array(t=t, data=data)

        assert "Q_selected" in result
        assert result["Q_selected_source"] in ("demod", "profile", "none")
        assert result["Q_selected_regime"] in (
            "coherent_short",
            "long_or_drifting",
            "plateau_dominated",
            "indeterminate",
        )
