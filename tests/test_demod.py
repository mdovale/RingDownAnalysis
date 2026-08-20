"""
Tests for the segmented-demodulation Q estimator (ringdownanalysis.demod).

The scenario names (E1, E2b, E2c, E3, E4, E5) follow the controlled
experiments of the 2026-08-18 investigation notebook
(notebooks/20260818_EDU_PreVibe_vs_PostVibe_RingDown.ipynb). All records are
time-scaled by 10x (tau 3700 s -> 370 s, drift rates x10) so each test record
generates and analyzes in well under a second; the dimensionless products that
control estimator behavior (drift x tau, decay fraction per window, plateau /
initial-amplitude ratio) are preserved.

Tolerances are calibrated empirically: with a driven plateau the accuracy of
any envelope-based estimator on a single record is limited by the plateau's
slow amplitude wander (correlation time ~ tau), so plateau cases assert a
loose single-seed bound plus a tight multi-seed median bound.
"""

from pathlib import Path

import numpy as np
import pytest

from ringdownanalysis.demod import (
    COHERENCE_RATIO_THRESHOLD,
    SegmentedDemodEstimator,
    SegmentedDemodResult,
)
from ringdownanalysis.signal import generate_pathological_ringdown

F0 = 7.6699
FS = 30.0
TAU = 370.0
A0 = 600.0
SIGMA_W = 1.5
Q_TRUE = np.pi * F0 * TAU
HOUR = 360.0  # scaled hour (1/10 of real time)
PLATEAU_RMS = 16.5 / np.sqrt(2)
SEED = 20260818

FIXTURES = Path(__file__).parent / "fixtures"


def make_record(**kwargs):
    defaults = dict(
        f0=F0,
        fs=FS,
        duration=3 * HOUR,
        a0=A0,
        tau=TAU,
        sigma_white=SIGMA_W,
        rng=np.random.default_rng(SEED),
    )
    defaults.update(kwargs)
    return generate_pathological_ringdown(**defaults)


@pytest.fixture(scope="module")
def estimator():
    return SegmentedDemodEstimator()


@pytest.fixture(scope="module")
def e3_6h_record():
    return make_record(duration=6 * HOUR, plateau_rms=PLATEAU_RMS)


class TestConstructorValidation:
    def test_rejects_small_min_segments(self):
        with pytest.raises(ValueError, match="min_segments"):
            SegmentedDemodEstimator(min_segments=3)

    def test_rejects_small_min_decay_segments(self):
        with pytest.raises(ValueError, match="min_decay_segments"):
            SegmentedDemodEstimator(min_decay_segments=2)

    def test_rejects_floor_threshold_at_most_one(self):
        with pytest.raises(ValueError, match="floor_threshold"):
            SegmentedDemodEstimator(floor_threshold=1.0)

    def test_rejects_flatness_ratio_at_most_one(self):
        with pytest.raises(ValueError, match="plateau_flatness_ratio"):
            SegmentedDemodEstimator(plateau_flatness_ratio=0.9)

    def test_rejects_nonpositive_seg_duration(self):
        with pytest.raises(ValueError, match="seg_duration"):
            SegmentedDemodEstimator(seg_duration=-5.0)

    def test_rejects_few_scan_points(self):
        with pytest.raises(ValueError, match="f_scan_points"):
            SegmentedDemodEstimator(f_scan_points=3)


class TestInputValidation:
    def test_rejects_mismatched_lengths(self, estimator):
        with pytest.raises(ValueError, match="same length"):
            estimator.estimate(np.arange(10.0), np.zeros(9), FS)

    def test_rejects_two_dimensional_input(self, estimator):
        with pytest.raises(ValueError, match="one-dimensional"):
            estimator.estimate(np.zeros((5, 2)), np.zeros((5, 2)), FS)

    def test_rejects_nonfinite_data(self, estimator):
        t = np.arange(100.0) / FS
        data = np.ones(100)
        data[3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            estimator.estimate(t, data, FS)

    def test_rejects_bad_fs(self, estimator):
        t = np.arange(100.0) / FS
        with pytest.raises(ValueError, match="fs"):
            estimator.estimate(t, np.ones(100), -1.0)

    def test_rejects_nonmonotonic_time(self, estimator):
        t = np.arange(100.0) / FS
        t[50] = t[49]
        with pytest.raises(ValueError, match="strictly increasing"):
            estimator.estimate(t, np.ones(100), FS)

    def test_short_record_is_invalid_not_crash(self, estimator):
        t = np.arange(int(30 * FS)) / FS
        x = np.cos(2 * np.pi * F0 * t)
        result = estimator.estimate(t, x, FS)
        assert not result.valid
        assert result.status == "invalid"
        assert result.reasons[0] in ("demod_record_too_short", "demod_insufficient_segments")


class TestIdealRecord:
    """E1: clean exponential decay, white noise only."""

    @pytest.fixture(scope="class")
    def result(self, estimator) -> SegmentedDemodResult:
        t, x = make_record()
        return estimator.estimate(t, x, FS)

    def test_q_accurate(self, result):
        assert result.valid
        assert result.status == "valid"
        assert pytest.approx(Q_TRUE, rel=0.005) == result.Q

    def test_tau_and_f_consistent(self, result):
        assert result.tau == pytest.approx(TAU, rel=0.005)
        assert result.f_mean == pytest.approx(F0, abs=1e-3)
        assert pytest.approx(np.pi * result.f_mean * result.tau) == result.Q

    def test_ci_brackets_truth(self, result):
        low, high = result.Q_ci95
        assert low < Q_TRUE < high

    def test_coherence_ratio_small(self, result):
        assert result.coherence_ratio < 0.05

    def test_segment_diagnostics_populated(self, result):
        assert result.n_segments >= 90
        assert len(result.t_mid) == result.n_segments
        assert result.decay_mask.dtype == bool
        assert np.all(result.amplitude > 0)

    def test_bootstrap_is_deterministic(self, estimator, result):
        t, x = make_record()
        again = estimator.estimate(t, x, FS)
        assert again.Q_ci95 == result.Q_ci95


class TestFrequencyDriftImmunity:
    """E2b/E2c: linear and measured f(t) drift leave the estimate unbiased."""

    def test_linear_drift_unbiased_and_gated(self, estimator):
        t, x = make_record(linear_drift=1e-5)
        result = estimator.estimate(t, x, FS)
        assert result.valid
        assert pytest.approx(Q_TRUE, rel=0.01) == result.Q
        # The drift gate must flag this record as unusable for coherent fits.
        assert result.coherence_ratio > COHERENCE_RATIO_THRESHOLD
        assert result.drift_hz == pytest.approx(1e-5 * result.t_mid[result.decay_mask][-1], rel=0.2)

    def test_measured_previbe_trajectory_unbiased(self, estimator):
        fix = np.load(FIXTURES / "previbe_f_trajectory.npz")
        t_traj = fix["t_mid"] / 10.0
        f_traj = F0 + 10.0 * (fix["f"] - fix["f"][0])
        t, x = make_record(f_trajectory=(t_traj, f_traj))
        result = estimator.estimate(t, x, FS)
        assert result.valid
        assert pytest.approx(Q_TRUE, rel=0.01) == result.Q
        assert result.coherence_ratio > COHERENCE_RATIO_THRESHOLD

    def test_frequency_pull_reported(self, estimator):
        t, x = make_record(freq_pull=-2.06e-5)
        result = estimator.estimate(t, x, FS)
        assert pytest.approx(Q_TRUE, rel=0.01) == result.Q
        # f decreases with amplitude, so df/dln(A) must come out negative.
        assert result.f_pull_per_efold < 0


class TestDrivenPlateau:
    """E3: ambient-driven plateau biases naive envelope fits."""

    def test_3h_window_accuracy(self, estimator):
        t, x = make_record(plateau_rms=PLATEAU_RMS)
        result = estimator.estimate(t, x, FS)
        assert result.valid
        assert pytest.approx(Q_TRUE, rel=0.08) == result.Q

    def test_6h_window_detects_plateau(self, estimator, e3_6h_record):
        t, x = e3_6h_record
        result = estimator.estimate(t, x, FS)
        assert result.plateau_detected
        assert result.plateau_amplitude == pytest.approx(16.5, rel=0.25)
        # Single-realization accuracy is limited by the plateau's slow
        # amplitude wander near the cutoff (see multi-seed test below).
        assert pytest.approx(Q_TRUE, rel=0.20) == result.Q

    def test_multi_seed_median_unbiased(self, estimator):
        ratios = []
        for seed in range(8):
            t, x = make_record(
                duration=6 * HOUR,
                plateau_rms=PLATEAU_RMS,
                rng=np.random.default_rng(1000 + seed),
            )
            result = estimator.estimate(t, x, FS)
            assert result.Q is not None
            ratios.append(result.Q / Q_TRUE)
        assert float(np.median(ratios)) == pytest.approx(1.0, abs=0.05)
        assert max(abs(r - 1.0) for r in ratios) < 0.30

    def test_plateau_dominated_window_returns_no_q(self, estimator, e3_6h_record):
        t, x = e3_6h_record
        mask = t >= 4.3 * TAU
        result = estimator.estimate(t[mask], x[mask], FS)
        assert result.Q is None
        assert not result.valid
        assert result.status == "plateau_dominated"
        assert "demod_plateau_dominated_window" in result.reasons

    def test_short_window_warns_about_span(self, estimator, e3_6h_record):
        t, x = e3_6h_record
        n = int(1 * HOUR * FS)
        result = estimator.estimate(t[:n], x[:n], FS)
        assert result.status == "warning"
        assert "demod_decay_window_shorter_than_tau" in result.reasons
        assert pytest.approx(Q_TRUE, rel=0.10) == result.Q


class TestEduTwin:
    """E4: drift + plateau + baseline wander together (the real-data twin)."""

    def test_q_within_ten_percent(self, estimator):
        t, x = make_record(
            freq_pull=-2.06e-5,
            plateau_rms=PLATEAU_RMS,
            baseline_wander_rms=10.0,
        )
        result = estimator.estimate(t, x, FS)
        assert result.valid
        assert pytest.approx(Q_TRUE, rel=0.10) == result.Q
        assert result.coherence_ratio > COHERENCE_RATIO_THRESHOLD


class TestAmplitudeDependentDamping:
    """E5: local Q must resolve the amplitude dependence."""

    @pytest.fixture(scope="class")
    def result(self, estimator) -> SegmentedDemodResult:
        t, x = make_record(tau=520.0, damping_beta=2.17e-6)
        return estimator.estimate(t, x, FS)

    def test_banded_q_increases_as_amplitude_decreases(self, result):
        bands = result.q_vs_amplitude
        assert len(bands) >= 3
        # Bands are generated from high to low amplitude; Q must increase.
        q_values = [band.Q for band in bands]
        assert q_values == sorted(q_values)
        assert q_values[-1] / q_values[0] > 1.2

    def test_band_metadata_consistent(self, result):
        for band in result.q_vs_amplitude:
            assert band.amplitude_low < band.amplitude_mid < band.amplitude_high
            assert band.n_segments >= 4
            assert band.tau > 0


class TestExplicitAmplitudeBands:
    def test_user_bands_are_respected(self, estimator):
        t, x = make_record()
        result = estimator.estimate(t, x, FS, amplitude_bands=[(100.0, 400.0)])
        assert len(result.q_vs_amplitude) == 1
        band = result.q_vs_amplitude[0]
        assert band.amplitude_low == 100.0
        assert band.amplitude_high == 400.0
        assert pytest.approx(Q_TRUE, rel=0.05) == band.Q


class TestRobustLineFit:
    def test_matches_least_squares_on_clean_line(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 10.0, 60)
        y = 2.0 - 0.5 * x + rng.normal(0, 0.01, 60)
        slope, intercept = SegmentedDemodEstimator._fit_line_robust(x, y)
        assert slope == pytest.approx(-0.5, abs=0.01)
        assert intercept == pytest.approx(2.0, abs=0.02)

    def test_resists_contiguous_outlier_block(self):
        x = np.linspace(0.0, 10.0, 60)
        y = 2.0 - 0.5 * x
        y[-10:] += 1.5  # 17 % contamination, block-correlated like a revival
        slope, _ = SegmentedDemodEstimator._fit_line_robust(x, y)
        assert slope == pytest.approx(-0.5, abs=0.05)


class TestDemodulate:
    def test_per_segment_frequency_accuracy(self, estimator):
        t, x = make_record()
        segments = estimator.demodulate(t, x, FS, f_init=F0, seg_duration=45.0)
        assert segments.shape[1] == 4
        t_mid, f_seg, amplitude, sigma = segments.T
        early = t_mid < TAU
        assert np.allclose(f_seg[early], F0, atol=1e-3)
        expected = A0 * np.exp(-t_mid[early] / TAU)
        assert np.allclose(amplitude[early], expected, rtol=0.05)
        assert np.all(sigma[early] > 0)
