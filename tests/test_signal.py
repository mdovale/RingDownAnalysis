"""
Unit tests for RingDownSignal class.
"""

import numpy as np
import pytest

from ringdownanalysis.signal import (
    RingDownSignal,
    generate_driven_plateau,
    generate_pathological_ringdown,
)


class TestRingDownSignal:
    """Test RingDownSignal class."""

    def test_init(self):
        """Test signal initialization."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0)

        assert signal.f0 == 5.0
        assert signal.fs == 100.0
        assert signal.N == 1000
        assert signal.A0 == 1.0
        assert signal.snr_db == 60.0
        assert signal.Q == 10000.0
        assert signal.tau > 0
        assert signal.sigma > 0

    def test_init_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="f0 must be positive"):
            RingDownSignal(f0=-1.0, fs=100.0, N=1000)

        with pytest.raises(ValueError, match="fs must be positive"):
            RingDownSignal(f0=5.0, fs=-100.0, N=1000)

        with pytest.raises(ValueError, match="N must be positive"):
            RingDownSignal(f0=5.0, fs=100.0, N=-1000)

        with pytest.raises(ValueError, match="A0 must be positive"):
            RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=-1.0)

        with pytest.raises(ValueError, match="Q must be positive"):
            RingDownSignal(f0=5.0, fs=100.0, N=1000, Q=-10000.0)

    def test_tau_computation(self):
        """Test tau computation from Q and f0."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, Q=10000.0)
        expected_tau = 10000.0 / (np.pi * 5.0)
        assert abs(signal.tau - expected_tau) < 1e-10

    def test_sigma_computation(self):
        """Test sigma computation from SNR."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0)
        rho0 = 10.0 ** (60.0 / 10.0)
        expected_sigma = np.sqrt((1.0**2 / 2.0) / rho0)
        assert abs(signal.sigma - expected_sigma) < 1e-10

    def test_time_array(self):
        """Test time array property."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000)
        t = signal.t

        assert len(t) == 1000
        assert t[0] == 0.0
        assert abs(t[-1] - 9.99) < 0.01
        assert abs(signal.T - 10.0) < 0.01

    def test_generate(self):
        """Test signal generation."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        t, x, phi0 = signal.generate(rng=rng)

        assert len(t) == 1000
        assert len(x) == 1000
        assert -np.pi <= phi0 <= np.pi
        assert signal.get_phase() == phi0
        assert np.array_equal(signal.get_signal(), x)

    def test_generate_with_phase(self):
        """Test signal generation with specified phase."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000)
        t, x, phi0 = signal.generate(phi0=0.5)

        assert phi0 == 0.5

    def test_signal_properties(self):
        """Test signal has expected properties."""
        signal = RingDownSignal(f0=5.0, fs=100.0, N=1000, A0=1.0, snr_db=60.0, Q=10000.0)
        rng = np.random.default_rng(42)
        t, x, phi0 = signal.generate(rng=rng)

        # Signal should decay exponentially
        # Check that later samples have smaller amplitude on average
        early_std = np.std(x[:100])
        late_std = np.std(x[-100:])
        assert late_std < early_std  # Should decay


class TestGenerateDrivenPlateau:
    """AR(2) driven-plateau fixture generator."""

    def test_plateau_rms_and_tone(self):
        """Plateau has the requested RMS and is narrowband at f0."""
        fs, f0, tau = 100.0, 7.67, 300.0
        n = 60_000
        rng = np.random.default_rng(20260818)
        x = generate_driven_plateau(n, fs, f0, tau, rms_target=10.0, rng=rng)
        assert np.std(x[n // 2 :]) == pytest.approx(10.0, rel=1e-6)
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        f_peak = freqs[np.argmax(spec[1:]) + 1]
        assert f_peak == pytest.approx(f0, abs=0.05)

    def test_deterministic_with_seed(self):
        """Same seed gives the same plateau realization."""
        a = generate_driven_plateau(5000, 100.0, 7.67, 300.0, 5.0, np.random.default_rng(7))
        b = generate_driven_plateau(5000, 100.0, 7.67, 300.0, 5.0, np.random.default_rng(7))
        assert np.array_equal(a, b)


class TestGeneratePathologicalRingdown:
    """Controlled-pathology ring-down generator (investigation fixtures)."""

    COMMON = {"f0": 7.6699, "fs": 30.0, "duration": 600.0, "a0": 600.0, "tau": 200.0}

    def test_ideal_case_matches_exponential(self):
        """With no pathologies the envelope is a clean exponential at f0."""
        t, x = generate_pathological_ringdown(**self.COMMON, rng=np.random.default_rng(1))
        assert len(t) == len(x) == int(600.0 * 30.0)
        expected = 600.0 * np.exp(-t / 200.0)
        assert np.all(np.abs(x) <= expected * (1 + 1e-9) + 1e-9)
        # Peaks reach the envelope somewhere
        assert np.max(np.abs(x[:100]) / expected[:100]) > 0.9

    def test_linear_drift_changes_instantaneous_frequency(self):
        """Linear drift shifts the late-record tone relative to the early one."""
        drift = 1e-4  # Hz/s -> 0.06 Hz over the record
        t, x = generate_pathological_ringdown(
            **{**self.COMMON, "tau": 1e9},  # no decay, isolate frequency behavior
            linear_drift=drift,
            rng=np.random.default_rng(2),
        )
        n = len(x)

        def peak_freq(seg):
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=8 * len(seg)))
            freqs = np.fft.rfftfreq(8 * len(seg), 1.0 / 30.0)
            return freqs[np.argmax(spec[1:]) + 1]

        f_early = peak_freq(x[: n // 4])
        f_late = peak_freq(x[-n // 4 :])
        expected_shift = drift * 600.0 * 0.75
        assert f_late - f_early == pytest.approx(expected_shift, rel=0.2)

    def test_f_trajectory_overrides_drift_terms(self):
        """An explicit f(t) trajectory is honored (interpolated onto samples)."""
        t_points = np.array([0.0, 600.0])
        f_points = np.array([7.6699, 7.6699])
        t, x = generate_pathological_ringdown(
            **self.COMMON,
            f_trajectory=(t_points, f_points),
            linear_drift=999.0,  # must be ignored
            rng=np.random.default_rng(3),
        )
        t_ref, x_ref = generate_pathological_ringdown(**self.COMMON, rng=np.random.default_rng(3))
        assert np.allclose(x, x_ref)

    def test_amplitude_dependent_damping_is_non_exponential(self):
        """damping_beta>0 decays faster at high amplitude than the tau0 exponential."""
        t, x = generate_pathological_ringdown(
            **self.COMMON, damping_beta=2e-5, rng=np.random.default_rng(4)
        )
        envelope0 = 600.0 * np.exp(-t / 200.0)
        # Early decay is faster than the zero-amplitude exponential...
        n_quarter = len(t) // 4
        assert np.max(np.abs(x[n_quarter : 2 * n_quarter])) < 0.9 * np.max(
            envelope0[n_quarter : 2 * n_quarter]
        )
        # ...but still decays monotonically overall.
        assert np.max(np.abs(x[-n_quarter:])) < np.max(np.abs(x[:n_quarter]))

    def test_plateau_prevents_decay_to_zero(self):
        """An added driven plateau keeps a late-record oscillation level."""
        t, x = generate_pathological_ringdown(
            **{**self.COMMON, "tau": 60.0},  # decay meets the plateau early
            plateau_rms=10.0,
            rng=np.random.default_rng(5),
        )
        late = x[-len(x) // 10 :]
        assert np.std(late) == pytest.approx(10.0, rel=0.5)

    def test_deterministic_with_seed(self):
        """Same seed and settings give identical records."""
        kwargs = {
            **self.COMMON,
            "sigma_white": 1.5,
            "plateau_rms": 5.0,
            "baseline_wander_rms": 3.0,
        }
        _, a = generate_pathological_ringdown(**kwargs, rng=np.random.default_rng(6))
        _, b = generate_pathological_ringdown(**kwargs, rng=np.random.default_rng(6))
        assert np.array_equal(a, b)
