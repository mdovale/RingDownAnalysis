"""
Ring-down signal generation and parameter management.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


class RingDownSignal:
    """
    Encapsulates ring-down signal parameters and generation.

    A ring-down signal is an exponentially decaying sinusoid:
    x(t) = A0 * exp(-t/tau) * cos(2*pi*f0*t + phi0) + noise

    Attributes:
    -----------
    f0 : float
        Frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    N : int
        Number of samples
    A0 : float
        Initial amplitude
    snr_db : float
        Initial signal-to-noise ratio (dB)
    Q : float
        Quality factor
    tau : float
        Decay time constant (s), computed from Q and f0
    sigma : float
        Noise standard deviation, computed from SNR
    """

    def __init__(
        self,
        f0: float,
        fs: float,
        N: int,
        A0: float = 1.0,
        snr_db: float = 60.0,
        Q: float = 10000.0,
    ):
        """
        Initialize ring-down signal parameters.

        Parameters:
        -----------
        f0 : float
            Frequency (Hz)
        fs : float
            Sampling frequency (Hz)
        N : int
            Number of samples
        A0 : float
            Initial amplitude (default: 1.0)
        snr_db : float
            Initial signal-to-noise ratio in dB (default: 60.0)
        Q : float
            Quality factor (default: 10000.0)
        """
        if f0 <= 0:
            raise ValueError("f0 must be positive")
        if fs <= 0:
            raise ValueError("fs must be positive")
        if N <= 0:
            raise ValueError("N must be positive")
        if A0 <= 0:
            raise ValueError("A0 must be positive")
        if Q <= 0:
            raise ValueError("Q must be positive")

        self.f0 = float(f0)
        self.fs = float(fs)
        self.N = int(N)
        self.A0 = float(A0)
        self.snr_db = float(snr_db)
        self.Q = float(Q)

        # Compute derived parameters
        self.tau = self.Q / (np.pi * self.f0)
        rho0 = 10.0 ** (self.snr_db / 10.0)
        self.sigma = np.sqrt((self.A0**2 / 2.0) / rho0)

        # Signal arrays (generated on demand)
        self._t = None
        self._x = None
        self._phi0 = None

    @property
    def t(self) -> np.ndarray:
        """Time array (s)."""
        if self._t is None:
            self._t = np.arange(self.N) / self.fs
        return self._t

    @property
    def T(self) -> float:
        """Total observation time (s)."""
        return self.N / self.fs

    def generate(
        self,
        phi0: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Generate a noisy ring-down signal.

        Parameters:
        -----------
        phi0 : float, optional
            Initial phase (rad). If None, random phase is used.
        rng : np.random.Generator, optional
            Random number generator. If None, default RNG is used.

        Returns:
        --------
        t : np.ndarray
            Time array (s)
        x : np.ndarray
            Noisy signal
        phi0 : float
            Initial phase used (rad)
        """
        if rng is None:
            rng = np.random.default_rng()

        if phi0 is None:
            phi0 = rng.uniform(-np.pi, np.pi)

        self._phi0 = phi0
        t = self.t

        # Generate ring-down signal: A(t) = A0 * exp(-t/tau)
        A_t = self.A0 * np.exp(-t / self.tau)
        s = A_t * np.cos(2.0 * np.pi * self.f0 * t + phi0)
        w = rng.normal(0.0, self.sigma, size=self.N)
        x = s + w

        self._x = x
        return t, x, phi0

    def get_signal(self) -> np.ndarray | None:
        """Get the generated signal if available."""
        return self._x

    def get_phase(self) -> float | None:
        """Get the initial phase if signal was generated."""
        return self._phi0


def generate_driven_plateau(
    n: int,
    fs: float,
    f0: float,
    tau: float,
    rms_target: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate an ambient-driven narrowband plateau signal.

    Simulates an AR(2) resonator (resonance f0, decay time tau) driven by
    white noise, i.e. the "never decays to zero" equilibrium oscillation
    observed on real ring-down records. The output is scaled so its RMS over
    the second half of the record equals ``rms_target``.
    """
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}")
    dt = 1.0 / fs
    w0 = 2.0 * np.pi * f0
    if w0 * dt >= np.pi:
        raise ValueError("f0 must be below the Nyquist frequency")
    # AR(2) poles at r*exp(+-i*w0*dt) with r set by the decay time, so the
    # driven resonance sits exactly at f0.
    r = 1.0 - dt / tau
    a = 2.0 * r * np.cos(w0 * dt)
    b = -(r * r)
    force = rng.normal(0.0, 1.0, n) * dt * dt
    # AR(2) recursion x[i] = a*x[i-1] + b*x[i-2] + force[i]
    x = lfilter([1.0], [1.0, -a, -b], force)
    scale = float(np.std(x[n // 2 :]))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Driven plateau simulation produced a degenerate signal")
    return x * (rms_target / scale)


def _amplitude_with_nonlinear_damping(
    t: np.ndarray,
    fs: float,
    a0: float,
    tau0: float,
    beta: float,
    coarse: int = 100,
) -> np.ndarray:
    """Integrate dA/dt = -A*(1/tau0 + beta*A) on a coarse grid, then interpolate."""
    n = len(t)
    a_current = float(a0)
    a_list = [a_current]
    for _ in range(n // coarse + 1):
        a_current = a_current * np.exp(-coarse / fs * (1.0 / tau0 + beta * a_current))
        a_list.append(a_current)
    t_coarse = np.arange(len(a_list)) * coarse / fs
    return np.interp(t, t_coarse, np.array(a_list))


def generate_pathological_ringdown(
    *,
    f0: float,
    fs: float,
    duration: float,
    a0: float = 1.0,
    tau: float = 1.0,
    sigma_white: float = 0.0,
    freq_pull: float = 0.0,
    linear_drift: float = 0.0,
    f_trajectory: tuple[np.ndarray, np.ndarray] | None = None,
    plateau_rms: float = 0.0,
    baseline_wander_rms: float = 0.0,
    damping_beta: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a ring-down record with controlled real-world pathologies.

    Starting from an ideal decaying sinusoid, each keyword adds one measured
    pathology of real resonator records (see the 2026-08-18 Q-estimation
    investigation): frequency drift/pull, an ambient-driven plateau,
    amplitude-dependent damping, and baseline wander.

    Parameters:
    -----------
    f0 : float
        Base frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    duration : float
        Record duration (s).
    a0 : float
        Initial amplitude.
    tau : float
        Decay time constant (s). With ``damping_beta`` set this is the
        zero-amplitude decay time tau0.
    sigma_white : float
        White-noise standard deviation.
    freq_pull : float
        Amplitude-proportional frequency pull coefficient (Hz per amplitude
        unit): f(t) = f0 + freq_pull * A(t).
    linear_drift : float
        Linear-in-time frequency drift rate (Hz/s).
    f_trajectory : (t_points, f_points), optional
        Explicit instantaneous-frequency trajectory, interpolated onto the
        sample times. Overrides freq_pull and linear_drift.
    plateau_rms : float
        RMS of an added ambient-driven AR(2) plateau (same f0, tau).
    baseline_wander_rms : float
        RMS of an added slow random-walk baseline drift.
    damping_beta : float, optional
        Amplitude-dependent damping coefficient beta in
        1/tau(A) = 1/tau + beta*A. None means single-exponential decay.
    rng : np.random.Generator, optional
        Random generator; required (directly or via default_rng()) whenever a
        stochastic ingredient is enabled. Pass a seeded generator for
        deterministic fixtures.

    Returns:
    --------
    (t, x) : tuple of np.ndarray
        Time array (s, starting at 0) and the generated signal.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(round(duration * fs))
    if n < 3:
        raise ValueError("duration * fs must be at least 3 samples")
    t = np.arange(n) / fs

    if damping_beta is None:
        amplitude = a0 * np.exp(-t / tau)
    else:
        amplitude = _amplitude_with_nonlinear_damping(t, fs, a0, tau, float(damping_beta))

    if f_trajectory is not None:
        t_points, f_points = f_trajectory
        f_inst = np.interp(t, np.asarray(t_points, dtype=float), np.asarray(f_points, dtype=float))
    else:
        f_inst = f0 + freq_pull * amplitude + linear_drift * t

    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    x = amplitude * np.cos(phase)
    if sigma_white > 0:
        x = x + rng.normal(0.0, sigma_white, n)
    if plateau_rms > 0:
        x = x + generate_driven_plateau(n, fs, f0, tau, plateau_rms, rng)
    if baseline_wander_rms > 0:
        wander = np.cumsum(rng.normal(0.0, 1.0, n))
        kernel = max(int(60 * fs), 1)
        wander = (
            np.convolve(wander / np.std(wander), np.ones(kernel) / kernel, mode="same")
            * baseline_wander_rms
        )
        x = x + wander
    return t, x
