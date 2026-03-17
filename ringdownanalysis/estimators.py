"""
Frequency estimation methods for ring-down signals.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal.windows import kaiser

logger = logging.getLogger(__name__)


def _validate_signal_input(x: np.ndarray) -> None:
    """
    Validate signal array for frequency estimation.

    Raises:
    --------
    TypeError
        If x is not a numpy ndarray
    ValueError
        If x is empty, has wrong dtype, or contains NaN/Inf
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray for signal x, got {type(x).__name__}")
    if x.ndim != 1:
        raise ValueError(f"Signal x must be 1-dimensional, got shape {x.shape}")
    if len(x) == 0:
        raise ValueError("Signal x cannot be empty; need at least one sample for estimation")
    if len(x) == 1:
        raise ValueError(
            "Signal x must have at least 2 samples; single-sample signals "
            "cannot be used for frequency estimation"
        )
    if not np.issubdtype(x.dtype, np.floating) and not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"Signal x must have numeric dtype (float or int), got {x.dtype}")
    x_float = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(x_float)):
        raise ValueError("Signal x contains NaN; cannot perform frequency estimation")
    if np.any(np.isinf(x_float)):
        raise ValueError("Signal x contains Inf; cannot perform frequency estimation")


def _validate_fs(fs: float) -> None:
    """Validate sampling frequency fs. Raises ValueError if invalid."""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling frequency fs must be positive and finite, got {fs}")


class EstimationResult(NamedTuple):
    """Result of frequency, tau, and Q estimation."""

    f: float
    """Estimated frequency (Hz)"""
    tau: float | None
    """Estimated decay time constant (s), or None if not estimated"""
    Q: float | None
    """Estimated quality factor, or None if tau is not available"""
    success: bool = True
    """Whether the estimator converged without falling back to the initializer"""
    used_fallback: bool = False
    """Whether the result fell back to a heuristic initializer"""
    message: str | None = None
    """Fit termination or fallback message"""
    nfev: int | None = None
    """Number of function evaluations used by the fit, if available"""


class FrequencyEstimator(ABC):
    """Base class for frequency estimators."""

    @abstractmethod
    def estimate(self, x: np.ndarray, fs: float, **kwargs) -> float:
        """
        Estimate frequency from signal.

        Parameters:
        -----------
        x : np.ndarray
            Signal samples
        fs : float
            Sampling frequency (Hz)
        ``**kwargs``
            Additional method-specific parameters

        Returns:
        --------
        float
            Estimated frequency (Hz)
        """
        pass


def _lorentzian_func(f: np.ndarray, A: float, f0: float, gamma: float, offset: float) -> np.ndarray:
    """Lorentzian function for power spectrum fitting."""
    return A / ((f - f0) ** 2 + (gamma / 2.0) ** 2) + offset


def _amplitude_floor(x: np.ndarray) -> float:
    """Return a small positive amplitude scale suitable for fit bounds."""
    scale = max(
        float(np.std(x) * np.sqrt(2.0)),
        float(np.max(np.abs(x))) * 1e-6 if len(x) > 0 else 0.0,
        np.finfo(np.float64).eps,
    )
    return scale


def _sanitize_initial_parameters(
    x: np.ndarray,
    fs: float,
    initial_params: tuple,
) -> tuple[float, float, float, float]:
    """Sanitize heuristic initialization before passing it to bounded solvers."""
    f0_init, phi0_init, A0_init, c0 = initial_params

    if not np.isfinite(f0_init) or f0_init < 0:
        f0_init = fs / max(len(x), 2)
    else:
        f0_init = float(min(max(f0_init, 0.0), 0.5 * fs))

    if not np.isfinite(phi0_init):
        phi0_init = 0.0
    else:
        phi0_init = float(np.arctan2(np.sin(phi0_init), np.cos(phi0_init)))

    if not np.isfinite(c0):
        c0 = float(np.mean(x))
    else:
        c0 = float(c0)

    A0_init = float(abs(A0_init)) if np.isfinite(A0_init) else 0.0
    A0_init = max(A0_init, _amplitude_floor(x))

    return f0_init, phi0_init, A0_init, c0


def _sanitize_tau_guess(tau_init: float | None, t: np.ndarray) -> tuple[float, float, float]:
    """Return a feasible tau initialization and matching lower/upper bounds."""
    eps = float(np.finfo(np.float64).eps)
    tau_lower = float(max(float(t[1]), eps))
    tau_upper_default = float(max(10.0 * float(t[-1]), tau_lower * 10.0))

    if tau_init is None or not np.isfinite(tau_init) or tau_init <= 0:
        tau_guess = float(max(0.5 * float(t[-1]), tau_lower * 2.0))
    else:
        tau_guess = float(tau_init)

    tau_upper = float(max(tau_upper_default, tau_guess * 1.1))
    tau_guess = float(min(max(tau_guess, tau_lower * 1.01), tau_upper * 0.99))

    return tau_guess, tau_lower, tau_upper


def _has_resolved_ac_content(x: np.ndarray) -> bool:
    """Return True when the demeaned signal has meaningful AC content."""
    x_demean = x - np.mean(x)
    threshold = max(float(np.max(np.abs(x))) * 1e-12, np.finfo(np.float64).eps * 10.0)
    return float(np.max(np.abs(x_demean))) > threshold


def _fit_lorentzian_to_peak(
    P: np.ndarray,
    k: int,
    fs: float,
    N_dft: int,
    n_points: int = 7,
) -> float:
    """
    Fit a Lorentzian function to the power spectrum around the peak.

    For ring-down signals, the Fourier transform has a Lorentzian shape,
    so fitting a Lorentzian is more appropriate than parabolic interpolation.
    """
    # Determine range of bins to use
    # Scale number of points based on frequency resolution to ensure consistent frequency coverage
    # Target ~1.2e-3 Hz frequency range for fitting (balanced: enough points without too much noise)
    df = fs / N_dft
    target_freq_range_fit = 1.2e-3  # Target frequency range for fitting
    n_points_scaled = max(n_points, int(target_freq_range_fit / df))
    # Ensure odd number for symmetric range around peak
    if n_points_scaled % 2 == 0:
        n_points_scaled += 1

    half_range = n_points_scaled // 2
    k_start = max(0, k - half_range)
    k_end = min(len(P), k + half_range + 1)

    # Extract frequency and power values
    k_indices = np.arange(k_start, k_end)
    f_bins = k_indices * fs / N_dft
    P_bins = P[k_indices]

    # Initial parameter guesses
    P_max = P[k]
    f0_init = k * fs / N_dft

    # Estimate gamma (FWHM) from half-maximum points using vectorized search
    half_max = P_max / 2.0
    left_idx = k
    right_idx = k

    # Scale search range based on frequency resolution to ensure consistent frequency coverage
    # Use a frequency-based range (~1e-3 Hz) converted to bins, with minimum of 10 bins
    # This ensures the search range scales properly with zero-padding
    df = fs / N_dft
    target_freq_range = 1e-3  # Target ~1 mHz frequency range for search
    search_bins = max(10, int(target_freq_range / df))

    # Vectorized search for left half-maximum point
    left_range_start = max(0, k - search_bins)
    left_range = np.arange(k - 1, left_range_start - 1, -1)
    if len(left_range) > 0:
        left_mask = P[left_range] < half_max
        if np.any(left_mask):
            left_idx = left_range[np.argmax(left_mask)]  # First True from left

    # Vectorized search for right half-maximum point
    right_range_end = min(len(P), k + search_bins)
    right_range = np.arange(k + 1, right_range_end)
    if len(right_range) > 0:
        right_mask = P[right_range] < half_max
        if np.any(right_mask):
            right_idx = right_range[np.argmax(right_mask)]  # First True from left

    if right_idx > left_idx:
        gamma_init = (right_idx - left_idx) * fs / N_dft
    else:
        gamma_init = 2.0 * fs / N_dft

    # Estimate background offset from edges
    offset_init = np.min([P[0], P[-1], np.mean(P[: max(1, len(P) // 20)])])

    # Initial amplitude guess
    A_init = P_max * (gamma_init / 2.0) ** 2

    # Fit Lorentzian
    try:
        # Relax frequency bounds slightly to allow interpolation beyond bin edges
        # This helps when the true peak is between bins
        df = fs / N_dft
        f_low = max(0.0, f_bins[0] - 0.5 * df)  # Allow half bin width below
        f_high = min(
            0.5 * fs, f_bins[-1] + 0.5 * df
        )  # Allow half bin width above, but not beyond Nyquist

        # Improve gamma bounds: use more reasonable range based on typical ring-down behavior
        # Gamma should be positive and typically in range of 0.1x to 5x the fitting range
        gamma_max = max((f_bins[-1] - f_bins[0]) * 5.0, 1e-3)  # At least 1 mHz

        popt, _ = curve_fit(
            _lorentzian_func,
            f_bins,
            P_bins,
            p0=[A_init, f0_init, gamma_init, offset_init],
            bounds=(
                [
                    0.0,
                    f_low,
                    1e-6,  # Minimum gamma (very small but positive)
                    -np.inf,
                ],
                [
                    np.inf,
                    f_high,
                    gamma_max,
                    np.inf,
                ],
            ),
            maxfev=500,  # Increased for better convergence
            method="trf",  # Trust Region Reflective algorithm, more robust
        )

        A_fit, f0_fit, gamma_fit, offset_fit = popt

        # Calculate delta (offset from bin k)
        delta = (f0_fit - f0_init) / (fs / N_dft)

        # Clip delta to reasonable range (±1 bin width) to prevent outliers
        # With zero-padding, we can allow slightly larger range for better interpolation
        delta = np.clip(delta, -1.0, 1.0)

        return delta
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "lorentzian_fit_failed",
                extra={
                    "event": "lorentzian_fit_failed",
                    "error_type": type(e).__name__,
                    "k": int(k),
                    "n_points": n_points_scaled,
                },
            )
        return 0.0


def _estimate_initial_parameters_from_dft(x: np.ndarray, fs: float) -> tuple:
    """Estimate initial frequency, phase, amplitude, and DC offset from DFT."""
    N = len(x)
    x_demean = x - np.mean(x)
    X = np.fft.rfft(x_demean * np.hanning(N))
    mag2 = np.abs(X) ** 2

    # Skip DC component (k=0) when finding peak
    # Use view instead of copy to avoid memory allocation
    k = int(np.argmax(mag2[1:]) + 1)  # Skip first element, add 1 to index

    # Use Lorentzian fitting for initial guess if possible
    if k > 0 and k < len(mag2) - 1:
        delta = _fit_lorentzian_to_peak(mag2, k, fs, N, n_points=7)
        k_interp = k + delta
    else:
        k_interp = k

    f0_init = k_interp * fs / N
    phi0_init = np.angle(X[k])

    # Initial amplitude estimation with sanity check
    A0_init = np.sqrt(2.0) * np.sqrt(mag2[k] / N)
    if A0_init < 0.1 * np.std(x) or A0_init > 10 * np.std(x):
        A0_init = np.std(x) * np.sqrt(2.0)
    A0_init = max(float(A0_init), _amplitude_floor(x))

    c0 = np.mean(x)

    return _sanitize_initial_parameters(x, fs, (f0_init, phi0_init, A0_init, c0))


def _estimate_initial_tau_from_envelope(x: np.ndarray, t: np.ndarray) -> float:
    """Estimate initial tau from signal envelope decay using RMS in windows."""
    N = len(x)
    if N < 10:
        return max(float(t[-1]) / 2.0, float(t[1]))

    window_size = max(1, min(1000, N // 10))
    n_windows = N // window_size

    if n_windows == 0:
        return max(float(t[-1]) / 2.0, float(t[1]))

    # Vectorized RMS calculation using reshape and std along axis
    # Pad or truncate to make evenly divisible
    x_padded = x[: n_windows * window_size]
    if len(x_padded) > 0:
        # Reshape to (n_windows, window_size) and compute std along axis=1
        x_reshaped = x_padded.reshape(n_windows, window_size)
        rms_values = np.std(x_reshaped, axis=1)
    else:
        # Fallback for very short signals
        rms_values = np.array([np.std(x)])
        n_windows = 1
        window_size = N

    rms_peak = np.max(rms_values)
    decay_idx = np.where(rms_values < rms_peak * np.exp(-1))[0]

    if len(decay_idx) > 0 and decay_idx[0] > 0:
        return max(float(t[decay_idx[0] * window_size]), float(t[1]))
    else:
        return max(float(t[-1]) / 2.0, float(t[1]))


class NLSFrequencyEstimator(FrequencyEstimator):
    """
    Frequency estimation using nonlinear least squares with ring-down model.
    """

    def __init__(self, tau_known: float | None = None):
        """
        Initialize NLS frequency estimator.

        Parameters:
        -----------
        tau_known : float, optional
            Known decay time constant. If None, tau is estimated along with other parameters.
        """
        self.tau_known = tau_known

    @staticmethod
    def _frequency_sanity_passes(f_hat: float, f0_init: float, fs: float) -> bool:
        """Return True when the fitted frequency is physically and numerically plausible."""
        if not np.isfinite(f_hat) or f_hat < 0 or f_hat > 0.5 * fs:
            return False
        if f0_init <= 0:
            return True
        return abs(f_hat - f0_init) <= 0.5 * f0_init

    def _estimate_known_tau_full(
        self,
        x: np.ndarray,
        fs: float,
        initial_params: tuple,
        **kwargs,
    ) -> EstimationResult:
        """Estimate frequency when tau is fixed a priori."""
        assert self.tau_known is not None
        tau = self.tau_known
        N = len(x)
        t = np.arange(N) / fs
        f0_init, phi0_init, A0_init, c0 = _sanitize_initial_parameters(x, fs, initial_params)

        if not _has_resolved_ac_content(x):
            return EstimationResult(
                f=f0_init,
                tau=tau,
                Q=np.pi * f0_init * tau,
                success=False,
                used_fallback=True,
                message="Signal has no resolved AC content after demeaning",
                nfev=0,
            )

        def residuals(p):
            A0, f, phi, c = p
            return (A0 * np.exp(-t / tau) * np.cos(2.0 * np.pi * f * t + phi) + c) - x

        df = fs / N
        f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
        f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))
        amp_upper = max(10.0 * A0_init, _amplitude_floor(x) * 10.0)

        ls_kwargs = {
            "method": "trf",
            "verbose": 0,
            "ftol": kwargs.get("ftol", 1e-8),
            "xtol": kwargs.get("xtol", 1e-8),
            "gtol": kwargs.get("gtol", 1e-8),
            "max_nfev": kwargs.get("max_nfev", 500),
        }
        res = least_squares(
            residuals,
            x0=np.array([A0_init, f0_init, phi0_init, c0]),
            bounds=([0.0, f_low, -np.pi, -np.inf], [amp_upper, f_high, np.pi, np.inf]),
            **ls_kwargs,
        )

        if not res.success:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "nls_estimation_failed",
                    extra={
                        "event": "nls_estimation_failed",
                        "method": "nls_tau_known",
                        "fit_message": res.message,
                        "nfev": res.nfev,
                    },
                )
            return EstimationResult(
                f=f0_init,
                tau=tau,
                Q=np.pi * f0_init * tau,
                success=False,
                used_fallback=True,
                message=f"NLS tau-known fit failed: {res.message}",
                nfev=res.nfev,
            )

        _, f_hat, _, _ = res.x
        if not self._frequency_sanity_passes(float(f_hat), f0_init, fs):
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "nls_sanity_check_failed",
                    extra={
                        "event": "nls_sanity_check_failed",
                        "f_hat": float(f_hat),
                        "f0_init": float(f0_init),
                        "fs": float(fs),
                    },
                )
            return EstimationResult(
                f=f0_init,
                tau=tau,
                Q=np.pi * f0_init * tau,
                success=False,
                used_fallback=True,
                message="NLS tau-known fit failed frequency sanity check",
                nfev=res.nfev,
            )

        return EstimationResult(
            f=float(f_hat),
            tau=float(tau),
            Q=float(np.pi * f_hat * tau),
            success=True,
            used_fallback=False,
            message=res.message,
            nfev=res.nfev,
        )

    def _estimate_unknown_tau_full(
        self,
        x: np.ndarray,
        fs: float,
        initial_params: tuple,
        **kwargs,
    ) -> EstimationResult:
        """Estimate frequency and tau jointly with NLS."""
        N = len(x)
        t = np.arange(N) / fs
        f0_init, phi0_init, A0_init, c0 = _sanitize_initial_parameters(x, fs, initial_params)

        if not _has_resolved_ac_content(x):
            return EstimationResult(
                f=f0_init,
                tau=None,
                Q=None,
                success=False,
                used_fallback=True,
                message="Signal has no resolved AC content after demeaning",
                nfev=0,
            )
        tau_seed = kwargs.get("tau_init")
        if tau_seed is None:
            tau_seed = _estimate_initial_tau_from_envelope(x, t)
        tau_init, tau_lower, tau_upper = _sanitize_tau_guess(tau_seed, t)

        def residuals(p):
            A0, f, phi, tau, c = p
            return (A0 * np.exp(-t / tau) * np.cos(2.0 * np.pi * f * t + phi) + c) - x

        df = fs / N
        f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
        f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))
        amp_upper = max(10.0 * A0_init, _amplitude_floor(x) * 10.0)

        ls_kwargs = {
            "method": "trf",
            "verbose": 0,
            "ftol": kwargs.get("ftol", 1e-8),
            "xtol": kwargs.get("xtol", 1e-8),
            "gtol": kwargs.get("gtol", 1e-8),
            "max_nfev": kwargs.get("max_nfev", 500),
        }
        res = least_squares(
            residuals,
            x0=np.array([A0_init, f0_init, phi0_init, tau_init, c0]),
            bounds=(
                [0.0, f_low, -np.pi, tau_lower, -np.inf],
                [amp_upper, f_high, np.pi, tau_upper, np.inf],
            ),
            **ls_kwargs,
        )

        if not res.success:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "nls_full_estimation_failed",
                    extra={
                        "event": "nls_full_estimation_failed",
                        "method": "nls_tau_unknown",
                        "fit_message": res.message,
                        "nfev": res.nfev,
                    },
                )
            return EstimationResult(
                f=f0_init,
                tau=None,
                Q=None,
                success=False,
                used_fallback=True,
                message=f"NLS tau-unknown fit failed: {res.message}",
                nfev=res.nfev,
            )

        _, f_hat, _, tau_hat, _ = res.x

        if not self._frequency_sanity_passes(float(f_hat), f0_init, fs):
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "nls_full_sanity_check_failed",
                    extra={
                        "event": "nls_full_sanity_check_failed",
                        "f_hat": float(f_hat),
                        "f0_init": float(f0_init),
                        "fs": float(fs),
                    },
                )
            return EstimationResult(
                f=f0_init,
                tau=None,
                Q=None,
                success=False,
                used_fallback=True,
                message="NLS tau-unknown fit failed frequency sanity check",
                nfev=res.nfev,
            )

        if not np.isfinite(tau_hat) or tau_hat <= 0 or tau_hat < tau_lower or tau_hat > tau_upper:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "nls_full_tau_sanity_check_failed",
                    extra={
                        "event": "nls_full_tau_sanity_check_failed",
                        "tau_hat": float(tau_hat),
                        "tau_init": float(tau_init),
                        "t_max": float(t[-1]),
                    },
                )
            return EstimationResult(
                f=float(f_hat),
                tau=None,
                Q=None,
                success=False,
                used_fallback=True,
                message="NLS tau-unknown fit failed tau sanity check",
                nfev=res.nfev,
            )

        return EstimationResult(
            f=float(f_hat),
            tau=float(tau_hat),
            Q=float(np.pi * f_hat * tau_hat),
            success=True,
            used_fallback=False,
            message=res.message,
            nfev=res.nfev,
        )

    def estimate(self, x: np.ndarray, fs: float, **kwargs) -> float:
        """
        Estimate frequency using nonlinear least squares.

        Parameters:
        -----------
        x : np.ndarray
            Signal samples
        fs : float
            Sampling frequency (Hz)
        ``**kwargs``
            Additional parameters:
            - initial_params: Optional tuple of (f0_init, phi0_init, A0_init, c0) to avoid redundant FFT
            - tau_init: Optional initial guess for tau (s). If None, estimated from envelope.
            - max_nfev, ftol, xtol, gtol: Optional fit convergence parameters for least_squares

        Returns:
        --------
        float
            Estimated frequency (Hz)

        Raises:
        -------
        ValueError
            If x is empty, has wrong dtype, contains NaN/Inf, or fs is invalid
        """
        _validate_signal_input(x)
        _validate_fs(fs)
        fit_kwargs = dict(kwargs)
        initial_params = kwargs.get("initial_params")
        fit_kwargs.pop("initial_params", None)
        if self.tau_known is not None:
            if initial_params is None:
                initial_params = _estimate_initial_parameters_from_dft(x, fs)
            result = self._estimate_known_tau_full(x, fs, initial_params, **fit_kwargs)
        else:
            if initial_params is None:
                initial_params = _estimate_initial_parameters_from_dft(x, fs)
            result = self._estimate_unknown_tau_full(x, fs, initial_params, **fit_kwargs)

        return float(result.f)

    def estimate_full(self, x: np.ndarray, fs: float, **kwargs) -> EstimationResult:
        """
        Estimate frequency, tau, and Q using nonlinear least squares.

        When tau_known is None, this method extracts both frequency and tau
        from the joint NLS fit and computes Q = π f τ.

        Parameters:
        -----------
        x : np.ndarray
            Signal samples
        fs : float
            Sampling frequency (Hz)
        ``**kwargs``
            Additional parameters:
            - initial_params: Optional tuple of (f0_init, phi0_init, A0_init, c0) to avoid redundant FFT
            - tau_init: Optional initial guess for tau (s). If None, estimated from envelope.
            - max_nfev, ftol, xtol, gtol: Optional fit convergence parameters for least_squares

        Returns:
        --------
        EstimationResult
            Named tuple with (f, tau, Q) estimates

        Raises:
        -------
        ValueError
            If x is empty, has wrong dtype, contains NaN/Inf, or fs is invalid
        """
        _validate_signal_input(x)
        _validate_fs(fs)
        fit_kwargs = dict(kwargs)
        initial_params = kwargs.get("initial_params")
        fit_kwargs.pop("initial_params", None)
        if initial_params is None:
            initial_params = _estimate_initial_parameters_from_dft(x, fs)

        if self.tau_known is not None:
            return self._estimate_known_tau_full(x, fs, initial_params, **fit_kwargs)
        return self._estimate_unknown_tau_full(x, fs, initial_params, **fit_kwargs)


class DFTFrequencyEstimator(FrequencyEstimator):
    """
    Frequency estimation using DFT peak fitting with Lorentzian function.
    """

    def __init__(
        self,
        window: str = "rect",
        use_zeropad: bool = True,
        pad_factor: int = 4,
        lorentzian_points: int = 7,
        kaiser_beta: float = 9.0,
        f_min: float = 0.0,
    ):
        """
        Initialize DFT frequency estimator.

        Parameters:
        -----------
        window : str
            Window type: 'rect' (default), 'hann', 'kaiser', or 'blackman'
        use_zeropad : bool
            Use zero-padding for finer frequency grid (default: False)
        pad_factor : int
            Zero-padding factor: DFT size = pad_factor * N (default: 4)
        lorentzian_points : int
            Number of points around peak to use for Lorentzian fitting (default: 7)
        kaiser_beta : float
            Kaiser window beta parameter (default: 9.0)
        f_min : float
            Minimum frequency (Hz) to consider when searching for the peak.
            Bins below f_min are excluded. Use this for phase/cumulative data
            where a ramp dominates the low-frequency spectrum (default: 0.0).
        """
        self.window = window
        self.use_zeropad = use_zeropad
        self.pad_factor = pad_factor
        self.lorentzian_points = lorentzian_points
        self.kaiser_beta = kaiser_beta
        self.f_min = f_min

    def _window_values(self, N: int) -> np.ndarray:
        """Return the configured analysis window."""
        if self.window == "kaiser":
            return kaiser(N, self.kaiser_beta)
        if self.window == "hann":
            return np.hanning(N)
        if self.window == "rect":
            return np.ones(N)
        if self.window == "blackman":
            return np.blackman(N)
        raise ValueError(f"Unknown window: {self.window}")

    def _estimate_frequency_result(self, x: np.ndarray, fs: float) -> EstimationResult:
        """Estimate frequency and expose whether the Lorentzian interpolation succeeded."""
        N = len(x)
        if not _has_resolved_ac_content(x):
            f_fallback = fs / max(N, 2)
            return EstimationResult(
                f=float(f_fallback),
                tau=None,
                Q=None,
                success=False,
                used_fallback=True,
                message="Signal has no resolved AC content after demeaning",
                nfev=0,
            )
        x = x - np.mean(x)
        w = self._window_values(N)
        xw = x * w

        if self.use_zeropad:
            N_pad = self.pad_factor * N
            xw_pad = np.zeros(N_pad, dtype=xw.dtype)
            xw_pad[:N] = xw
            N_dft = N_pad
        else:
            xw_pad = xw
            N_dft = N

        X = np.fft.rfft(xw_pad)
        P = np.abs(X) ** 2

        k_min = 1
        if self.f_min > 0:
            k_min = max(1, int(np.ceil(self.f_min * N_dft / fs)))
        if k_min >= len(P):
            k_min = max(1, len(P) - 1)
        k = int(np.argmax(P[k_min:]) + k_min)

        if k <= 0 or k >= len(P) - 1:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "dft_peak_at_edge",
                    extra={
                        "event": "dft_peak_at_edge",
                        "k": int(k),
                        "n_bins": len(P),
                    },
                )
            f_hat = float(k * fs / N_dft)
            return EstimationResult(
                f=f_hat,
                tau=None,
                Q=None,
                success=False,
                used_fallback=True,
                message="DFT peak occurred at the FFT edge; skipping Lorentzian interpolation",
                nfev=None,
            )

        delta = _fit_lorentzian_to_peak(P, k, fs, N_dft, n_points=self.lorentzian_points)
        f_hat = float((k + delta) * fs / N_dft)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dft_estimated",
                extra={
                    "event": "dft_estimated",
                    "f_hat": float(f_hat),
                    "k": int(k),
                    "delta": float(delta),
                },
            )

        return EstimationResult(
            f=f_hat,
            tau=None,
            Q=None,
            success=True,
            used_fallback=False,
            message="DFT frequency estimated via Lorentzian-interpolated FFT peak",
            nfev=None,
        )

    def estimate(self, x: np.ndarray, fs: float, **kwargs) -> float:
        """
        Estimate frequency using DFT with Lorentzian fitting.

        Parameters:
        -----------
        x : np.ndarray
            Signal samples
        fs : float
            Sampling frequency (Hz)
        ``**kwargs``
            Additional parameters (ignored)

        Returns:
        --------
        float
            Estimated frequency (Hz)

        Raises:
        -------
        ValueError
            If x is empty, has wrong dtype, contains NaN/Inf, or fs is invalid
        """
        _validate_signal_input(x)
        _validate_fs(fs)
        return float(self._estimate_frequency_result(x, fs).f)

    def estimate_full(self, x: np.ndarray, fs: float, **kwargs) -> EstimationResult:
        """
        Estimate frequency, tau, and Q using a two-step approach:
        1. Estimate frequency via DFT (as in estimate())
        2. Estimate tau via NLS with frequency fixed to the DFT result

        Parameters:
        -----------
        x : np.ndarray
            Signal samples
        fs : float
            Sampling frequency (Hz)
        ``**kwargs``
            Additional parameters:
            - initial_params: Optional tuple of (f0_init, phi0_init, A0_init, c0) for NLS tau step
            - tau_init: Optional initial guess for tau (s). If None, estimated from envelope.
            - max_nfev, ftol, xtol, gtol: Optional fit convergence parameters for least_squares

        Returns:
        --------
        EstimationResult
            Named tuple with (f, tau, Q) estimates

        Raises:
        -------
        ValueError
            If x is empty, has wrong dtype, contains NaN/Inf, or fs is invalid
        """
        _validate_signal_input(x)
        _validate_fs(fs)
        # Step 1: Estimate frequency via DFT
        frequency_result = self._estimate_frequency_result(x, fs)
        f_hat = frequency_result.f

        # Step 2: Estimate tau via NLS with fixed frequency
        N = len(x)
        t = np.arange(N) / fs

        # Get initial parameter estimates for NLS
        initial_params = kwargs.get("initial_params")
        if initial_params is not None:
            _, phi0_init, A0_init, c0 = initial_params
        else:
            _, phi0_init, A0_init, c0 = _estimate_initial_parameters_from_dft(x, fs)
        _, phi0_init, A0_init, c0 = _sanitize_initial_parameters(
            x, fs, (f_hat, phi0_init, A0_init, c0)
        )

        # Initial tau guess
        tau_seed = kwargs.get("tau_init")
        if tau_seed is None:
            tau_seed = _estimate_initial_tau_from_envelope(x, t)
        tau_init, tau_lower, tau_upper = _sanitize_tau_guess(tau_seed, t)

        # NLS fit with fixed frequency: estimate (A0, phi, tau, c)
        def residuals(p):
            A0, phi, tau, c = p
            return (A0 * np.exp(-t / tau) * np.cos(2.0 * np.pi * f_hat * t + phi) + c) - x

        lb = [0.0, -np.pi, tau_lower, -np.inf]
        ub = [max(10.0 * A0_init, _amplitude_floor(x) * 10.0), np.pi, tau_upper, np.inf]

        ls_kwargs = {
            "method": "trf",
            "verbose": 0,
            "ftol": kwargs.get("ftol", 1e-8),
            "xtol": kwargs.get("xtol", 1e-8),
            "gtol": kwargs.get("gtol", 1e-8),
            "max_nfev": kwargs.get("max_nfev", 500),
        }
        res = least_squares(
            residuals,
            x0=np.array([A0_init, phi0_init, tau_init, c0]),
            bounds=(lb, ub),
            **ls_kwargs,
        )

        if not res.success:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "dft_full_tau_estimation_failed",
                    extra={
                        "event": "dft_full_tau_estimation_failed",
                        "fit_message": res.message,
                        "nfev": res.nfev,
                    },
                )
            return EstimationResult(
                f=f_hat,
                tau=None,
                Q=None,
                success=False,
                used_fallback=True,
                message=f"DFT tau fit failed: {res.message}",
                nfev=res.nfev,
            )

        _, _, tau_hat, _ = res.x

        # Sanity check on tau
        if tau_hat <= 0 or tau_hat > tau_upper or tau_hat < tau_lower:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "dft_full_tau_sanity_check_failed",
                    extra={
                        "event": "dft_full_tau_sanity_check_failed",
                        "tau_hat": float(tau_hat),
                        "tau_init": float(tau_init),
                        "t_max": float(t[-1]),
                    },
                )
            return EstimationResult(
                f=f_hat,
                tau=None,
                Q=None,
                success=False,
                used_fallback=True,
                message="DFT tau fit failed tau sanity check",
                nfev=res.nfev,
            )

        Q_hat = np.pi * f_hat * tau_hat
        return EstimationResult(
            f=float(f_hat),
            tau=float(tau_hat),
            Q=float(Q_hat),
            success=frequency_result.success,
            used_fallback=frequency_result.used_fallback,
            message=frequency_result.message
            if frequency_result.success
            else frequency_result.message,
            nfev=res.nfev,
        )
