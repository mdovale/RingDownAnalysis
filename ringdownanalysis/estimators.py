"""
Frequency estimation methods for ring-down signals.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal.windows import kaiser

logger = logging.getLogger(__name__)


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
        **kwargs
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
    X = np.fft.rfft(x * np.hanning(N))
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

    c0 = np.mean(x)

    return f0_init, phi0_init, A0_init, c0


def _estimate_initial_tau_from_envelope(x: np.ndarray, t: np.ndarray) -> float:
    """Estimate initial tau from signal envelope decay using RMS in windows."""
    N = len(x)
    window_size = min(1000, N // 10)
    n_windows = N // window_size

    if n_windows == 0:
        return t[-1] / 2.0

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
        return t[decay_idx[0] * window_size]
    else:
        return t[-1] / 2.0


class NLSFrequencyEstimator(FrequencyEstimator):
    """
    Frequency estimation using nonlinear least squares with ring-down model.
    """

    def __init__(self, tau_known: Optional[float] = None):
        """
        Initialize NLS frequency estimator.

        Parameters:
        -----------
        tau_known : float, optional
            Known decay time constant. If None, tau is estimated along with other parameters.
        """
        self.tau_known = tau_known

    def estimate(self, x: np.ndarray, fs: float, **kwargs) -> float:
        """
        Estimate frequency using nonlinear least squares.

        Parameters:
        -----------
        x : np.ndarray
            Signal samples
        fs : float
            Sampling frequency (Hz)
        **kwargs
            Additional parameters:
            - initial_params: Optional tuple of (f0_init, phi0_init, A0_init, c0) to avoid redundant FFT

        Returns:
        --------
        float
            Estimated frequency (Hz)
        """
        N = len(x)
        t = np.arange(N) / fs

        # Get initial parameter estimates (use cached if provided)
        initial_params = kwargs.get("initial_params")
        if initial_params is not None:
            f0_init, phi0_init, A0_init, c0 = initial_params
        else:
            f0_init, phi0_init, A0_init, c0 = _estimate_initial_parameters_from_dft(x, fs)

        if self.tau_known is not None:
            # Known tau: estimate (A0, f, phi, c)
            def residuals(p):
                A0, f, phi, c = p
                return (
                    A0 * np.exp(-t / self.tau_known) * np.cos(2.0 * np.pi * f * t + phi) + c
                ) - x

            # Tighter frequency bounds
            df = fs / N
            f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
            f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))

            lb = [0.0, f_low, -np.pi, -np.inf]
            ub = [10.0 * A0_init, f_high, np.pi, np.inf]

            res = least_squares(
                residuals,
                x0=np.array([A0_init, f0_init, phi0_init, c0]),
                bounds=(lb, ub),
                method="trf",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=100,  # Optimized: typical convergence in 5-10 nfev, 100 provides safety margin
                verbose=0,
            )

            if not res.success:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "nls_estimation_failed",
                        extra={
                            "event": "nls_estimation_failed",
                            "method": "nls_tau_known",
                            "message": res.message,
                            "nfev": res.nfev,
                        },
                    )
                return f0_init

            _, f_hat, _, _ = res.x
        else:
            # Unknown tau: estimate (A0, f, phi, tau, c)
            tau_init = _estimate_initial_tau_from_envelope(x, t)

            def residuals(p):
                A0, f, phi, tau, c = p
                return (A0 * np.exp(-t / tau) * np.cos(2.0 * np.pi * f * t + phi) + c) - x

            df = fs / N
            f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
            f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))

            lb = [0.0, f_low, -np.pi, t[1], -np.inf]
            ub = [10.0 * A0_init, f_high, np.pi, 10.0 * t[-1], np.inf]

            res = least_squares(
                residuals,
                x0=np.array([A0_init, f0_init, phi0_init, tau_init, c0]),
                bounds=(lb, ub),
                method="trf",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=150,  # Optimized: typical convergence in 6-12 nfev, 150 provides safety margin
                verbose=0,
            )

            if not res.success:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "nls_estimation_failed",
                        extra={
                            "event": "nls_estimation_failed",
                            "method": "nls_tau_unknown",
                            "message": res.message,
                            "nfev": res.nfev,
                        },
                    )
                return f0_init

            _, f_hat, _, _, _ = res.x

        # Sanity check
        if f_hat < 0 or f_hat > 0.5 * fs or abs(f_hat - f0_init) > 0.5 * f0_init:
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
            return f0_init

        return float(f_hat)


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
        """
        self.window = window
        self.use_zeropad = use_zeropad
        self.pad_factor = pad_factor
        self.lorentzian_points = lorentzian_points
        self.kaiser_beta = kaiser_beta

    def estimate(self, x: np.ndarray, fs: float, **kwargs) -> float:
        """
        Estimate frequency using DFT with Lorentzian fitting.

        Parameters:
        -----------
        x : np.ndarray
            Signal samples
        fs : float
            Sampling frequency (Hz)
        **kwargs
            Additional parameters (ignored)

        Returns:
        --------
        float
            Estimated frequency (Hz)
        """
        N = len(x)

        # Apply window
        if self.window == "kaiser":
            w = kaiser(N, self.kaiser_beta)
        elif self.window == "hann":
            w = np.hanning(N)
        elif self.window == "rect":
            w = np.ones(N)
        elif self.window == "blackman":
            w = np.blackman(N)
        else:
            raise ValueError(f"Unknown window: {self.window}")

        xw = x * w

        # Zero-padding for finer frequency grid
        if self.use_zeropad:
            N_pad = self.pad_factor * N
            xw_pad = np.zeros(N_pad, dtype=xw.dtype)
            xw_pad[:N] = xw
            N_dft = N_pad
        else:
            xw_pad = xw
            N_dft = N

        # Compute one-sided DFT
        X = np.fft.rfft(xw_pad)
        P = np.abs(X) ** 2

        # Find peak bin (skip DC component k=0)
        # Use view instead of copy to avoid memory allocation
        k = int(np.argmax(P[1:]) + 1)  # Skip first element, add 1 to index

        # Guard against edges
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
            return float(k * fs / N_dft)

        # Fit Lorentzian to power spectrum around peak
        delta = _fit_lorentzian_to_peak(P, k, fs, N_dft, n_points=self.lorentzian_points)

        f_hat = (k + delta) * fs / N_dft

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

        return float(f_hat)
