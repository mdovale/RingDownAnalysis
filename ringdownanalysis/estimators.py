"""
Frequency estimation methods for ring-down signals.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from scipy.optimize import least_squares, curve_fit
from scipy.signal.windows import kaiser


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
    return A / ((f - f0)**2 + (gamma / 2.0)**2) + offset


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
    half_range = n_points // 2
    k_start = max(0, k - half_range)
    k_end = min(len(P), k + half_range + 1)
    
    # Extract frequency and power values
    k_indices = np.arange(k_start, k_end)
    f_bins = k_indices * fs / N_dft
    P_bins = P[k_indices]
    
    # Initial parameter guesses
    P_max = P[k]
    f0_init = k * fs / N_dft
    
    # Estimate gamma (FWHM) from half-maximum points
    half_max = P_max / 2.0
    left_idx = k
    right_idx = k
    for i in range(k - 1, max(0, k - 10), -1):
        if P[i] < half_max:
            left_idx = i
            break
    for i in range(k + 1, min(len(P), k + 10)):
        if P[i] < half_max:
            right_idx = i
            break
    
    if right_idx > left_idx:
        gamma_init = (right_idx - left_idx) * fs / N_dft
    else:
        gamma_init = 2.0 * fs / N_dft
    
    # Estimate background offset from edges
    offset_init = np.min([P[0], P[-1], np.mean(P[:max(1, len(P)//20)])])
    
    # Initial amplitude guess
    A_init = P_max * (gamma_init / 2.0)**2
    
    # Fit Lorentzian
    try:
        popt, _ = curve_fit(
            _lorentzian_func,
            f_bins,
            P_bins,
            p0=[A_init, f0_init, gamma_init, offset_init],
            bounds=([
                0.0,
                f_bins[0],
                0.0,
                -np.inf,
            ], [
                np.inf,
                f_bins[-1],
                (f_bins[-1] - f_bins[0]) * 2.0,
                np.inf,
            ]),
            maxfev=1000,
        )
        
        A_fit, f0_fit, gamma_fit, offset_fit = popt
        
        # Calculate delta (offset from bin k)
        delta = (f0_fit - f0_init) / (fs / N_dft)
        
        # Clip delta to reasonable range
        delta = np.clip(delta, -0.5, 0.5)
        
        return delta
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return 0.0


def _estimate_initial_parameters_from_dft(x: np.ndarray, fs: float) -> tuple:
    """Estimate initial frequency, phase, amplitude, and DC offset from DFT."""
    N = len(x)
    X = np.fft.rfft(x * np.hanning(N))
    mag2 = np.abs(X) ** 2
    
    # Skip DC component (k=0) when finding peak
    mag2_no_dc = mag2.copy()
    mag2_no_dc[0] = 0.0
    k = int(np.argmax(mag2_no_dc))
    
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
    
    rms_values = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        rms_values.append(np.std(x[start:end]))
    
    rms_values = np.array(rms_values)
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
            Additional parameters (ignored)
        
        Returns:
        --------
        float
            Estimated frequency (Hz)
        """
        N = len(x)
        t = np.arange(N) / fs
        
        # Get initial parameter estimates
        f0_init, phi0_init, A0_init, c0 = _estimate_initial_parameters_from_dft(x, fs)
        
        if self.tau_known is not None:
            # Known tau: estimate (A0, f, phi, c)
            def residuals(p):
                A0, f, phi, c = p
                return (A0 * np.exp(-t / self.tau_known) * np.cos(2.0 * np.pi * f * t + phi) + c) - x
            
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
                max_nfev=500,
                verbose=0,
            )
            
            if not res.success:
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
                max_nfev=1000,
                verbose=0,
            )
            
            if not res.success:
                return f0_init
            
            _, f_hat, _, _, _ = res.x
        
        # Sanity check
        if f_hat < 0 or f_hat > 0.5 * fs or abs(f_hat - f0_init) > 0.5 * f0_init:
            return f0_init
        
        return float(f_hat)


class DFTFrequencyEstimator(FrequencyEstimator):
    """
    Frequency estimation using DFT peak fitting with Lorentzian function.
    """
    
    def __init__(
        self,
        window: str = "kaiser",
        use_zeropad: bool = False,
        pad_factor: int = 4,
        lorentzian_points: int = 7,
        kaiser_beta: float = 9.0,
    ):
        """
        Initialize DFT frequency estimator.
        
        Parameters:
        -----------
        window : str
            Window type: 'kaiser' (default), 'hann', 'rect', or 'blackman'
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
        P_no_dc = P.copy()
        P_no_dc[0] = 0.0
        k = int(np.argmax(P_no_dc))
        
        # Guard against edges
        if k <= 0 or k >= len(P) - 1:
            return float(k * fs / N_dft)
        
        # Fit Lorentzian to power spectrum around peak
        delta = _fit_lorentzian_to_peak(P, k, fs, N_dft, n_points=self.lorentzian_points)
        
        f_hat = (k + delta) * fs / N_dft
        return float(f_hat)

