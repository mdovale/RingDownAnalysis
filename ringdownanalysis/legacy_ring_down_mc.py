"""
Frequency Estimation Analysis for Ring-Down Signals: NLS and Optimized DFT Methods

This script performs Monte Carlo analysis comparing two frequency estimation methods for ring-down signals:
1. Nonlinear least squares (NLS) with ring-down model
2. Optimized DFT peak fitting (with Kaiser window, Lorentzian fitting for ring-down signals, and zero-padding)

Results are compared against the Cramér-Rao lower bound (CRLB) calculated from the explicit
Fisher information matrix for ring-down signals.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal.windows import kaiser

# Import plotting style and functions (applies automatically on import)
from .plots import (
    plot_aggregate_results,
    plot_individual_results,
    plot_performance_comparison,
)

try:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as FutureTimeoutError

    HAS_MULTIPROCESSING = True
except ImportError:
    HAS_MULTIPROCESSING = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Fallback: create a dummy tqdm that just passes through
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


# ============================================================================
# Utility functions
# ============================================================================


def db_to_lin(x_db: float) -> float:
    """Convert dB to linear scale."""
    return 10.0 ** (x_db / 10.0)


def crlb_var_f_ringdown_explicit(A0: float, sigma: float, fs: float, N: int, tau: float) -> float:
    """
    Cramér-Rao lower bound for frequency estimation variance with ring-down,
    calculated from explicit Fisher information matrix.

    For the ring-down model with known tau, the Fisher information matrix
    elements involve weighted sums S_0, S_1, S_2 of exp(-2t_n/tau) with
    different powers of t_n.

    Parameters:
    -----------
    A0 : float
        Initial amplitude of the sinusoid
    sigma : float
        Standard deviation of additive white Gaussian noise
    fs : float
        Sampling frequency (Hz)
    N : int
        Number of samples
    tau : float
        Decay time constant (s)

    Returns:
    --------
    float
        Lower bound on Var(f_hat)
    """
    Ts = 1.0 / fs
    T = N * Ts
    t = np.arange(N) * Ts

    # Calculate weighted sums explicitly
    # S_0 = sum_n exp(-2t_n/tau)
    # S_1 = sum_n t_n exp(-2t_n/tau)
    # S_2 = sum_n t_n^2 exp(-2t_n/tau)
    exp_factor = np.exp(-2.0 * t / tau)
    S_0 = np.sum(exp_factor)
    S_1 = np.sum(t * exp_factor)
    S_2 = np.sum(t**2 * exp_factor)

    # Effective Fisher information for omega (frequency in rad/s)
    # I_eff(omega) = (A0^2/sigma^2) * (S_2 - S_1^2/S_0)
    # This comes from the Schur complement accounting for nuisance parameters
    I_eff_omega = (A0**2 / sigma**2) * (S_2 - S_1**2 / S_0)

    if I_eff_omega < 1e-30:
        # Fallback for degenerate case
        return np.inf

    # CRLB for frequency in Hz: Var(f) = Var(omega)/(2pi)^2
    crlb_var_f = 1.0 / ((2.0 * np.pi) ** 2 * I_eff_omega)

    return crlb_var_f


# ============================================================================
# Signal generation
# ============================================================================


def generate_ringdown(
    f0: float,
    fs: float,
    N: int,
    A0: float = 1.0,
    snr_db: float = 60.0,
    Q: float = 10000.0,
    rng: np.random.Generator | None = None,
):
    """
    Generate a noisy ring-down signal (exponentially decaying sinusoid).

    Parameters:
    -----------
    f0 : float
        Frequency of the sinusoid (Hz)
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
    rng : np.random.Generator, optional
        Random number generator

    Returns:
    --------
    t : np.ndarray
        Time array (s)
    x : np.ndarray
        Noisy signal
    sigma : float
        Noise standard deviation
    phi0 : float
        Initial phase (rad)
    tau : float
        Decay time constant (s)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Time array
    t = np.arange(N) / fs

    # Decay time constant from Q-factor
    tau = Q / (np.pi * f0)

    # Random initial phase
    phi0 = rng.uniform(-np.pi, np.pi)

    # Set noise level from initial SNR
    rho0 = db_to_lin(snr_db)
    sigma = np.sqrt((A0**2 / 2.0) / rho0)

    # Generate ring-down signal: A(t) = A0 * exp(-t/tau)
    A_t = A0 * np.exp(-t / tau)
    s = A_t * np.cos(2.0 * np.pi * f0 * t + phi0)
    w = rng.normal(0.0, sigma, size=N)
    x = s + w

    return t, x, sigma, phi0, tau


# ============================================================================
# Frequency estimation methods
# ============================================================================


def _lorentzian_func(f: np.ndarray, A: float, f0: float, gamma: float, offset: float) -> np.ndarray:
    """
    Lorentzian function for power spectrum fitting.

    P(f) = A / ((f - f0)^2 + (gamma/2)^2) + offset

    Parameters:
    -----------
    f : np.ndarray
        Frequency array
    A : float
        Amplitude parameter
    f0 : float
        Center frequency
    gamma : float
        Full width at half maximum (FWHM)
    offset : float
        Background offset

    Returns:
    --------
    np.ndarray
        Power values
    """
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

    Parameters:
    -----------
    P : np.ndarray
        Power spectrum (magnitude squared)
    k : int
        Peak bin index
    fs : float
        Sampling frequency (Hz)
    N_dft : int
        DFT size
    n_points : int
        Number of points around peak to use for fitting (default: 7, i.e., k-3 to k+3)

    Returns:
    --------
    float
        Estimated frequency offset (delta) from bin k
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
    # Find where power drops to half of peak
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
        # Fallback: use a reasonable default based on typical ring-down width
        gamma_init = 2.0 * fs / N_dft

    # Estimate background offset from edges
    offset_init = np.min([P[0], P[-1], np.mean(P[: max(1, len(P) // 20)])])

    # Initial amplitude guess
    A_init = P_max * (gamma_init / 2.0) ** 2

    # Fit Lorentzian
    try:
        # Use curve_fit for robust fitting
        popt, _ = curve_fit(
            _lorentzian_func,
            f_bins,
            P_bins,
            p0=[A_init, f0_init, gamma_init, offset_init],
            bounds=(
                [
                    0.0,  # A > 0
                    f_bins[0],  # f0 within range
                    0.0,  # gamma > 0
                    -np.inf,  # offset can be negative
                ],
                [
                    np.inf,  # A
                    f_bins[-1],  # f0 within range
                    (f_bins[-1] - f_bins[0]) * 2.0,  # gamma reasonable upper bound
                    np.inf,  # offset
                ],
            ),
            maxfev=1000,
        )

        A_fit, f0_fit, gamma_fit, offset_fit = popt

        # Calculate delta (offset from bin k)
        delta = (f0_fit - f0_init) / (fs / N_dft)

        # Clip delta to reasonable range
        delta = np.clip(delta, -0.5, 0.5)

        return delta
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        # If fitting fails, fall back to simple peak location
        return 0.0


def estimate_freq_nls_ringdown(x: np.ndarray, fs: float, tau_known: float = None) -> float:
    """
    Estimate frequency using nonlinear least squares with ring-down model.

    Parameters:
    -----------
    x : np.ndarray
        Signal samples
    fs : float
        Sampling frequency (Hz)
    tau_known : float, optional
        Known decay time constant. If None, tau is estimated along with other parameters.

    Returns:
    --------
    float
        Estimated frequency (Hz)
    """
    N = len(x)
    t = np.arange(N) / fs

    # Initial frequency guess from DFT peak with Lorentzian fitting
    X = np.fft.rfft(x * np.hanning(N))
    mag2 = np.abs(X) ** 2
    k = int(np.argmax(mag2))
    if k > 0 and k < len(mag2) - 1:
        # Use Lorentzian fitting for initial guess (more appropriate for ring-down)
        delta = _fit_lorentzian_to_peak(mag2, k, fs, N, n_points=7)
        k_interp = k + delta
    else:
        k_interp = k
    f0_init = k_interp * fs / N

    # Initial phase estimation using DFT phase at peak
    phi0_init = np.angle(X[k])

    # Initial amplitude estimation
    A0_init = np.sqrt(2.0) * np.sqrt(mag2[k] / N)
    if A0_init < 0.1 * np.std(x) or A0_init > 10 * np.std(x):
        A0_init = np.std(x) * np.sqrt(2.0)

    c0 = np.mean(x)

    if tau_known is not None:
        # Known tau: estimate (A0, f, phi, c)
        def residuals(p):
            A0, f, phi, c = p
            return (A0 * np.exp(-t / tau_known) * np.cos(2.0 * np.pi * f * t + phi) + c) - x

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
        # Initial tau guess: estimate from signal envelope decay
        # Use a simple approach: find where RMS in windows drops significantly
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
        if len(decay_idx) > 0:
            tau_init = t[decay_idx[0] * window_size] if decay_idx[0] > 0 else t[-1] / 2.0
        else:
            tau_init = t[-1] / 2.0

        def residuals(p):
            A0, f, phi, tau, c = p
            return (A0 * np.exp(-t / tau) * np.cos(2.0 * np.pi * f * t + phi) + c) - x

        df = fs / N
        f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
        f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))

        lb = [0.0, f_low, -np.pi, t[1], -np.inf]  # tau must be > 0
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


def estimate_freq_dft_optimized(
    x: np.ndarray,
    fs: float,
    window: str = "kaiser",
    use_zeropad: bool = True,
    pad_factor: int = 4,
    lorentzian_points: int = 7,
    kaiser_beta: float = 9.0,
) -> float:
    """
    Optimized DFT-based frequency estimation with Lorentzian fitting for ring-down signals.

    For ring-down signals, the Fourier transform has a Lorentzian shape, so fitting
    a Lorentzian function is more appropriate than parabolic interpolation.

    Currently implemented:
    - Kaiser window with optimized parameters for high side-lobe suppression (default)
    - Zero-padding for finer frequency grid (when use_zeropad=True)
    - Lorentzian fitting for peak location (appropriate for ring-down signals)

    Parameters:
    -----------
    x : np.ndarray
        Signal samples
    fs : float
        Sampling frequency (Hz)
    window : str
        Window type: 'kaiser' (default), 'hann', 'rect', or 'blackman'
    use_zeropad : bool
        Use zero-padding for finer frequency grid (default: True)
    pad_factor : int
        Zero-padding factor: DFT size = pad_factor * N (default: 4)
    lorentzian_points : int
        Number of points around peak to use for Lorentzian fitting (default: 7)
    kaiser_beta : float
        Kaiser window beta parameter for side-lobe suppression (default: 9.0)
        Higher values provide better side-lobe suppression but wider main lobe

    Returns:
    --------
    float
        Estimated frequency (Hz)
    """
    N = len(x)

    # Apply window
    if window == "kaiser":
        w = kaiser(N, kaiser_beta)
    elif window == "hann":
        w = np.hanning(N)
    elif window == "rect":
        w = np.ones(N)
    elif window == "blackman":
        w = np.blackman(N)
    else:
        raise ValueError(f"Unknown window: {window}")

    xw = x * w

    # Zero-padding for finer frequency grid
    if use_zeropad:
        N_pad = pad_factor * N
        xw_pad = np.zeros(N_pad, dtype=xw.dtype)
        xw_pad[:N] = xw
        N_dft = N_pad
    else:
        xw_pad = xw
        N_dft = N

    # Compute one-sided DFT
    X = np.fft.rfft(xw_pad)
    P = np.abs(X) ** 2

    # Find peak bin
    k = int(np.argmax(P))

    # Guard against edges
    if k <= 0 or k >= len(P) - 1:
        return float(k * fs / N_dft)

    # Fit Lorentzian to power spectrum around peak
    # This is more appropriate for ring-down signals than parabolic interpolation
    delta = _fit_lorentzian_to_peak(P, k, fs, N_dft, n_points=lorentzian_points)

    f_hat = (k + delta) * fs / N_dft
    return float(f_hat)


# Backward compatibility: keep the old function name
def estimate_freq_dft(
    x: np.ndarray, fs: float, window: str = "kaiser", kaiser_beta: float = 9.0
) -> float:
    """
    Estimate frequency using DFT peak fitting with Lorentzian function.

    This is a wrapper around estimate_freq_dft_optimized.
    Uses Lorentzian fitting for peak location, which is more appropriate for ring-down
    signals than parabolic interpolation since the Fourier transform has a Lorentzian shape.
    Zero-padding disabled (was found to be less optimal).
    Default window is Kaiser with beta=9.0 for high side-lobe suppression.

    Parameters:
    -----------
    x : np.ndarray
        Signal samples
    fs : float
        Sampling frequency (Hz)
    window : str
        Window type: 'kaiser' (default), 'hann', 'rect', or 'blackman'
    kaiser_beta : float
        Kaiser window beta parameter (default: 9.0)

    Returns:
    --------
    float
        Estimated frequency (Hz)
    """
    # Use Lorentzian fitting for ring-down signals
    # Zero-padding was tested but found to be slightly less optimal, so disabled
    return estimate_freq_dft_optimized(
        x,
        fs,
        window=window,
        use_zeropad=False,
        pad_factor=1,
        lorentzian_points=7,
        kaiser_beta=kaiser_beta,
    )


# ============================================================================
# Monte Carlo analysis
# ============================================================================


def _process_single_trial(args):
    """
    Process a single Monte Carlo trial (worker function for parallel processing).

    Parameters:
    -----------
    args : tuple
        (trial_idx, f0, fs, N, A0, snr_db, Q, base_seed)

    Returns:
    --------
    tuple
        (trial_idx, error_nls, error_dft, success_flags)
    """
    trial_idx, f0, fs, N, A0, snr_db, Q, base_seed = args

    # Create independent RNG for this trial
    rng = np.random.default_rng(base_seed + trial_idx)

    # Generate ring-down signal
    try:
        _, x, _, _, tau = generate_ringdown(f0, fs, N, A0, snr_db, Q, rng)
    except Exception as e:
        return (trial_idx, None, None, {"generate": False, "error": str(e)})

    results = {"nls": None, "dft": None}
    errors = {}

    # Estimate frequency using each method
    try:
        # Use known tau for NLS (more realistic if Q is known)
        f_hat_nls = estimate_freq_nls_ringdown(x, fs, tau_known=None)
        errors["nls"] = f_hat_nls - f0
        results["nls"] = True
    except Exception:
        errors["nls"] = None
        results["nls"] = False

    try:
        # Use optimized DFT with Lorentzian fitting
        f_hat_dft = estimate_freq_dft(x, fs, window="kaiser")
        errors["dft"] = f_hat_dft - f0
        results["dft"] = True
    except Exception:
        errors["dft"] = None
        results["dft"] = False

    return (trial_idx, errors["nls"], errors["dft"], results)


def monte_carlo_analysis(
    f0: float = 5.0,
    fs: float = 100.0,
    N: int = 1_000_000,
    A0: float = 1.0,
    snr_db: float = 60.0,
    Q: float = 10000.0,
    n_mc: int = 100,
    seed: int = 42,
    n_workers: int = None,
    timeout_per_trial: float = 30.0,
):
    """
    Perform Monte Carlo analysis of frequency estimation methods for ring-down signals.

    Parameters:
    -----------
    f0 : float
        True frequency (Hz), default: 5.0
    fs : float
        Sampling frequency (Hz), default: 100.0
    N : int
        Number of samples, default: 1_000_000
    A0 : float
        Initial signal amplitude, default: 1.0
    snr_db : float
        Initial signal-to-noise ratio (dB), default: 60.0
    Q : float
        Quality factor, default: 10000.0
    n_mc : int
        Number of Monte Carlo runs, default: 100
    seed : int
        Random seed, default: 42
    n_workers : int, optional
        Number of parallel workers. If None, uses all available CPUs.
    timeout_per_trial : float, optional
        Maximum time (seconds) allowed per trial before timing out. Default: 30.0.

    Returns:
    --------
    dict
        Results dictionary containing:
        - 'f0': true frequency
        - 'Q': quality factor
        - 'tau': decay time constant
        - 'crlb_std': CRLB standard deviation from explicit Fisher information
        - 'errors_nls': frequency errors for NLS method
        - 'errors_dft': frequency errors for DFT method
        - 'stats': statistics for each method
    """
    # Calculate decay time constant
    tau = Q / (np.pi * f0)
    T = N / fs

    # Calculate CRLB from explicit Fisher information
    rho0 = db_to_lin(snr_db)
    sigma = np.sqrt((A0**2 / 2.0) / rho0)
    crlb_var = crlb_var_f_ringdown_explicit(A0, sigma, fs, N, tau)
    crlb_std = np.sqrt(crlb_var)

    # Use parallel processing if available
    use_parallel = HAS_MULTIPROCESSING and n_mc > 10

    if use_parallel:
        if n_workers is None:
            n_workers = mp.cpu_count()
        print(f"Running {n_mc} Monte Carlo trials using {n_workers} workers...")
    else:
        n_workers = 1
        if not HAS_MULTIPROCESSING:
            print(
                f"Running {n_mc} Monte Carlo trials (sequential, multiprocessing not available)..."
            )
        else:
            print(f"Running {n_mc} Monte Carlo trials (sequential)...")

    print(f"Parameters: f0={f0:.3f} Hz, fs={fs:.1f} Hz, N={N}, initial SNR={snr_db:.1f} dB")
    print(f"           Q={Q:.1e}, tau={tau:.2f} s, T/tau={T / tau:.2f}")
    print(f"CRLB std(f_hat) = {crlb_std:.6e} Hz (from explicit Fisher information)")
    print("Using optimized DFT: Kaiser window (beta=9.0) + Lorentzian fitting\n")

    # Prepare arguments for processing
    trial_args = [(i, f0, fs, N, A0, snr_db, Q, seed) for i in range(n_mc)]

    # Storage for errors
    errors_dict = {i: {"nls": None, "dft": None} for i in range(n_mc)}
    failure_counts = {"nls": 0, "dft": 0}

    # Process trials with or without parallelization
    if use_parallel:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_single_trial, args): args[0] for args in trial_args}
            timeout_count = 0

            with tqdm(total=n_mc, desc="Monte Carlo trials", unit="trial") as pbar:
                for future in as_completed(futures):
                    try:
                        trial_idx, err_nls, err_dft, success = future.result(
                            timeout=timeout_per_trial
                        )
                        errors_dict[trial_idx] = {
                            "nls": err_nls,
                            "dft": err_dft,
                        }

                        if err_nls is None:
                            failure_counts["nls"] += 1
                        if err_dft is None:
                            failure_counts["dft"] += 1
                    except FutureTimeoutError:
                        timeout_count += 1
                        trial_idx = futures[future]
                        print(
                            f"\n  Warning: Trial {trial_idx} timed out after {timeout_per_trial}s"
                        )
                        errors_dict[trial_idx] = {"nls": None, "dft": None}
                        failure_counts["nls"] += 1
                        failure_counts["dft"] += 1
                    except Exception as e:
                        trial_idx = futures[future]
                        print(f"\n  Warning: Trial {trial_idx} failed with error: {e}")
                        errors_dict[trial_idx] = {"nls": None, "dft": None}
                        failure_counts["nls"] += 1
                        failure_counts["dft"] += 1

                    pbar.update(1)

            if timeout_count > 0:
                print(f"\n  Total timeouts: {timeout_count} out of {n_mc} trials")
    else:
        if HAS_TQDM:
            iterator = tqdm(trial_args, desc="Monte Carlo trials", unit="trial")
        else:
            iterator = trial_args

        for args in iterator:
            trial_idx, err_nls, err_dft, success = _process_single_trial(args)
            errors_dict[trial_idx] = {
                "nls": err_nls,
                "dft": err_dft,
            }

            if err_nls is None:
                failure_counts["nls"] += 1
            if err_dft is None:
                failure_counts["dft"] += 1

            if not HAS_TQDM and (trial_idx + 1) % 20 == 0:
                print(f"  Completed {trial_idx + 1}/{n_mc} trials...")

    # Extract errors
    errors_nls = [errors_dict[i]["nls"] for i in range(n_mc) if errors_dict[i]["nls"] is not None]
    errors_dft = [errors_dict[i]["dft"] for i in range(n_mc) if errors_dict[i]["dft"] is not None]

    errors_nls = np.array(errors_nls)
    errors_dft = np.array(errors_dft)

    # Report failures
    if any(failure_counts.values()):
        print("\nWarnings:")
        for method, count in failure_counts.items():
            if count > 0:
                print(f"  {method.upper()}: {count} failures out of {n_mc} trials")

    # Calculate statistics
    stats = {
        "nls": {
            "mean": np.mean(errors_nls) if len(errors_nls) > 0 else np.nan,
            "std": np.std(errors_nls, ddof=1) if len(errors_nls) > 0 else np.nan,
            "rmse": np.sqrt(np.mean(errors_nls**2)) if len(errors_nls) > 0 else np.nan,
        },
        "dft": {
            "mean": np.mean(errors_dft) if len(errors_dft) > 0 else np.nan,
            "std": np.std(errors_dft, ddof=1) if len(errors_dft) > 0 else np.nan,
            "rmse": np.sqrt(np.mean(errors_dft**2)) if len(errors_dft) > 0 else np.nan,
        },
    }

    print("\nResults:")
    print(f"  NLS:  std={stats['nls']['std']:.6e} Hz, bias={stats['nls']['mean']:.6e} Hz")
    print(f"  DFT:  std={stats['dft']['std']:.6e} Hz, bias={stats['dft']['mean']:.6e} Hz")
    print(f"  CRLB: std={crlb_std:.6e} Hz")

    return {
        "f0": f0,
        "Q": Q,
        "tau": tau,
        "fs": fs,
        "N": N,
        "snr_db": snr_db,
        "crlb_std": crlb_std,
        "errors_nls": errors_nls,
        "errors_dft": errors_dft,
        "stats": stats,
    }


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Default parameters
    f0 = 5.0  # Hz
    fs = 100.0  # Hz
    N = 1_000_000  # samples
    snr_db = 60.0  # dB (initial SNR)
    Q = 10000.0  # Quality factor
    n_mc = 100  # Monte Carlo trials

    print("=" * 70)
    print("Frequency Estimation Analysis for Ring-Down Signals")
    print("NLS and Optimized DFT Methods (with Lorentzian Fitting)")
    print("=" * 70)
    print()

    # Run Monte Carlo analysis
    results = monte_carlo_analysis(
        f0=f0,
        fs=fs,
        N=N,
        A0=1.0,
        snr_db=snr_db,
        Q=Q,
        n_mc=n_mc,
    )

    print()
    print("=" * 70)
    print("Generating plots...")
    print("=" * 70)

    # Generate plots
    fig1 = plot_individual_results(results)
    plt.savefig("freq_estimation_ringdown_v6_individual.pdf", bbox_inches="tight")
    print("  Saved: freq_estimation_ringdown_v6_individual.pdf")

    fig2 = plot_aggregate_results(results)
    plt.savefig("freq_estimation_ringdown_v6_aggregate.pdf", bbox_inches="tight")
    print("  Saved: freq_estimation_ringdown_v6_aggregate.pdf")

    fig3 = plot_performance_comparison(results)
    plt.savefig("freq_estimation_ringdown_v6_performance.pdf", bbox_inches="tight")
    print("  Saved: freq_estimation_ringdown_v6_performance.pdf")

    plt.show()

    print()
    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)
