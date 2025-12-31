"""
Monte Carlo analysis for comparing frequency estimation methods.
"""

import logging
from typing import Dict, Optional

import numpy as np

try:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as FutureTimeoutError

    HAS_MULTIPROCESSING = True
except ImportError:
    HAS_MULTIPROCESSING = False

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


from .crlb import CRLBCalculator
from .estimators import DFTFrequencyEstimator, FrequencyEstimator, NLSFrequencyEstimator
from .signal import RingDownSignal


def _estimate_freq_and_tau_nls(x: np.ndarray, fs: float):
    """
    Estimate both frequency and tau using NLS (when tau is unknown).

    This is a helper function that performs the full NLS fit and returns
    both f and tau estimates, which are needed to compute Q.

    Parameters:
    -----------
    x : np.ndarray
        Signal samples
    fs : float
        Sampling frequency (Hz)

    Returns:
    --------
    tuple
        (f_hat, tau_hat) or (None, None) if estimation fails
    """
    from scipy.optimize import least_squares

    from .estimators import (
        _estimate_initial_parameters_from_dft,
        _estimate_initial_tau_from_envelope,
    )

    N = len(x)
    t = np.arange(N) / fs

    # Get initial parameter estimates
    f0_init, phi0_init, A0_init, c0 = _estimate_initial_parameters_from_dft(x, fs)

    # Initial tau guess from envelope decay
    tau_init = _estimate_initial_tau_from_envelope(x, t)

    def residuals(p):
        A0, f, phi, tau, c = p
        return (A0 * np.exp(-t / tau) * np.cos(2.0 * np.pi * f * t + phi) + c) - x

    df = fs / N
    f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
    f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))

    lb = [0.0, f_low, -np.pi, t[1], -np.inf]
    ub = [10.0 * A0_init, f_high, np.pi, 10.0 * t[-1], np.inf]

    try:
        res = least_squares(
            residuals,
            x0=np.array([A0_init, f0_init, phi0_init, tau_init, c0]),
            bounds=(lb, ub),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=150,
            verbose=0,
        )

        if not res.success:
            return (None, None)

        _, f_hat, _, tau_hat, _ = res.x

        # Sanity check
        if f_hat < 0 or f_hat > 0.5 * fs or abs(f_hat - f0_init) > 0.5 * f0_init:
            return (None, None)
        if tau_hat <= 0 or tau_hat > 10.0 * t[-1]:
            return (None, None)

        return (float(f_hat), float(tau_hat))
    except Exception:
        return (None, None)


def _process_single_trial(args):
    """
    Process a single Monte Carlo trial (worker function for parallel processing).

    Parameters:
    -----------
    args : tuple
        (trial_idx, signal_params, nls_estimator, dft_estimator, base_seed)

    Returns:
    --------
    tuple
        (trial_idx, error_nls, error_dft, error_q_nls, success_flags)
    """
    trial_idx, signal_params, nls_estimator, dft_estimator, base_seed = args

    # Create independent RNG for this trial
    rng = np.random.default_rng(base_seed + trial_idx)

    # Create signal
    signal = RingDownSignal(**signal_params)

    # Generate ring-down signal
    try:
        _, x, _ = signal.generate(rng=rng)
    except Exception as e:
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                "mc_trial_signal_generation_failed",
                extra={
                    "event": "mc_trial_signal_generation_failed",
                    "trial_idx": trial_idx,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                },
            )
        return (trial_idx, None, None, None, {"generate": False, "error": str(e)})

    errors = {}
    results = {"nls": None, "dft": None}

    # Estimate frequency, tau, and Q using estimate_full() for streamlined Q estimation
    error_q_nls = None
    try:
        result_nls = nls_estimator.estimate_full(x, signal.fs)
        errors["nls"] = result_nls.f - signal.f0
        results["nls"] = True
        # Extract Q error if available
        if result_nls.Q is not None:
            error_q_nls = result_nls.Q - signal.Q
    except Exception as e:
        errors["nls"] = None
        results["nls"] = False
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "mc_trial_nls_failed",
                extra={
                    "event": "mc_trial_nls_failed",
                    "trial_idx": trial_idx,
                    "error_type": type(e).__name__,
                },
            )

    try:
        result_dft = dft_estimator.estimate_full(x, signal.fs)
        errors["dft"] = result_dft.f - signal.f0
        results["dft"] = True
    except Exception as e:
        errors["dft"] = None
        results["dft"] = False
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "mc_trial_dft_failed",
                extra={
                    "event": "mc_trial_dft_failed",
                    "trial_idx": trial_idx,
                    "error_type": type(e).__name__,
                },
            )

    return (trial_idx, errors["nls"], errors["dft"], error_q_nls, results)


class MonteCarloAnalyzer:
    """
    Performs Monte Carlo analysis comparing frequency estimation methods.
    """

    def __init__(
        self,
        nls_estimator: Optional[FrequencyEstimator] = None,
        dft_estimator: Optional[FrequencyEstimator] = None,
    ):
        """
        Initialize Monte Carlo analyzer.

        Parameters:
        -----------
        nls_estimator : FrequencyEstimator, optional
            NLS frequency estimator. If None, creates default.
        dft_estimator : FrequencyEstimator, optional
            DFT frequency estimator. If None, creates default.
        """
        self.nls_estimator = nls_estimator or NLSFrequencyEstimator(tau_known=None)
        self.dft_estimator = dft_estimator or DFTFrequencyEstimator(
            window="rect", use_zeropad=False
        )
        self.crlb_calc = CRLBCalculator()

    def run(
        self,
        f0: float = 5.0,
        fs: float = 100.0,
        N: int = 1_000_000,
        A0: float = 1.0,
        snr_db: float = 60.0,
        Q: float = 10000.0,
        n_mc: int = 100,
        seed: int = 42,
        n_workers: Optional[int] = None,
        timeout_per_trial: float = 30.0,
    ) -> Dict:
        """
        Perform Monte Carlo analysis.

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
            - 'crlb_std': CRLB standard deviation for frequency
            - 'crlb_std_q': CRLB standard deviation for Q
            - 'errors_nls': frequency errors for NLS method
            - 'errors_dft': frequency errors for DFT method
            - 'errors_q_nls': Q errors for NLS method
            - 'stats': statistics for each method
        """
        # Create signal to compute derived parameters
        signal = RingDownSignal(f0=f0, fs=fs, N=N, A0=A0, snr_db=snr_db, Q=Q)
        tau = signal.tau
        T = signal.T

        # Calculate CRLB for frequency
        crlb_var = self.crlb_calc.variance(A0, signal.sigma, fs, N, tau)
        crlb_std = np.sqrt(crlb_var) if np.isfinite(crlb_var) else np.inf

        # Calculate CRLB for Q
        crlb_var_q = self.crlb_calc.q_variance(A0, signal.sigma, fs, N, tau, f0)
        crlb_std_q = np.sqrt(crlb_var_q) if np.isfinite(crlb_var_q) else np.inf

        # Use parallel processing if available
        use_parallel = HAS_MULTIPROCESSING and n_mc > 10

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "mc_analysis_start",
                extra={
                    "event": "mc_analysis_start",
                    "n_trials": n_mc,
                    "f0": float(f0),
                    "fs": float(fs),
                    "N": int(N),
                    "snr_db": float(snr_db),
                    "Q": float(Q),
                    "tau": float(tau),
                    "crlb_std": float(crlb_std) if np.isfinite(crlb_std) else None,
                    "use_parallel": use_parallel,
                    "n_workers": n_workers if use_parallel else 1,
                },
            )

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
        print(f"CRLB std(Q_hat) = {crlb_std_q:.6e} (from explicit Fisher information)")
        print("Using optimized DFT: Rectangular window (no window) + Lorentzian fitting\n")

        # Prepare arguments for processing
        signal_params = {
            "f0": f0,
            "fs": fs,
            "N": N,
            "A0": A0,
            "snr_db": snr_db,
            "Q": Q,
        }
        trial_args = [
            (i, signal_params, self.nls_estimator, self.dft_estimator, seed) for i in range(n_mc)
        ]

        # Storage for errors
        errors_dict = {i: {"nls": None, "dft": None, "q_nls": None} for i in range(n_mc)}
        failure_counts = {"nls": 0, "dft": 0, "q_nls": 0}

        # Process trials with or without parallelization
        if use_parallel:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_process_single_trial, args): args[0] for args in trial_args
                }
                timeout_count = 0

                with tqdm(total=n_mc, desc="Monte Carlo trials", unit="trial") as pbar:
                    for future in as_completed(futures):
                        try:
                            trial_idx, err_nls, err_dft, err_q_nls, success = future.result(
                                timeout=timeout_per_trial
                            )
                            errors_dict[trial_idx] = {
                                "nls": err_nls,
                                "dft": err_dft,
                                "q_nls": err_q_nls,
                            }

                            if err_nls is None:
                                failure_counts["nls"] += 1
                            if err_dft is None:
                                failure_counts["dft"] += 1
                            if err_q_nls is None:
                                failure_counts["q_nls"] += 1
                        except FutureTimeoutError:
                            timeout_count += 1
                            trial_idx = futures[future]
                            logger.warning(
                                "mc_trial_timeout",
                                extra={
                                    "event": "mc_trial_timeout",
                                    "trial_idx": trial_idx,
                                    "timeout": timeout_per_trial,
                                },
                            )
                            print(
                                f"\n  Warning: Trial {trial_idx} timed out after {timeout_per_trial}s"
                            )
                            errors_dict[trial_idx] = {"nls": None, "dft": None, "q_nls": None}
                            failure_counts["nls"] += 1
                            failure_counts["dft"] += 1
                            failure_counts["q_nls"] += 1
                        except Exception as e:
                            trial_idx = futures[future]
                            logger.error(
                                "mc_trial_error",
                                extra={
                                    "event": "mc_trial_error",
                                    "trial_idx": trial_idx,
                                    "error_type": type(e).__name__,
                                    "error_msg": str(e),
                                },
                                exc_info=True,
                            )
                            print(f"\n  Warning: Trial {trial_idx} failed with error: {e}")
                            errors_dict[trial_idx] = {"nls": None, "dft": None, "q_nls": None}
                            failure_counts["nls"] += 1
                            failure_counts["dft"] += 1
                            failure_counts["q_nls"] += 1

                        pbar.update(1)

                if timeout_count > 0:
                    logger.warning(
                        "mc_timeouts_summary",
                        extra={
                            "event": "mc_timeouts_summary",
                            "n_timeouts": timeout_count,
                            "n_total": n_mc,
                        },
                    )
                    print(f"\n  Total timeouts: {timeout_count} out of {n_mc} trials")
        else:
            if HAS_TQDM:
                iterator = tqdm(trial_args, desc="Monte Carlo trials", unit="trial")
            else:
                iterator = trial_args

            for args in iterator:
                trial_idx, err_nls, err_dft, err_q_nls, success = _process_single_trial(args)
                errors_dict[trial_idx] = {
                    "nls": err_nls,
                    "dft": err_dft,
                    "q_nls": err_q_nls,
                }

                if err_nls is None:
                    failure_counts["nls"] += 1
                if err_dft is None:
                    failure_counts["dft"] += 1
                if err_q_nls is None:
                    failure_counts["q_nls"] += 1

                if not HAS_TQDM and (trial_idx + 1) % 20 == 0:
                    print(f"  Completed {trial_idx + 1}/{n_mc} trials...")

        # Extract errors
        errors_nls = [
            errors_dict[i]["nls"] for i in range(n_mc) if errors_dict[i]["nls"] is not None
        ]
        errors_dft = [
            errors_dict[i]["dft"] for i in range(n_mc) if errors_dict[i]["dft"] is not None
        ]
        errors_q_nls = [
            errors_dict[i]["q_nls"] for i in range(n_mc) if errors_dict[i]["q_nls"] is not None
        ]

        errors_nls = np.array(errors_nls)
        errors_dft = np.array(errors_dft)
        errors_q_nls = np.array(errors_q_nls)

        # Report failures
        if any(failure_counts.values()):
            logger.warning(
                "mc_failures_summary",
                extra={
                    "event": "mc_failures_summary",
                    "nls_failures": failure_counts["nls"],
                    "dft_failures": failure_counts["dft"],
                    "n_total": n_mc,
                },
            )
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
            "q_nls": {
                "mean": np.mean(errors_q_nls) if len(errors_q_nls) > 0 else np.nan,
                "std": np.std(errors_q_nls, ddof=1) if len(errors_q_nls) > 0 else np.nan,
                "rmse": np.sqrt(np.mean(errors_q_nls**2)) if len(errors_q_nls) > 0 else np.nan,
            },
        }

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "mc_analysis_complete",
                extra={
                    "event": "mc_analysis_complete",
                    "n_trials": n_mc,
                    "nls_std": float(stats["nls"]["std"])
                    if not np.isnan(stats["nls"]["std"])
                    else None,
                    "nls_bias": float(stats["nls"]["mean"])
                    if not np.isnan(stats["nls"]["mean"])
                    else None,
                    "dft_std": float(stats["dft"]["std"])
                    if not np.isnan(stats["dft"]["std"])
                    else None,
                    "dft_bias": float(stats["dft"]["mean"])
                    if not np.isnan(stats["dft"]["mean"])
                    else None,
                    "crlb_std": float(crlb_std) if np.isfinite(crlb_std) else None,
                    "q_nls_std": float(stats["q_nls"]["std"])
                    if len(errors_q_nls) > 0 and not np.isnan(stats["q_nls"]["std"])
                    else None,
                    "q_nls_bias": float(stats["q_nls"]["mean"])
                    if len(errors_q_nls) > 0 and not np.isnan(stats["q_nls"]["mean"])
                    else None,
                    "crlb_std_q": float(crlb_std_q) if np.isfinite(crlb_std_q) else None,
                },
            )

        print("\nResults:")
        print(f"  NLS:  std={stats['nls']['std']:.6e} Hz, bias={stats['nls']['mean']:.6e} Hz")
        print(f"  DFT:  std={stats['dft']['std']:.6e} Hz, bias={stats['dft']['mean']:.6e} Hz")
        print(f"  CRLB: std={crlb_std:.6e} Hz")
        if len(errors_q_nls) > 0:
            print(f"  NLS Q: std={stats['q_nls']['std']:.6e}, bias={stats['q_nls']['mean']:.6e}")
            print(f"  CRLB Q: std={crlb_std_q:.6e}")

        return {
            "f0": f0,
            "Q": Q,
            "tau": tau,
            "fs": fs,
            "N": N,
            "snr_db": snr_db,
            "crlb_std": crlb_std,
            "crlb_std_q": crlb_std_q,
            "errors_nls": errors_nls,
            "errors_dft": errors_dft,
            "errors_q_nls": errors_q_nls,
            "stats": stats,
        }
