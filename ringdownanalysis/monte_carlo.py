"""
Monte Carlo analysis for comparing frequency estimation methods.
"""

import numpy as np
from typing import Dict, Optional
try:
    from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
    import multiprocessing as mp
    HAS_MULTIPROCESSING = True
except ImportError:
    HAS_MULTIPROCESSING = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get('total', 0))

from .signal import RingDownSignal
from .estimators import FrequencyEstimator, NLSFrequencyEstimator, DFTFrequencyEstimator
from .crlb import CRLBCalculator


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
        (trial_idx, error_nls, error_dft, success_flags)
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
        return (trial_idx, None, None, {"generate": False, "error": str(e)})
    
    errors = {}
    results = {"nls": None, "dft": None}
    
    # Estimate frequency using each method
    try:
        f_hat_nls = nls_estimator.estimate(x, signal.fs)
        errors["nls"] = f_hat_nls - signal.f0
        results["nls"] = True
    except Exception as e:
        errors["nls"] = None
        results["nls"] = False
    
    try:
        f_hat_dft = dft_estimator.estimate(x, signal.fs)
        errors["dft"] = f_hat_dft - signal.f0
        results["dft"] = True
    except Exception as e:
        errors["dft"] = None
        results["dft"] = False
    
    return (trial_idx, errors["nls"], errors["dft"], results)


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
        self.dft_estimator = dft_estimator or DFTFrequencyEstimator(window="kaiser", use_zeropad=False)
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
            - 'crlb_std': CRLB standard deviation
            - 'errors_nls': frequency errors for NLS method
            - 'errors_dft': frequency errors for DFT method
            - 'stats': statistics for each method
        """
        # Create signal to compute derived parameters
        signal = RingDownSignal(f0=f0, fs=fs, N=N, A0=A0, snr_db=snr_db, Q=Q)
        tau = signal.tau
        T = signal.T
        
        # Calculate CRLB
        crlb_var = self.crlb_calc.variance(A0, signal.sigma, fs, N, tau)
        crlb_std = np.sqrt(crlb_var) if np.isfinite(crlb_var) else np.inf
        
        # Use parallel processing if available
        use_parallel = HAS_MULTIPROCESSING and n_mc > 10
        
        if use_parallel:
            if n_workers is None:
                n_workers = mp.cpu_count()
            print(f"Running {n_mc} Monte Carlo trials using {n_workers} workers...")
        else:
            n_workers = 1
            if not HAS_MULTIPROCESSING:
                print(f"Running {n_mc} Monte Carlo trials (sequential, multiprocessing not available)...")
            else:
                print(f"Running {n_mc} Monte Carlo trials (sequential)...")
        
        print(f"Parameters: f0={f0:.3f} Hz, fs={fs:.1f} Hz, N={N}, initial SNR={snr_db:.1f} dB")
        print(f"           Q={Q:.1e}, tau={tau:.2f} s, T/tau={T/tau:.2f}")
        print(f"CRLB std(f_hat) = {crlb_std:.6e} Hz (from explicit Fisher information)")
        print(f"Using optimized DFT: Kaiser window (beta=9.0) + Lorentzian fitting\n")
        
        # Prepare arguments for processing
        signal_params = {
            'f0': f0,
            'fs': fs,
            'N': N,
            'A0': A0,
            'snr_db': snr_db,
            'Q': Q,
        }
        trial_args = [
            (i, signal_params, self.nls_estimator, self.dft_estimator, seed)
            for i in range(n_mc)
        ]
        
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
                            trial_idx, err_nls, err_dft, success = future.result(timeout=timeout_per_trial)
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
                            print(f"\n  Warning: Trial {trial_idx} timed out after {timeout_per_trial}s")
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
                    print(f"  Completed {trial_idx+1}/{n_mc} trials...")
        
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

