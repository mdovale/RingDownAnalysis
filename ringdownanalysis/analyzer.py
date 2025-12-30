"""
Analysis pipeline for real ring-down measurement data.
"""

import logging
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from scipy.optimize import least_squares

from .data_loader import RingDownDataLoader
from .estimators import NLSFrequencyEstimator, DFTFrequencyEstimator, _estimate_initial_parameters_from_dft, _estimate_initial_tau_from_envelope
from .crlb import CRLBCalculator

logger = logging.getLogger(__name__)


class RingDownAnalyzer:
    """
    Analyzes real ring-down measurement data.
    
    Performs the following pipeline:
    1. Load data from file
    2. Estimate tau from full data using NLS
    3. Crop data to max_tau_multiplier*tau to avoid long noisy tail
    4. Estimate frequency using NLS and DFT methods
    5. Estimate noise parameters for CRLB calculation
    6. Calculate CRLB
    """
    
    def __init__(
        self,
        nls_estimator: Optional[NLSFrequencyEstimator] = None,
        dft_estimator: Optional[DFTFrequencyEstimator] = None,
    ):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        nls_estimator : NLSFrequencyEstimator, optional
            NLS frequency estimator. If None, creates default (tau unknown).
        dft_estimator : DFTFrequencyEstimator, optional
            DFT frequency estimator. If None, creates default (Kaiser window).
        """
        self.nls_estimator = nls_estimator or NLSFrequencyEstimator(tau_known=None)
        self.dft_estimator = dft_estimator or DFTFrequencyEstimator(window="kaiser")
        self.crlb_calc = CRLBCalculator()
    
    def estimate_tau(
        self,
        data: np.ndarray,
        t: np.ndarray,
        fs: float,
        initial_params: Optional[tuple] = None,
    ) -> float:
        """
        Estimate tau from full data using NLS fit.
        
        Parameters:
        -----------
        data : np.ndarray
            Signal data
        t : np.ndarray
            Time array (s)
        fs : float
            Sampling frequency (Hz)
        
        Returns:
        --------
        float
            Estimated tau value in seconds
        """
        N = len(data)
        t_norm = t - t[0]
        
        # Get initial parameter estimates (use cached if provided)
        if initial_params is not None:
            f0_init, phi0_init, A0_init, c0 = initial_params
        else:
            f0_init, phi0_init, A0_init, c0 = _estimate_initial_parameters_from_dft(data, fs)
        
        # Initial tau guess from envelope decay
        tau_init = _estimate_initial_tau_from_envelope(data, t_norm)
        
        # NLS fit to estimate tau: fit (A0, f, phi, tau, c)
        def residuals_tau(p):
            A0, f, phi, tau, c = p
            return (A0 * np.exp(-t_norm / tau) * np.cos(2.0 * np.pi * f * t_norm + phi) + c) - data
        
        df = fs / N
        f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
        f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))
        
        lb = [0.0, f_low, -np.pi, t_norm[1], -np.inf]
        ub = [10.0 * A0_init, f_high, np.pi, 10.0 * t_norm[-1], np.inf]
        
        res_tau = least_squares(
            residuals_tau,
            x0=np.array([A0_init, f0_init, phi0_init, tau_init, c0]),
            bounds=(lb, ub),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=150,  # Optimized: typical convergence in 6-12 nfev, 150 provides safety margin
            verbose=0,
        )
        
        if res_tau.success:
            _, _, _, tau_est, _ = res_tau.x
            # Sanity check
            if tau_est <= 0 or not np.isfinite(tau_est) or tau_est > 10.0 * t_norm[-1] or tau_est < t_norm[1]:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "tau_sanity_check_failed",
                        extra={
                            "event": "tau_sanity_check_failed",
                            "tau_est": float(tau_est),
                            "tau_init": float(tau_init),
                            "t_max": float(t_norm[-1]),
                        },
                    )
                return tau_init
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "tau_estimated",
                    extra={
                        "event": "tau_estimated",
                        "tau_est": float(tau_est),
                        "tau_init": float(tau_init),
                        "nfev": res_tau.nfev,
                    },
                )
            return tau_est
        else:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "tau_estimation_failed",
                    extra={
                        "event": "tau_estimation_failed",
                        "tau_init": float(tau_init),
                        "message": res_tau.message,
                        "nfev": res_tau.nfev,
                    },
                )
            return tau_init
    
    def crop_data_to_tau(
        self,
        t: np.ndarray,
        data: np.ndarray,
        tau_est: float,
        min_samples: int = 100,
        max_tau_multiplier: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop data to max_tau_multiplier*tau_est to avoid long noisy tail affecting frequency estimation.
        
        Parameters:
        -----------
        t : np.ndarray
            Time array
        data : np.ndarray
            Signal array
        tau_est : float
            Estimated tau value in seconds
        min_samples : int
            Minimum number of samples required. If cropped data is shorter, return original.
        max_tau_multiplier : float
            Multiplier for tau_est to determine maximum record length. Default is 1.0.
        
        Returns:
        --------
        (t_crop, data_cropped) : tuple
            Cropped time and signal arrays
        """
        t_crop_max = max_tau_multiplier * tau_est
        crop_idx = t <= t_crop_max
        t_crop = t[crop_idx]
        data_cropped = data[crop_idx]
        
        # If cropped data is too short, return original
        # Use views instead of copies when possible
        if len(t_crop) < min_samples:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "data_crop_too_short",
                    extra={
                        "event": "data_crop_too_short",
                        "n_cropped": len(t_crop),
                        "min_samples": min_samples,
                        "tau_est": float(tau_est),
                    },
                )
            return t, data
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "data_cropped",
                extra={
                    "event": "data_cropped",
                    "n_original": len(t),
                    "n_cropped": len(t_crop),
                    "tau_est": float(tau_est),
                    "crop_time": float(t_crop[-1]) if len(t_crop) > 0 else 0.0,
                },
            )
        
        return t_crop, data_cropped
    
    def estimate_noise_parameters(
        self,
        data_cropped: np.ndarray,
        t_crop: np.ndarray,
        tau_est: float,
        fs: float,
        initial_params: Optional[tuple] = None,
    ) -> tuple[float, float]:
        """
        Estimate A0 (initial amplitude) and sigma (noise std) from cropped data.
        
        Parameters:
        -----------
        data_cropped : np.ndarray
            Cropped signal data
        t_crop : np.ndarray
            Cropped time array
        tau_est : float
            Estimated tau value in seconds
        fs : float
            Sampling frequency (Hz)
        
        Returns:
        --------
        (A0_est, sigma_est) : tuple
            Estimated A0 and sigma
        """
        N_crop = len(data_cropped)
        t_crop_norm = t_crop - t_crop[0]
        
        # Initial estimate from first portion
        n_init = min(1000, N_crop // 10)
        A0_est = np.sqrt(2.0) * np.std(data_cropped[:n_init])
        
        # Fit model to get residuals for noise estimation
        def model_residuals(p):
            A0, f, phi, c = p
            return (A0 * np.exp(-t_crop_norm / tau_est) * np.cos(2.0 * np.pi * f * t_crop_norm + phi) + c) - data_cropped
        
        # Get initial guesses (use cached if provided)
        if initial_params is not None:
            f0_init, phi0_init, A0_init, c0 = initial_params
        else:
            f0_init, phi0_init, A0_init, c0 = _estimate_initial_parameters_from_dft(data_cropped, fs)
        
        # Quick fit to get residuals
        df = fs / N_crop
        f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
        f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))
        
        res_fit = least_squares(
            model_residuals,
            x0=np.array([A0_init, f0_init, phi0_init, c0]),
            bounds=([0.0, f_low, -np.pi, -np.inf], [10.0 * A0_init, f_high, np.pi, np.inf]),
            method="trf",
            ftol=1e-6,
            max_nfev=100,  # Optimized: typical convergence in 5-10 nfev, 100 provides safety margin
            verbose=0,
        )
        
        if res_fit.success:
            residuals = res_fit.fun
            sigma_est = np.std(residuals)
            A0_est = res_fit.x[0]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "noise_parameters_estimated",
                    extra={
                        "event": "noise_parameters_estimated",
                        "A0_est": float(A0_est),
                        "sigma_est": float(sigma_est),
                        "nfev": res_fit.nfev,
                    },
                )
        else:
            # Fallback: estimate noise from tail
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "noise_estimation_fallback",
                    extra={
                        "event": "noise_estimation_fallback",
                        "message": res_fit.message,
                        "nfev": res_fit.nfev,
                    },
                )
            tail_start = max(int(0.8 * len(data_cropped)), len(data_cropped) - 1000)
            sigma_est = np.std(data_cropped[tail_start:])
            A0_est = np.sqrt(2.0) * np.std(data_cropped[:n_init])
        
        return A0_est, sigma_est
    
    def analyze_file(self, filepath: str, max_tau_multiplier: float = 1.0) -> Dict:
        """
        Process a single data file and return analysis results.
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
        max_tau_multiplier : float
            Multiplier for tau_est to determine maximum record length when cropping data.
            Default is 1.0.
        
        Returns:
        --------
        dict
            Results dictionary with all analysis data
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_start",
                extra={
                    "event": "analysis_start",
                    "filepath": str(filepath),
                },
            )
        
        # Load data
        t, data, V2, file_type = RingDownDataLoader.load(filepath)
        
        # Calculate sampling frequency
        fs = 1.0 / np.mean(np.diff(t))
        
        # Compute initial parameters once and reuse them
        initial_params_full = _estimate_initial_parameters_from_dft(data, fs)
        
        # Estimate tau from full data (using cached initial params)
        tau_est = self.estimate_tau(data, t, fs, initial_params=initial_params_full)
        
        # Crop data to max_tau_multiplier*tau_est
        t_crop, data_cropped = self.crop_data_to_tau(t, data, tau_est, min_samples=1000, max_tau_multiplier=max_tau_multiplier)
        
        # Warn if cropped data is too short
        min_samples_for_analysis = 1000
        if len(t_crop) < min_samples_for_analysis:
            t_crop = t
            data_cropped = data
        
        # Compute initial parameters for cropped data once
        initial_params_cropped = _estimate_initial_parameters_from_dft(data_cropped, fs)
        
        # Estimate frequencies on cropped data
        f_nls = self.nls_estimator.estimate(data_cropped, fs, initial_params=initial_params_cropped)
        f_dft = self.dft_estimator.estimate(data_cropped, fs)
        
        # Estimate noise parameters (using cached initial params)
        A0_est, sigma_est = self.estimate_noise_parameters(data_cropped, t_crop, tau_est, fs, initial_params=initial_params_cropped)
        
        # Calculate CRLB
        N_crop = len(data_cropped)
        crlb_var_f = self.crlb_calc.variance(A0_est, sigma_est, fs, N_crop, tau_est)
        crlb_std_f = np.sqrt(crlb_var_f) if np.isfinite(crlb_var_f) else np.inf
        
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_complete",
                extra={
                    "event": "analysis_complete",
                    "filepath": str(filepath),
                    "f_nls": float(f_nls),
                    "f_dft": float(f_dft),
                    "tau_est": float(tau_est),
                    "crlb_std_f": float(crlb_std_f) if np.isfinite(crlb_std_f) else None,
                },
            )
        
        return {
            'filename': Path(filepath).name,
            'type': file_type,
            't': t,
            'data': data,
            'V2': V2,
            't_crop': t_crop,
            'data_cropped': data_cropped,
            'fs': fs,
            'tau_est': tau_est,
            'f_nls': f_nls,
            'f_dft': f_dft,
            'A0_est': A0_est,
            'sigma_est': sigma_est,
            'crlb_std_f': crlb_std_f,
            'N': len(t),
            'N_crop': len(t_crop),
            'T': t[-1],
            'T_crop': t_crop[-1] if len(t_crop) > 0 else 0
        }

