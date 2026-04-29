"""
Analysis pipeline for real ring-down measurement data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import least_squares

from .crlb import CRLBCalculator
from .data_loader import RingDownDataLoader
from .estimators import (
    DFTFrequencyEstimator,
    NLSFrequencyEstimator,
    _estimate_initial_parameters_from_dft,
    _estimate_initial_tau_from_envelope,
    _estimate_initial_tau_with_method,
    _sanitize_initial_parameters,
    _sanitize_tau_guess,
)

logger = logging.getLogger(__name__)


class NoiseEstimate(NamedTuple):
    """Noise and amplitude estimates for plug-in uncertainty calculations."""

    A0: float
    sigma: float
    sigma_mle: float
    noise_dof: int
    success: bool
    method: str
    message: str | None = None


def _parse_array_input(
    t: np.ndarray | None = None,
    data: np.ndarray | None = None,
    fs: float | None = None,
    *,
    time_col: str | int = 0,
    data_col: str | int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse array-like inputs into (t, data) numpy arrays.

    Supports:
    - (t, data): two array-likes
    - (data, fs): data array and sampling frequency (t = arange(len(data))/fs)
    - (data=DataFrame): extract t and data from columns via time_col, data_col

    Returns:
    --------
    t : np.ndarray
        Time array (s), starting from 0
    data : np.ndarray
        Signal array
    """
    import pandas as pd

    if data is None:
        raise ValueError("data is required")

    # DataFrame: extract time and signal columns
    if isinstance(data, pd.DataFrame):
        t_arr = np.asarray(
            data.iloc[:, time_col] if isinstance(time_col, int) else data[time_col],
            dtype=np.float64,
        )
        data_arr = np.asarray(
            data.iloc[:, data_col] if isinstance(data_col, int) else data[data_col],
            dtype=np.float64,
        )
        t_arr = t_arr - t_arr[0]
    elif t is not None:
        t_arr = np.asarray(t, dtype=np.float64)
        data_arr = np.asarray(data, dtype=np.float64)
    elif fs is not None:
        data_arr = np.asarray(data, dtype=np.float64)
        t_arr = np.arange(len(data_arr), dtype=np.float64) / fs
    else:
        raise ValueError("Either t or fs must be provided when data is not a DataFrame")

    if len(t_arr) != len(data_arr):
        raise ValueError(f"t and data must have same length, got {len(t_arr)} and {len(data_arr)}")
    if len(t_arr) < 2:
        raise ValueError("At least 2 samples required for analysis")
    if not np.all(np.isfinite(data_arr)):
        raise ValueError("Signal data must contain only finite values")

    return t_arr, data_arr


def _validate_signal_data(data: np.ndarray, *, source: str = "Signal data") -> np.ndarray:
    """Return a 1D finite signal array or raise a clear ValueError."""
    data_arr = np.asarray(data, dtype=np.float64)
    if data_arr.ndim != 1:
        raise ValueError(f"{source} must be 1-dimensional, got shape {data_arr.shape}")
    if len(data_arr) < 2:
        raise ValueError("At least 2 samples required for analysis")
    if not np.all(np.isfinite(data_arr)):
        raise ValueError(f"{source} must contain only finite values")
    return data_arr


def _validate_max_tau_multiplier(max_tau_multiplier: float) -> float:
    """Validate crop multiplier before it can silently produce an empty crop."""
    value = float(max_tau_multiplier)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"max_tau_multiplier must be positive and finite, got {max_tau_multiplier}"
        )
    return value


def _near_bound(
    value: float | None, lower: float, upper: float, *, rtol: float = 1e-4
) -> tuple[bool, bool]:
    """Return lower/upper bound-hit flags for tau-like estimates."""
    if value is None or not np.isfinite(value):
        return False, False
    span = max(abs(upper - lower), abs(upper), abs(lower), 1.0)
    tol = rtol * span
    return abs(float(value) - lower) <= tol, abs(float(value) - upper) <= tol


def _validate_uniform_timebase(
    t: np.ndarray,
    *,
    rtol: float = 5e-3,
    atol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Validate a strictly increasing, approximately uniform time base.

    Nonuniform timestamps are intentionally unsupported because the downstream
    estimators assume a fixed sample interval throughout the pipeline.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ValueError(f"Time array must be 1-dimensional, got shape {t_arr.shape}")
    if len(t_arr) < 2:
        raise ValueError("At least 2 samples required for analysis")
    if not np.all(np.isfinite(t_arr)):
        raise ValueError("Time array must contain only finite values")

    t_norm = t_arr - t_arr[0]
    dt = np.diff(t_norm)
    if not np.all(np.isfinite(dt)):
        raise ValueError("Time array differences must be finite")
    if np.any(dt <= 0):
        raise ValueError("Time array must be strictly increasing with no duplicate timestamps")

    dt_ref = float(np.median(dt))
    if not np.isfinite(dt_ref) or dt_ref <= 0:
        raise ValueError("Could not determine a valid sampling interval from the time array")

    if not np.allclose(dt, dt_ref, rtol=rtol, atol=max(atol, abs(dt_ref) * rtol)):
        max_rel_dev = float(np.max(np.abs(dt - dt_ref) / dt_ref))
        raise ValueError(
            "Nonuniform timestamps are not supported; "
            f"max relative sample-interval deviation was {max_rel_dev:.3e}"
        )

    return t_norm, float(1.0 / dt_ref)


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
        nls_estimator: NLSFrequencyEstimator | None = None,
        dft_estimator: DFTFrequencyEstimator | None = None,
    ):
        """
        Initialize analyzer.

        Parameters:
        -----------
        nls_estimator : NLSFrequencyEstimator, optional
            NLS frequency estimator. If None, creates default (tau unknown).
        dft_estimator : DFTFrequencyEstimator, optional
            DFT frequency estimator. If None, creates default (rectangular window).
        """
        self.nls_estimator = nls_estimator or NLSFrequencyEstimator(tau_known=None)
        self.dft_estimator = dft_estimator or DFTFrequencyEstimator(window="rect")
        self.crlb_calc = CRLBCalculator()

    def estimate_tau(
        self,
        data: np.ndarray,
        t: np.ndarray,
        fs: float,
        initial_params: tuple | None = None,
        *,
        tau_init: float | None = None,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
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
        initial_params : tuple, optional
            (f0_init, phi0_init, A0_init, c0) to avoid redundant DFT. If None, estimated from data.
        tau_init : float, optional
            Initial guess for tau (s). If None, estimated from envelope decay.
        max_nfev : int, optional
            Maximum number of function evaluations for the fit. Default 150.
        ftol, xtol, gtol : float, optional
            Convergence tolerances for least_squares. Defaults 1e-8.

        Returns:
        --------
        float
            Estimated tau value in seconds
        """
        N = len(data)
        t_norm = t - t[0]

        # Get initial parameter estimates (use cached if provided)
        if initial_params is not None:
            f0_init, phi0_init, A0_init, c0 = _sanitize_initial_parameters(data, fs, initial_params)
        else:
            f0_init, phi0_init, A0_init, c0 = _estimate_initial_parameters_from_dft(data, fs)

        # Initial tau guess
        tau_seed = tau_init
        if tau_seed is None:
            tau_seed = _estimate_initial_tau_from_envelope(data, t_norm)
        tau_init_fit, tau_lower, tau_upper = _sanitize_tau_guess(tau_seed, t_norm)

        # NLS fit to estimate tau: fit (A0, f, phi, tau, c)
        def residuals_tau(p):
            A0, f, phi, tau, c = p
            return (A0 * np.exp(-t_norm / tau) * np.cos(2.0 * np.pi * f * t_norm + phi) + c) - data

        df = fs / N
        f_low = max(0.0, f0_init - max(0.2 * f0_init, 2 * df))
        f_high = min(0.5 * fs, f0_init + max(0.2 * f0_init, 2 * df))

        amp_upper = max(10.0 * A0_init, np.finfo(np.float64).eps * 10.0)
        lb = [0.0, f_low, -np.pi, tau_lower, -np.inf]
        ub = [amp_upper, f_high, np.pi, tau_upper, np.inf]

        ls_kwargs: dict = {"method": "trf", "verbose": 0}
        if max_nfev is not None:
            ls_kwargs["max_nfev"] = max_nfev
        else:
            ls_kwargs["max_nfev"] = 150
        if ftol is not None:
            ls_kwargs["ftol"] = ftol
        else:
            ls_kwargs["ftol"] = 1e-8
        if xtol is not None:
            ls_kwargs["xtol"] = xtol
        else:
            ls_kwargs["xtol"] = 1e-8
        if gtol is not None:
            ls_kwargs["gtol"] = gtol
        else:
            ls_kwargs["gtol"] = 1e-8

        res_tau = least_squares(
            residuals_tau,
            x0=np.array([A0_init, f0_init, phi0_init, tau_init_fit, c0]),
            bounds=(lb, ub),
            **ls_kwargs,
        )

        if res_tau.success:
            _, _, _, tau_est, _ = res_tau.x
            if (
                tau_est <= 0
                or not np.isfinite(tau_est)
                or tau_est > tau_upper
                or tau_est < tau_lower
            ):
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "tau_sanity_check_failed",
                        extra={
                            "event": "tau_sanity_check_failed",
                            "tau_est": float(tau_est),
                            "tau_init": float(tau_init_fit),
                            "t_max": float(t_norm[-1]),
                        },
                    )
                return tau_init_fit
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "tau_estimated",
                    extra={
                        "event": "tau_estimated",
                        "tau_est": float(tau_est),
                        "tau_init": float(tau_init_fit),
                        "nfev": res_tau.nfev,
                    },
                )
            return float(tau_est)

        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                "tau_estimation_failed",
                extra={
                    "event": "tau_estimation_failed",
                    "tau_init": float(tau_init_fit),
                    "fit_message": res_tau.message,
                    "nfev": res_tau.nfev,
                },
            )
        return tau_init_fit

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
        max_tau_multiplier = _validate_max_tau_multiplier(max_tau_multiplier)
        if not np.isfinite(tau_est) or tau_est <= 0:
            raise ValueError(f"tau_est must be positive and finite, got {tau_est}")
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
        tau_model: float,
        f_model: float,
    ) -> NoiseEstimate:
        """
        Estimate amplitude and noise for plug-in uncertainty calculations.

        The uncertainty model fixes the analyzed frequency and decay constant,
        then solves for the nuisance parameters in a linear least-squares model.
        Noise is reported with a degrees-of-freedom correction:
        sigma = sqrt(RSS / (N - p)) with p = 3 nuisance parameters.

        Parameters:
        -----------
        data_cropped : np.ndarray
            Cropped signal data
        t_crop : np.ndarray
            Cropped time array
        tau_model : float
            Decay constant associated with the analyzed cropped-stage model.
        f_model : float
            Frequency associated with the analyzed cropped-stage model.

        Returns:
        --------
        NoiseEstimate
            Estimated amplitude and noise summary
        """
        N_crop = len(data_cropped)
        t_crop_norm = t_crop - t_crop[0]
        if N_crop < 4:
            raise ValueError("At least 4 cropped samples are required for noise estimation")
        if not np.isfinite(tau_model) or tau_model <= 0:
            raise ValueError(f"tau_model must be positive and finite, got {tau_model}")
        if not np.isfinite(f_model) or f_model < 0:
            raise ValueError(f"f_model must be non-negative and finite, got {f_model}")

        exp_term = np.exp(-t_crop_norm / tau_model)
        omega_t = 2.0 * np.pi * f_model * t_crop_norm
        design = np.column_stack(
            [
                exp_term * np.cos(omega_t),
                exp_term * np.sin(omega_t),
                np.ones_like(t_crop_norm),
            ]
        )

        try:
            coeffs, _, rank, _ = np.linalg.lstsq(design, data_cropped, rcond=None)
            if rank < design.shape[1]:
                raise np.linalg.LinAlgError("Design matrix is rank-deficient")

            fitted = design @ coeffs
            residuals = data_cropped - fitted
            rss = float(np.sum(residuals**2))
            dof = N_crop - design.shape[1]
            if dof <= 0:
                raise ValueError("Noise-model degrees of freedom must be positive")

            a_cos, b_sin, _ = coeffs
            A0_est = float(np.hypot(a_cos, b_sin))
            sigma_mle = float(np.sqrt(rss / N_crop))
            sigma_est = float(np.sqrt(rss / dof))

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "noise_parameters_estimated",
                    extra={
                        "event": "noise_parameters_estimated",
                        "A0_est": A0_est,
                        "sigma_est": sigma_est,
                        "sigma_mle": sigma_mle,
                        "noise_dof": int(dof),
                    },
                )

            a0_safe: float = max(float(A0_est), float(np.finfo(np.float64).eps))
            return NoiseEstimate(
                A0=a0_safe,
                sigma=sigma_est,
                sigma_mle=sigma_mle,
                noise_dof=int(dof),
                success=True,
                method="fixed_frequency_tau_linear_lstsq",
                message=None,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "noise_estimation_fallback",
                    extra={
                        "event": "noise_estimation_fallback",
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc),
                    },
                )

            tail_start = max(int(0.8 * N_crop), N_crop - min(1000, N_crop))
            tail = data_cropped[tail_start:]
            if len(tail) < 2:
                tail = data_cropped
            sigma_est = float(np.std(tail, ddof=1)) if len(tail) > 1 else np.inf
            sigma_mle = float(np.std(tail, ddof=0)) if len(tail) > 0 else np.inf
            n_init = max(1, min(1000, max(1, N_crop // 10)))
            A0_est = max(
                float(np.sqrt(2.0) * np.std(data_cropped[:n_init])),
                np.finfo(np.float64).eps,
            )
            noise_dof = max(len(tail) - 1, 0)
            return NoiseEstimate(
                A0=A0_est,
                sigma=sigma_est,
                sigma_mle=sigma_mle,
                noise_dof=int(noise_dof),
                success=False,
                method="tail_std_fallback",
                message=str(exc),
            )

    def _run_analysis_pipeline(
        self,
        t: np.ndarray,
        data: np.ndarray,
        fs: float,
        max_tau_multiplier: float,
        *,
        initial_params: tuple | None = None,
        tau_init: float | None = None,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
    ) -> dict:
        """Run the full analysis pipeline on (t, data) arrays."""
        max_tau_multiplier = _validate_max_tau_multiplier(max_tau_multiplier)
        if initial_params is not None:
            initial_params_full = _sanitize_initial_parameters(data, fs, initial_params)
            initial_params_cropped = initial_params_full
        else:
            initial_params_full = _estimate_initial_parameters_from_dft(data, fs)
            initial_params_cropped = None  # Will compute after crop

        t_norm = t - t[0]
        if tau_init is None:
            tau_initialization = _estimate_initial_tau_with_method(data, t_norm)
        else:
            tau_initialization = (float(tau_init), "user_provided_tau_init")
        tau_seed = float(tau_initialization[0])
        tau_seed_method = str(tau_initialization[1])
        tau_full_init, tau_full_lower, tau_full_upper = _sanitize_tau_guess(tau_seed, t_norm)

        tau_est = self.estimate_tau(
            data,
            t,
            fs,
            initial_params=initial_params_full,
            tau_init=tau_seed,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        t_crop, data_cropped = self.crop_data_to_tau(
            t, data, tau_est, min_samples=1000, max_tau_multiplier=max_tau_multiplier
        )

        min_samples_for_analysis = 1000
        if len(t_crop) < min_samples_for_analysis:
            t_crop = t
            data_cropped = data

        if initial_params_cropped is None:
            initial_params_cropped = _estimate_initial_parameters_from_dft(data_cropped, fs)

        t_crop_norm = t_crop - t_crop[0]
        tau_cropped_seed = tau_est
        tau_cropped_init, tau_cropped_lower, tau_cropped_upper = _sanitize_tau_guess(
            tau_cropped_seed,
            t_crop_norm,
        )

        fit_kwargs = {}
        if max_nfev is not None:
            fit_kwargs["max_nfev"] = max_nfev
        if ftol is not None:
            fit_kwargs["ftol"] = ftol
        if xtol is not None:
            fit_kwargs["xtol"] = xtol
        if gtol is not None:
            fit_kwargs["gtol"] = gtol
        fit_kwargs["tau_init"] = tau_cropped_init

        result_nls = self.nls_estimator.estimate_full(
            data_cropped,
            fs,
            initial_params=initial_params_cropped,
            **fit_kwargs,
        )
        result_dft = self.dft_estimator.estimate_full(
            data_cropped,
            fs,
            initial_params=initial_params_cropped,
            **fit_kwargs,
        )

        f_nls = result_nls.f
        f_dft = result_dft.f
        Q_nls = result_nls.Q
        Q_dft = result_dft.Q
        tau_nls = result_nls.tau
        tau_dft = result_dft.tau
        tau_model = tau_nls if tau_nls is not None else tau_dft if tau_dft is not None else tau_est
        tau_est_at_lower, tau_est_at_upper = _near_bound(tau_est, tau_full_lower, tau_full_upper)
        tau_nls_at_lower, tau_nls_at_upper = _near_bound(
            tau_nls, tau_cropped_lower, tau_cropped_upper
        )
        tau_dft_at_lower, tau_dft_at_upper = _near_bound(
            tau_dft, tau_cropped_lower, tau_cropped_upper
        )
        T_over_tau_est = float(t[-1] / tau_est) if tau_est > 0 else np.inf
        tau_est_low_confidence = bool(
            tau_seed_method.startswith("record_half_duration_fallback")
            or T_over_tau_est < 1.0
            or tau_est_at_lower
            or tau_est_at_upper
        )
        noise_estimate = self.estimate_noise_parameters(
            data_cropped,
            t_crop,
            tau_model,
            f_nls,
        )

        N_crop = len(data_cropped)
        plugin_crlb_var_f = self.crlb_calc.variance(
            noise_estimate.A0,
            noise_estimate.sigma,
            fs,
            N_crop,
            tau_model,
        )
        plugin_crlb_std_f = (
            float(np.sqrt(plugin_crlb_var_f)) if np.isfinite(plugin_crlb_var_f) else np.inf
        )
        uncertainty_valid = (
            noise_estimate.success
            and np.isfinite(plugin_crlb_std_f)
            and plugin_crlb_std_f > 0
            and result_nls.success
        )

        return {
            "t": t,
            "data": data,
            "V2": None,
            "t_crop": t_crop,
            "data_cropped": data_cropped,
            "fs": fs,
            "tau_seed": tau_seed,
            "tau_seed_method": tau_seed_method,
            "tau_full_init": tau_full_init,
            "tau_full_lower": tau_full_lower,
            "tau_full_upper": tau_full_upper,
            "tau_est": tau_est,
            "tau_est_at_lower_bound": tau_est_at_lower,
            "tau_est_at_upper_bound": tau_est_at_upper,
            "T_over_tau_est": T_over_tau_est,
            "tau_est_low_confidence": tau_est_low_confidence,
            "tau_cropped_seed": tau_cropped_seed,
            "tau_cropped_seed_source": "full_record_tau_est",
            "tau_cropped_init": tau_cropped_init,
            "tau_cropped_lower": tau_cropped_lower,
            "tau_cropped_upper": tau_cropped_upper,
            "tau_nls": tau_nls,
            "tau_dft": tau_dft,
            "tau_model": tau_model,
            "tau_nls_at_lower_bound": tau_nls_at_lower,
            "tau_nls_at_upper_bound": tau_nls_at_upper,
            "tau_dft_at_lower_bound": tau_dft_at_lower,
            "tau_dft_at_upper_bound": tau_dft_at_upper,
            "f_nls": f_nls,
            "f_dft": f_dft,
            "Q_nls": Q_nls,
            "Q_dft": Q_dft,
            "Q_pre_crop": float(np.pi * f_nls * tau_est) if np.isfinite(f_nls) else np.nan,
            "nls_success": result_nls.success,
            "dft_success": result_dft.success,
            "nls_used_fallback": result_nls.used_fallback,
            "dft_used_fallback": result_dft.used_fallback,
            "nls_message": result_nls.message,
            "dft_message": result_dft.message,
            "A0_est": noise_estimate.A0,
            "sigma_est": noise_estimate.sigma,
            "sigma_mle_est": noise_estimate.sigma_mle,
            "noise_dof": noise_estimate.noise_dof,
            "noise_estimation_success": noise_estimate.success,
            "noise_estimation_method": noise_estimate.method,
            "noise_estimation_message": noise_estimate.message,
            "plugin_crlb_var_f": plugin_crlb_var_f,
            "plugin_crlb_std_f": plugin_crlb_std_f,
            "uncertainty_std_f": plugin_crlb_std_f,
            "uncertainty_method": (
                "plugin_crlb_fitted_tau_with_residual_dof_correction"
                if uncertainty_valid
                else "unavailable"
            ),
            "uncertainty_description": (
                "Plug-in frequency diagnostic computed from fitted/cropped data; "
                "crlb_std_f is a backward-compatible alias."
            ),
            "uncertainty_valid": uncertainty_valid,
            "crlb_std_f": plugin_crlb_std_f,
            "crlb_std_f_is_alias": True,
            "N": len(t),
            "N_crop": len(t_crop),
            "T": t[-1],
            "T_crop": t_crop[-1] if len(t_crop) > 0 else 0,
        }

    def analyze_array(
        self,
        t: np.ndarray | None = None,
        data: np.ndarray | None = None,
        fs: float | None = None,
        *,
        time_col: str | int = 0,
        data_col: str | int = 1,
        max_tau_multiplier: float = 1.0,
        initial_params: tuple | None = None,
        tau_init: float | None = None,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
    ) -> dict:
        """
        Analyze ring-down data from numpy arrays or pandas Series/DataFrame.

        Parameters:
        -----------
        t : np.ndarray or pd.Series, optional
            Time array (s). Required unless data is a DataFrame or fs is provided.
        data : np.ndarray, pd.Series, or pd.DataFrame
            Signal data. Required.
        fs : float, optional
            Sampling frequency (Hz). If provided with t=None, time is inferred as
            t = np.arange(len(data)) / fs.
        time_col : str or int, optional
            Column index or name for time when data is a DataFrame. Default 0.
        data_col : str or int, optional
            Column index or name for signal when data is a DataFrame. Default 1.
        max_tau_multiplier : float
            Multiplier for tau_est when cropping data. Default 1.0.
        initial_params : tuple, optional
            (f0_init, phi0_init, A0_init, c0) to avoid redundant DFT. If None, estimated from data.
        tau_init : float, optional
            Initial guess for tau (s). If None, estimated from envelope decay.
        max_nfev : int, optional
            Maximum function evaluations for NLS fits. Increase for noisy data (e.g. 300–500).
        ftol, xtol, gtol : float, optional
            Convergence tolerances for least_squares. Relax (e.g. 1e-6) for noisy data.

        Returns:
        --------
        dict
            Results dictionary (same structure as analyze_file, minus filename/type).

        Raises:
        -------
        ValueError
            If data is invalid, lengths mismatch, or neither t nor fs provided.

        Examples:
        ---------
        >>> # From numpy arrays
        >>> result = analyzer.analyze_array(t, data)
        >>> # From data and sampling rate
        >>> result = analyzer.analyze_array(data=data, fs=1000.0)
        >>> # From pandas DataFrame
        >>> result = analyzer.analyze_array(data=df, time_col="time", data_col="phase")
        >>> # Noisy data: relax tolerances and increase max_nfev
        >>> result = analyzer.analyze_array(t, data, max_nfev=500, ftol=1e-6)
        """
        t_arr, data_arr = _parse_array_input(
            t=t, data=data, fs=fs, time_col=time_col, data_col=data_col
        )
        data_arr = _validate_signal_data(data_arr)
        if t is not None or hasattr(data, "iloc"):
            t_arr, fs = _validate_uniform_timebase(t_arr)
        else:
            t_arr = t_arr - t_arr[0]
            if fs is None or not np.isfinite(fs) or fs <= 0:
                raise ValueError(f"Sampling frequency fs must be positive and finite, got {fs}")
            fs = float(fs)

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_start",
                extra={
                    "event": "analysis_start",
                    "source": "array",
                    "n_samples": len(t_arr),
                },
            )

        result = self._run_analysis_pipeline(
            t_arr,
            data_arr,
            fs,
            max_tau_multiplier,
            initial_params=initial_params,
            tau_init=tau_init,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_complete",
                extra={
                    "event": "analysis_complete",
                    "source": "array",
                    "f_nls": float(result["f_nls"]),
                    "f_dft": float(result["f_dft"]),
                    "tau_est": float(result["tau_est"]),
                    "uncertainty_std_f": float(result["uncertainty_std_f"])
                    if np.isfinite(result["uncertainty_std_f"])
                    else None,
                },
            )

        return result

    def analyze_file(
        self,
        filepath: str,
        max_tau_multiplier: float = 1.0,
        *,
        initial_params: tuple | None = None,
        tau_init: float | None = None,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
    ) -> dict:
        """
        Process a single data file and return analysis results.

        Parameters:
        -----------
        filepath : str
            Path to the data file
        max_tau_multiplier : float
            Multiplier for tau_est to determine maximum record length when cropping data.
            Default is 1.0.
        initial_params : tuple, optional
            (f0_init, phi0_init, A0_init, c0) to avoid redundant DFT. If None, estimated from data.
        tau_init : float, optional
            Initial guess for tau (s). If None, estimated from envelope decay.
        max_nfev : int, optional
            Maximum function evaluations for NLS fits. Increase for noisy data (e.g. 300–500).
        ftol, xtol, gtol : float, optional
            Convergence tolerances for least_squares. Relax (e.g. 1e-6) for noisy data.

        Returns:
        --------
        dict
            Results dictionary with all analysis data

        Raises:
        -------
        FileNotFoundError
            If the file does not exist
        ValueError
            If the file format is unsupported, data is invalid (e.g., empty CSV,
            malformed MAT structure), or file exceeds size limit
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_start",
                extra={
                    "event": "analysis_start",
                    "filepath": str(filepath),
                },
            )

        max_tau_multiplier = _validate_max_tau_multiplier(max_tau_multiplier)
        t, data, V2, file_type = RingDownDataLoader.load(filepath)
        t, fs = _validate_uniform_timebase(t)
        data = _validate_signal_data(data, source=f"Signal data in {filepath}")

        result = self._run_analysis_pipeline(
            t,
            data,
            fs,
            max_tau_multiplier,
            initial_params=initial_params,
            tau_init=tau_init,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        result["filename"] = Path(filepath).name
        result["type"] = file_type
        result["V2"] = V2

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "analysis_complete",
                extra={
                    "event": "analysis_complete",
                    "filepath": str(filepath),
                    "f_nls": float(result["f_nls"]),
                    "f_dft": float(result["f_dft"]),
                    "tau_est": float(result["tau_est"]),
                    "uncertainty_std_f": float(result["uncertainty_std_f"])
                    if np.isfinite(result["uncertainty_std_f"])
                    else None,
                },
            )

        return result
