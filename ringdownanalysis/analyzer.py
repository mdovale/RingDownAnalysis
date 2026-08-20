"""
Analysis pipeline for real ring-down measurement data.
"""

from __future__ import annotations

import dataclasses
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
from .q_envelope import q_envelope_diagnostic
from .q_profile import ProfileQEstimator

logger = logging.getLogger(__name__)


class TauEstimate(NamedTuple):
    """Full-record tau estimate with an explicit fit-success flag."""

    tau: float
    fit_success: bool


class NoiseEstimate(NamedTuple):
    """Noise and amplitude estimates for plug-in uncertainty calculations."""

    A0: float
    sigma: float
    sigma_mle: float
    noise_dof: int
    success: bool
    method: str
    message: str | None = None


class QAssessment(NamedTuple):
    """Validity assessment for a raw Q estimate."""

    value: float | None
    raw: float | None
    valid: bool
    status: str
    reasons: list[str]
    raw_to_pre_crop_ratio: float | None


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


def _apply_array_detrend(data: np.ndarray, detrend: str | None) -> np.ndarray:
    """Apply optional preprocessing for array inputs."""
    if detrend is None:
        return data
    if detrend == "constant":
        return data - float(np.mean(data))
    raise ValueError("detrend must be 'constant' or None")


def _validate_max_tau_multiplier(max_tau_multiplier: float) -> float:
    """Validate crop multiplier before it can silently produce an empty crop."""
    value = float(max_tau_multiplier)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"max_tau_multiplier must be positive and finite, got {max_tau_multiplier}"
        )
    return value


def _is_positive_finite(value: float | None) -> bool:
    """Return True for finite, strictly positive scalar values."""
    return value is not None and np.isfinite(value) and value > 0


def _near_bound(
    value: float | None, lower: float, upper: float, *, rtol: float = 1e-4
) -> tuple[bool, bool]:
    """Return lower/upper bound-hit flags for tau-like estimates."""
    if value is None or not np.isfinite(value):
        return False, False
    span = max(abs(upper - lower), abs(upper), abs(lower), 1.0)
    tol = rtol * span
    return abs(float(value) - lower) <= tol, abs(float(value) - upper) <= tol


def _assess_q_estimate(
    *,
    method: str,
    raw_q: float | None,
    tau: float | None,
    success: bool,
    used_fallback: bool,
    tau_at_lower_bound: bool,
    tau_at_upper_bound: bool,
    tau_est_low_confidence: bool,
    q_pre_crop: float,
    t_fit: np.ndarray,
) -> QAssessment:
    """
    Classify a raw Q estimate before exposing it as a user-facing value.

    Raw optimizer output remains available, but Q is only marked valid when
    the fitted tau is identifiable enough to avoid known bound/crop artifacts.
    """
    hard_reasons: list[str] = []
    warning_reasons: list[str] = []

    if not success:
        hard_reasons.append(f"{method}_fit_failed")
    if used_fallback:
        hard_reasons.append(f"{method}_used_fallback")
    if not _is_positive_finite(tau):
        hard_reasons.append(f"{method}_tau_missing_or_nonpositive")
    if tau_at_lower_bound:
        hard_reasons.append(f"{method}_tau_at_lower_bound")
    if tau_at_upper_bound:
        hard_reasons.append(f"{method}_tau_at_upper_bound")
    if not _is_positive_finite(raw_q):
        hard_reasons.append(f"{method}_q_missing_or_nonpositive")

    raw_to_pre_crop_ratio: float | None = None
    if _is_positive_finite(raw_q) and _is_positive_finite(q_pre_crop):
        assert raw_q is not None
        raw_to_pre_crop_ratio = float(raw_q / q_pre_crop)
        if raw_to_pre_crop_ratio > 5.0:
            hard_reasons.append(f"{method}_q_raw_vs_pre_crop_ratio_gt_5")
        elif raw_to_pre_crop_ratio > 2.0:
            warning_reasons.append(f"{method}_q_raw_vs_pre_crop_ratio_gt_2")
        elif raw_to_pre_crop_ratio < 0.5:
            warning_reasons.append(f"{method}_q_raw_vs_pre_crop_ratio_lt_0_5")

    if tau_est_low_confidence:
        warning_reasons.append("tau_est_low_confidence")

    if _is_positive_finite(tau) and len(t_fit) > 0:
        assert tau is not None
        fit_duration = float(t_fit[-1] - t_fit[0])
        if fit_duration / float(tau) < 1.0:
            warning_reasons.append(f"{method}_fit_window_shorter_than_tau")

    if hard_reasons:
        return QAssessment(
            value=None,
            raw=float(raw_q) if raw_q is not None and np.isfinite(raw_q) else raw_q,
            valid=False,
            status="invalid",
            reasons=hard_reasons + warning_reasons,
            raw_to_pre_crop_ratio=raw_to_pre_crop_ratio,
        )

    if warning_reasons:
        return QAssessment(
            value=None,
            raw=float(raw_q) if raw_q is not None else None,
            valid=False,
            status="warning",
            reasons=warning_reasons,
            raw_to_pre_crop_ratio=raw_to_pre_crop_ratio,
        )

    return QAssessment(
        value=float(raw_q) if raw_q is not None else None,
        raw=float(raw_q) if raw_q is not None else None,
        valid=True,
        status="valid",
        reasons=[],
        raw_to_pre_crop_ratio=raw_to_pre_crop_ratio,
    )


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
    5. Profile Q over log(tau) with separable least squares
    6. Estimate noise parameters for CRLB calculation
    7. Calculate CRLB
    """

    def __init__(
        self,
        nls_estimator: NLSFrequencyEstimator | None = None,
        dft_estimator: DFTFrequencyEstimator | None = None,
        q_profile_estimator: ProfileQEstimator | None = None,
    ):
        """
        Initialize analyzer.

        Parameters:
        -----------
        nls_estimator : NLSFrequencyEstimator, optional
            NLS frequency estimator. If None, creates default (tau unknown).
        dft_estimator : DFTFrequencyEstimator, optional
            DFT frequency estimator. If None, creates default (rectangular window).
        q_profile_estimator : ProfileQEstimator, optional
            Profile-likelihood Q estimator. If None, creates default.
        """
        self.nls_estimator = nls_estimator or NLSFrequencyEstimator(tau_known=None)
        self.dft_estimator = dft_estimator or DFTFrequencyEstimator(window="rect")
        self.q_profile_estimator = q_profile_estimator or ProfileQEstimator()
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
            Estimated tau value in seconds. On fit failure the (possibly
            user-provided) seed is returned; use estimate_tau_with_status to
            distinguish that case.
        """
        return self.estimate_tau_with_status(
            data,
            t,
            fs,
            initial_params=initial_params,
            tau_init=tau_init,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        ).tau

    def estimate_tau_with_status(
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
    ) -> TauEstimate:
        """
        Estimate tau from full data using NLS, reporting fit success explicitly.

        Same fit as estimate_tau, but the returned TauEstimate carries
        fit_success=False when the optimizer failed or the fitted tau failed
        its sanity checks — in both cases the returned tau is the seed value,
        which is otherwise indistinguishable from a genuine estimate.
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
                return TauEstimate(tau=float(tau_init_fit), fit_success=False)
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
            return TauEstimate(tau=float(tau_est), fit_success=True)

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
        return TauEstimate(tau=float(tau_init_fit), fit_success=False)

    def crop_data_to_tau(
        self,
        t: np.ndarray,
        data: np.ndarray,
        tau_est: float,
        min_samples: int = 100,
        max_tau_multiplier: float = 3.0,
        min_duration: float | None = None,
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
            Multiplier for tau_est to determine maximum record length. Default is 3.0.
        min_duration : float, optional
            Never crop the record shorter than this duration (s). Used by the
            pipeline to guarantee at least one envelope decay time of data even
            when the coherent tau fit collapses to a small value.

        Returns:
        --------
        (t_crop, data_cropped) : tuple
            Cropped time and signal arrays
        """
        max_tau_multiplier = _validate_max_tau_multiplier(max_tau_multiplier)
        if not np.isfinite(tau_est) or tau_est <= 0:
            raise ValueError(f"tau_est must be positive and finite, got {tau_est}")
        t_crop_max = max_tau_multiplier * tau_est
        if min_duration is not None and np.isfinite(min_duration) and min_duration > 0:
            t_crop_max = max(t_crop_max, float(min_duration))
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

        tau_estimate = self.estimate_tau_with_status(
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
        tau_est = float(tau_estimate.tau)
        tau_est_fit_success = bool(tau_estimate.fit_success)

        # Crop cascade guard: the coherent full-record tau fit can collapse to a
        # small value on decoherent data (frequency drift), which would silently
        # crop away almost all of the selected window. Cross-check tau_est
        # against the incoherent envelope tau before trusting it for cropping.
        f0_init_full = float(initial_params_full[0])
        envelope_precrop = q_envelope_diagnostic(t, data, f0_init_full)
        tau_envelope_precrop = (
            float(envelope_precrop.tau)
            if envelope_precrop.valid and envelope_precrop.tau is not None
            else None
        )
        record_duration = float(t[-1] - t[0])
        tau_crop = tau_est
        tau_crop_source = "tau_est"
        tau_est_envelope_ratio: float | None = None
        # An envelope tau longer than the record is not identifiable (flat or
        # barely decaying envelope) and must not override the coherent fit.
        tau_envelope_identifiable = (
            tau_envelope_precrop is not None and tau_envelope_precrop <= record_duration
        )
        if tau_envelope_precrop is not None and _is_positive_finite(tau_est):
            tau_est_envelope_ratio = float(
                max(tau_est / tau_envelope_precrop, tau_envelope_precrop / tau_est)
            )
        if tau_envelope_identifiable and tau_est_envelope_ratio is not None:
            assert tau_envelope_precrop is not None
            if tau_est_envelope_ratio > 3.0:
                tau_crop = tau_envelope_precrop
                tau_crop_source = "envelope_tau_disagreement_fallback"
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "tau_est_envelope_disagreement",
                        extra={
                            "event": "tau_est_envelope_disagreement",
                            "tau_est": float(tau_est),
                            "tau_envelope": tau_envelope_precrop,
                            "ratio": tau_est_envelope_ratio,
                        },
                    )

        t_crop, data_cropped = self.crop_data_to_tau(
            t,
            data,
            tau_crop,
            min_samples=1000,
            max_tau_multiplier=max_tau_multiplier,
            min_duration=tau_envelope_precrop if tau_envelope_identifiable else None,
        )

        min_samples_for_analysis = 1000
        if len(t_crop) < min_samples_for_analysis:
            t_crop = t
            data_cropped = data

        if initial_params_cropped is None:
            initial_params_cropped = _estimate_initial_parameters_from_dft(data_cropped, fs)

        t_crop_norm = t_crop - t_crop[0]
        tau_cropped_seed = tau_crop
        tau_cropped_seed_source = (
            "full_record_tau_est" if tau_crop_source == "tau_est" else "envelope_precrop"
        )
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
        Q_nls_raw = result_nls.Q
        Q_dft_raw = result_dft.Q
        tau_nls = result_nls.tau
        tau_dft = result_dft.tau
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
            or tau_crop_source != "tau_est"
            or not tau_est_fit_success
        )
        Q_pre_crop = float(np.pi * f_nls * tau_est) if np.isfinite(f_nls) else np.nan
        q_nls_assessment = _assess_q_estimate(
            method="nls",
            raw_q=Q_nls_raw,
            tau=tau_nls,
            success=result_nls.success,
            used_fallback=result_nls.used_fallback,
            tau_at_lower_bound=tau_nls_at_lower,
            tau_at_upper_bound=tau_nls_at_upper,
            tau_est_low_confidence=tau_est_low_confidence,
            q_pre_crop=Q_pre_crop,
            t_fit=t_crop,
        )
        q_dft_assessment = _assess_q_estimate(
            method="dft",
            raw_q=Q_dft_raw,
            tau=tau_dft,
            success=result_dft.success,
            used_fallback=result_dft.used_fallback,
            tau_at_lower_bound=tau_dft_at_lower,
            tau_at_upper_bound=tau_dft_at_upper,
            tau_est_low_confidence=tau_est_low_confidence,
            q_pre_crop=Q_pre_crop,
            t_fit=t_crop,
        )
        profile_f_init = f_nls if np.isfinite(f_nls) and f_nls > 0 else f_dft
        q_envelope_seed = q_envelope_diagnostic(t, data, profile_f_init)
        if q_envelope_seed.valid and q_envelope_seed.tau is not None:
            profile_tau_init = q_envelope_seed.tau
            profile_tau_init_source = "q_envelope"
        else:
            profile_tau_init = tau_est
            profile_tau_init_source = "tau_est"
        q_profile = self.q_profile_estimator.estimate(
            t,
            data,
            fs,
            f_init=profile_f_init,
            tau_init=profile_tau_init,
        )
        if _is_positive_finite(q_profile.Q):
            candidate_envelope_q = q_profile.Q
        elif _is_positive_finite(q_nls_assessment.raw):
            candidate_envelope_q = q_nls_assessment.raw
        elif _is_positive_finite(q_dft_assessment.raw):
            candidate_envelope_q = q_dft_assessment.raw
        else:
            candidate_envelope_q = None
        q_envelope = q_envelope_diagnostic(t, data, profile_f_init, q=candidate_envelope_q)
        # Envelope-mismatch gate: a coherent profile Q that disagrees with the
        # measured envelope slope must not be reported as a valid finite Q.
        # The raw optimizer value stays available in Q_profile_raw.
        q_profile_raw = q_profile.Q
        if (
            q_profile.valid
            and _is_positive_finite(q_profile.Q)
            and candidate_envelope_q == q_profile.Q
            and q_envelope.candidate_agrees is False
        ):
            q_profile = dataclasses.replace(
                q_profile,
                Q=None,
                valid=False,
                status="warning",
                reasons=[*q_profile.reasons, "envelope_mismatch"],
            )
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "q_profile_envelope_mismatch",
                    extra={
                        "event": "q_profile_envelope_mismatch",
                        "q_profile_raw": float(q_profile_raw) if q_profile_raw else None,
                        "q_envelope": q_envelope.Q,
                        "slope_mismatch": q_envelope.candidate_slope_mismatch,
                    },
                )
        if q_nls_assessment.valid and tau_nls is not None:
            tau_model = tau_nls
        elif q_dft_assessment.valid and tau_dft is not None:
            tau_model = tau_dft
        else:
            tau_model = tau_est
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
            "tau_est_fit_success": tau_est_fit_success,
            "tau_est_at_lower_bound": tau_est_at_lower,
            "tau_est_at_upper_bound": tau_est_at_upper,
            "T_over_tau_est": T_over_tau_est,
            "tau_est_low_confidence": tau_est_low_confidence,
            "tau_envelope_precrop": tau_envelope_precrop,
            "tau_crop": tau_crop,
            "tau_crop_source": tau_crop_source,
            "tau_est_envelope_ratio": tau_est_envelope_ratio,
            "tau_cropped_seed": tau_cropped_seed,
            "tau_cropped_seed_source": tau_cropped_seed_source,
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
            "Q_nls": q_nls_assessment.value,
            "Q_dft": q_dft_assessment.value,
            "Q_nls_raw": q_nls_assessment.raw,
            "Q_dft_raw": q_dft_assessment.raw,
            "Q_nls_valid": q_nls_assessment.valid,
            "Q_dft_valid": q_dft_assessment.valid,
            "Q_nls_status": q_nls_assessment.status,
            "Q_dft_status": q_dft_assessment.status,
            "Q_nls_reasons": q_nls_assessment.reasons,
            "Q_dft_reasons": q_dft_assessment.reasons,
            "Q_nls_raw_to_pre_crop_ratio": q_nls_assessment.raw_to_pre_crop_ratio,
            "Q_dft_raw_to_pre_crop_ratio": q_dft_assessment.raw_to_pre_crop_ratio,
            "Q_pre_crop": Q_pre_crop,
            "Q_profile": q_profile.Q,
            "Q_profile_raw": q_profile_raw,
            "Q_profile_valid": q_profile.valid,
            "Q_profile_status": q_profile.status,
            "Q_profile_reasons": q_profile.reasons,
            "tau_profile": q_profile.tau_hat,
            "f_profile": q_profile.f_hat,
            "Q_profile_ci95": q_profile.ci95,
            "Q_profile_lower_limit_95": q_profile.lower_limit_95,
            "Q_profile_upper_limit_95": q_profile.upper_limit_95,
            "Q_profile_method": q_profile.method,
            "Q_profile_tau_init": profile_tau_init,
            "Q_profile_tau_init_source": profile_tau_init_source,
            "Q_profile_rss_min": q_profile.rss_min,
            "Q_profile_sigma": q_profile.sigma,
            "Q_profile_dof": q_profile.dof,
            "Q_profile_n_grid": q_profile.n_grid,
            "Q_profile_tau_grid": q_profile.profile_tau,
            "Q_profile_q_grid": q_profile.profile_q,
            "Q_profile_delta": q_profile.profile_delta,
            "tau_envelope": q_envelope.tau,
            "Q_envelope": q_envelope.Q,
            "Q_envelope_valid": q_envelope.valid,
            "Q_envelope_status": q_envelope.status,
            "Q_envelope_reasons": q_envelope.reasons,
            "Q_envelope_method": q_envelope.method,
            "Q_envelope_n_windows": q_envelope.n_windows,
            "Q_envelope_n_windows_used": q_envelope.n_windows_used,
            "Q_envelope_log_amplitude_slope": q_envelope.log_amplitude_slope,
            "Q_envelope_log_amplitude_intercept": q_envelope.log_amplitude_intercept,
            "Q_envelope_log_amplitude_rmse": q_envelope.log_amplitude_rmse,
            "Q_envelope_slope_stderr": q_envelope.slope_stderr,
            "Q_envelope_candidate_Q": q_envelope.candidate_q,
            "Q_envelope_candidate_tau": q_envelope.candidate_tau,
            "Q_envelope_candidate_log_rmse": q_envelope.candidate_log_rmse,
            "Q_envelope_candidate_slope_mismatch": q_envelope.candidate_slope_mismatch,
            "Q_envelope_candidate_agrees": q_envelope.candidate_agrees,
            "Q_envelope_t_mid": q_envelope.t_mid,
            "Q_envelope_amplitude": q_envelope.amplitude,
            "Q_envelope_used": q_envelope.used,
            "Q_envelope_fit_amplitude": q_envelope.fitted_amplitude,
            "Q_envelope_candidate_amplitude": q_envelope.candidate_amplitude,
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
        max_tau_multiplier: float = 3.0,
        detrend: str | None = None,
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
            Multiplier for tau_est when cropping data. Default 3.0 to avoid
            treating a one-tau crop as sufficient evidence for Q.
        detrend : {"constant", None}, optional
            If "constant", subtract the array mean before analysis. File inputs
            already remove a constant phase offset during loading.
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
        data_arr = _apply_array_detrend(data_arr, detrend)
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

    def q_sensitivity(
        self,
        t: np.ndarray,
        data: np.ndarray,
        *,
        start_offsets: list[float],
        durations: list[float],
        max_tau_multipliers: list[float] | None = None,
        detrend: str | None = None,
        initial_params: tuple | None = None,
        tau_init: float | None = None,
        max_nfev: int | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        gtol: float | None = None,
    ) -> list[dict]:
        """
        Run Q reliability sensitivity over start offsets, durations, and crop multipliers.

        Returns a list of DataFrame-ready records containing raw Q, validated Q,
        status/reasons, and core tau/frequency diagnostics for each window.
        """
        t_arr, data_arr = _parse_array_input(t=t, data=data)
        data_arr = _validate_signal_data(data_arr)
        data_arr = _apply_array_detrend(data_arr, detrend)
        t_arr, _ = _validate_uniform_timebase(t_arr)

        multipliers = max_tau_multipliers if max_tau_multipliers is not None else [3.0]
        multipliers = [_validate_max_tau_multiplier(multiplier) for multiplier in multipliers]
        starts = [float(start) for start in start_offsets]
        window_durations = [float(duration) for duration in durations]

        records: list[dict] = []
        for start in starts:
            if not np.isfinite(start) or start < 0:
                raise ValueError(f"start offsets must be non-negative and finite, got {start}")
            for duration in window_durations:
                if not np.isfinite(duration) or duration <= 0:
                    raise ValueError(f"durations must be positive and finite, got {duration}")
                stop = start + duration
                window_mask = (t_arr >= start) & (t_arr <= stop)
                if int(np.count_nonzero(window_mask)) < 2:
                    raise ValueError(
                        f"window start={start} duration={duration} contains fewer than 2 samples"
                    )

                t_window = t_arr[window_mask]
                data_window = data_arr[window_mask]
                for multiplier in multipliers:
                    result = self.analyze_array(
                        t=t_window,
                        data=data_window,
                        max_tau_multiplier=multiplier,
                        initial_params=initial_params,
                        tau_init=tau_init,
                        max_nfev=max_nfev,
                        ftol=ftol,
                        xtol=xtol,
                        gtol=gtol,
                    )
                    records.append(
                        {
                            "start_offset": start,
                            "duration": duration,
                            "max_tau_multiplier": multiplier,
                            "N": result["N"],
                            "N_crop": result["N_crop"],
                            "T": result["T"],
                            "T_crop": result["T_crop"],
                            "tau_est": result["tau_est"],
                            "tau_est_fit_success": result["tau_est_fit_success"],
                            "tau_est_low_confidence": result["tau_est_low_confidence"],
                            "tau_nls": result["tau_nls"],
                            "tau_dft": result["tau_dft"],
                            "tau_nls_at_lower_bound": result["tau_nls_at_lower_bound"],
                            "tau_nls_at_upper_bound": result["tau_nls_at_upper_bound"],
                            "tau_dft_at_lower_bound": result["tau_dft_at_lower_bound"],
                            "tau_dft_at_upper_bound": result["tau_dft_at_upper_bound"],
                            "f_nls": result["f_nls"],
                            "f_dft": result["f_dft"],
                            "Q_pre_crop": result["Q_pre_crop"],
                            "Q_nls": result["Q_nls"],
                            "Q_nls_raw": result["Q_nls_raw"],
                            "Q_nls_valid": result["Q_nls_valid"],
                            "Q_nls_status": result["Q_nls_status"],
                            "Q_nls_reasons": result["Q_nls_reasons"],
                            "Q_nls_raw_to_pre_crop_ratio": result["Q_nls_raw_to_pre_crop_ratio"],
                            "Q_dft": result["Q_dft"],
                            "Q_dft_raw": result["Q_dft_raw"],
                            "Q_dft_valid": result["Q_dft_valid"],
                            "Q_dft_status": result["Q_dft_status"],
                            "Q_dft_reasons": result["Q_dft_reasons"],
                            "Q_dft_raw_to_pre_crop_ratio": result["Q_dft_raw_to_pre_crop_ratio"],
                            "Q_profile": result["Q_profile"],
                            "Q_profile_raw": result["Q_profile_raw"],
                            "Q_profile_valid": result["Q_profile_valid"],
                            "Q_profile_status": result["Q_profile_status"],
                            "Q_profile_reasons": result["Q_profile_reasons"],
                            "tau_profile": result["tau_profile"],
                            "f_profile": result["f_profile"],
                            "Q_profile_ci95": result["Q_profile_ci95"],
                            "Q_profile_lower_limit_95": result["Q_profile_lower_limit_95"],
                            "Q_profile_upper_limit_95": result["Q_profile_upper_limit_95"],
                            "Q_profile_method": result["Q_profile_method"],
                            "Q_profile_tau_init": result["Q_profile_tau_init"],
                            "Q_profile_tau_init_source": result["Q_profile_tau_init_source"],
                            "tau_envelope": result["tau_envelope"],
                            "Q_envelope": result["Q_envelope"],
                            "Q_envelope_valid": result["Q_envelope_valid"],
                            "Q_envelope_status": result["Q_envelope_status"],
                            "Q_envelope_reasons": result["Q_envelope_reasons"],
                            "Q_envelope_candidate_log_rmse": result[
                                "Q_envelope_candidate_log_rmse"
                            ],
                            "Q_envelope_candidate_slope_mismatch": result[
                                "Q_envelope_candidate_slope_mismatch"
                            ],
                            "Q_envelope_candidate_agrees": result["Q_envelope_candidate_agrees"],
                        }
                    )

        return records

    def analyze_file(
        self,
        filepath: str,
        max_tau_multiplier: float = 3.0,
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
            Multiplier for tau_est to determine maximum record length when
            cropping data. Default is 3.0 to avoid treating a one-tau crop as
            sufficient evidence for Q.
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
