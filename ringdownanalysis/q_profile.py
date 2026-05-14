"""
Profile-likelihood Q estimation with separable least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .estimators import (
    _estimate_initial_parameters_from_dft,
    _estimate_initial_tau_from_envelope,
)

_CHI2_95_ONE_PARAMETER = 3.841458820694124


@dataclass(frozen=True)
class QProfileResult:
    """Profile-Q result with finite estimates, intervals, or one-sided limits."""

    f_hat: float | None
    tau_hat: float | None
    Q: float | None
    valid: bool
    status: str
    reasons: list[str]
    ci95: tuple[float, float] | None
    lower_limit_95: float | None
    upper_limit_95: float | None
    profile_tau: np.ndarray
    profile_q: np.ndarray
    profile_delta: np.ndarray
    rss_min: float
    sigma: float
    dof: int
    n_grid: int
    method: str


@dataclass(frozen=True)
class _ProjectionFit:
    """Linear least-squares fit for one fixed frequency and tau."""

    tau: float
    rss: float
    sigma: float
    dof: int
    amplitude: float
    rank: int


class ProfileQEstimator:
    """
    Estimate Q by profiling decay time with variable projection.

    For each candidate tau, the sinusoid coefficients and DC offset are solved
    by linear least squares. The reported finite Q is only valid when the
    profiled likelihood interval closes on both sides of the optimum.
    """

    def __init__(
        self,
        *,
        n_grid: int = 161,
        chi2_threshold: float = _CHI2_95_ONE_PARAMETER,
        tau_max_record_multiplier: float = 100.0,
        tau_init_span: float = 20.0,
    ):
        if n_grid < 25:
            raise ValueError(f"n_grid must be at least 25, got {n_grid}")
        if not np.isfinite(chi2_threshold) or chi2_threshold <= 0:
            raise ValueError(f"chi2_threshold must be positive and finite, got {chi2_threshold}")
        if not np.isfinite(tau_max_record_multiplier) or tau_max_record_multiplier <= 1:
            raise ValueError(
                "tau_max_record_multiplier must be finite and greater than 1, "
                f"got {tau_max_record_multiplier}"
            )
        if not np.isfinite(tau_init_span) or tau_init_span <= 1:
            raise ValueError(
                f"tau_init_span must be finite and greater than 1, got {tau_init_span}"
            )

        self.n_grid = int(n_grid)
        self.chi2_threshold = float(chi2_threshold)
        self.tau_max_record_multiplier = float(tau_max_record_multiplier)
        self.tau_init_span = float(tau_init_span)

    @staticmethod
    def _invalid_result(
        *,
        status: str,
        reasons: list[str],
        method: str,
        f_hat: float | None = None,
        tau_hat: float | None = None,
    ) -> QProfileResult:
        return QProfileResult(
            f_hat=f_hat,
            tau_hat=tau_hat,
            Q=None,
            valid=False,
            status=status,
            reasons=reasons,
            ci95=None,
            lower_limit_95=None,
            upper_limit_95=None,
            profile_tau=np.array([], dtype=float),
            profile_q=np.array([], dtype=float),
            profile_delta=np.array([], dtype=float),
            rss_min=np.nan,
            sigma=np.nan,
            dof=0,
            n_grid=0,
            method=method,
        )

    @staticmethod
    def _has_resolved_ac_content(data: np.ndarray) -> bool:
        demeaned = data - np.mean(data)
        threshold = max(float(np.max(np.abs(data))) * 1e-12, np.finfo(np.float64).eps * 10.0)
        return float(np.max(np.abs(demeaned))) > threshold

    @staticmethod
    def _fit_fixed_tau(t: np.ndarray, data: np.ndarray, f_hat: float, tau: float) -> _ProjectionFit:
        exp_term = np.exp(-t / tau)
        omega_t = 2.0 * np.pi * f_hat * t
        design = np.column_stack(
            [
                exp_term * np.cos(omega_t),
                exp_term * np.sin(omega_t),
                np.ones_like(t),
            ]
        )
        coeffs, _, rank, _ = np.linalg.lstsq(design, data, rcond=None)
        if rank < design.shape[1]:
            raise np.linalg.LinAlgError("Profile-Q design matrix is rank-deficient")

        residuals = data - design @ coeffs
        rss = float(np.sum(residuals**2))
        dof = int(len(data) - design.shape[1])
        if dof <= 0:
            raise ValueError("Profile-Q degrees of freedom must be positive")

        a_cos, b_sin, _ = coeffs
        sigma = float(np.sqrt(max(rss, 0.0) / dof))
        amplitude = float(np.hypot(a_cos, b_sin))
        return _ProjectionFit(
            tau=float(tau),
            rss=rss,
            sigma=sigma,
            dof=dof,
            amplitude=amplitude,
            rank=int(rank),
        )

    @staticmethod
    def _crossing_tau(
        tau0: float,
        delta0: float,
        tau1: float,
        delta1: float,
        threshold: float,
    ) -> float:
        log_tau0 = float(np.log(tau0))
        log_tau1 = float(np.log(tau1))
        if delta1 == delta0:
            return float(np.exp(0.5 * (log_tau0 + log_tau1)))
        frac = (threshold - delta0) / (delta1 - delta0)
        frac = float(np.clip(frac, 0.0, 1.0))
        return float(np.exp(log_tau0 + frac * (log_tau1 - log_tau0)))

    def _tau_bounds(
        self,
        t: np.ndarray,
        tau_init: float | None,
        tau_bounds: tuple[float, float] | None,
    ) -> tuple[float, float]:
        dt = float(np.median(np.diff(t)))
        duration = float(t[-1] - t[0])
        lower_floor = max(dt, np.finfo(np.float64).eps)

        if tau_bounds is not None:
            tau_min, tau_max = float(tau_bounds[0]), float(tau_bounds[1])
        else:
            tau_seed = float(tau_init) if tau_init is not None and np.isfinite(tau_init) else np.nan
            if not np.isfinite(tau_seed) or tau_seed <= 0:
                tau_seed = _estimate_initial_tau_from_envelope(np.ones_like(t), t)
            tau_min = max(lower_floor, tau_seed / self.tau_init_span)
            tau_max = max(
                tau_min * 10.0,
                tau_seed * self.tau_init_span,
                duration * self.tau_max_record_multiplier,
            )

        if (
            not np.isfinite(tau_min)
            or not np.isfinite(tau_max)
            or tau_min <= 0
            or tau_max <= tau_min
        ):
            raise ValueError(f"Invalid tau bounds for profile Q: ({tau_min}, {tau_max})")
        return float(max(tau_min, lower_floor)), float(tau_max)

    def estimate(
        self,
        t: np.ndarray,
        data: np.ndarray,
        fs: float,
        *,
        f_init: float | None = None,
        tau_init: float | None = None,
        tau_bounds: tuple[float, float] | None = None,
        n_grid: int | None = None,
    ) -> QProfileResult:
        """
        Profile tau and Q for a single ring-down record.

        The current implementation fixes frequency at ``f_init`` when supplied,
        or at a DFT initializer otherwise, then profiles ``log(tau)``.
        """
        method = "fixed_frequency_log_tau_variable_projection"
        t_arr = np.asarray(t, dtype=np.float64)
        data_arr = np.asarray(data, dtype=np.float64)
        if t_arr.ndim != 1 or data_arr.ndim != 1:
            raise ValueError("t and data must be one-dimensional arrays")
        if len(t_arr) != len(data_arr):
            raise ValueError(
                f"t and data must have same length, got {len(t_arr)} and {len(data_arr)}"
            )
        if len(t_arr) < 5:
            return self._invalid_result(
                status="invalid",
                reasons=["profile_insufficient_samples"],
                method=method,
            )
        if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(data_arr)):
            raise ValueError("t and data must contain only finite values")
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"Sampling frequency fs must be positive and finite, got {fs}")

        t_norm = t_arr - t_arr[0]
        if np.any(np.diff(t_norm) <= 0):
            raise ValueError("t must be strictly increasing")
        if not self._has_resolved_ac_content(data_arr):
            return self._invalid_result(
                status="invalid",
                reasons=["profile_no_resolved_ac_content"],
                method=method,
            )

        if f_init is None or not np.isfinite(f_init) or f_init <= 0:
            f_hat = float(_estimate_initial_parameters_from_dft(data_arr, fs)[0])
        else:
            f_hat = float(f_init)
        if not np.isfinite(f_hat) or f_hat <= 0 or f_hat >= 0.5 * fs:
            return self._invalid_result(
                status="failed",
                reasons=["profile_frequency_missing_or_out_of_range"],
                method=method,
                f_hat=f_hat if np.isfinite(f_hat) else None,
            )

        if tau_init is None or not np.isfinite(tau_init) or tau_init <= 0:
            tau_init = float(_estimate_initial_tau_from_envelope(data_arr, t_norm))
        tau_min, tau_max = self._tau_bounds(t_norm, tau_init, tau_bounds)
        n_profile_grid = int(n_grid or self.n_grid)
        if n_profile_grid < 25:
            raise ValueError(f"n_grid must be at least 25, got {n_profile_grid}")

        log_tau_min = float(np.log(tau_min))
        log_tau_max = float(np.log(tau_max))
        tau_grid = np.exp(np.linspace(log_tau_min, log_tau_max, n_profile_grid))

        def fit_at_log_tau(log_tau: float) -> _ProjectionFit:
            tau = float(np.exp(log_tau))
            return self._fit_fixed_tau(t_norm, data_arr, f_hat, tau)

        fits: list[_ProjectionFit] = []
        try:
            for tau in tau_grid:
                fits.append(self._fit_fixed_tau(t_norm, data_arr, f_hat, float(tau)))
        except (np.linalg.LinAlgError, ValueError) as exc:
            return self._invalid_result(
                status="failed",
                reasons=[f"profile_lstsq_failed:{type(exc).__name__}"],
                method=method,
                f_hat=f_hat,
            )

        rss_grid = np.array([fit.rss for fit in fits], dtype=float)
        best_grid = fits[int(np.argmin(rss_grid))]
        best_fit = best_grid
        try:
            opt = minimize_scalar(
                lambda log_tau: fit_at_log_tau(log_tau).rss,
                bounds=(log_tau_min, log_tau_max),
                method="bounded",
                options={"xatol": 1e-5},
            )
            if opt.success and np.isfinite(opt.fun):
                best_fit = fit_at_log_tau(float(opt.x))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            best_fit = best_grid

        tau_profile = np.array([fit.tau for fit in fits] + [best_fit.tau], dtype=float)
        rss_profile = np.array([fit.rss for fit in fits] + [best_fit.rss], dtype=float)
        order = np.argsort(tau_profile)
        tau_profile = tau_profile[order]
        rss_profile = rss_profile[order]

        rss_scale = max(float(np.sum((data_arr - np.mean(data_arr)) ** 2)), 1.0)
        rss_floor = np.finfo(np.float64).tiny * rss_scale
        rss_min = float(max(best_fit.rss, rss_floor))
        profile_delta = best_fit.dof * np.log(np.maximum(rss_profile, rss_floor) / rss_min)
        profile_delta = np.maximum(profile_delta, 0.0)
        profile_q = np.pi * f_hat * tau_profile
        best_index = int(np.argmin(np.abs(tau_profile - best_fit.tau)))

        reasons: list[str] = []
        if best_index == 0:
            reasons.append("profile_min_at_lower_tau_bound")
        if best_index == len(tau_profile) - 1:
            reasons.append("profile_min_at_upper_tau_bound")
        if best_fit.amplitude <= max(float(np.std(data_arr)) * 1e-12, np.finfo(np.float64).eps):
            return self._invalid_result(
                status="invalid",
                reasons=["profile_amplitude_unresolved"],
                method=method,
                f_hat=f_hat,
                tau_hat=best_fit.tau,
            )

        inside = profile_delta <= self.chi2_threshold
        if not bool(inside[best_index]):
            return self._invalid_result(
                status="failed",
                reasons=["profile_minimum_not_inside_threshold_region"],
                method=method,
                f_hat=f_hat,
                tau_hat=best_fit.tau,
            )

        left = best_index
        while left > 0 and inside[left - 1]:
            left -= 1
        right = best_index
        while right < len(inside) - 1 and inside[right + 1]:
            right += 1

        lower_tau_95: float | None = None
        upper_tau_95: float | None = None
        if left > 0:
            lower_tau_95 = self._crossing_tau(
                tau_profile[left - 1],
                profile_delta[left - 1],
                tau_profile[left],
                profile_delta[left],
                self.chi2_threshold,
            )
        if right < len(profile_delta) - 1:
            upper_tau_95 = self._crossing_tau(
                tau_profile[right],
                profile_delta[right],
                tau_profile[right + 1],
                profile_delta[right + 1],
                self.chi2_threshold,
            )

        q_hat = float(np.pi * f_hat * best_fit.tau)
        lower_q_95 = float(np.pi * f_hat * lower_tau_95) if lower_tau_95 is not None else None
        upper_q_95 = float(np.pi * f_hat * upper_tau_95) if upper_tau_95 is not None else None

        valid = lower_q_95 is not None and upper_q_95 is not None and not reasons
        if valid:
            assert lower_q_95 is not None
            assert upper_q_95 is not None
            status = "valid"
            q_value: float | None = q_hat
            ci95 = (lower_q_95, upper_q_95)
            lower_limit_95 = None
            upper_limit_95 = None
        elif lower_q_95 is not None and upper_q_95 is None:
            status = "lower_limit"
            q_value = None
            ci95 = None
            lower_limit_95 = lower_q_95
            upper_limit_95 = None
            reasons.append("profile_open_high")
        elif lower_q_95 is None and upper_q_95 is not None:
            status = "upper_limit"
            q_value = None
            ci95 = None
            lower_limit_95 = None
            upper_limit_95 = upper_q_95
            reasons.append("profile_open_low")
        else:
            status = "unbounded"
            q_value = None
            ci95 = None
            lower_limit_95 = None
            upper_limit_95 = None
            reasons.append("profile_does_not_cross_threshold")

        return QProfileResult(
            f_hat=f_hat,
            tau_hat=float(best_fit.tau),
            Q=q_value,
            valid=valid,
            status=status,
            reasons=reasons,
            ci95=ci95,
            lower_limit_95=lower_limit_95,
            upper_limit_95=upper_limit_95,
            profile_tau=tau_profile,
            profile_q=profile_q,
            profile_delta=profile_delta,
            rss_min=float(best_fit.rss),
            sigma=float(best_fit.sigma),
            dof=int(best_fit.dof),
            n_grid=int(len(tau_profile)),
            method=method,
        )
