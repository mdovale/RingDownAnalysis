"""
Profile-likelihood Q estimation with separable least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from ._gridfit import (
    RSS_CANCELLATION_FRACTION,
    BlockIndex,
    geometric_sum,
    is_uniformly_sampled,
)
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


class _ProfileScan:
    """
    Fixed-frequency decaying-tone least squares over a batch of trial taus.

    Every fit in a profile scan solves the same 3-parameter problem
    ``y ~ exp(-t/tau) * (a*cos(w*t) + b*sin(w*t)) + c`` with the frequency, the
    time base and the data all held fixed, so nothing but the envelope changes
    from one trial tau to the next. Two properties of that structure remove
    almost all of the work:

    - the fit only needs the Gram matrix and the data projections, not the
      design matrix, and the residual sum of squares follows from them as
      ``||y||^2 - b'x``;
    - on a uniform time grid the model is a geometric sequence in the sample
      index, so the Gram matrix is closed-form
      (:func:`~ringdownanalysis._gridfit.geometric_sum`) and the projections
      for the whole batch of taus come from two matrix products
      (:class:`~ringdownanalysis._gridfit.BlockIndex`).

    Together these replace a length-n exponential, a fresh design matrix and an
    SVD least-squares solve per grid point with one pass over the data for the
    entire scan.

    The uniform path evaluates the model on the nominal grid ``k*dt``, with
    ``dt`` taken from the record endpoints; the uniformity test bounds the
    resulting phase error (see
    :data:`~ringdownanalysis._gridfit.UNIFORM_TOLERANCE_SAMPLES`). Records that
    fail it fall back to an explicit design matrix and ``lstsq``, sharing the
    tone basis across the scan.
    """

    __slots__ = (
        "_cos_wt",
        "_index",
        "_matrix",
        "_phase_step",
        "_rss_floor",
        "_sin_wt",
        "data",
        "dt",
        "f_hat",
        "n",
        "sy",
        "t",
        "uniform",
        "yy",
    )

    def __init__(self, t: np.ndarray, data: np.ndarray, f_hat: float):
        self.t = t
        self.data = data
        self.f_hat = float(f_hat)
        self.n = len(data)
        self.yy = float(data @ data)
        self.sy = float(np.sum(data))
        self._rss_floor = RSS_CANCELLATION_FRACTION * self.yy
        self.dt = float(t[-1] - t[0]) / (self.n - 1) if self.n > 1 else 0.0
        self.uniform = self.dt > 0.0 and is_uniformly_sampled(t, 1.0 / self.dt)
        if self.uniform:
            self._index = BlockIndex(data)
            self._phase_step = 2.0 * np.pi * self.f_hat * self.dt
        else:
            omega_t = 2.0 * np.pi * self.f_hat * t
            self._cos_wt = np.cos(omega_t)
            self._sin_wt = np.sin(omega_t)
            self._matrix = np.empty((self.n, 3), dtype=np.float64)
            self._matrix[:, 2] = 1.0

    def fit(self, taus: np.ndarray) -> list[_ProjectionFit]:
        """
        Fit every trial tau, cheapest available path first.

        Raises ``LinAlgError`` if the model is degenerate at any trial tau and
        ``ValueError`` if the record cannot support the three parameters, which
        is how the caller learns the profile is unusable.
        """
        taus = np.atleast_1d(np.asarray(taus, dtype=np.float64))
        dof = self.n - 3
        if dof <= 0:
            raise ValueError("Profile-Q degrees of freedom must be positive")
        if self.uniform:
            return self._fit_uniform(taus, dof)
        return [self._fit_direct(float(tau), dof) for tau in taus]

    # -- uniform grid --------------------------------------------------

    def _fit_uniform(self, taus: np.ndarray, dof: int) -> list[_ProjectionFit]:
        # Per-sample log-ratio of the model: decay plus phase advance.
        log_ratio = -(self.dt / taus) + 1j * self._phase_step
        envelope = geometric_sum(self.n, log_ratio)
        squared = geometric_sum(self.n, 2.0 * log_ratio)
        # sum exp(-2t/tau): the same series with the tone removed.
        envelope2 = geometric_sum(self.n, 2.0 * log_ratio.real).real
        projection = self._index.project(log_ratio)

        m = len(taus)
        n = float(self.n)
        gram = np.empty((m, 3, 3), dtype=np.float64)
        # sum e^(-2t/tau) cos^2 = (sum e^(-2t/tau) + sum e^(-2t/tau) cos 2wt)/2,
        # and likewise for sin^2 and the cross term.
        gram[:, 0, 0] = 0.5 * (envelope2 + squared.real)
        gram[:, 1, 1] = 0.5 * (envelope2 - squared.real)
        gram[:, 0, 1] = gram[:, 1, 0] = 0.5 * squared.imag
        gram[:, 0, 2] = gram[:, 2, 0] = envelope.real
        gram[:, 1, 2] = gram[:, 2, 1] = envelope.imag
        gram[:, 2, 2] = n
        rhs = np.stack([projection.real, projection.imag, np.full(m, self.sy)], axis=1)

        coef = np.linalg.solve(gram, rhs[..., None])[..., 0]
        rss = self.yy - np.einsum("mi,mi->m", rhs, coef)
        if not np.all(np.isfinite(rss)):
            raise np.linalg.LinAlgError("Profile-Q normal equations are degenerate")

        fits: list[_ProjectionFit] = []
        for idx in range(m):
            value = float(rss[idx])
            if value < self._rss_floor:
                value = self._explicit_rss(float(taus[idx]), coef[idx])
            fits.append(
                _ProjectionFit(
                    tau=float(taus[idx]),
                    rss=value,
                    sigma=float(np.sqrt(max(value, 0.0) / dof)),
                    dof=dof,
                    amplitude=float(np.hypot(coef[idx, 0], coef[idx, 1])),
                    rank=3,
                )
            )
        return fits

    def _explicit_rss(self, tau: float, coef: np.ndarray) -> float:
        """Residual sum of squares formed directly, for near-perfect fits."""
        grid = np.arange(self.n) * self.dt
        theta = (2.0 * np.pi * self.f_hat) * grid
        model = np.exp(-grid / tau) * (coef[0] * np.cos(theta) + coef[1] * np.sin(theta)) + coef[2]
        resid = self.data - model
        return float(resid @ resid)

    # -- non-uniform fallback ------------------------------------------

    def _fit_direct(self, tau: float, dof: int) -> _ProjectionFit:
        exp_term = np.exp(-self.t / tau)
        np.multiply(exp_term, self._cos_wt, out=self._matrix[:, 0])
        np.multiply(exp_term, self._sin_wt, out=self._matrix[:, 1])
        coeffs, _, rank, _ = np.linalg.lstsq(self._matrix, self.data, rcond=None)
        if rank < 3:
            raise np.linalg.LinAlgError("Profile-Q design matrix is rank-deficient")
        residuals = self.data - self._matrix @ coeffs
        rss = float(np.sum(residuals**2))
        return _ProjectionFit(
            tau=tau,
            rss=rss,
            sigma=float(np.sqrt(max(rss, 0.0) / dof)),
            dof=dof,
            amplitude=float(np.hypot(coeffs[0], coeffs[1])),
            rank=int(rank),
        )


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
    def _make_scan(t: np.ndarray, data: np.ndarray, f_hat: float) -> _ProfileScan:
        """
        Build the fitting engine for one profile scan.

        Overridable so that alternative fit implementations can be substituted
        wholesale (the benchmarks' frozen reference does this).
        """
        return _ProfileScan(t, data, f_hat)

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

        # Built once and shared by every fit in the scan below.
        scan = self._make_scan(t_norm, data_arr, f_hat)

        def fit_at_log_tau(log_tau: float) -> _ProjectionFit:
            return scan.fit(np.array([np.exp(log_tau)]))[0]

        try:
            fits = scan.fit(tau_grid)
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
