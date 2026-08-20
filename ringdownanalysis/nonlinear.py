"""
Nonlinear-damping and frequency-pull model for demodulated ring-down data.

Real resonators can show amplitude-dependent damping — the local decay rate
follows 1/tau(A) = 1/tau0 + beta * A — and an amplitude-dependent resonance
frequency (anharmonic pull), f(A) = f_zero + pull * A. Any single-exponential
Q estimate on such data is a window-dependent average. This module fits both
amplitude laws on the per-segment output of the SegmentedDemodEstimator,
turning the nonlinearity from a nuisance into a measurement: it reports the
zero-amplitude decay time tau0 (and Q0 = pi * f_zero * tau0), the damping
coefficient beta, the zero-amplitude frequency and the pull coefficient.

The amplitude ODE dA/dt = -A * (1/tau0 + beta*A) is a Bernoulli equation with
the closed-form solution

    A(t) = 1 / [ (1/A0 + beta*tau0) * exp(t/tau0) - beta*tau0 ],

which is fit to the floor-corrected segment amplitudes in log space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .demod import SegmentedDemodResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NonlinearDampingResult:
    """Result of the nonlinear-damping / frequency-pull fit."""

    #: Zero-amplitude decay time (s).
    tau0: float | None
    #: Amplitude-dependent damping coefficient: 1/tau(A) = 1/tau0 + beta*A.
    beta: float | None
    #: Fitted initial amplitude.
    a0: float | None
    #: Zero-amplitude Q = pi * f_zero * tau0.
    Q0: float | None
    #: Zero-amplitude (extrapolated) frequency (Hz).
    f_zero: float | None
    #: Frequency-pull coefficient (Hz per amplitude unit): f(A) = f_zero + pull*A.
    f_pull: float | None
    # Approximate 1-sigma standard errors (from the fit covariance; they do
    # not account for correlated segment-to-segment systematics).
    tau0_stderr: float | None
    beta_stderr: float | None
    f_zero_stderr: float | None
    f_pull_stderr: float | None
    valid: bool
    status: str
    reasons: list[str]
    #: RMS of the log-amplitude fit residual.
    log_residual_rms: float | None
    n_segments: int
    # Fit inputs and model prediction for plotting.
    t_fit: np.ndarray
    amplitude_fit: np.ndarray
    model_amplitude: np.ndarray
    method: str = "bernoulli_nonlinear_damping_log_fit"

    def tau_at(self, amplitude: float | np.ndarray):
        """Local decay time tau(A) = 1 / (1/tau0 + beta*A)."""
        if self.tau0 is None or self.beta is None:
            raise ValueError("tau_at requires a successful fit (tau0 and beta)")
        return 1.0 / (1.0 / self.tau0 + self.beta * np.asarray(amplitude, dtype=float))

    def q_at(self, amplitude: float | np.ndarray):
        """Local quality factor Q(A) = pi * f(A) * tau(A)."""
        if self.f_zero is None or self.f_pull is None:
            raise ValueError("q_at requires a successful frequency fit")
        amp = np.asarray(amplitude, dtype=float)
        return np.pi * (self.f_zero + self.f_pull * amp) * self.tau_at(amp)


def _model_log_amplitude(t: np.ndarray, log_a0: float, log_tau0: float, beta: float) -> np.ndarray:
    """log A(t) for the Bernoulli decay; +inf-safe for invalid parameters."""
    a0 = np.exp(log_a0)
    tau0 = np.exp(log_tau0)
    u = (1.0 / a0 + beta * tau0) * np.exp(t / tau0) - beta * tau0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(u > 0, -np.log(np.maximum(u, np.finfo(np.float64).tiny)), np.nan)
    return out


def _empty_result(status: str, reasons: list[str], n_segments: int = 0) -> NonlinearDampingResult:
    empty = np.array([], dtype=float)
    return NonlinearDampingResult(
        tau0=None,
        beta=None,
        a0=None,
        Q0=None,
        f_zero=None,
        f_pull=None,
        tau0_stderr=None,
        beta_stderr=None,
        f_zero_stderr=None,
        f_pull_stderr=None,
        valid=False,
        status=status,
        reasons=reasons,
        log_residual_rms=None,
        n_segments=n_segments,
        t_fit=empty,
        amplitude_fit=empty,
        model_amplitude=empty,
    )


def fit_nonlinear_damping(
    demod_result: SegmentedDemodResult,
    *,
    min_segments: int = 12,
    min_amplitude_efolds: float = 1.0,
) -> NonlinearDampingResult:
    """
    Fit the amplitude-dependent damping and frequency-pull laws.

    Parameters:
    -----------
    demod_result : SegmentedDemodResult
        Output of SegmentedDemodEstimator.estimate on the record. The fit
        uses the decay-region segments and floor-corrected amplitudes.
    min_segments : int
        Minimum number of decay segments required for the fit.
    min_amplitude_efolds : float
        Minimum log-amplitude span of the decay region; below this the
        damping law is not identifiable (beta and 1/tau0 are degenerate).

    Returns:
    --------
    NonlinearDampingResult
        (tau0, beta, Q0), (f_zero, pull), standard errors, and the model
        prediction over the fitted segments.
    """
    mask = demod_result.decay_mask
    n_fit = int(np.count_nonzero(mask))
    if n_fit < min_segments:
        return _empty_result("invalid", ["nonlinear_insufficient_decay_segments"], n_fit)

    t_fit = demod_result.t_mid[mask]
    amp_fit = demod_result.amplitude_corrected[mask]
    f_fit = demod_result.f_seg[mask]
    log_amp = np.log(amp_fit)

    span_efolds = float(np.ptp(log_amp))
    if span_efolds < min_amplitude_efolds:
        return _empty_result("invalid", ["nonlinear_amplitude_span_too_small"], n_fit)

    # Seed from the single-exponential fit; beta = 0.
    slope0, intercept0 = np.polyfit(t_fit, log_amp, 1)
    if slope0 >= 0:
        return _empty_result("invalid", ["nonlinear_nondecaying_amplitude"], n_fit)
    log_a0_seed = float(intercept0)
    log_tau0_seed = float(np.log(-1.0 / slope0))
    a_max = float(np.max(amp_fit))
    # |beta| is capped so the amplitude-dependent rate cannot exceed ~100x the
    # single-exponential rate at the largest observed amplitude.
    beta_cap = 100.0 * (-slope0) / a_max

    def residuals(params: np.ndarray) -> np.ndarray:
        model = _model_log_amplitude(t_fit, params[0], params[1], params[2])
        res = log_amp - model
        return np.where(np.isfinite(res), res, 1e3)

    fit = least_squares(
        residuals,
        x0=np.array([log_a0_seed, log_tau0_seed, 0.0]),
        bounds=(
            np.array([log_a0_seed - 5.0, log_tau0_seed - 5.0, -beta_cap]),
            np.array([log_a0_seed + 5.0, log_tau0_seed + 5.0, beta_cap]),
        ),
        x_scale=np.array([1.0, 1.0, max(beta_cap / 100.0, np.finfo(np.float64).tiny)]),
    )
    if not fit.success:
        return _empty_result("invalid", ["nonlinear_fit_failed"], n_fit)

    log_a0_hat, log_tau0_hat, beta_hat = (float(v) for v in fit.x)
    a0_hat = float(np.exp(log_a0_hat))
    tau0_hat = float(np.exp(log_tau0_hat))
    model_log = _model_log_amplitude(t_fit, log_a0_hat, log_tau0_hat, beta_hat)
    resid = log_amp - model_log
    log_residual_rms = float(np.sqrt(np.mean(resid**2)))

    # Approximate parameter covariance from the Jacobian at the solution.
    tau0_stderr: float | None = None
    beta_stderr: float | None = None
    dof = n_fit - 3
    if dof > 0:
        jtj = fit.jac.T @ fit.jac
        try:
            cov = np.linalg.inv(jtj) * float(np.sum(resid**2) / dof)
            log_tau0_var = float(cov[1, 1])
            beta_var = float(cov[2, 2])
            if log_tau0_var >= 0:
                tau0_stderr = tau0_hat * float(np.sqrt(log_tau0_var))
            if beta_var >= 0:
                beta_stderr = float(np.sqrt(beta_var))
        except np.linalg.LinAlgError:
            pass

    # Frequency-pull law: linear in amplitude, f(A) = f_zero + pull * A.
    pull_slope, f_zero_hat = (float(v) for v in np.polyfit(amp_fit, f_fit, 1))
    f_resid = f_fit - (pull_slope * amp_fit + f_zero_hat)
    f_dof = n_fit - 2
    sxx = float(np.sum((amp_fit - np.mean(amp_fit)) ** 2))
    if f_dof > 0 and sxx > 0:
        s2 = float(np.sum(f_resid**2) / f_dof)
        f_pull_stderr = float(np.sqrt(s2 / sxx))
        f_zero_stderr = float(np.sqrt(s2 * (1.0 / n_fit + np.mean(amp_fit) ** 2 / sxx)))
    else:
        f_pull_stderr = None
        f_zero_stderr = None

    q0 = float(np.pi * f_zero_hat * tau0_hat)

    reasons: list[str] = []
    status = "valid"
    if beta_cap > 0 and abs(beta_hat) >= 0.999 * beta_cap:
        status = "warning"
        reasons.append("nonlinear_beta_at_bound")
    if span_efolds < 2.0:
        status = "warning"
        reasons.append("nonlinear_amplitude_span_below_two_efolds")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "nonlinear_damping_fit",
            extra={
                "event": "nonlinear_damping_fit",
                "tau0": tau0_hat,
                "beta": beta_hat,
                "q0": q0,
                "f_zero": f_zero_hat,
                "f_pull": pull_slope,
                "n_segments": n_fit,
            },
        )

    return NonlinearDampingResult(
        tau0=tau0_hat,
        beta=beta_hat,
        a0=a0_hat,
        Q0=q0,
        f_zero=f_zero_hat,
        f_pull=pull_slope,
        tau0_stderr=tau0_stderr,
        beta_stderr=beta_stderr,
        f_zero_stderr=f_zero_stderr,
        f_pull_stderr=f_pull_stderr,
        valid=status == "valid",
        status=status,
        reasons=reasons,
        log_residual_rms=log_residual_rms,
        n_segments=n_fit,
        t_fit=t_fit,
        amplitude_fit=amp_fit,
        model_amplitude=np.exp(model_log),
    )
