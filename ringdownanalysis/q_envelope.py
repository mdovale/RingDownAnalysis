"""
Peak-to-peak envelope diagnostics for Q and decay-time estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QEnvelopeDiagnostic:
    """Envelope-derived tau/Q estimate and optional candidate-Q agreement."""

    tau: float | None
    Q: float | None
    valid: bool
    status: str
    reasons: list[str]
    n_windows: int
    n_windows_used: int
    t_mid: np.ndarray
    amplitude: np.ndarray
    used: np.ndarray
    fitted_amplitude: np.ndarray
    candidate_q: float | None
    candidate_tau: float | None
    candidate_amplitude: np.ndarray
    log_amplitude_slope: float | None
    log_amplitude_intercept: float | None
    log_amplitude_rmse: float | None
    slope_stderr: float | None
    candidate_log_rmse: float | None
    candidate_slope_mismatch: float | None
    candidate_agrees: bool | None
    method: str


def _empty_result(status: str, reasons: list[str], method: str) -> QEnvelopeDiagnostic:
    empty = np.array([], dtype=float)
    return QEnvelopeDiagnostic(
        tau=None,
        Q=None,
        valid=False,
        status=status,
        reasons=reasons,
        n_windows=0,
        n_windows_used=0,
        t_mid=empty,
        amplitude=empty,
        used=np.array([], dtype=bool),
        fitted_amplitude=empty,
        candidate_q=None,
        candidate_tau=None,
        candidate_amplitude=empty,
        log_amplitude_slope=None,
        log_amplitude_intercept=None,
        log_amplitude_rmse=None,
        slope_stderr=None,
        candidate_log_rmse=None,
        candidate_slope_mismatch=None,
        candidate_agrees=None,
        method=method,
    )


def _window_peak_to_peak_amplitudes(
    t: np.ndarray,
    data: np.ndarray,
    *,
    f_hz: float,
    cycles_per_window: float,
    step_cycles: float,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    duration = float(t[-1] - t[0])
    dt = float(np.median(np.diff(t)))
    period = 1.0 / f_hz
    window_duration = max(cycles_per_window * period, 5.0 * dt)
    if window_duration >= duration:
        return np.array([], dtype=float), np.array([], dtype=float)

    step_duration = max(step_cycles * period, dt)
    n_possible = int(np.floor((duration - window_duration) / step_duration)) + 1
    if n_possible <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_windows = max(1, min(int(max_windows), n_possible))
    starts = np.linspace(0.0, duration - window_duration, n_windows)
    t_mid = starts + 0.5 * window_duration
    amplitudes = np.empty(n_windows, dtype=float)

    for i, start in enumerate(starts):
        left = int(np.searchsorted(t, start, side="left"))
        right = int(np.searchsorted(t, start + window_duration, side="right"))
        segment = data[left:right]
        if len(segment) < 2:
            amplitudes[i] = np.nan
        elif len(segment) >= 5:
            amplitudes[i] = 0.5 * float(np.percentile(segment, 95.0) - np.percentile(segment, 5.0))
        else:
            amplitudes[i] = 0.5 * float(np.max(segment) - np.min(segment))

    return t_mid, amplitudes


def _robust_log_slope(
    t_mid: np.ndarray,
    amplitude: np.ndarray,
    used: np.ndarray,
    *,
    min_windows: int,
) -> tuple[float, float, np.ndarray]:
    fit_t_abs = t_mid[used]
    fit_t = fit_t_abs - fit_t_abs[0]
    log_amp = np.log(amplitude[used])
    slope, intercept = np.polyfit(fit_t, log_amp, deg=1)
    residual = log_amp - (slope * fit_t + intercept)

    median_residual = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median_residual)))
    scale = 1.4826 * mad
    if np.isfinite(scale) and scale > 0:
        keep = np.abs(residual - median_residual) <= max(3.0 * scale, 0.25)
        if int(np.count_nonzero(keep)) >= min_windows and not bool(np.all(keep)):
            used_indices = np.flatnonzero(used)
            used = np.zeros_like(used, dtype=bool)
            used[used_indices[keep]] = True
            fit_t_abs = t_mid[used]
            fit_t = fit_t_abs - fit_t_abs[0]
            log_amp = np.log(amplitude[used])
            slope, intercept = np.polyfit(fit_t, log_amp, deg=1)

    return float(slope), float(intercept), used


def _slope_stderr(fit_t: np.ndarray, residual: np.ndarray) -> float | None:
    dof = len(fit_t) - 2
    if dof <= 0:
        return None
    sxx = float(np.sum((fit_t - np.mean(fit_t)) ** 2))
    if sxx <= 0 or not np.isfinite(sxx):
        return None
    sigma = float(np.sqrt(np.sum(residual**2) / dof))
    return float(sigma / np.sqrt(sxx))


def q_envelope_diagnostic(
    t: np.ndarray,
    data: np.ndarray,
    f_hz: float,
    q: float | None = None,
    *,
    cycles_per_window: float = 3.0,
    step_cycles: float = 1.0,
    max_windows: int = 500,
    min_windows: int = 5,
    min_amplitude_fraction: float = 0.05,
    candidate_log_rmse_warning: float = 0.5,
    candidate_slope_mismatch_warning: float = 0.75,
) -> QEnvelopeDiagnostic:
    """
    Estimate the ring-down envelope and optionally compare a candidate Q.

    The envelope is measured as robust peak-to-peak amplitude in sliding windows
    of several cycles. A straight line is fit to log-amplitude versus time over
    high-amplitude windows, yielding an independent tau/Q diagnostic.
    """
    method = "sliding_peak_to_peak_log_envelope"
    t_arr = np.asarray(t, dtype=np.float64)
    data_arr = np.asarray(data, dtype=np.float64)
    if t_arr.ndim != 1 or data_arr.ndim != 1:
        raise ValueError("t and data must be one-dimensional arrays")
    if len(t_arr) != len(data_arr):
        raise ValueError(f"t and data must have same length, got {len(t_arr)} and {len(data_arr)}")
    if len(t_arr) < 3:
        return _empty_result("invalid", ["envelope_insufficient_samples"], method)
    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(data_arr)):
        raise ValueError("t and data must contain only finite values")
    if not np.isfinite(f_hz) or f_hz <= 0:
        return _empty_result("invalid", ["envelope_frequency_missing_or_nonpositive"], method)
    if cycles_per_window <= 0 or step_cycles <= 0:
        raise ValueError("cycles_per_window and step_cycles must be positive")
    if max_windows < 1:
        raise ValueError(f"max_windows must be positive, got {max_windows}")
    if min_windows < 2:
        raise ValueError(f"min_windows must be at least 2, got {min_windows}")

    t_norm = t_arr - t_arr[0]
    if np.any(np.diff(t_norm) <= 0):
        raise ValueError("t must be strictly increasing")

    t_mid, amplitude = _window_peak_to_peak_amplitudes(
        t_norm,
        data_arr,
        f_hz=f_hz,
        cycles_per_window=cycles_per_window,
        step_cycles=step_cycles,
        max_windows=max_windows,
    )
    n_windows = len(t_mid)
    if n_windows < min_windows:
        return _empty_result("invalid", ["envelope_insufficient_windows"], method)

    finite_amp = np.isfinite(amplitude) & (amplitude > 0)
    if int(np.count_nonzero(finite_amp)) < min_windows:
        return _empty_result("invalid", ["envelope_insufficient_positive_windows"], method)

    max_amp = float(np.nanmax(amplitude[finite_amp]))
    amplitude_floor = max(max_amp * float(min_amplitude_fraction), np.finfo(np.float64).eps)
    used = finite_amp & (amplitude >= amplitude_floor)
    if int(np.count_nonzero(used)) < min_windows:
        relaxed_floor = max(max_amp * 0.01, np.finfo(np.float64).eps)
        used = finite_amp & (amplitude >= relaxed_floor)
    if int(np.count_nonzero(used)) < min_windows:
        return _empty_result("invalid", ["envelope_insufficient_high_snr_windows"], method)

    slope, intercept, used = _robust_log_slope(t_mid, amplitude, used, min_windows=min_windows)
    fit_t_abs = t_mid[used]
    fit_t = fit_t_abs - fit_t_abs[0]
    log_amp = np.log(amplitude[used])
    fitted_log_used = slope * fit_t + intercept
    residual = log_amp - fitted_log_used
    log_rmse = float(np.sqrt(np.mean(residual**2)))
    stderr = _slope_stderr(fit_t, residual)

    first_used_t = float(fit_t_abs[0])
    fitted_log_all = intercept + slope * (t_mid - first_used_t)
    fitted_amplitude = np.exp(fitted_log_all)

    reasons: list[str] = []
    if slope >= 0 or not np.isfinite(slope):
        return QEnvelopeDiagnostic(
            tau=None,
            Q=None,
            valid=False,
            status="invalid",
            reasons=["envelope_nondecaying_log_slope"],
            n_windows=n_windows,
            n_windows_used=int(np.count_nonzero(used)),
            t_mid=t_mid,
            amplitude=amplitude,
            used=used,
            fitted_amplitude=fitted_amplitude,
            candidate_q=None,
            candidate_tau=None,
            candidate_amplitude=np.full_like(t_mid, np.nan),
            log_amplitude_slope=float(slope) if np.isfinite(slope) else None,
            log_amplitude_intercept=float(intercept) if np.isfinite(intercept) else None,
            log_amplitude_rmse=log_rmse,
            slope_stderr=stderr,
            candidate_log_rmse=None,
            candidate_slope_mismatch=None,
            candidate_agrees=None,
            method=method,
        )

    tau = float(-1.0 / slope)
    Q = float(np.pi * f_hz * tau)
    candidate_q: float | None = None
    candidate_tau: float | None = None
    candidate_amplitude = np.full_like(t_mid, np.nan)
    candidate_log_rmse: float | None = None
    candidate_slope_mismatch: float | None = None
    candidate_agrees: bool | None = None
    status = "valid"

    if q is not None and np.isfinite(q) and q > 0:
        candidate_q = float(q)
        candidate_tau = float(candidate_q / (np.pi * f_hz))
        candidate_slope = -1.0 / candidate_tau
        anchor_log_amp = float(np.log(amplitude[used][0]))
        candidate_log = anchor_log_amp + candidate_slope * (t_mid - first_used_t)
        candidate_amplitude = np.exp(candidate_log)
        candidate_residual = log_amp - candidate_log[used]
        candidate_log_rmse = float(np.sqrt(np.mean(candidate_residual**2)))
        candidate_slope_mismatch = float(
            abs(candidate_slope - slope) / max(abs(slope), np.finfo(np.float64).eps)
        )
        candidate_agrees = bool(
            candidate_log_rmse <= candidate_log_rmse_warning
            and candidate_slope_mismatch <= candidate_slope_mismatch_warning
        )
        if not candidate_agrees:
            status = "warning"
            reasons.append("candidate_q_envelope_mismatch")

    return QEnvelopeDiagnostic(
        tau=tau,
        Q=Q,
        valid=True,
        status=status,
        reasons=reasons,
        n_windows=n_windows,
        n_windows_used=int(np.count_nonzero(used)),
        t_mid=t_mid,
        amplitude=amplitude,
        used=used,
        fitted_amplitude=fitted_amplitude,
        candidate_q=candidate_q,
        candidate_tau=candidate_tau,
        candidate_amplitude=candidate_amplitude,
        log_amplitude_slope=slope,
        log_amplitude_intercept=intercept,
        log_amplitude_rmse=log_rmse,
        slope_stderr=stderr,
        candidate_log_rmse=candidate_log_rmse,
        candidate_slope_mismatch=candidate_slope_mismatch,
        candidate_agrees=candidate_agrees,
        method=method,
    )
