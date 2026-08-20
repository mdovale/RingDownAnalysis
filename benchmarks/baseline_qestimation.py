"""
Frozen pre-optimization reference for the Q-estimation stack.

This module preserves the hot-path method bodies of
``ringdownanalysis.demod.SegmentedDemodEstimator`` and
``ringdownanalysis.q_profile.ProfileQEstimator`` exactly as they were on
2026-08-19, before the performance work of 2026-08-20. It exists so
``bench_qestimation.py`` can measure the speedup and verify numerical agreement
in a single process, without checking out old revisions.

Only the methods that were actually optimized are overridden; everything else
is inherited, so the comparison isolates the optimization rather than drifting
with unrelated changes.

DO NOT EDIT to "keep up" with the library. If an estimator's numerical recipe
is deliberately changed, this baseline stops being the right comparison point
and the benchmark's agreement check is supposed to fail and say so.
"""

from __future__ import annotations

import numpy as np

from ringdownanalysis.demod import SegmentedDemodEstimator
from ringdownanalysis.q_profile import ProfileQEstimator, _ProjectionFit


class BaselineSegmentedDemodEstimator(SegmentedDemodEstimator):
    """Original (unoptimized) implementations of the demod hot paths."""

    # -- Original: per-frequency np.linalg.lstsq scan over a fresh design
    # matrix, seeded by a full 4n-point zero-padded FFT.
    # ``uniform`` is accepted and ignored: the inherited segment loop supplies
    # it, and the original code had no fast path to select.
    def _demodulate_segment(
        self,
        ts: np.ndarray,
        ys: np.ndarray,
        fs: float,
        f_band: tuple[float, float],
        *,
        uniform: bool | None = None,
    ) -> tuple[float, float, float] | None:
        ys = ys - np.mean(ys)
        ys = ys - np.polyval(np.polyfit(ts, ys, 1), ts)
        n_fft = 4 * len(ys)
        spec = np.abs(np.fft.rfft(ys * np.hanning(len(ys)), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        band = (freqs > f_band[0]) & (freqs < f_band[1])
        if not np.any(band):
            return None
        f_seed = float(freqs[band][np.argmax(spec[band])])

        def scan(f_center: float, half_span: float) -> tuple[float, float, float]:
            best_rss = np.inf
            best_f = f_center
            best_amp = 0.0
            for f in f_center + np.linspace(-half_span, half_span, self.f_scan_points):
                if f <= 0:
                    continue
                design = np.column_stack(
                    [
                        np.cos(2.0 * np.pi * f * ts),
                        np.sin(2.0 * np.pi * f * ts),
                        np.ones_like(ts),
                    ]
                )
                coef, _, _, _ = np.linalg.lstsq(design, ys, rcond=None)
                rss = float(np.sum((ys - design @ coef) ** 2))
                if rss < best_rss:
                    best_rss = rss
                    best_f = float(f)
                    best_amp = float(np.hypot(coef[0], coef[1]))
            return best_rss, best_f, best_amp

        seg_span = float(ts[-1] - ts[0])
        half_span_coarse = 0.25 / seg_span
        _, f_coarse, _ = scan(f_seed, half_span_coarse)
        rss, f_hat, a_hat = scan(f_coarse, half_span_coarse / 10.0)
        dof = len(ys) - 3
        sigma = float(np.sqrt(rss / dof)) if dof > 0 else np.nan
        return f_hat, a_hat, sigma

    # -- Original: pair indices rebuilt on every call.
    @staticmethod
    def _fit_line_robust(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        n = len(x)
        i, j = np.triu_indices(n, k=1)
        slopes = (y[j] - y[i]) / (x[j] - x[i])
        slope = float(np.median(slopes))
        intercept = float(np.median(y - slope * x))
        return slope, intercept

    # -- Original: Python loop over bootstrap resamples, full refit per resample.
    def _bootstrap_q_ci(
        self,
        t_fit: np.ndarray,
        log_amp: np.ndarray,
        slope: float,
        intercept: float,
        f_mean: float,
    ) -> tuple[float, float] | None:
        n = len(t_fit)
        if n < 4 or self.n_bootstrap < 10:
            return None
        residual = log_amp - (slope * t_fit + intercept)
        block = max(2, n // 8)
        n_blocks = int(np.ceil(n / block))
        rng = np.random.default_rng(self.bootstrap_seed)
        q_samples = []
        for _ in range(self.n_bootstrap):
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            resampled = np.concatenate([residual[s : s + block] for s in starts])[:n]
            log_amp_star = slope * t_fit + intercept + resampled
            slope_star = self._fit_line_robust(t_fit, log_amp_star)[0]
            if slope_star < 0:
                q_samples.append(np.pi * f_mean * (-1.0 / slope_star))
        if len(q_samples) < max(10, self.n_bootstrap // 2):
            return None
        lower, upper = np.percentile(q_samples, [2.5, 97.5])
        return float(lower), float(upper)


class _BaselineProfileScan:
    """Original profile scan: a full design matrix and an SVD solve per tau."""

    def __init__(self, t: np.ndarray, data: np.ndarray, f_hat: float):
        self.t = t
        self.data = data
        self.f_hat = float(f_hat)

    def fit(self, taus: np.ndarray) -> list[_ProjectionFit]:
        return [self._fit_one(float(tau)) for tau in np.atleast_1d(taus)]

    def _fit_one(self, tau: float) -> _ProjectionFit:
        exp_term = np.exp(-self.t / tau)
        omega_t = 2.0 * np.pi * self.f_hat * self.t
        design_matrix = np.column_stack(
            [
                exp_term * np.cos(omega_t),
                exp_term * np.sin(omega_t),
                np.ones_like(self.t),
            ]
        )
        coeffs, _, rank, _ = np.linalg.lstsq(design_matrix, self.data, rcond=None)
        if rank < design_matrix.shape[1]:
            raise np.linalg.LinAlgError("Profile-Q design matrix is rank-deficient")

        residuals = self.data - design_matrix @ coeffs
        rss = float(np.sum(residuals**2))
        dof = int(len(self.data) - design_matrix.shape[1])
        if dof <= 0:
            raise ValueError("Profile-Q degrees of freedom must be positive")

        a_cos, b_sin, _ = coeffs
        return _ProjectionFit(
            tau=float(tau),
            rss=rss,
            sigma=float(np.sqrt(max(rss, 0.0) / dof)),
            dof=dof,
            amplitude=float(np.hypot(a_cos, b_sin)),
            rank=int(rank),
        )


class BaselineProfileQEstimator(ProfileQEstimator):
    """Original (unoptimized) fixed-tau fit: full design rebuilt per grid point."""

    @staticmethod
    def _make_scan(t: np.ndarray, data: np.ndarray, f_hat: float):
        return _BaselineProfileScan(t, data, f_hat)
