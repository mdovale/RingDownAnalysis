"""
Segmented-demodulation Q estimation.

This module implements the incoherent, drift-immune Q estimator recommended by
the 2026-08-18 Q-estimation investigation
(docs/investigations/20260818_q_estimation_failure_investigation.md). Instead
of fitting a single globally phase-coherent model — which breaks catastrophically
when the resonance frequency drifts during the decay — the record is split into
short segments and each segment is demodulated independently:

1. Per segment: remove mean and linear baseline, locate the tone with a
   zero-padded FFT, refine frequency with a two-stage linear least-squares scan
   of ``a*cos + b*sin + c`` over f. This yields amplitude A_i, frequency f_i and
   residual RMS sigma_i as functions of time.
2. Estimate the ambient-driven plateau (noise-floor) level from the late-record
   amplitude distribution and detect whether the record actually reaches it.
3. Select the contiguous early decay region with A > k * A_floor (default k=3)
   and fit log(sqrt(A^2 - A_floor^2)) versus time (floor-corrected log-linear
   fit). The fit is unweighted: segment scatter is dominated by systematic
   decay-rate curvature, not by per-segment statistical noise, so uncertainties
   come from a residual block bootstrap over segments rather than from a
   white-noise model.

The estimator additionally reports amplitude-resolved local Q values, the
frequency-vs-amplitude pull coefficient, the frequency drift over the decay and
the dimensionless ``coherence_ratio`` = |drift| * tau used to gate coherent
estimators (they are only trustworthy for coherence_ratio << 0.01).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import ZoomFFT

from ._gridfit import (
    RSS_CANCELLATION_FRACTION,
    BlockIndex,
    is_uniformly_sampled,
    tone_sums,
)
from .estimators import _estimate_initial_parameters_from_dft

logger = logging.getLogger(__name__)

#: Coherent estimators (NLS/DFT/profile) tolerate a frequency error of about
#: delta_f * tau < 0.01 before their Q bias exceeds ~2 % (investigation §5.3).
COHERENCE_RATIO_THRESHOLD = 0.01

#: Working-set cap (array elements) for the batched block bootstrap, which
#: builds a (resamples x segment pairs) matrix of pairwise slopes.
_BOOTSTRAP_ELEMENT_BUDGET = 1_000_000


@lru_cache(maxsize=8)
def _hann_window(n: int) -> np.ndarray:
    """
    Cached read-only Hann window.

    Every segment of a record has the same length up to a sample, so a tiny
    cache removes the per-segment window construction entirely.
    """
    window = np.hanning(n)
    window.setflags(write=False)
    return window


@lru_cache(maxsize=8)
def _band_transform(n: int, m: int, f_start: float, f_stop: float, fs: float) -> ZoomFFT:
    """
    Cached chirp-z plan for m equally spaced DFT bins of a length-n segment.

    Building the plan costs more than applying it, and every segment of a
    record shares a length, so the plan is cached rather than rebuilt.
    """
    return ZoomFFT(n, [f_start, f_stop], m=m, fs=fs, endpoint=True)


@lru_cache(maxsize=8)
def _pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached upper-triangle index pairs (i < j) for Theil-Sen slopes."""
    i, j = np.triu_indices(n, k=1)
    i.setflags(write=False)
    j.setflags(write=False)
    return i, j


def _detrend_linear(ts: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Remove the least-squares mean and linear trend.

    Closed form rather than polyfit/polyval, which solves the same 2-parameter
    problem through a general SVD least-squares path.
    """
    t_mean = float(np.mean(ts))
    y_mean = float(np.mean(ys))
    centered_t = ts - t_mean
    stt = float(centered_t @ centered_t)
    slope = float(centered_t @ ys) / stt if stt > 0 else 0.0
    return ys - (y_mean + slope * centered_t)


class _ToneFitter:
    """
    Least-squares tone fits over a batch of trial frequencies.

    Fits ``y ~ a*cos(2*pi*f*t) + b*sin(2*pi*f*t) + c`` for many f, sharing all
    per-segment precomputation. The normal equations of this 3-parameter model
    are built from six sufficient statistics per frequency:

    - the data projections ``sum y*cos``, ``sum y*sin``;
    - the pure sums ``sum cos``, ``sum sin`` and their double-angle
      counterparts, which give the Gram matrix via
      ``sum cos^2 = (n + sum cos 2t)/2`` and friends.

    On a uniform sampling grid the pure sums are closed-form (see
    :func:`~ringdownanalysis._gridfit.tone_sums`) and the data projections
    factor through the two-index decomposition of
    :class:`~ringdownanalysis._gridfit.BlockIndex`, which evaluates the whole
    frequency batch with two real matrix products and only O(sqrt(n))
    transcendental calls per frequency. That is what makes the scan cheap
    enough to stop dominating the estimator.

    Both shortcuts need the sample times to be exactly ``k/fs``, so the uniform
    path evaluates the model on that nominal grid rather than on ``ts``; the
    caller's uniformity test (``UNIFORM_TOLERANCE_SAMPLES``) is what bounds the
    resulting error. Non-uniform sample times fall back to direct evaluation at
    the given ``ts``, which is still one pass per frequency instead of a general
    least-squares solve.
    """

    def __init__(self, ts: np.ndarray, ys: np.ndarray, fs: float, *, uniform: bool):
        self.ts = ts
        self.ys = ys
        self.fs = float(fs)
        self.n = len(ys)
        self.uniform = bool(uniform)
        self.yy = float(ys @ ys)
        self.sy = float(np.sum(ys))
        self._rss_floor = RSS_CANCELLATION_FRACTION * self.yy
        self._index = BlockIndex(ys) if self.uniform else None

    # -- sufficient statistics -----------------------------------------

    def _statistics(self, freqs: np.ndarray) -> tuple[np.ndarray, ...]:
        if self.uniform:
            angle = (2.0 * np.pi / self.fs) * freqs
            sum_cos, sum_sin = tone_sums(self.n, angle)
            sum_cos2, sum_sin2 = tone_sums(self.n, 2.0 * angle)
            proj_cos, proj_sin = self._project_uniform(angle)
        else:
            proj_cos, proj_sin, sum_cos, sum_sin, sum_cos2, sum_sin2 = self._statistics_direct(
                freqs
            )
        return proj_cos, proj_sin, sum_cos, sum_sin, sum_cos2, sum_sin2

    def _project_uniform(self, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Batched ``sum_k y_k exp(i*a*k)`` by two-factor index decomposition.

        The exponent is purely imaginary here, so this evaluates the block
        factors with cosine and sine directly rather than through the complex
        exponential of :meth:`BlockIndex.project`.
        """
        assert self._index is not None
        outer = np.outer(angle, self._index.outer_index)
        inner = np.outer(angle, self._index.inner_index)
        partial_re = np.cos(outer) @ self._index.blocked
        partial_im = np.sin(outer) @ self._index.blocked
        inner_re = np.cos(inner)
        inner_im = np.sin(inner)
        proj_cos = np.sum(partial_re * inner_re - partial_im * inner_im, axis=1)
        proj_sin = np.sum(partial_re * inner_im + partial_im * inner_re, axis=1)
        return proj_cos, proj_sin

    def _statistics_direct(self, freqs: np.ndarray) -> tuple[np.ndarray, ...]:
        """Non-uniform fallback: one pass over the segment per trial frequency."""
        m = len(freqs)
        out = np.empty((6, m), dtype=np.float64)
        for idx, freq in enumerate(freqs):
            theta = (2.0 * np.pi * float(freq)) * self.ts
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            out[0, idx] = self.ys @ cos_t
            out[1, idx] = self.ys @ sin_t
            out[2, idx] = np.sum(cos_t)
            out[3, idx] = np.sum(sin_t)
            # Double-angle identities avoid a second pair of trig evaluations.
            out[4, idx] = (cos_t - sin_t) @ (cos_t + sin_t)
            out[5, idx] = 2.0 * (cos_t @ sin_t)
        return tuple(out)

    # -- batched solve -------------------------------------------------

    def best(self, freqs: np.ndarray) -> tuple[float, float, float]:
        """
        Return ``(rss, f, amplitude)`` at the RSS-minimizing trial frequency.

        Ties go to the earliest frequency in ``freqs``, matching a sequential
        scan that keeps a strict improvement.
        """
        freqs = np.asarray(freqs, dtype=float)
        fallback_f = float(freqs[len(freqs) // 2])
        positive = freqs > 0.0
        if not np.any(positive):
            return np.inf, fallback_f, 0.0
        trial = freqs[positive]

        proj_cos, proj_sin, sum_cos, sum_sin, sum_cos2, sum_sin2 = self._statistics(trial)
        n = float(self.n)
        gram = np.empty((len(trial), 3, 3), dtype=np.float64)
        gram[:, 0, 0] = 0.5 * (n + sum_cos2)
        gram[:, 1, 1] = 0.5 * (n - sum_cos2)
        gram[:, 0, 1] = gram[:, 1, 0] = 0.5 * sum_sin2
        gram[:, 0, 2] = gram[:, 2, 0] = sum_cos
        gram[:, 1, 2] = gram[:, 2, 1] = sum_sin
        gram[:, 2, 2] = n
        rhs = np.stack([proj_cos, proj_sin, np.full(len(trial), self.sy)], axis=1)

        try:
            coef = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # A degenerate trial frequency (e.g. a tone aliased to DC) makes
            # one system singular; the pseudo-inverse keeps the rest usable.
            coef = np.einsum("nij,nj->ni", np.linalg.pinv(gram), rhs)

        rss = self.yy - np.einsum("ni,ni->n", rhs, coef)
        rss = np.where(np.isfinite(rss), rss, np.inf)
        k = int(np.argmin(rss))
        if not np.isfinite(rss[k]):
            return np.inf, fallback_f, 0.0

        best_f = float(trial[k])
        best_amp = float(np.hypot(coef[k, 0], coef[k, 1]))
        best_rss = float(rss[k])
        if best_rss < self._rss_floor:
            best_rss = self._explicit_rss(best_f, coef[k])
        return best_rss, best_f, best_amp

    def _explicit_rss(self, freq: float, coef: np.ndarray) -> float:
        """Residual sum of squares formed directly, for near-perfect fits."""
        model_time = np.arange(self.n) / self.fs if self.uniform else self.ts
        theta = (2.0 * np.pi * freq) * model_time
        resid = self.ys - (coef[0] * np.cos(theta) + coef[1] * np.sin(theta) + coef[2])
        return float(resid @ resid)


@dataclass(frozen=True)
class AmplitudeBandQ:
    """Local decay time and Q measured over one amplitude band of the decay."""

    amplitude_low: float
    amplitude_high: float
    amplitude_mid: float
    n_segments: int
    tau: float
    Q: float


@dataclass(frozen=True)
class SegmentedDemodResult:
    """Result of segmented-demodulation Q estimation."""

    Q: float | None
    tau: float | None
    f_mean: float | None
    Q_ci95: tuple[float, float] | None
    valid: bool
    status: str
    reasons: list[str]
    # Per-segment diagnostics
    t_mid: np.ndarray
    f_seg: np.ndarray
    amplitude: np.ndarray
    sigma_seg: np.ndarray
    amplitude_corrected: np.ndarray
    decay_mask: np.ndarray
    # Plateau and drift diagnostics
    plateau_amplitude: float | None
    plateau_detected: bool
    drift_hz: float | None
    drift_hz_stderr: float | None
    coherence_ratio: float | None
    #: 2-sigma lower bound on coherence_ratio; use this to gate coherent
    #: estimators so that drift-measurement noise cannot fire the gate on a
    #: genuinely coherent record.
    coherence_ratio_lower: float | None
    # Amplitude-resolved outputs
    q_vs_amplitude: list[AmplitudeBandQ]
    f_pull_per_efold: float | None
    # Fit internals
    log_slope: float | None
    log_intercept: float | None
    slope_stderr: float | None
    seg_duration: float
    n_segments: int
    method: str


def _empty_result(status: str, reasons: list[str], *, seg_duration: float = 0.0, method: str):
    empty = np.array([], dtype=float)
    return SegmentedDemodResult(
        Q=None,
        tau=None,
        f_mean=None,
        Q_ci95=None,
        valid=False,
        status=status,
        reasons=reasons,
        t_mid=empty,
        f_seg=empty,
        amplitude=empty,
        sigma_seg=empty,
        amplitude_corrected=empty,
        decay_mask=np.array([], dtype=bool),
        plateau_amplitude=None,
        plateau_detected=False,
        drift_hz=None,
        drift_hz_stderr=None,
        coherence_ratio=None,
        coherence_ratio_lower=None,
        q_vs_amplitude=[],
        f_pull_per_efold=None,
        log_slope=None,
        log_intercept=None,
        slope_stderr=None,
        seg_duration=seg_duration,
        n_segments=0,
        method=method,
    )


class SegmentedDemodEstimator:
    """
    Estimate Q by incoherent segmented demodulation of the decay envelope.

    Immune to the frequency drift and phase decoherence that bias globally
    phase-coherent estimators (NLS/DFT/profile), and explicitly models the
    ambient-driven plateau that biases naive envelope-slope fits.
    """

    def __init__(
        self,
        *,
        seg_duration: float | None = None,
        target_segments: int = 96,
        min_segments: int = 8,
        min_decay_segments: int = 6,
        min_cycles_per_segment: float = 20.0,
        min_samples_per_segment: int = 100,
        band_halfwidth_rel: float = 0.05,
        floor_threshold: float = 3.0,
        min_amplitude_fraction: float = 0.05,
        plateau_flatness_ratio: float = 1.5,
        n_bootstrap: int = 200,
        bootstrap_seed: int = 20260818,
        f_scan_points: int = 21,
    ):
        """
        Initialize the estimator.

        Parameters:
        -----------
        seg_duration : float, optional
            Segment duration (s). If None, the record is split into about
            ``target_segments`` segments, subject to the per-segment minimum
            cycle and sample counts.
        target_segments : int
            Number of segments to aim for when seg_duration is None.
        min_segments : int
            Minimum number of demodulated segments for any estimate.
        min_decay_segments : int
            Minimum number of segments in the decay region for a finite Q.
        min_cycles_per_segment : float
            Lower bound on oscillation cycles per segment.
        min_samples_per_segment : int
            Lower bound on samples per segment.
        band_halfwidth_rel : float
            Relative half-width of the FFT search band around f_init.
        floor_threshold : float
            Decay region keeps segments with amplitude > floor_threshold *
            plateau amplitude (k in the investigation report).
        min_amplitude_fraction : float
            Amplitude cutoff (fraction of the maximum segment amplitude) used
            when no plateau is detected.
        plateau_flatness_ratio : float
            Quartile-median amplitude ratio below which the late record is
            considered flat (and above which the early record is considered
            decaying). Median ratios are used because the driven plateau
            wanders slowly and defeats slope-based flatness tests.
        n_bootstrap : int
            Number of residual block-bootstrap resamples for the Q CI.
        bootstrap_seed : int
            Seed for the deterministic bootstrap generator.
        f_scan_points : int
            Points per stage of the two-stage frequency refinement scan.
        """
        if min_segments < 4:
            raise ValueError(f"min_segments must be at least 4, got {min_segments}")
        if min_decay_segments < 3:
            raise ValueError(f"min_decay_segments must be at least 3, got {min_decay_segments}")
        if floor_threshold <= 1.0:
            raise ValueError(f"floor_threshold must be greater than 1, got {floor_threshold}")
        if plateau_flatness_ratio <= 1.0:
            raise ValueError(
                f"plateau_flatness_ratio must be greater than 1, got {plateau_flatness_ratio}"
            )
        if f_scan_points < 5:
            raise ValueError(f"f_scan_points must be at least 5, got {f_scan_points}")
        if seg_duration is not None and (not np.isfinite(seg_duration) or seg_duration <= 0):
            raise ValueError(f"seg_duration must be positive and finite, got {seg_duration}")

        self.seg_duration = seg_duration
        self.target_segments = int(target_segments)
        self.min_segments = int(min_segments)
        self.min_decay_segments = int(min_decay_segments)
        self.min_cycles_per_segment = float(min_cycles_per_segment)
        self.min_samples_per_segment = int(min_samples_per_segment)
        self.band_halfwidth_rel = float(band_halfwidth_rel)
        self.floor_threshold = float(floor_threshold)
        self.min_amplitude_fraction = float(min_amplitude_fraction)
        self.plateau_flatness_ratio = float(plateau_flatness_ratio)
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_seed = int(bootstrap_seed)
        self.f_scan_points = int(f_scan_points)

    # ------------------------------------------------------------------
    # Per-segment demodulation
    # ------------------------------------------------------------------

    def _resolve_seg_duration(self, duration: float, f_hz: float, fs: float) -> float:
        min_seg = max(
            self.min_cycles_per_segment / f_hz,
            self.min_samples_per_segment / fs,
        )
        if self.seg_duration is not None:
            return max(float(self.seg_duration), min_seg)
        return max(duration / float(self.target_segments), min_seg)

    def _seed_frequency(
        self,
        ys: np.ndarray,
        fs: float,
        f_band: tuple[float, float],
    ) -> float | None:
        """
        Locate the tone on the 4x-zero-padded Hann-windowed spectrum.

        Zero-padding to 4x the segment length puts the spectral peak within
        0.125/T of the true tone, comfortably inside the +-0.25/T span of the
        refinement scan. Only the bins inside the search band are needed, and a
        chirp-z transform evaluates exactly those bins at a cost set by the
        segment length instead of the padded length -- the padded transform is
        both mostly discarded and, at 4n, prone to landing on an awkward
        factorization (4n = 2^2*37*151 for the EDU segment length) that forces
        a slow general-length FFT.
        """
        n = len(ys)
        n_fft = 4 * n
        bin_width = fs / n_fft
        n_bins = n_fft // 2 + 1
        # Strict band edges, matching a boolean mask on the bin frequencies.
        k_lo = min(max(int(np.floor(f_band[0] / bin_width)) + 1, 0), n_bins)
        k_hi = min(int(np.ceil(f_band[1] / bin_width)), n_bins)
        if k_hi <= k_lo:
            return None
        if k_hi - k_lo == 1:
            return float(k_lo * bin_width)
        transform = _band_transform(n, k_hi - k_lo, k_lo * bin_width, (k_hi - 1) * bin_width, fs)
        band = transform(ys * _hann_window(n))
        peak = int(np.argmax(band.real**2 + band.imag**2))
        return float((k_lo + peak) * bin_width)

    def _demodulate_segment(
        self,
        ts: np.ndarray,
        ys: np.ndarray,
        fs: float,
        f_band: tuple[float, float],
        *,
        uniform: bool | None = None,
    ) -> tuple[float, float, float] | None:
        """
        Return (f_hat, amplitude, sigma_resid) for one segment.

        ``uniform`` states whether the segment lies on the nominal 1/fs grid;
        None means detect it here. Callers that already know it for the whole
        record should pass it to avoid a redundant per-segment check.
        """
        ys = _detrend_linear(ts, ys)
        f_seed = self._seed_frequency(ys, fs, f_band)
        if f_seed is None:
            return None

        if uniform is None:
            uniform = is_uniformly_sampled(ts, fs)
        fitter = _ToneFitter(ts, ys, fs, uniform=uniform)
        offsets = np.linspace(-1.0, 1.0, self.f_scan_points)

        seg_span = float(ts[-1] - ts[0])
        half_span_coarse = 0.25 / seg_span
        _, f_coarse, _ = fitter.best(f_seed + half_span_coarse * offsets)
        rss, f_hat, a_hat = fitter.best(f_coarse + (0.1 * half_span_coarse) * offsets)
        dof = len(ys) - 3
        sigma = float(np.sqrt(max(rss, 0.0) / dof)) if dof > 0 else np.nan
        return f_hat, a_hat, sigma

    def demodulate(
        self,
        t: np.ndarray,
        data: np.ndarray,
        fs: float,
        *,
        f_init: float,
        seg_duration: float,
        uniform: bool | None = None,
    ) -> np.ndarray:
        """
        Demodulate the record in fixed-duration segments.

        Returns an array with rows (t_mid, f, amplitude, sigma_resid), one per
        successfully demodulated segment; t_mid is relative to the first sample.

        ``uniform`` states whether the record lies on the nominal 1/fs sampling
        grid, which enables a much faster exact tone fit. None means detect it
        once here.
        """
        t_norm = t - t[0]
        duration = float(t_norm[-1])
        f_band = (
            f_init * (1.0 - self.band_halfwidth_rel),
            f_init * (1.0 + self.band_halfwidth_rel),
        )
        if uniform is None:
            uniform = is_uniformly_sampled(t_norm, fs)
        rows = []
        n_seg = int(duration / seg_duration)
        for i in range(n_seg):
            start = i * seg_duration
            lo = int(np.searchsorted(t_norm, start))
            hi = int(np.searchsorted(t_norm, start + seg_duration))
            if hi - lo < self.min_samples_per_segment:
                continue
            ts = t_norm[lo:hi] - t_norm[lo]
            demod = self._demodulate_segment(ts, data[lo:hi], fs, f_band, uniform=uniform)
            if demod is None:
                continue
            f_hat, a_hat, sigma = demod
            rows.append((start + 0.5 * seg_duration, f_hat, a_hat, sigma))
        return np.array(rows, dtype=float).reshape(-1, 4)

    # ------------------------------------------------------------------
    # Plateau detection and decay-region selection
    # ------------------------------------------------------------------

    def _detect_plateau(
        self, t_mid: np.ndarray, amplitude: np.ndarray
    ) -> tuple[float | None, bool, bool, bool]:
        """
        Classify the late-record amplitude behavior.

        Returns (floor, plateau_detected, plateau_dominated, dynamic_range_ok).
        Quartile medians of the amplitude are used instead of local slopes
        because the driven plateau wanders slowly (correlation time ~ tau) and
        defeats slope-based flatness tests.

        - plateau_detected: the record decays well above the floor and is flat
          late — the late median is a usable floor for the corrected fit.
        - plateau_dominated: the record is flat late but never rises
          meaningfully above the floor — there is no measurable free decay in
          this window.
        - dynamic_range_ok: the record spans enough amplitude above the late
          median for a floor-corrected decay fit to make sense. The range is
          measured from a robust peak (median of the top-decile amplitudes),
          not from the first-quartile median: a long plateau tail after a
          perfectly good decay must not dilute the usable dynamic range
          (e.g. a 17 h record whose decay is only the first 4 h).
        """
        quartile = np.ceil(4.0 * (t_mid - t_mid[0]) / max(t_mid[-1] - t_mid[0], 1e-30))
        quartile = np.clip(quartile, 1, 4)
        medians = []
        for q in (1, 2, 3, 4):
            values = amplitude[quartile == q]
            if len(values) < 2:
                return None, False, False, False
            medians.append(float(np.median(values)))
        q1, q2, q3, q4 = medians
        if min(medians) <= 0 or not all(np.isfinite(medians)):
            return None, False, False, False

        floor = q4
        flat_late = (q3 / q4) < self.plateau_flatness_ratio
        early_decaying = (q1 / q2) > self.plateau_flatness_ratio
        n_peak = max(3, len(amplitude) // 10)
        a_peak = float(np.median(np.sort(amplitude)[-n_peak:]))
        dynamic_range_ok = a_peak / floor >= self.floor_threshold**2

        if flat_late and dynamic_range_ok:
            return floor, True, False, True
        if flat_late and early_decaying and not dynamic_range_ok:
            return floor, False, True, False
        return floor, False, False, dynamic_range_ok

    @staticmethod
    def _fit_line_robust(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Theil-Sen line fit (median of pairwise slopes).

        The decay-region log amplitudes carry heavy-tailed, block-correlated
        contamination near the cutoff (plateau revivals inflate a contiguous
        run of segments); a least-squares fit is pulled by such blocks while
        the median-of-slopes estimator tolerates them up to ~29 % of points.
        """
        i, j = _pair_indices(len(x))
        slopes = (y[j] - y[i]) / (x[j] - x[i])
        slope = float(np.median(slopes))
        intercept = float(np.median(y - slope * x))
        return slope, intercept

    def _decay_mask(
        self,
        t_mid: np.ndarray,
        amplitude: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        """Contiguous early region above the amplitude cutoff."""
        above = amplitude > cutoff
        below_idx = np.flatnonzero(~above)
        t_end = t_mid[below_idx[0]] if len(below_idx) else t_mid[-1] + 1.0
        return above & (t_mid < t_end)

    def _refine_decay_mask(
        self,
        t_mid: np.ndarray,
        log_amp_corrected: np.ndarray,
        decay_mask: np.ndarray,
        cutoff: float,
    ) -> tuple[np.ndarray, float, float]:
        """
        Iteratively trim the decay region to the fitted cutoff crossing.

        The measured amplitude can stay above the cutoff long after the true
        decay has crossed it when the driven plateau wanders high (revivals);
        trimming at the *fitted* crossing keeps such plateau-dominated tail
        segments out of the fit.
        """
        mask = decay_mask.copy()
        slope, intercept = self._fit_line_robust(t_mid[mask], log_amp_corrected[mask])
        log_cutoff = float(np.log(max(cutoff, np.finfo(np.float64).tiny)))
        for _ in range(3):
            if slope >= 0:
                break
            t_cross = (log_cutoff - intercept) / slope
            new_mask = mask & (t_mid <= t_cross)
            if int(np.count_nonzero(new_mask)) < self.min_decay_segments:
                break
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
            slope, intercept = self._fit_line_robust(t_mid[mask], log_amp_corrected[mask])
        return mask, float(slope), float(intercept)

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_q_ci(
        self,
        t_fit: np.ndarray,
        log_amp: np.ndarray,
        slope: float,
        intercept: float,
        f_mean: float,
    ) -> tuple[float, float] | None:
        """
        Residual block-bootstrap 95 % CI for Q.

        All resamples are drawn and refitted as one batched array operation.
        Drawing the whole (n_bootstrap, n_blocks) index matrix in a single call
        consumes the generator in the same order as one draw per resample, so
        the CI is bit-identical to the sequential formulation.
        """
        n = len(t_fit)
        if n < 4 or self.n_bootstrap < 10:
            return None
        residual = log_amp - (slope * t_fit + intercept)
        fitted = slope * t_fit + intercept
        block = max(2, n // 8)
        n_blocks = int(np.ceil(n / block))
        rng = np.random.default_rng(self.bootstrap_seed)
        starts = rng.integers(0, n - block + 1, size=(self.n_bootstrap, n_blocks))
        offsets = np.arange(block)

        i, j = _pair_indices(n)
        delta_t = t_fit[j] - t_fit[i]
        chunk = max(1, _BOOTSTRAP_ELEMENT_BUDGET // max(len(i), 1))
        slopes = np.empty(self.n_bootstrap, dtype=np.float64)
        for lo in range(0, self.n_bootstrap, chunk):
            batch_starts = starts[lo : lo + chunk]
            index = (batch_starts[:, :, None] + offsets).reshape(len(batch_starts), -1)[:, :n]
            resampled = fitted + residual[index]
            pair_slopes = resampled[:, j]
            pair_slopes -= resampled[:, i]
            pair_slopes /= delta_t
            slopes[lo : lo + len(batch_starts)] = np.median(
                pair_slopes, axis=1, overwrite_input=True
            )

        decaying = slopes < 0
        if int(np.count_nonzero(decaying)) < max(10, self.n_bootstrap // 2):
            return None
        q_samples = np.pi * f_mean * (-1.0 / slopes[decaying])
        lower, upper = np.percentile(q_samples, [2.5, 97.5])
        return float(lower), float(upper)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def estimate(
        self,
        t: np.ndarray,
        data: np.ndarray,
        fs: float,
        *,
        f_init: float | None = None,
        amplitude_bands: list[tuple[float, float]] | None = None,
    ) -> SegmentedDemodResult:
        """
        Estimate Q from a ring-down record by segmented demodulation.

        Parameters:
        -----------
        t : np.ndarray
            Time array (s), strictly increasing.
        data : np.ndarray
            Signal array.
        fs : float
            Sampling frequency (Hz).
        f_init : float, optional
            Approximate tone frequency (Hz). Estimated from a DFT of the full
            record when omitted.
        amplitude_bands : list of (low, high), optional
            Explicit amplitude bands (signal units) for the amplitude-resolved
            local Q table. Auto-generated octave bands when omitted.

        Returns:
        --------
        SegmentedDemodResult
            Q/tau with bootstrap CI, per-segment diagnostics, plateau and drift
            diagnostics, amplitude-resolved Q, and the coherence_ratio gate.
        """
        method = "segmented_demodulation_floor_corrected_log_linear"
        t_arr = np.asarray(t, dtype=np.float64)
        data_arr = np.asarray(data, dtype=np.float64)
        if t_arr.ndim != 1 or data_arr.ndim != 1:
            raise ValueError("t and data must be one-dimensional arrays")
        if len(t_arr) != len(data_arr):
            raise ValueError(
                f"t and data must have same length, got {len(t_arr)} and {len(data_arr)}"
            )
        if len(t_arr) < 3:
            return _empty_result("invalid", ["demod_insufficient_samples"], method=method)
        if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(data_arr)):
            raise ValueError("t and data must contain only finite values")
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"Sampling frequency fs must be positive and finite, got {fs}")
        t_norm = t_arr - t_arr[0]
        # Uniform sampling is checked first because it both enables the fast
        # tone fit and implies strict monotonicity (every sample within 1e-4 of
        # a sampling interval of an increasing grid), which saves a pass over
        # the record on the common path.
        uniform_sampling = is_uniformly_sampled(t_norm, fs)
        if not uniform_sampling and np.any(np.diff(t_norm) <= 0):
            raise ValueError("t must be strictly increasing")

        if f_init is None or not np.isfinite(f_init) or f_init <= 0:
            f_hz = float(_estimate_initial_parameters_from_dft(data_arr, fs)[0])
        else:
            f_hz = float(f_init)
        if not np.isfinite(f_hz) or f_hz <= 0 or f_hz >= 0.5 * fs:
            return _empty_result("invalid", ["demod_frequency_unresolved"], method=method)

        duration = float(t_norm[-1])
        seg_duration = self._resolve_seg_duration(duration, f_hz, fs)
        if duration / seg_duration < self.min_segments:
            return _empty_result(
                "invalid",
                ["demod_record_too_short"],
                seg_duration=seg_duration,
                method=method,
            )

        segments = self.demodulate(
            t_norm,
            data_arr,
            fs,
            f_init=f_hz,
            seg_duration=seg_duration,
            uniform=uniform_sampling,
        )
        if len(segments) < self.min_segments:
            return _empty_result(
                "invalid",
                ["demod_insufficient_segments"],
                seg_duration=seg_duration,
                method=method,
            )
        t_mid, f_seg, amplitude, sigma_seg = segments.T

        floor, plateau_detected, plateau_dominated, dynamic_range_ok = self._detect_plateau(
            t_mid, amplitude
        )
        if plateau_detected and floor is not None:
            cutoff = self.floor_threshold * float(floor)
            floor_for_correction = float(floor)
        else:
            cutoff = self.min_amplitude_fraction * float(np.max(amplitude))
            floor_for_correction = 0.0
        decay_mask = self._decay_mask(t_mid, amplitude, cutoff)
        amplitude_corrected = np.sqrt(
            np.maximum(amplitude**2 - floor_for_correction**2, np.finfo(np.float64).tiny)
        )

        def _diagnostic_result(status: str, reasons: list[str]) -> SegmentedDemodResult:
            return SegmentedDemodResult(
                Q=None,
                tau=None,
                f_mean=None,
                Q_ci95=None,
                valid=False,
                status=status,
                reasons=reasons,
                t_mid=t_mid,
                f_seg=f_seg,
                amplitude=amplitude,
                sigma_seg=sigma_seg,
                amplitude_corrected=amplitude_corrected,
                decay_mask=decay_mask,
                plateau_amplitude=floor,
                plateau_detected=plateau_detected,
                drift_hz=None,
                drift_hz_stderr=None,
                coherence_ratio=None,
                coherence_ratio_lower=None,
                q_vs_amplitude=[],
                f_pull_per_efold=None,
                log_slope=None,
                log_intercept=None,
                slope_stderr=None,
                seg_duration=seg_duration,
                n_segments=len(t_mid),
                method=method,
            )

        if plateau_dominated:
            # The window contains (almost) nothing but the driven plateau:
            # there is no measurable free decay, so no finite Q is honest.
            return _diagnostic_result("plateau_dominated", ["demod_plateau_dominated_window"])

        n_decay = int(np.count_nonzero(decay_mask))
        if n_decay < self.min_decay_segments:
            if plateau_detected:
                return _diagnostic_result("plateau_dominated", ["demod_plateau_dominated_window"])
            return _diagnostic_result("invalid", ["demod_insufficient_decay_segments"])

        log_amp_corrected_all = np.log(amplitude_corrected)
        decay_mask, slope, intercept = self._refine_decay_mask(
            t_mid, log_amp_corrected_all, decay_mask, cutoff
        )

        # Second-pass plateau detection: a strong plateau revival (slow
        # amplitude excursion of the driven floor) can defeat the quartile
        # flatness test. If the late-record median sits well above the
        # extrapolated fitted decay, the record does end in a plateau: redo
        # the selection and fit floor-corrected.
        if (
            not plateau_detected
            and floor is not None
            and dynamic_range_ok
            and slope < 0
            and floor > 2.0 * float(np.exp(slope * t_mid[-1] + intercept))
        ):
            plateau_detected = True
            floor_for_correction = float(floor)
            cutoff = self.floor_threshold * float(floor)
            decay_mask = self._decay_mask(t_mid, amplitude, cutoff)
            amplitude_corrected = np.sqrt(
                np.maximum(amplitude**2 - floor_for_correction**2, np.finfo(np.float64).tiny)
            )
            if int(np.count_nonzero(decay_mask)) < self.min_decay_segments:
                return _diagnostic_result("plateau_dominated", ["demod_plateau_dominated_window"])
            log_amp_corrected_all = np.log(amplitude_corrected)
            decay_mask, slope, intercept = self._refine_decay_mask(
                t_mid, log_amp_corrected_all, decay_mask, cutoff
            )

        n_decay = int(np.count_nonzero(decay_mask))
        t_fit = t_mid[decay_mask]
        log_amp = log_amp_corrected_all[decay_mask]
        if slope >= 0 or not np.isfinite(slope):
            return _diagnostic_result("invalid", ["demod_nondecaying_amplitude"])

        residual = log_amp - (slope * t_fit + intercept)
        sxx = float(np.sum((t_fit - np.mean(t_fit)) ** 2))
        dof = n_decay - 2
        slope_stderr = (
            float(np.sqrt(np.sum(residual**2) / dof / sxx)) if dof > 0 and sxx > 0 else None
        )

        tau = float(-1.0 / slope)
        f_mean = float(np.mean(f_seg[decay_mask]))
        q_value = float(np.pi * f_mean * tau)

        # Frequency drift over the decay region and the coherence gate.
        f_decay = f_seg[decay_mask]
        drift_slope, drift_intercept = np.polyfit(t_fit, f_decay, 1)
        decay_span = float(t_fit[-1] - t_fit[0])
        drift_hz = float(drift_slope * decay_span)
        f_resid = f_decay - (drift_slope * t_fit + drift_intercept)
        drift_dof = n_decay - 2
        if drift_dof > 0 and sxx > 0:
            drift_slope_stderr = float(np.sqrt(np.sum(f_resid**2) / drift_dof / sxx))
            drift_hz_stderr = float(drift_slope_stderr * decay_span)
        else:
            drift_hz_stderr = None
        coherence_ratio = float(abs(drift_hz) * tau)
        if drift_hz_stderr is not None:
            coherence_ratio_lower = float(max(0.0, abs(drift_hz) - 2.0 * drift_hz_stderr) * tau)
        else:
            coherence_ratio_lower = None

        # Frequency-pull coefficient (Hz per amplitude e-fold).
        log_amp_raw = np.log(amplitude[decay_mask])
        if float(np.ptp(log_amp_raw)) > 0.1:
            f_pull_per_efold = float(np.polyfit(log_amp_raw, f_seg[decay_mask], 1)[0])
        else:
            f_pull_per_efold = None

        q_ci95 = self._bootstrap_q_ci(t_fit, log_amp, slope, intercept, f_mean)

        q_vs_amplitude = self._q_vs_amplitude(
            t_mid,
            f_seg,
            amplitude,
            amplitude_corrected,
            decay_mask,
            amplitude_bands,
        )

        reasons: list[str] = []
        status = "valid"
        if decay_span < tau:
            status = "warning"
            reasons.append("demod_decay_window_shorter_than_tau")
        if q_ci95 is None:
            status = "warning"
            reasons.append("demod_bootstrap_ci_unavailable")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "demod_estimated",
                extra={
                    "event": "demod_estimated",
                    "q": q_value,
                    "tau": tau,
                    "f_mean": f_mean,
                    "n_decay_segments": n_decay,
                    "coherence_ratio": coherence_ratio,
                },
            )

        return SegmentedDemodResult(
            Q=q_value,
            tau=tau,
            f_mean=f_mean,
            Q_ci95=q_ci95,
            valid=status == "valid",
            status=status,
            reasons=reasons,
            t_mid=t_mid,
            f_seg=f_seg,
            amplitude=amplitude,
            sigma_seg=sigma_seg,
            amplitude_corrected=amplitude_corrected,
            decay_mask=decay_mask,
            plateau_amplitude=floor,
            plateau_detected=plateau_detected,
            drift_hz=drift_hz,
            drift_hz_stderr=drift_hz_stderr,
            coherence_ratio=coherence_ratio,
            coherence_ratio_lower=coherence_ratio_lower,
            q_vs_amplitude=q_vs_amplitude,
            f_pull_per_efold=f_pull_per_efold,
            log_slope=slope,
            log_intercept=intercept,
            slope_stderr=slope_stderr,
            seg_duration=seg_duration,
            n_segments=len(t_mid),
            method=method,
        )

    # ------------------------------------------------------------------
    # Amplitude-resolved Q
    # ------------------------------------------------------------------

    def _q_vs_amplitude(
        self,
        t_mid: np.ndarray,
        f_seg: np.ndarray,
        amplitude: np.ndarray,
        amplitude_corrected: np.ndarray,
        decay_mask: np.ndarray,
        amplitude_bands: list[tuple[float, float]] | None,
    ) -> list[AmplitudeBandQ]:
        if not np.any(decay_mask):
            return []
        if amplitude_bands is None:
            a_high = float(np.max(amplitude[decay_mask]))
            a_low = float(np.min(amplitude[decay_mask]))
            if a_low <= 0 or a_high <= a_low:
                return []
            bands = []
            edge = a_high
            while edge > a_low * (1.0 + 1e-9):
                lower_edge = max(edge / 2.0, a_low)
                bands.append((lower_edge, edge))
                edge = lower_edge
        else:
            bands = [(float(lo), float(hi)) for lo, hi in amplitude_bands]

        out: list[AmplitudeBandQ] = []
        for lo, hi in bands:
            in_band = decay_mask & (amplitude >= lo) & (amplitude < hi)
            n_band = int(np.count_nonzero(in_band))
            if n_band < 4:
                continue
            slope_band = self._fit_line_robust(
                t_mid[in_band], np.log(amplitude_corrected[in_band])
            )[0]
            if slope_band >= 0 or not np.isfinite(slope_band):
                continue
            tau_band = -1.0 / slope_band
            q_band = float(np.pi * np.mean(f_seg[in_band]) * tau_band)
            out.append(
                AmplitudeBandQ(
                    amplitude_low=lo,
                    amplitude_high=hi,
                    amplitude_mid=float(np.sqrt(lo * hi)),
                    n_segments=n_band,
                    tau=float(tau_band),
                    Q=q_band,
                )
            )
        return out
