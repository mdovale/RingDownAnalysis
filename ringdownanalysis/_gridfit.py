"""
Uniform-grid primitives shared by the least-squares tone estimators.

Both the segmented-demodulation estimator (:mod:`ringdownanalysis.demod`) and
the profile-likelihood estimator (:mod:`ringdownanalysis.q_profile`) fit models
of the form ``a*cos(w*t) + b*sin(w*t) + c``, optionally with an exponential
envelope, over a scan of trial parameters. When the sample times lie on the
nominal grid ``t_k = k/fs``, the model is a geometric sequence in ``k`` and the
whole scan collapses onto two closed-form ingredients:

- the Gram matrix, whose entries are geometric sums with no data dependence and
  therefore cost O(1) per trial parameter (:func:`geometric_sum`);
- the data projections, which factor through the two-index decomposition
  ``k = p*B + b`` (:class:`BlockIndex`) and so evaluate an entire batch of
  trial parameters with two real matrix products.

This module holds only the parts both estimators need. Each estimator keeps its
own exponent conventions on top: the demodulator scans frequency at zero decay,
the profile estimator scans decay at fixed frequency.
"""

from __future__ import annotations

import numpy as np

#: A record counts as uniformly sampled when every sample time is within this
#: fraction of a sampling interval of the nominal grid k/fs, which lets the fit
#: evaluate its model on that grid.
#:
#: The tolerance is set by the phase error it admits: a timing offset of
#: ``tol/fs`` shifts the model phase by at most ``pi*tol`` radians (at Nyquist;
#: proportionally less for an oversampled tone), so 1e-3 caps the phase error at
#: 3e-3 rad and the induced amplitude error near 5e-6 relative -- orders of
#: magnitude below the scatter that limits either estimator. Real phasemeter
#: exports need this headroom: the EDU records' time column carries ~2e-7 s of
#: quantization noise, about 1e-4 of a sample.
UNIFORM_TOLERANCE_SAMPLES = 1e-3

#: Below this fraction of the total signal power, a residual sum of squares
#: computed by subtraction (``||y||^2 - b'x``) has lost too many significant
#: digits to be trusted, and the residual is formed explicitly instead. Only
#: reached on essentially noiseless records.
RSS_CANCELLATION_FRACTION = 1e-8

#: Geometric weights below this magnitude are flushed to zero. They contribute
#: less than one part in 1e290 of any sum taken here, and letting them decay
#: into the subnormal range would cost far more than they are worth on hardware
#: that traps subnormal arithmetic.
_UNDERFLOW_FLOOR = 1e-290


def is_uniformly_sampled(t: np.ndarray, fs: float) -> bool:
    """
    Whether the sample times lie on the nominal grid ``t[0] + k/fs``.

    The check is on the accumulated offset, not on consecutive differences: a
    systematic bias of even 1e-6 of a sample per step accumulates to a
    significant phase error over 1e5 samples. Times are normalized to the first
    sample before the comparison so that a large absolute time origin cannot
    swamp the deviation being measured.
    """
    n = len(t)
    if n < 2:
        return True
    # Reduced in place: the record can be tens of millions of samples.
    grid = np.arange(n, dtype=np.float64)
    grid /= fs
    deviation = t - t[0]
    deviation -= grid
    worst = max(float(deviation.max()), -float(deviation.min()))
    return worst <= UNIFORM_TOLERANCE_SAMPLES / fs


def tone_sums(n: int, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form ``(sum_k cos(a*k), sum_k sin(a*k))`` for k < n, per angle a.

    The geometric series sums to the Dirichlet kernel
    ``exp(i*a*(n-1)/2) * sin(a*n/2) / sin(a/2)``, so the Gram matrix of the
    undamped tone model costs O(1) per trial frequency on a uniform sampling
    grid.

    At a multiple of 2*pi the kernel is 0/0 and is replaced by its limit,
    ``n * cos(a*n/2) / cos(a/2)``. Using ``n`` alone would be wrong by a sign
    whenever ``a`` is an odd multiple of 2*pi, which is where the accompanying
    phase factor is -1.
    """
    half = 0.5 * angle
    sin_half = np.sin(half)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.where(
            np.abs(sin_half) > 1e-13,
            np.sin(half * n) / sin_half,
            n * np.cos(half * n) / np.cos(half),
        )
    phase = half * (n - 1)
    return kernel * np.cos(phase), kernel * np.sin(phase)


def _expm1_complex(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """
    ``exp(real + 1j*imag) - 1`` without the cancellation of a plain subtraction.

    Splitting the real part as ``expm1(x)*cos(y) - 2*sin(y/2)**2`` keeps both
    terms individually accurate for small arguments. For the decaying case used
    here (``real <= 0``) the two terms cannot cancel: both are non-positive
    wherever ``cos(y)`` is, and where ``cos(y) < 0`` the second term dominates.
    """
    half_sin = np.sin(0.5 * imag)
    real_part = np.expm1(real) * np.cos(imag) - 2.0 * half_sin * half_sin
    return real_part + 1j * (np.exp(real) * np.sin(imag))


def geometric_sum(n: int, log_ratio: np.ndarray) -> np.ndarray:
    """
    Closed-form ``sum_{k<n} exp(k*log_ratio)`` for a batch of complex ratios.

    Evaluated as ``expm1(n*w)/expm1(w)`` so that a ratio close to one -- a decay
    time much longer than the record, which is exactly the regime a Q profile
    explores at its upper bound -- keeps full relative precision instead of
    losing it to the subtraction ``1 - exp(w)``.
    """
    log_ratio = np.asarray(log_ratio, dtype=np.complex128)
    real = log_ratio.real
    imag = log_ratio.imag
    denominator = _expm1_complex(real, imag)
    numerator = _expm1_complex(n * real, n * imag)
    # A unit ratio has every term equal to one; the quotient is 0/0 there.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator == 0.0, float(n), numerator / denominator)


class BlockIndex:
    """
    Two-factor decomposition of the sample index for batched projections.

    Writing ``k = p*B + b`` with ``B ~ sqrt(n)`` turns a geometric projection
    into a matrix product::

        sum_k y_k z**k = sum_b z**b * sum_p y_{pB+b} (z**B)**p

    so a batch of M trial ratios costs two real ``(M, n_blocks) x (n_blocks,
    block)`` products plus only O(M*sqrt(n)) transcendental calls, and touches
    the data once instead of once per trial. The data is zero-padded to a full
    rectangle, which adds nothing to the sums.
    """

    __slots__ = ("block", "blocked", "inner_index", "n", "n_blocks", "outer_index")

    def __init__(self, y: np.ndarray):
        self.n = len(y)
        self.block = int(np.ceil(np.sqrt(self.n)))
        self.n_blocks = -(-self.n // self.block)
        padded = np.zeros(self.n_blocks * self.block, dtype=np.float64)
        padded[: self.n] = y
        self.blocked = padded.reshape(self.n_blocks, self.block)
        self.outer_index = np.arange(self.n_blocks) * self.block
        self.inner_index = np.arange(self.block)

    def project(self, log_ratio: np.ndarray) -> np.ndarray:
        """
        Batched ``sum_k y_k exp(k*log_ratio)`` for complex ratios.

        ``log_ratio`` has shape (M,); the result has shape (M,).
        """
        log_ratio = np.asarray(log_ratio, dtype=np.complex128)[:, None]
        outer = np.exp(log_ratio * self.outer_index)
        inner = np.exp(log_ratio * self.inner_index)
        # Weights this small are indistinguishable from zero in the sum, but
        # would otherwise become subnormal and slow the products down.
        np.copyto(outer, 0.0, where=np.abs(outer) < _UNDERFLOW_FLOOR)
        # Two real products rather than one complex product: the data is real,
        # so a complex matmul would spend three quarters of its work on zeros.
        partial_real = outer.real @ self.blocked
        partial_imag = outer.imag @ self.blocked
        return np.sum(
            (partial_real + 1j * partial_imag) * inner,
            axis=1,
        )
