"""
Real-data regression tests for the segmented-demodulation estimator.

These pin the library against the EDU R1 ring-down records analyzed in
notebooks/20260818_EDU_PreVibe_vs_PostVibe_RingDown.ipynb (reference numbers
from the investigation report, docs/investigations/
20260818_q_estimation_failure_investigation.md).

They are opt-in: they need the raw phasemeter exports (not tracked in the
repo), the ``mokutools`` loader, and several minutes of runtime. Enable with

    RINGDOWN_REAL_DATA_TESTS=1 pytest tests/test_real_data_regression.py

Note the mokutools ``start_time`` convention: it is an offset from the first
sample of the file, not an absolute time (docs/data_format.md).

On reference values: the notebook's whole-decay fit uses an unweighted
least-squares slope; the shipped estimator uses a Theil-Sen (median-of-slopes)
fit that weights the genuinely curved decay differently. On these records the
two recipes differ by 5-10 % — that is decay-rate curvature, not estimator
error — so each estimator is pinned tightly against its *own* reference and
only loosely against the notebook's.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from ringdownanalysis.demod import SegmentedDemodEstimator

DATA_DIR = Path(__file__).parent.parent / "data" / "ODIN"
PRE_FILE = DATA_DIR / "20260321_EDU_R1.csv.zip"
POST_FILE = DATA_DIR / "20260325_EDU_R1.csv.zip"

_ENABLED = bool(os.environ.get("RINGDOWN_REAL_DATA_TESTS"))
_HAVE_DATA = PRE_FILE.exists() and POST_FILE.exists()
_HAVE_MOKUTOOLS = importlib.util.find_spec("mokutools") is not None

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_ENABLED and _HAVE_DATA and _HAVE_MOKUTOOLS),
        reason=(
            "real-data regression tests are opt-in: set RINGDOWN_REAL_DATA_TESTS=1 "
            "and provide data/ODIN/2026032{1,5}_EDU_R1.csv.zip plus mokutools"
        ),
    ),
]

#: Approximate R1 resonance. Passed as f_init because the raw records carry
#: large baseline wander; a full-record DFT without a band hint can lock onto
#: the wander instead of the tone (the notebook constrains its FFT search band
#: to 7.5-7.85 Hz for the same reason).
F_NOMINAL = 7.6699

# Pinned values measured with the shipped SegmentedDemodEstimator (Theil-Sen
# whole-decay fit) on 2026-08-19. Tight tolerance: these are regression pins.
Q_PINNED = {"Pre-Vibe": 8.13e4, "Post-Vibe": 9.54e4}
# Whole-decay references from the notebook's least-squares recipe. Loose
# tolerance: recipe-dependent on a curved decay (see module docstring).
Q_NOTEBOOK = {"Pre-Vibe": 8.92e4, "Post-Vibe": 1.005e5}
DRIFT_MHZ_RANGE = {"Pre-Vibe": (0.6, 1.6), "Post-Vibe": (0.1, 0.6)}
RINGDOWN_CHANNEL = {"Pre-Vibe": 1, "Post-Vibe": 2}


@pytest.fixture(scope="module")
def records():
    from mokutools.phasemeter import MokuPhasemeterObject

    out = {}
    for label, path in {"Pre-Vibe": PRE_FILE, "Post-Vibe": POST_FILE}.items():
        pm = MokuPhasemeterObject(filename=str(path), start_time=0, duration=3600 * 24)
        t = pm.df["time"].values
        ch = RINGDOWN_CHANNEL[label]
        out[label] = (t - t[0], pm.df[f"{ch}_cycles"].values, float(pm.fs))
    return out


@pytest.fixture(scope="module")
def demod_results(records):
    estimator = SegmentedDemodEstimator()
    return {
        label: estimator.estimate(t, y, fs, f_init=F_NOMINAL)
        for label, (t, y, fs) in records.items()
    }


@pytest.mark.parametrize("label", ["Pre-Vibe", "Post-Vibe"])
class TestWholeDecayReference:
    def test_q_matches_pinned_value(self, demod_results, label):
        result = demod_results[label]
        assert result.status == "valid"
        assert pytest.approx(Q_PINNED[label], rel=0.03) == result.Q

    def test_q_consistent_with_notebook_reference(self, demod_results, label):
        result = demod_results[label]
        assert pytest.approx(Q_NOTEBOOK[label], rel=0.15) == result.Q

    def test_plateau_detected_in_reference_range(self, demod_results, label):
        result = demod_results[label]
        assert result.plateau_detected
        assert 14.0 <= result.plateau_amplitude <= 20.0

    def test_drift_sign_and_magnitude(self, demod_results, label):
        result = demod_results[label]
        low, high = DRIFT_MHZ_RANGE[label]
        assert low <= result.drift_hz * 1e3 <= high
        # Both records must be gated out of coherent estimation.
        assert result.coherence_ratio_lower > 0.01

    def test_amplitude_resolved_q_increases_at_low_amplitude(self, demod_results, label):
        bands = demod_results[label].q_vs_amplitude
        assert len(bands) >= 2
        # Bands run from high to low amplitude; the EDU resonators show
        # local Q rising roughly 6.5e4 -> 1.8e5 over the decay.
        assert bands[-1].Q > bands[0].Q
        for band in bands:
            assert 4e4 <= band.Q <= 2.5e5


class TestLateWindowSpecimen:
    def test_post_vibe_offset_window_reports_low_amplitude_q(self, records):
        """The Post-Vibe +9000 s window that breaks the coherent pipeline.

        The coherent estimators collapse on this window (tau_est -> 36 s or
        1051 s depending on float round-off; investigation section 5.1). The
        demod estimator instead measures the genuine low-amplitude local Q,
        which must land inside the physical Q(A) band for A ~ 60 cycles.
        """
        t, y, fs = records["Post-Vibe"]
        mask = (t >= 9000.0) & (t <= 9000.0 + 3.5 * 3600.0)
        result = SegmentedDemodEstimator().estimate(t[mask], y[mask], fs, f_init=F_NOMINAL)
        assert result.Q is not None
        assert 1.0e5 <= result.Q <= 1.9e5


class TestWindowStability:
    def test_previbe_demod_q_stable_across_standard_windows(self, records):
        t, y, fs = records["Pre-Vibe"]
        estimator = SegmentedDemodEstimator()
        q_values = []
        for start in (0.0, 1800.0):
            for hours in (1.0, 3.0, 6.0):
                mask = (t >= start) & (t <= start + hours * 3600.0)
                result = estimator.estimate(t[mask], y[mask], fs, f_init=F_NOMINAL)
                if result.Q is not None:
                    q_values.append(result.Q)
        assert len(q_values) >= 5
        spread = max(q_values) / min(q_values)
        # Q_profile scatters 12x across these windows. The demod estimator's
        # residual spread (~1.3x) is physical, not statistical: short early
        # windows sample the high-amplitude (low-Q) part of the genuinely
        # amplitude-dependent decay, long windows average further down it.
        assert spread < 1.5

    def test_previbe_windows_stay_inside_physical_band(self, records):
        t, y, fs = records["Pre-Vibe"]
        estimator = SegmentedDemodEstimator()
        for start in (0.0, 1800.0):
            for hours in (1.0, 3.0, 6.0):
                mask = (t >= start) & (t <= start + hours * 3600.0)
                result = estimator.estimate(t[mask], y[mask], fs, f_init=F_NOMINAL)
                if result.Q is not None:
                    # Physical Q(A) range measured in the notebook: 6.5e4-1.8e5.
                    assert 6.0e4 <= result.Q <= 1.9e5


def test_records_fixture_sanity(records):
    for label, (t, y, fs) in records.items():
        assert np.all(np.isfinite(y))
        assert fs == pytest.approx(149.01, abs=0.1)
        assert t[-1] > 7 * 3600.0, f"{label} record unexpectedly short"
