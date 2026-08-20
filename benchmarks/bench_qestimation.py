#!/usr/bin/env python3
"""
Performance evaluation for the Q-estimation stack.

Measures the optimized library against the frozen pre-optimization reference in
``baseline_qestimation.py`` in a single process, on the same synthetic records,
and reports both the speedup and the numerical agreement. Agreement matters as
much as speed here: the optimizations are algebraic rewrites of the same
estimators, not different estimators, so anything above float round-off is a
bug rather than a trade-off.

Usage:
    python benchmarks/bench_qestimation.py                  # default workloads
    python benchmarks/bench_qestimation.py --sizes 1h 3h    # pick workloads
    python benchmarks/bench_qestimation.py --repeat 5       # more timing repeats
    python benchmarks/bench_qestimation.py --real-data      # add the EDU records
    python benchmarks/bench_qestimation.py --json out.json  # save raw numbers

The headline number is the total-time speedup of
``SegmentedDemodEstimator.estimate``, the entry point the pipeline and the
batch analyzer call once per record.

Exit status is non-zero when any output moves by more than its allowed
deviation, so this doubles as an equivalence check. The allowance is round-off
(``AGREEMENT_THRESHOLD``) for synthetic records, looser for real records, whose
timing quantization the fast path deliberately models on the nominal grid, and
looser again for the profile likelihood, which multiplies round-off by the
degrees of freedom (see ``FIELD_THRESHOLDS``).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from baseline_qestimation import (  # noqa: E402
    BaselineProfileQEstimator,
    BaselineSegmentedDemodEstimator,
)

from ringdownanalysis.analyzer import RingDownAnalyzer  # noqa: E402
from ringdownanalysis.demod import SegmentedDemodEstimator  # noqa: E402
from ringdownanalysis.nonlinear import fit_nonlinear_damping  # noqa: E402
from ringdownanalysis.q_profile import (  # noqa: E402
    _CHI2_95_ONE_PARAMETER,
    ProfileQEstimator,
)
from ringdownanalysis.signal import generate_pathological_ringdown  # noqa: E402

# EDU R1 record parameters (data/ODIN/2026032{1,5}_EDU_R1.csv.zip): these set
# the shape of the real workload -- heavily oversampled tone (fs/f0 ~ 19),
# multi-hour records, mHz-scale drift and an ambient-driven plateau.
F0 = 7.6699
FS = 149.01
TAU = 3700.0
A0 = 1000.0
SIGMA_WHITE = 2.5
PLATEAU_RMS = 16.5 / np.sqrt(2)

#: Relative deviation above which the optimized code is not the same estimator.
#: Synthetic records sit exactly on the k/fs grid, so the optimized fast path is
#: algebraically identical to the baseline and only float round-off separates
#: them.
AGREEMENT_THRESHOLD = 1e-9

#: Real phasemeter exports carry ~1e-4 of a sample of timing quantization, and
#: the optimized fast path models them on the nominal k/fs grid (see
#: demod._UNIFORM_TOLERANCE_SAMPLES). That is a physical modelling choice, not
#: round-off, so real-data agreement is checked at a correspondingly looser
#: bound -- still ~7 orders of magnitude below the estimator's own uncertainty.
REAL_DATA_AGREEMENT_THRESHOLD = 1e-6

#: The profile likelihood is ``dof * log(rss / rss_min)``, so it multiplies the
#: round-off of a residual sum of squares by the degrees of freedom -- 3.2e6 on
#: the six-hour record. Round-off alone therefore lands near 1e-9 of the field's
#: own scale, which is why it gets a looser bound than the rest. What this bound
#: still guarantees is what the profile is for: a shift of 1e-7 of the
#: chi-square threshold cannot move a confidence-interval edge.
FIELD_THRESHOLDS: dict[str, float] = {"profile_delta": 1e-7}

#: Real EDU ring-down records: (filename under data/ODIN, phasemeter channel).
REAL_RECORDS: dict[str, tuple[str, int]] = {
    "pre-vibe": ("20260321_EDU_R1.csv.zip", 1),
    "post-vibe": ("20260325_EDU_R1.csv.zip", 2),
}

#: Workloads named by record duration. "unit" mirrors the time-scaled records
#: used by tests/test_demod.py (fs = 30 Hz), the rest are EDU-scale.
WORKLOADS: dict[str, dict] = {
    "unit": {"duration": 3 * 360.0, "fs": 30.0, "tau": 370.0, "a0": 600.0, "sigma": 1.5},
    "1h": {"duration": 3600.0, "fs": FS, "tau": TAU, "a0": A0, "sigma": SIGMA_WHITE},
    "3h": {"duration": 3 * 3600.0, "fs": FS, "tau": TAU, "a0": A0, "sigma": SIGMA_WHITE},
    "6h": {"duration": 6 * 3600.0, "fs": FS, "tau": TAU, "a0": A0, "sigma": SIGMA_WHITE},
}
DEFAULT_SIZES = ["unit", "1h", "3h", "6h"]


@dataclass
class Timing:
    """One before/after timing comparison."""

    label: str
    n_samples: int
    baseline_s: float
    optimized_s: float

    @property
    def speedup(self) -> float:
        return self.baseline_s / self.optimized_s


def make_record(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Build an EDU-like pathological record: drift + plateau + white noise."""
    spec = WORKLOADS[name]
    duration = float(spec["duration"])
    t, x = generate_pathological_ringdown(
        f0=F0,
        fs=spec["fs"],
        duration=duration,
        a0=spec["a0"],
        tau=spec["tau"],
        sigma_white=spec["sigma"],
        # +1.1 mHz over the record, i.e. the measured Pre-Vibe drift.
        linear_drift=1.1e-3 / duration,
        plateau_rms=PLATEAU_RMS * spec["a0"] / A0,
        rng=np.random.default_rng(20260820),
    )
    return t, x, float(spec["fs"])


def timed(fn, repeat: int) -> tuple[float, object]:
    """Return (best-of-`repeat` seconds, last result). Best-of rejects noise."""
    out = fn()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return best, out


# ---------------------------------------------------------------------------
# Agreement checking
# ---------------------------------------------------------------------------


def _rel_diff(a, b) -> float:
    """Max relative difference between two scalars or arrays (0.0 if both None)."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    a_arr = np.atleast_1d(np.asarray(a, dtype=float))
    b_arr = np.atleast_1d(np.asarray(b, dtype=float))
    if a_arr.shape != b_arr.shape:
        return float("inf")
    scale = np.maximum(np.abs(a_arr), np.abs(b_arr))
    scale = np.where(scale > 0, scale, 1.0)
    return float(np.max(np.abs(a_arr - b_arr) / scale))


def compare_demod(base, opt) -> dict[str, float]:
    """Relative deviation of every numerical output of the demod estimator."""
    deviations = {
        "Q": _rel_diff(base.Q, opt.Q),
        "tau": _rel_diff(base.tau, opt.tau),
        "f_mean": _rel_diff(base.f_mean, opt.f_mean),
        "Q_ci95": _rel_diff(base.Q_ci95, opt.Q_ci95),
        "amplitude": _rel_diff(base.amplitude, opt.amplitude),
        "f_seg": _rel_diff(base.f_seg, opt.f_seg),
        "sigma_seg": _rel_diff(base.sigma_seg, opt.sigma_seg),
        "t_mid": _rel_diff(base.t_mid, opt.t_mid),
        "drift_hz": _rel_diff(base.drift_hz, opt.drift_hz),
        "coherence_ratio": _rel_diff(base.coherence_ratio, opt.coherence_ratio),
        "plateau_amplitude": _rel_diff(base.plateau_amplitude, opt.plateau_amplitude),
        "slope_stderr": _rel_diff(base.slope_stderr, opt.slope_stderr),
        "f_pull_per_efold": _rel_diff(base.f_pull_per_efold, opt.f_pull_per_efold),
        "q_vs_amplitude": _rel_diff(
            [b.Q for b in base.q_vs_amplitude], [o.Q for o in opt.q_vs_amplitude]
        ),
    }
    if base.status != opt.status:
        deviations["status_mismatch"] = float("inf")
    if not np.array_equal(base.decay_mask, opt.decay_mask):
        deviations["decay_mask_mismatch"] = float("inf")
    return deviations


def _delta_diff(base: np.ndarray, opt: np.ndarray) -> float:
    """
    Deviation of the profile likelihood, measured against its decision scale.

    ``profile_delta`` is only ever compared with the chi-square threshold, and
    it passes through zero at the optimum, so a plain relative deviation is
    both meaningless there and far stricter than anything the estimator can
    resolve. Scaling by the threshold answers the question that matters: could
    this shift move a confidence-interval edge?
    """
    if base.shape != opt.shape:
        return float("inf")
    scale = np.maximum(np.maximum(np.abs(base), np.abs(opt)), _CHI2_95_ONE_PARAMETER)
    return float(np.max(np.abs(base - opt) / scale))


def compare_profile(base, opt) -> dict[str, float]:
    """Relative deviation of every numerical output of the profile estimator."""
    deviations = {
        "Q": _rel_diff(base.Q, opt.Q),
        "tau_hat": _rel_diff(base.tau_hat, opt.tau_hat),
        "f_hat": _rel_diff(base.f_hat, opt.f_hat),
        "ci95": _rel_diff(base.ci95, opt.ci95),
        "lower_limit_95": _rel_diff(base.lower_limit_95, opt.lower_limit_95),
        "upper_limit_95": _rel_diff(base.upper_limit_95, opt.upper_limit_95),
        "rss_min": _rel_diff(base.rss_min, opt.rss_min),
        "sigma": _rel_diff(base.sigma, opt.sigma),
        "profile_tau": _rel_diff(base.profile_tau, opt.profile_tau),
        "profile_delta": _delta_diff(base.profile_delta, opt.profile_delta),
    }
    if base.status != opt.status:
        deviations["status_mismatch"] = float("inf")
    return deviations


def limit_for(field: str, *, real: bool) -> float:
    """The deviation this field is allowed, given where the record came from."""
    base = REAL_DATA_AGREEMENT_THRESHOLD if real else AGREEMENT_THRESHOLD
    return max(base, FIELD_THRESHOLDS.get(field, 0.0))


def worst_field(deviations: dict[str, float], *, real: bool) -> tuple[str, float, float]:
    """The field closest to its own limit, with that deviation and limit."""
    key = max(deviations, key=lambda k: deviations[k] / limit_for(k, real=real))
    return key, deviations[key], limit_for(key, real=real)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_demod_estimate(sizes: list[str], repeat: int) -> tuple[list[Timing], dict[str, dict]]:
    """End-to-end SegmentedDemodEstimator.estimate: the headline measurement."""
    timings: list[Timing] = []
    agreement: dict[str, dict] = {}
    for name in sizes:
        t, x, fs = make_record(name)
        base_est = BaselineSegmentedDemodEstimator()
        opt_est = SegmentedDemodEstimator()
        base_s, base_res = timed(partial(base_est.estimate, t, x, fs, f_init=F0), repeat)
        opt_s, opt_res = timed(partial(opt_est.estimate, t, x, fs, f_init=F0), repeat)
        timings.append(Timing(f"demod.estimate[{name}]", len(t), base_s, opt_s))
        agreement[f"demod[{name}]"] = {
            "deviations": compare_demod(base_res, opt_res),
            "baseline": base_res.Q,
            "optimized": opt_res.Q,
            "status": opt_res.status,
        }
    return timings, agreement


def bench_profile(sizes: list[str], repeat: int) -> tuple[list[Timing], dict[str, dict]]:
    """Profile-likelihood Q estimator, the pipeline's dominant coherent fit."""
    timings: list[Timing] = []
    agreement: dict[str, dict] = {}
    for name in sizes:
        t, x, fs = make_record(name)
        tau_init = WORKLOADS[name]["tau"]
        base_est = BaselineProfileQEstimator()
        opt_est = ProfileQEstimator()
        base_s, base_res = timed(
            partial(base_est.estimate, t, x, fs, f_init=F0, tau_init=tau_init), repeat
        )
        opt_s, opt_res = timed(
            partial(opt_est.estimate, t, x, fs, f_init=F0, tau_init=tau_init), repeat
        )
        timings.append(Timing(f"profile.estimate[{name}]", len(t), base_s, opt_s))
        agreement[f"profile[{name}]"] = {
            "deviations": compare_profile(base_res, opt_res),
            "baseline": base_res.Q,
            "optimized": opt_res.Q,
            "status": opt_res.status,
        }
    return timings, agreement


def bench_segment(sizes: list[str], repeat: int) -> list[Timing]:
    """Per-segment demodulation, the inner loop of the demod estimator."""
    timings: list[Timing] = []
    for name in sizes:
        t, x, fs = make_record(name)
        opt_est = SegmentedDemodEstimator()
        base_est = BaselineSegmentedDemodEstimator()
        seg_duration = opt_est._resolve_seg_duration(float(t[-1] - t[0]), F0, fs)
        n_seg = min(int(seg_duration * fs), len(x))
        ts = t[:n_seg] - t[0]
        ys = x[:n_seg]
        band = (F0 * 0.95, F0 * 1.05)
        base_s, _ = timed(partial(base_est._demodulate_segment, ts, ys, fs, band), repeat)
        opt_s, _ = timed(partial(opt_est._demodulate_segment, ts, ys, fs, band), repeat)
        timings.append(Timing(f"demodulate_segment[{name}]", n_seg, base_s, opt_s))
    return timings


def bench_bootstrap(repeat: int) -> list[Timing]:
    """Residual block bootstrap for the Q confidence interval."""
    timings: list[Timing] = []
    for n_seg in (24, 64, 96):
        rng = np.random.default_rng(7)
        t_fit = np.linspace(0.0, 3000.0, n_seg)
        log_amp = 6.9 - t_fit / TAU + rng.normal(0.0, 0.05, n_seg)
        base_est = BaselineSegmentedDemodEstimator()
        opt_est = SegmentedDemodEstimator()
        args = (t_fit, log_amp, -1.0 / TAU, 6.9, F0)
        base_s, base_ci = timed(partial(base_est._bootstrap_q_ci, *args), repeat)
        opt_s, opt_ci = timed(partial(opt_est._bootstrap_q_ci, *args), repeat)
        timings.append(Timing(f"bootstrap_q_ci[n={n_seg}]", n_seg, base_s, opt_s))
        if _rel_diff(base_ci, opt_ci) > AGREEMENT_THRESHOLD:
            print(f"  WARNING bootstrap CI moved at n={n_seg}: {base_ci} -> {opt_ci}")
    return timings


def bench_pipeline(sizes: list[str], repeat: int) -> list[Timing]:
    """Full analyzer pipeline, which runs every estimator on one record."""
    timings: list[Timing] = []
    for name in sizes:
        t, x, fs = make_record(name)
        base = RingDownAnalyzer(
            demod_estimator=BaselineSegmentedDemodEstimator(),
            q_profile_estimator=BaselineProfileQEstimator(),
        )
        opt = RingDownAnalyzer()
        base_s, base_res = timed(partial(base.analyze_array, t, x, fs), repeat)
        opt_s, opt_res = timed(partial(opt.analyze_array, t, x, fs), repeat)
        timings.append(Timing(f"analyze_array[{name}]", len(t), base_s, opt_s))
        for field in ("Q_demod", "Q_profile"):
            deviation = _rel_diff(base_res.get(field), opt_res.get(field))
            if deviation > AGREEMENT_THRESHOLD:
                print(f"  WARNING pipeline {field} deviation {deviation:.2e} at {name}")
    return timings


def load_real_record(key: str) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Load an EDU ring-down record, or None when the data/loader is absent."""
    filename, channel = REAL_RECORDS[key]
    path = Path(__file__).parent.parent / "data" / "ODIN" / filename
    if not path.exists():
        return None
    try:
        from mokutools.phasemeter import MokuPhasemeterObject
    except ImportError:
        return None
    # start_time is an offset from the first sample, not an absolute time.
    pm = MokuPhasemeterObject(filename=str(path), start_time=0, duration=3600 * 24)
    t = pm.df["time"].values
    return t - t[0], pm.df[f"{channel}_cycles"].values, float(pm.fs)


def bench_real_data(repeat: int) -> tuple[list[Timing], dict[str, dict]]:
    """The measurement that matters: the actual multi-hour EDU records."""
    timings: list[Timing] = []
    agreement: dict[str, dict] = {}
    for key in REAL_RECORDS:
        record = load_real_record(key)
        if record is None:
            print(f"  skipping real record {key}: data or mokutools unavailable")
            continue
        t, y, fs = record
        base_est = BaselineSegmentedDemodEstimator()
        opt_est = SegmentedDemodEstimator()
        base_s, base_res = timed(partial(base_est.estimate, t, y, fs, f_init=F0), repeat)
        opt_s, opt_res = timed(partial(opt_est.estimate, t, y, fs, f_init=F0), repeat)
        timings.append(Timing(f"demod.estimate[{key}]", len(t), base_s, opt_s))
        agreement[f"real[{key}]"] = {
            "deviations": compare_demod(base_res, opt_res),
            "baseline": base_res.Q,
            "optimized": opt_res.Q,
            "status": opt_res.status,
        }
    return timings, agreement


def bench_nonlinear(sizes: list[str], repeat: int) -> list[Timing]:
    """Nonlinear-damping fit: not optimized, timed for context."""
    timings: list[Timing] = []
    for name in sizes:
        t, x, fs = make_record(name)
        demod = SegmentedDemodEstimator().estimate(t, x, fs, f_init=F0)
        if int(np.count_nonzero(demod.decay_mask)) < 12:
            continue
        opt_s, _ = timed(partial(fit_nonlinear_damping, demod), repeat)
        timings.append(Timing(f"fit_nonlinear_damping[{name}]", demod.n_segments, opt_s, opt_s))
    return timings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(title: str, timings: list[Timing]) -> None:
    if not timings:
        return
    print(f"\n{title}")
    print("-" * 78)
    print(f"{'workload':30s} {'samples':>10s} {'before':>11s} {'after':>11s} {'speedup':>9s}")
    for row in timings:
        print(
            f"{row.label:30s} {row.n_samples:10,d} "
            f"{row.baseline_s * 1e3:9.2f}ms {row.optimized_s * 1e3:9.2f}ms "
            f"{row.speedup:8.1f}x"
        )


def total_speedup(timings: list[Timing]) -> float:
    """Total-time speedup: sum(before) / sum(after) over the workload set."""
    before = sum(row.baseline_s for row in timings)
    after = sum(row.optimized_s for row in timings)
    return before / after if after > 0 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=list(WORKLOADS), default=DEFAULT_SIZES)
    parser.add_argument("--repeat", type=int, default=3, help="timing repeats (best-of)")
    parser.add_argument("--json", type=str, help="path to save raw results")
    parser.add_argument("--skip-pipeline", action="store_true", help="skip full-pipeline timing")
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="also time the EDU records in data/ODIN (needs mokutools; several minutes)",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("Q-ESTIMATION PERFORMANCE EVALUATION")
    print("=" * 78)
    print(f"generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"platform  : {platform.platform()}")
    print(f"python    : {platform.python_version()}   numpy {np.__version__}")
    print(f"workloads : {', '.join(args.sizes)}   (best of {args.repeat})")

    demod_timings, demod_agreement = bench_demod_estimate(args.sizes, args.repeat)
    profile_timings, profile_agreement = bench_profile(args.sizes, args.repeat)
    segment_timings = bench_segment(args.sizes, args.repeat)
    bootstrap_timings = bench_bootstrap(max(args.repeat, 5))
    pipeline_timings = [] if args.skip_pipeline else bench_pipeline(args.sizes, args.repeat)
    nonlinear_timings = bench_nonlinear(args.sizes, args.repeat)
    real_timings: list[Timing] = []
    real_agreement: dict[str, dict] = {}
    if args.real_data:
        real_timings, real_agreement = bench_real_data(args.repeat)

    print_table("A. SegmentedDemodEstimator.estimate (headline)", demod_timings)
    print_table("B. Per-segment demodulation (demod inner loop)", segment_timings)
    print_table("C. Bootstrap Q confidence interval", bootstrap_timings)
    print_table("D. ProfileQEstimator.estimate", profile_timings)
    print_table("E. Full analyzer pipeline (analyze_array)", pipeline_timings)
    print_table("F. Nonlinear-damping fit (not optimized, for context)", nonlinear_timings)
    print_table("G. Real EDU records (demod.estimate)", real_timings)

    agreement = {**demod_agreement, **profile_agreement, **real_agreement}
    print("\nNumerical agreement (deviation vs frozen baseline, as a fraction of its limit)")
    print("-" * 78)
    worst_ratio = 0.0
    for name, info in agreement.items():
        real = name.startswith("real[")
        key, value, limit = worst_field(info["deviations"], real=real)
        worst_ratio = max(worst_ratio, value / limit)
        base_q = "None" if info["baseline"] is None else f"{info['baseline']:.6g}"
        opt_q = "None" if info["optimized"] is None else f"{info['optimized']:.6g}"
        print(
            f"{name:18s} Q {base_q} -> {opt_q}   "
            f"worst: {key} {value:.2e} of {limit:.0e}   status={info['status']}"
        )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    speedups = {
        "demod.estimate": total_speedup(demod_timings),
        "demodulate_segment": total_speedup(segment_timings),
        "bootstrap_q_ci": total_speedup(bootstrap_timings),
        "profile.estimate": total_speedup(profile_timings),
    }
    if pipeline_timings:
        speedups["analyze_array"] = total_speedup(pipeline_timings)
    if real_timings:
        speedups["demod.estimate[real]"] = total_speedup(real_timings)
    for key, value in speedups.items():
        print(f"  {key:22s} {value:6.2f}x  (total time before / total time after)")

    headline = speedups.get("demod.estimate[real]", speedups["demod.estimate"])
    source = "real EDU records" if real_timings else "synthetic records"
    print(f"\n  HEADLINE: demod.estimate is {headline:.1f}x faster on {source}")
    passed = worst_ratio < 1.0
    print(f"  Worst deviation reached {100.0 * worst_ratio:.1f} % of its limit")
    print(f"  Agreement: {'PASS' if passed else 'FAIL'}")

    if args.json:
        payload = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "repeat": args.repeat,
            "timings": [
                asdict(row) | {"speedup": row.speedup}
                for row in demod_timings
                + segment_timings
                + bootstrap_timings
                + profile_timings
                + pipeline_timings
                + nonlinear_timings
                + real_timings
            ],
            "speedups": speedups,
            "agreement": agreement,
            "worst_deviation_fraction_of_limit": worst_ratio,
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n  raw results -> {out}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
