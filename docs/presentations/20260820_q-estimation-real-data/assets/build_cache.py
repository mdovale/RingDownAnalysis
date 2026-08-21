"""
Extract everything the deck's real-data figures need into one small cache.

The EDU phasemeter exports are hundreds of MB to >1 GB each and take minutes to
load, so figure generation must not touch them. This script loads each record
once, runs the shipped estimators, and writes a compact ``_cache/edu_cache.npz``
plus ``_cache/edu_cache.json`` (scalars) that ``make_figures.py`` consumes.

Run from the repository root with the project virtualenv:

    .venv/bin/python docs/presentations/20260820_q-estimation-real-data/assets/build_cache.py

Records (see docs/data_format.md for the mokutools ``start_time`` convention:
it is an offset from the first sample of the file, not an absolute timestamp):

- Pre-Vibe  : data/ODIN/20260321_EDU_R1.csv.zip, channel 1
- Post-Vibe : data/ODIN/20260325_EDU_R1.csv.zip, channel 2
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from ringdownanalysis import (  # noqa: E402
    RingDownAnalyzer,
    SegmentedDemodEstimator,
    fit_nonlinear_damping,
)
from ringdownanalysis.q_envelope import q_envelope_diagnostic  # noqa: E402

CACHE_DIR = Path(__file__).parent / "_cache"
DATA_DIR = REPO / "data" / "ODIN"

# Approximate R1 resonance. The raw records carry large baseline wander, so a
# full-record DFT without a band hint can lock onto the wander instead of the
# tone; every estimator here gets the same seed.
F_INIT = 7.67

RECORDS = {
    "pre": {
        "label": "Pre-Vibe",
        "file": "20260321_EDU_R1.csv.zip",
        "channel": 1,
        # Canonical analysis window from the investigation report (§4).
        "canonical_duration": 3.0 * 3600.0,
        "canonical_tau_init": 2000.0,
    },
    "post": {
        "label": "Post-Vibe",
        "file": "20260325_EDU_R1.csv.zip",
        "channel": 2,
        "canonical_duration": 3.5 * 3600.0,
        "canonical_tau_init": None,
    },
}

# Window sweep: early-release starts an experimenter would actually try
# (high-SNR part of the decay), not late offsets.
SWEEP_STARTS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
SWEEP_DURATIONS_H = (1.0, 2.0, 3.0, 4.0, 6.0)

# Post-Vibe crop-cascade specimen (report §4/§8-D/§12).
SPECIMEN_START = 9000.0
SPECIMEN_DURATION = 3.5 * 3600.0

# Matched comparison protocol. Comparing Q between two records is only
# meaningful at matched amplitude, because Q depends on amplitude here. The
# auto-chosen segment duration scales with record length (17 h for Post-Vibe
# against 8 h for Pre-Vibe) and the auto octave bands then land on different
# amplitudes per record, so the Pre/Post figures use one fixed segment duration
# and one shared set of amplitude bands for both records.
MATCHED_SEG_DURATION = 120.0
MATCHED_WINDOW = 6.0 * 3600.0
MATCHED_BANDS = [(60.0, 100.0), (100.0, 200.0), (200.0, 300.0), (300.0, 500.0)]

#: Plotting traces are decimated to this many points; the raw records hold
#: millions of samples and the slide only resolves the envelope band.
TRACE_POINTS = 60_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_record(spec: dict) -> tuple[np.ndarray, np.ndarray, float]:
    from mokutools.phasemeter import MokuPhasemeterObject

    path = DATA_DIR / spec["file"]
    log(f"loading {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
    t0 = time.time()
    pm = MokuPhasemeterObject(filename=str(path), start_time=0, duration=3600 * 24)
    t = np.asarray(pm.df["time"].values, dtype=np.float64)
    x = np.asarray(pm.df[f"{spec['channel']}_cycles"].values, dtype=np.float64)
    fs = float(pm.fs)
    log(f"  loaded {len(t):,} samples, fs={fs:.4f} Hz in {time.time() - t0:.0f} s")
    return t - t[0], x - float(np.mean(x)), fs


def decimate(t: np.ndarray, x: np.ndarray, n_points: int = TRACE_POINTS) -> tuple[np.ndarray, ...]:
    """Decimate for plotting while preserving the visible envelope band.

    Plain subsampling would alias the 7.67 Hz carrier into a beat pattern, so
    each output point carries the min and max of its block: the filled band
    between them is exactly what the slide shows.
    """
    step = max(1, len(t) // n_points)
    n_blocks = len(t) // step
    tb = t[: n_blocks * step].reshape(n_blocks, step)
    xb = x[: n_blocks * step].reshape(n_blocks, step)
    return tb[:, 0], xb.min(axis=1), xb.max(axis=1)


def window(t: np.ndarray, x: np.ndarray, start: float, duration: float) -> tuple[np.ndarray, ...]:
    lo = int(np.searchsorted(t, start))
    hi = int(np.searchsorted(t, start + duration))
    return t[lo:hi], x[lo:hi]


def demod_of(t: np.ndarray, x: np.ndarray, fs: float) -> object:
    return SegmentedDemodEstimator().estimate(t, x, fs, f_init=F_INIT)


#: Fields kept from a full pipeline run. ``Q_profile_raw`` is the ungated
#: optimizer value, i.e. exactly what the pipeline reported as ``Q_profile``
#: before the gates shipped, so before/after comparisons come from one run.
PIPELINE_FIELDS = (
    "f_nls",
    "tau_est",
    "tau_est_fit_success",
    "Q_profile",
    "Q_profile_raw",
    "Q_profile_status",
    "Q_profile_ci95",
    "Q_nls_raw",
    "Q_envelope",
    "Q_envelope_candidate_agrees",
    "Q_envelope_candidate_slope_mismatch",
    "Q_demod",
    "Q_demod_status",
    "Q_selected",
    "Q_selected_source",
    "Q_selected_regime",
    "Q_selected_status",
    "coherence_ratio",
    "coherence_gate_fired",
    "tau_est_low_confidence",
    "tau_envelope_precrop",
    "tau_crop",
    "tau_crop_source",
    "tau_est_envelope_ratio",
)


def pipeline_run(t: np.ndarray, x: np.ndarray, tau_init: float | None = None) -> dict:
    """Run the shipped analysis pipeline and keep the comparison fields."""
    res = RingDownAnalyzer().analyze_array(t=t, data=x, detrend="constant", tau_init=tau_init)
    out: dict[str, object] = {}
    for key in PIPELINE_FIELDS:
        value = res.get(key)
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, tuple):
            value = [float(v) for v in value]
        out[key] = value
    t_crop = res.get("t_crop")
    out["crop_duration"] = (
        float(t_crop[-1] - t_crop[0]) if t_crop is not None and len(t_crop) > 1 else None
    )
    return out


def segment_spectrum(
    t: np.ndarray, x: np.ndarray, fs: float, start: float, seg: float = 2000.0
) -> dict:
    """Zero-padded amplitude spectrum of one segment, for the mode survey."""
    tw, xw = window(t, x, start, seg)
    xw = xw - float(np.mean(xw))
    n_fft = 1 << (int(np.ceil(np.log2(len(xw)))) + 2)
    spec = np.abs(np.fft.rfft(xw * np.hanning(len(xw)), n=n_fft)) / len(xw)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    keep = (freq > 6.5) & (freq < 9.0)
    return {"freq": freq[keep], "amp": spec[keep], "t_start": float(tw[0])}


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, object] = {"f_init": F_INIT, "generated": time.strftime("%Y-%m-%d %H:%M")}

    for key, spec in RECORDS.items():
        t, x, fs = load_record(spec)
        rec: dict[str, object] = {"label": spec["label"], "fs": fs, "duration": float(t[-1])}

        td, xlo, xhi = decimate(t, x)
        arrays[f"{key}_trace_t"] = td
        arrays[f"{key}_trace_lo"] = xlo
        arrays[f"{key}_trace_hi"] = xhi

        log(f"{spec['label']}: full-record segmented demodulation")
        d = demod_of(t, x, fs)
        for name in (
            "t_mid",
            "f_seg",
            "amplitude",
            "sigma_seg",
            "amplitude_corrected",
            "decay_mask",
        ):
            arrays[f"{key}_demod_{name}"] = np.asarray(getattr(d, name))
        rec["demod"] = {
            "Q": d.Q,
            "tau": d.tau,
            "f_mean": d.f_mean,
            "Q_ci95": None if d.Q_ci95 is None else list(d.Q_ci95),
            "status": d.status,
            "plateau_amplitude": d.plateau_amplitude,
            "plateau_detected": bool(d.plateau_detected),
            "drift_hz": d.drift_hz,
            "coherence_ratio": d.coherence_ratio,
            "f_pull_per_efold": d.f_pull_per_efold,
            "log_slope": d.log_slope,
            "log_intercept": d.log_intercept,
            "seg_duration": d.seg_duration,
            "n_segments": d.n_segments,
        }
        rec["bands"] = [
            {
                "lo": b.amplitude_low,
                "hi": b.amplitude_high,
                "mid": b.amplitude_mid,
                "n": b.n_segments,
                "tau": b.tau,
                "Q": b.Q,
            }
            for b in d.q_vs_amplitude
        ]
        log(f"  Q={d.Q}, tau={d.tau}, drift={d.drift_hz}, plateau={d.plateau_amplitude}")

        try:
            nl = fit_nonlinear_damping(d)
            rec["nonlinear"] = {
                "tau0": nl.tau0,
                "beta": nl.beta,
                "Q0": nl.Q0,
                "f_zero": nl.f_zero,
                "f_pull": nl.f_pull,
            }
        except Exception as exc:  # pragma: no cover - diagnostic path
            rec["nonlinear"] = {"error": repr(exc)}
        log(f"  nonlinear: {rec['nonlinear']}")

        log(f"{spec['label']}: matched-protocol demodulation for the Pre/Post comparison")
        tm, xm = window(t, x, 0.0, MATCHED_WINDOW)
        dm = SegmentedDemodEstimator(seg_duration=MATCHED_SEG_DURATION).estimate(
            tm, xm, fs, f_init=F_INIT, amplitude_bands=MATCHED_BANDS
        )
        for name in ("t_mid", "f_seg", "amplitude", "amplitude_corrected", "decay_mask"):
            arrays[f"{key}_matched_{name}"] = np.asarray(getattr(dm, name))
        rec["matched"] = {
            "seg_duration": dm.seg_duration,
            "window": MATCHED_WINDOW,
            "Q": dm.Q,
            "tau": dm.tau,
            "Q_ci95": None if dm.Q_ci95 is None else list(dm.Q_ci95),
            "f_mean": dm.f_mean,
            "drift_hz": dm.drift_hz,
            "f_pull_per_efold": dm.f_pull_per_efold,
            "plateau_amplitude": dm.plateau_amplitude,
            "n_segments": dm.n_segments,
            "bands": [
                {
                    "lo": b.amplitude_low,
                    "hi": b.amplitude_high,
                    "mid": b.amplitude_mid,
                    "n": b.n_segments,
                    "tau": b.tau,
                    "Q": b.Q,
                }
                for b in dm.q_vs_amplitude
            ],
        }
        log(
            f"  matched Q={dm.Q}, {dm.n_segments} segments of {dm.seg_duration:.0f} s, "
            f"pull/efold={dm.f_pull_per_efold}"
        )
        for b in rec["matched"]["bands"]:
            log(f"    band {b['lo']:.0f}-{b['hi']:.0f}  n={b['n']}  Q={b['Q']}")

        log(f"{spec['label']}: canonical window through the full pipeline")
        tc, xc = window(t, x, 0.0, float(spec["canonical_duration"]))
        pipe = pipeline_run(tc, xc, spec["canonical_tau_init"])
        rec["canonical"] = {"duration": float(spec["canonical_duration"]), "pipeline": pipe}
        env = q_envelope_diagnostic(tc, xc, F_INIT, q=pipe["Q_profile_raw"])
        arrays[f"{key}_env_t"] = np.asarray(env.t_mid)
        arrays[f"{key}_env_a"] = np.asarray(env.amplitude)
        arrays[f"{key}_env_used"] = np.asarray(env.used)
        arrays[f"{key}_env_fit"] = np.asarray(env.fitted_amplitude)
        arrays[f"{key}_env_cand"] = np.asarray(env.candidate_amplitude)
        rec["canonical"]["envelope"] = {
            "Q": env.Q,
            "tau": env.tau,
            "candidate_agrees": env.candidate_agrees,
            "candidate_slope_mismatch": env.candidate_slope_mismatch,
            "n_windows_used": env.n_windows_used,
        }
        tcd, clo, chi = decimate(tc, xc, 30_000)
        arrays[f"{key}_canon_t"] = tcd
        arrays[f"{key}_canon_lo"] = clo
        arrays[f"{key}_canon_hi"] = chi
        log(
            f"  profile_raw={pipe['Q_profile_raw']} ({pipe['Q_profile_status']}), "
            f"selected={pipe['Q_selected']} via {pipe['Q_selected_source']}, "
            f"envelope Q={env.Q}, agrees={env.candidate_agrees}"
        )

        log(f"{spec['label']}: window sweep ({len(SWEEP_STARTS) * len(SWEEP_DURATIONS_H)} windows)")
        sweep = []
        for start in SWEEP_STARTS:
            for dur_h in SWEEP_DURATIONS_H:
                dur = dur_h * 3600.0
                if start + dur > float(t[-1]):
                    continue
                tw, xw = window(t, x, start, dur)
                row = {"start": start, "duration_h": dur_h}
                row.update(pipeline_run(tw, xw))
                sweep.append(row)
                log(
                    f"  start={start:.0f} dur={dur_h}h  profile_raw={row['Q_profile_raw']}  "
                    f"selected={row['Q_selected']} ({row['Q_selected_source']})  "
                    f"envelope={row['Q_envelope']}  gate={row['coherence_gate_fired']}"
                )
        rec["sweep"] = sweep

        if key == "post":
            log("Post-Vibe: crop-cascade specimen (offset 9000 s)")
            ts, xs = window(t, x, SPECIMEN_START, SPECIMEN_DURATION)
            env_s = q_envelope_diagnostic(ts, xs, F_INIT)
            arrays["specimen_env_t"] = np.asarray(env_s.t_mid)
            arrays["specimen_env_a"] = np.asarray(env_s.amplitude)
            tsd, slo, shi = decimate(ts, xs, 30_000)
            arrays["specimen_t"] = tsd
            arrays["specimen_lo"] = slo
            arrays["specimen_hi"] = shi
            pipe_s = pipeline_run(ts, xs)
            scalars["specimen"] = {
                "start": SPECIMEN_START,
                "duration": SPECIMEN_DURATION,
                "envelope_Q": env_s.Q,
                "envelope_tau": env_s.tau,
                "pipeline": pipe_s,
            }
            log(
                f"  envelope tau={env_s.tau}, tau_est={pipe_s['tau_est']}, "
                f"crop={pipe_s['crop_duration']} s from {pipe_s['tau_crop_source']}, "
                f"low_confidence={pipe_s['tau_est_low_confidence']}, "
                f"selected={pipe_s['Q_selected']} ({pipe_s['Q_selected_status']})"
            )

            seg_len = 2000.0
            late = segment_spectrum(t, x, fs, float(t[-1]) - seg_len, seg_len)
            arrays["plateau_spec_freq"] = late.pop("freq")
            arrays["plateau_spec_amp"] = late.pop("amp")
            early = segment_spectrum(t, x, fs, 0.0, seg_len)
            arrays["decay_spec_freq"] = early.pop("freq")
            arrays["decay_spec_amp"] = early.pop("amp")
            scalars["spectra"] = {"plateau": late, "decay": early, "seg_duration": seg_len}

        scalars[key] = rec
        del t, x

    np.savez_compressed(CACHE_DIR / "edu_cache.npz", **arrays)
    (CACHE_DIR / "edu_cache.json").write_text(json.dumps(scalars, indent=2, default=str))
    size = (CACHE_DIR / "edu_cache.npz").stat().st_size / 1e6
    log(f"wrote {CACHE_DIR / 'edu_cache.npz'} ({size:.1f} MB) and edu_cache.json")


if __name__ == "__main__":
    main()
