"""
Generate every figure in the deck as a vector PDF.

Real-data figures read only ``_cache/edu_cache.npz`` and ``_cache/edu_cache.json``
(built once by ``build_cache.py``), so this script runs in seconds and needs no
access to the multi-GB phasemeter exports.

Figures whose underlying experiment is a controlled synthetic study carry their
measured values as literal tables with the source cited in a comment; those
numbers come from the executed companion notebook and are reproduced in the
investigation report. Regenerating them here would mean re-running the whole
Monte-Carlo study for no change in the plotted values.

Run from the repository root with the project virtualenv:

    .venv/bin/python docs/presentations/20260820_q-estimation-real-data/assets/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).parent
CACHE = HERE / "_cache"
REPO = HERE.resolve().parents[3]
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Shared style. Palette matches the Beamer theme accents so figures and slide
# furniture read as one design. In-figure titles are suppressed throughout: the
# frame title carries the message.
# ---------------------------------------------------------------------------
BLUE = "#1F5FBF"
ORANGE = "#D1701A"
RED = "#B32020"
GREEN = "#2E7D4F"
INK = "#1A1A1A"
MUTE = "#5A5A5A"
FAINT = "#BFBFBF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "legend.fontsize": 12.5,
        "legend.frameon": False,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    }
)


# Figures are laid out at slide proportions and then drawn at FIGSCALE of that size.
# Font sizes are absolute points, so shrinking the canvas is what makes on-figure text
# read at the same size as the Beamer body text once the PDF is placed on a frame.
# Kept slightly smaller than the frame so bullet text has room beneath.
FIGSCALE = 0.72


def panels(*args, figsize: tuple[float, float], **kwargs):
    """``plt.subplots`` with the deck-wide canvas scale applied."""
    scaled = (figsize[0] * FIGSCALE, figsize[1] * FIGSCALE)
    return plt.subplots(*args, figsize=scaled, **kwargs)


def save(fig: plt.Figure, name: str) -> None:
    out = HERE / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def load_cache() -> tuple[dict, dict]:
    npz = CACHE / "edu_cache.npz"
    js = CACHE / "edu_cache.json"
    if not npz.exists() or not js.exists():
        raise SystemExit(f"cache missing; run build_cache.py first (looked in {CACHE})")
    return dict(np.load(npz)), json.loads(js.read_text())


def hours(t: np.ndarray) -> np.ndarray:
    return np.asarray(t) / 3600.0


def band(ax, t, lo, hi, color=FAINT, label=None) -> None:
    """Draw a decimated record as its min/max band."""
    ax.fill_between(hours(t), lo, hi, color=color, linewidth=0, label=label)


def envelope_from_q(t: np.ndarray, a0: float, f: float, q: float) -> np.ndarray:
    """Single-exponential amplitude envelope implied by a constant-$Q$ fit."""
    return float(a0) * np.exp(-np.pi * float(f) * np.asarray(t) / float(q))


def decimate_trace(t: np.ndarray, x: np.ndarray, n_points: int = 12_000) -> tuple[np.ndarray, ...]:
    """Min/max band for plotting an oscillating ring down without aliasing."""
    step = max(1, len(t) // n_points)
    n_blocks = len(t) // step
    tb = t[: n_blocks * step].reshape(n_blocks, step)
    xb = x[: n_blocks * step].reshape(n_blocks, step)
    return tb[:, 0], xb.min(axis=1), xb.max(axis=1)


def annotate(ax, text: str, xy, xytext, color=INK, fontsize=13, arrow=True) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        arrowprops=(
            dict(arrowstyle="->", color=color, linewidth=1.4, shrinkA=0, shrinkB=3)
            if arrow
            else None
        ),
    )


def _matched_synth_params() -> dict[str, float]:
    """Ideal ring down matched to Pre-Vibe sampling, frequency, Q, amplitude, SNR."""
    fs, f0, tau, a0, sigma = 149.0116, 7.6699, 3700.0, 600.0, 1.5
    return {
        "fs": fs,
        "f0": f0,
        "tau": tau,
        "a0": a0,
        "sigma": sigma,
        "q": float(np.pi * f0 * tau),
        "snr_db": float(20.0 * np.log10(a0 / sigma)),
    }


def _generate_matched_synth(duration_h: float = 3.0, seed: int = 1):
    from ringdownanalysis import RingDownSignal

    p = _matched_synth_params()
    n = int(duration_h * 3600.0 * p["fs"])
    sig = RingDownSignal(
        f0=p["f0"], fs=p["fs"], N=n, A0=p["a0"], snr_db=p["snr_db"], Q=p["q"]
    )
    ts, xs, _ = sig.generate(rng=np.random.default_rng(seed))
    return ts, xs, p


# ===========================================================================
# 1. Title strip — the real decay with the wrong fitted envelope over it
# ===========================================================================
def fig_title_strip(arr: dict, sc: dict) -> None:
    """
    Decorative title strip: the record's envelope against the fitted one.

    Cropped to the first 1.3 h. Over the full window both envelopes are small on a
    linear axis and the strip degenerates into a flat line, which hides the very
    disagreement the deck is about.
    """
    span = 0.9 * 3600.0
    fig, ax = panels(figsize=(11.0, 1.6))
    t = arr["pre_canon_t"]
    keep = t <= span
    band(ax, t[keep], arr["pre_canon_lo"][keep], arr["pre_canon_hi"][keep], color="#E4E4E4")

    env_t, env_a = arr["pre_env_t"], arr["pre_env_a"]
    cand = arr["pre_env_cand"]
    for sign in (1.0, -1.0):
        m = env_t <= span
        ax.plot(hours(env_t[m]), sign * env_a[m], color=BLUE, linewidth=2.0)
        good = m & np.isfinite(cand) & (cand > 0)
        ax.plot(hours(env_t[good]), sign * cand[good], color=RED, linewidth=2.0, linestyle="--")

    ax.set_xlim(0, 0.9)
    ax.axis("off")
    save(fig, "title-strip.pdf")


# ===========================================================================
# 2. Symptom — synthetic vs real (time series + log envelope)
# ===========================================================================
def fig_symptom(arr: dict, sc: dict) -> None:
    from ringdownanalysis.q_envelope import q_envelope_diagnostic
    from ringdownanalysis.q_profile import ProfileQEstimator

    ts, xs, p = _generate_matched_synth(duration_h=3.0, seed=1)
    res = ProfileQEstimator().estimate(ts, xs, p["fs"], f_init=p["f0"])
    env = q_envelope_diagnostic(ts, xs, p["f0"], q=res.Q)
    st, slo, shi = decimate_trace(ts, xs, 8_000)

    pipe = sc["pre"]["canonical"]["pipeline"]
    f_fit = float(pipe["f_nls"] or p["f0"])
    q_wrong = float(pipe["Q_profile_raw"])
    q_ref = float(pipe["Q_demod"] or sc["pre"]["canonical"]["envelope"]["Q"])
    a0_real = float(arr["pre_env_a"][np.isfinite(arr["pre_env_a"])][0])

    # 2x2: columns = synthetic | real; rows = oscillating record | log envelope
    fig, axes = panels(2, 2, figsize=(12.4, 6.4), sharex="col")

    # --- top row: ring down time series with ±Q envelopes
    ax = axes[0, 0]
    band(ax, st, slo, shi, color="#D8D8D8")
    te = np.linspace(0.0, 3.0 * 3600.0, 400)
    e_fit = envelope_from_q(te, p["a0"], p["f0"], res.Q)
    for sign in (1.0, -1.0):
        ax.plot(hours(te), sign * e_fit, color=GREEN, linewidth=1.8, linestyle="--")
    ax.set_ylabel("Signal (cycles)")
    ax.set_ylim(-750, 750)
    ax.text(
        0.97,
        0.92,
        "synthetic (matched SNR)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        color=MUTE,
    )
    ax.text(
        0.03,
        0.08,
        "coherent fit tracks the decay",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        color=GREEN,
        fontweight="bold",
    )

    ax = axes[0, 1]
    band(ax, arr["pre_canon_t"], arr["pre_canon_lo"], arr["pre_canon_hi"], color="#D8D8D8")
    te = np.linspace(0.0, float(arr["pre_canon_t"][-1]), 400)
    e_wrong = envelope_from_q(te, a0_real, f_fit, q_wrong)
    e_ref = envelope_from_q(te, a0_real, f_fit, q_ref)
    for sign in (1.0, -1.0):
        ax.plot(
            hours(te),
            sign * e_wrong,
            color=RED,
            linewidth=1.8,
            linestyle="--",
            label="coherent single-$Q$ envelope" if sign > 0 else None,
        )
        ax.plot(
            hours(te),
            sign * e_ref,
            color=GREEN,
            linewidth=1.6,
            label="incoherent reference envelope" if sign > 0 else None,
        )
    ax.set_ylim(-750, 750)
    ax.text(
        0.97,
        0.92,
        "ODIN EDU Pre-Vibe",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        color=MUTE,
    )
    ax.legend(loc="lower right", fontsize=10.5)

    # --- bottom row: log amplitude (slope = decay rate)
    ax = axes[1, 0]
    ax.plot(hours(env.t_mid), env.amplitude, ".", color="#9A9A9A", markersize=2.8, zorder=2)
    good = np.isfinite(env.candidate_amplitude) & (env.candidate_amplitude > 0)
    ax.plot(
        hours(env.t_mid[good]),
        env.candidate_amplitude[good],
        color=GREEN,
        linewidth=2.2,
        linestyle="--",
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Amplitude (cycles)")
    ax.set_xlabel("Time (h)")
    ax.set_xlim(0, 3.0)
    ax.text(
        0.03,
        0.08,
        f"$Q_{{\\rm fit}}/Q_{{\\rm true}}={res.Q / p['q']:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        color=GREEN,
        fontweight="bold",
    )

    ax = axes[1, 1]
    env_t, env_a, cand = arr["pre_env_t"], arr["pre_env_a"], arr["pre_env_cand"]
    ax.plot(hours(env_t), env_a, ".", color="#9A9A9A", markersize=2.8, zorder=2)
    good = np.isfinite(cand) & (cand > 0)
    ax.plot(hours(env_t[good]), cand[good], color=RED, linewidth=2.2, linestyle="--", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Time (h)")
    ax.set_xlim(0, 3.0)
    ax.text(
        0.03,
        0.08,
        "same coherent fit leaves the data",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        color=RED,
        fontweight="bold",
    )

    fig.tight_layout()
    save(fig, "symptom-synth-vs-real.pdf")


# ===========================================================================
# 3. Overlay — close-up of the envelope mismatch
# ===========================================================================
def fig_overlay(arr: dict, sc: dict) -> None:
    env = sc["pre"]["canonical"]["envelope"]

    fig, ax = panels(figsize=(11.6, 4.6))

    env_t, env_a, fit = arr["pre_env_t"], arr["pre_env_a"], arr["pre_env_fit"]
    ax.plot(hours(env_t), env_a, ".", color=FAINT, markersize=3.6)
    good_fit = np.isfinite(fit) & (fit > 0)
    ax.plot(hours(env_t[good_fit]), fit[good_fit], color=GREEN, linewidth=2.6)

    cand = arr["pre_env_cand"]
    good = np.isfinite(cand) & (cand > 0)
    ax.plot(hours(env_t[good]), cand[good], color=RED, linewidth=2.6, linestyle="--")

    ax.set_yscale("log")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Amplitude (cycles)")
    ax.set_xlim(0, hours(env_t)[-1])

    ratio = None
    mismatch = env.get("candidate_slope_mismatch")
    if mismatch:
        ratio = 1.0 / mismatch if mismatch < 1 else mismatch
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=FAINT,
                marker=".",
                linestyle="",
                markersize=9,
                label="measured envelope (peak-to-peak)",
            ),
            Line2D(
                [],
                [],
                color=GREEN,
                linewidth=2.6,
                label="log-linear envelope slope (incoherent)",
            ),
            Line2D(
                [],
                [],
                color=RED,
                linewidth=2.6,
                linestyle="--",
                label="coherent single-frequency fit",
            ),
        ],
        loc="lower left",
    )
    ax.text(
        0.66,
        0.70,
        f"slope {ratio:.1f}$\\times$ too steep" if ratio else "wrong decay rate",
        transform=ax.transAxes,
        color=RED,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.tight_layout()
    save(fig, "overlay-real.pdf")


# ===========================================================================
# 4. Pre/Post time series: wrong vs reference envelopes
# ===========================================================================
def fig_numbers_traces(arr: dict, sc: dict) -> None:
    """Canonical Pre-Vibe and Post-Vibe records with coherent and reference envelopes."""
    fig, axes = panels(1, 2, figsize=(12.4, 4.4), sharey=True)
    for ax, key in zip(axes, ("pre", "post"), strict=True):
        pipe = sc[key]["canonical"]["pipeline"]
        f_fit = float(pipe["f_nls"] or sc["f_init"])
        q_wrong = float(pipe["Q_profile_raw"])
        # Same reference as the numbers table: incoherent Q on this canonical window.
        q_ref = float(pipe["Q_demod"] or sc[key]["canonical"]["envelope"]["Q"])
        t = arr[f"{key}_canon_t"]
        a0 = float(arr[f"{key}_env_a"][np.isfinite(arr[f"{key}_env_a"])][0])
        band(ax, t, arr[f"{key}_canon_lo"], arr[f"{key}_canon_hi"], color="#D8D8D8")
        te = np.linspace(0.0, float(t[-1]), 500)
        e_wrong = envelope_from_q(te, a0, f_fit, q_wrong)
        e_ref = envelope_from_q(te, a0, f_fit, q_ref)
        for sign in (1.0, -1.0):
            ax.plot(
                hours(te),
                sign * e_wrong,
                color=RED,
                linewidth=1.7,
                linestyle="--",
                label="coherent fit" if sign > 0 else None,
            )
            ax.plot(
                hours(te),
                sign * e_ref,
                color=GREEN,
                linewidth=1.6,
                label="incoherent reference" if sign > 0 else None,
            )
        ax.set_xlabel("Time (h)")
        ax.set_ylim(-750, 750)
        ax.text(
            0.03,
            0.94,
            sc[key]["label"],
            transform=ax.transAxes,
            va="top",
            fontsize=13.5,
            fontweight="bold",
            color=MUTE,
        )
        err = q_ref / q_wrong
        ax.text(
            0.97,
            0.08,
            f"coherent $Q$ is {err:.1f}$\\times$ low",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12.5,
            color=RED,
            fontweight="bold",
        )
        if key == "pre":
            ax.legend(loc="upper right", fontsize=11)
            ax.set_ylabel("Signal (cycles)")
    fig.tight_layout()
    save(fig, "pre-post-traces.pdf")


# ===========================================================================
# 5 / 15. Window sweep, before and after
# ===========================================================================
def _sweep_arrays(rows: list[dict], field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dur = np.array([r["duration_h"] for r in rows], dtype=float)
    start = np.array([r["start"] for r in rows], dtype=float)
    val = np.array([np.nan if r.get(field) is None else r[field] for r in rows], dtype=float)
    return dur, start, val


def _q_band(sc: dict, key: str) -> tuple[float, float]:
    """Measured local-Q range from the matched amplitude bands."""
    qs = [b["Q"] for b in sc[key]["matched"]["bands"] if b["Q"] is not None]
    return (min(qs), max(qs)) if qs else (np.nan, np.nan)


def _log_axis_plain(ax, axis: str = "x") -> None:
    """Plain numeric labels on a log axis, with no minor-tick labels."""
    target = ax.get_xaxis() if axis == "x" else ax.get_yaxis()
    target.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    target.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _spread(values: np.ndarray) -> float:
    v = values[np.isfinite(values) & (values > 0)]
    return float(v.max() / v.min()) if len(v) > 1 else np.nan


# Early-release start offsets used in the cache rebuild.
_SWEEP_START_STYLE = {
    0.0: ("o", BLUE),
    20.0: ("s", "#4C78A8"),
    40.0: ("D", "#72B7B2"),
    60.0: ("^", ORANGE),
    80.0: ("v", "#E45756"),
    100.0: ("P", RED),
}


def fig_window_sweep(arr: dict, sc: dict, *, after: bool) -> None:
    """
    Window sweep as a ratio to the drift-immune reference.

    Starts are early offsets an experimenter would try (0–100 s). Duration varies
    1–6 h. Plotting the ratio rather than absolute Q puts both records on one axis.
    """
    colour = GREEN if after else RED
    field = "Q_selected" if after else "Q_profile_raw"

    fig, axes = panels(1, 2, figsize=(12.4, 4.5), sharey=True)
    for ax, key in zip(axes, ("pre", "post"), strict=True):
        rows = sc[key]["sweep"]
        ref = sc[key]["demod"]["Q"]
        lo, hi = _q_band(sc, key)
        ax.axhspan(lo / ref, hi / ref, color=FAINT, alpha=0.40, zorder=0)
        ax.axhline(1.0, color=INK, linewidth=1.5, zorder=1)

        dur, start, val = _sweep_arrays(rows, field)
        val = val / ref
        starts = sorted(set(float(s) for s in start))
        for s in starts:
            marker, mk_colour = _SWEEP_START_STYLE.get(s, ("o", colour))
            m = start == s
            ax.plot(
                dur[m],
                val[m],
                marker,
                color=mk_colour if not after else colour,
                markersize=8.5,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=3,
            )

        missing = ~np.isfinite(val)
        if after and missing.any():
            ax.plot(
                dur[missing],
                np.full(missing.sum(), 0.068),
                "x",
                color=MUTE,
                markersize=11,
                markeredgewidth=2.4,
                zorder=3,
                clip_on=False,
            )

        ax.set_yscale("log")
        ax.set_ylim(0.055, 6.5)
        ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(["0.1", "0.25", "0.5", "1", "2", "4"])
        ax.set_xlim(0.4, 6.6)
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xlabel("Window duration (h)")
        ax.text(
            0.035,
            0.955,
            sc[key]["label"],
            transform=ax.transAxes,
            va="top",
            fontsize=13.5,
            fontweight="bold",
            color=MUTE,
        )

        sp = _spread(val)
        med = float(np.nanmedian(val[np.isfinite(val)])) if np.isfinite(val).any() else np.nan
        lines = []
        if np.isfinite(sp):
            lines.append(f"spread $\\times${sp:.0f}" if sp > 10 else f"spread $\\times${sp:.2f}")
        if not after and np.isfinite(med) and (med < 0.75 or med > 1.4):
            lines.append(f"median {med:.2f}$\\times$")
        if lines:
            ax.text(
                0.965,
                0.045,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=13.5,
                fontweight="bold",
                color=colour,
            )

    axes[0].set_ylabel("Fitted $Q$ / reference $Q$")
    handles = [
        Line2D(
            [],
            [],
            color=c if not after else colour,
            marker=m,
            linestyle="",
            markersize=8,
            label=f"start {int(s)} s",
        )
        for s, (m, c) in _SWEEP_START_STYLE.items()
    ]
    handles.extend(
        [
            Line2D([], [], color=INK, linewidth=1.5, label="incoherent reference"),
            mpatches.Patch(color=FAINT, alpha=0.55, label="physical $Q(A)$ range"),
        ]
    )
    if after:
        handles.append(
            Line2D(
                [],
                [],
                color=MUTE,
                marker="x",
                linestyle="",
                markersize=10,
                markeredgewidth=2.4,
                label="refused: no finite $Q$",
            )
        )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4 if after else 4,
        fontsize=10.5,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    save(fig, "window-sweep-after.pdf" if after else "window-sweep-before.pdf")


# ===========================================================================
# 6 / 10. The three assumptions (early: questions; late: measured answers)
# ===========================================================================
def fig_assumptions(arr: dict, sc: dict, *, late: bool = False) -> None:
    pre, post = sc["pre"]["demod"], sc["post"]["demod"]
    drift_pre = 1e3 * abs(sc["pre"]["matched"]["drift_hz"])
    drift_post = 1e3 * abs(sc["post"]["matched"]["drift_hz"])
    q_lo, q_hi = _q_band(sc, "pre")
    q_lo = min(q_lo, _q_band(sc, "post")[0])
    q_hi = max(q_hi, _q_band(sc, "post")[1])
    plateau = 0.5 * (pre["plateau_amplitude"] + post["plateau_amplitude"])

    fig, ax = panels(figsize=(12.0, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.90,
        r"$x(t) = A_0\, e^{-t/\tau} \cos(2\pi f t + \phi) + c$",
        ha="center",
        va="center",
        fontsize=23,
    )
    ax.text(
        0.5,
        0.76,
        "phase-coherent single-exponential model used by every coherent fit",
        ha="center",
        va="center",
        fontsize=13,
        color=MUTE,
    )

    if late:
        rows = [
            (
                "constant $f$",
                f"drifts {drift_pre:.1f} / {drift_post:.1f} mHz\nduring the decay",
                "physics to report",
                GREEN,
            ),
            (
                "single $\\tau$",
                f"local $Q$ runs {q_lo / 1e3:.0f}k to {q_hi / 1e3:.0f}k\nwith amplitude",
                "physics to report",
                GREEN,
            ),
            (
                "decays to zero",
                f"settles at $\\approx${plateau:.0f} cycles\nof driven oscillation",
                "nuisance for fitting",
                ORANGE,
            ),
        ]
    else:
        rows = [
            ("constant $f$", "does $f$ hold\nacross the window?", "assumption", BLUE),
            ("single $\\tau$", "is one decay time\nenough?", "assumption", BLUE),
            ("decays to zero", "does the amplitude\nreach the noise floor?", "assumption", BLUE),
        ]

    for i, (assumption, detail, tag, tag_colour) in enumerate(rows):
        x = 0.18 + 0.32 * i
        ax.text(x, 0.58, assumption, ha="center", fontsize=15.5, fontweight="bold", color=BLUE)
        ax.annotate(
            "",
            xy=(x, 0.42),
            xytext=(x, 0.52),
            arrowprops=dict(arrowstyle="->", color=RED if late else MUTE, linewidth=1.8),
        )
        ax.text(
            x,
            0.30,
            detail,
            ha="center",
            va="center",
            fontsize=13,
            color=RED if late else MUTE,
        )
        ax.text(
            x,
            0.14,
            tag,
            ha="center",
            va="center",
            fontsize=12,
            color=tag_colour,
            fontweight="bold",
        )

    ax.text(
        0.5,
        0.02,
        "broadband noise: 1.3–1.9 cycles RMS against a 600-cycle release   (SNR $\\approx$ 400)",
        ha="center",
        fontsize=12.5,
        color=MUTE,
    )
    save(fig, "assumptions-card-late.pdf" if late else "assumptions-card.pdf")


# ===========================================================================
# 7. Frequency pull (real vs ideal synthetic)
# ===========================================================================
def _decay_segments(arr: dict, key: str, prefix: str = "matched") -> tuple[np.ndarray, ...]:
    mask = arr[f"{key}_{prefix}_decay_mask"].astype(bool)
    return (
        arr[f"{key}_{prefix}_t_mid"][mask],
        arr[f"{key}_{prefix}_f_seg"][mask],
        arr[f"{key}_{prefix}_amplitude"][mask],
    )


def _synth_segment_tone(
    ts: np.ndarray, xs: np.ndarray, fs: float, f0: float, seg: float = 120.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crude per-segment amplitude and frequency for an ideal synthetic reference."""
    n_seg = int(seg * fs)
    t_mid, f_seg, amp = [], [], []
    for i0 in range(0, len(ts) - n_seg, n_seg):
        tw = ts[i0 : i0 + n_seg]
        xw = xs[i0 : i0 + n_seg]
        # FFT peak near f0
        spec = np.fft.rfft(xw - np.mean(xw))
        freq = np.fft.rfftfreq(len(xw), d=1.0 / fs)
        band = (freq > f0 - 0.15) & (freq < f0 + 0.15)
        if not np.any(band):
            continue
        k = np.argmax(np.abs(spec[band]))
        f_hat = float(freq[band][k])
        # Peak-to-peak amplitude proxy
        a_hat = 0.5 * (np.percentile(xw, 95) - np.percentile(xw, 5))
        t_mid.append(0.5 * (tw[0] + tw[-1]))
        f_seg.append(f_hat)
        amp.append(a_hat)
    return np.asarray(t_mid), np.asarray(f_seg), np.asarray(amp)


def fig_drift(arr: dict, sc: dict) -> None:
    """
    Frequency versus amplitude for real records, with an ideal synthetic panel.

    The synthetic is matched to Pre-Vibe SNR and Q, so a constant-frequency model
    should produce a flat drift trace there.
    """
    ts, xs, p = _generate_matched_synth(duration_h=6.0, seed=2)
    st, sf, sa = _synth_segment_tone(ts, xs, p["fs"], p["f0"], seg=120.0)

    fig, axes = panels(1, 3, figsize=(12.8, 4.2), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})

    # Ideal synthetic: f vs A should be flat
    ax = axes[0]
    f0s = float(sf[np.argmax(sa)])
    ax.plot(sa, 1e3 * (sf - f0s), "o", color=GREEN, markersize=4.5, alpha=0.85)
    ax.axhline(0.0, color=INK, linewidth=1.0)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylabel("Frequency rise (mHz)")
    ax.set_xlabel("Amplitude (cycles)")
    ax.set_ylim(-0.25, 1.75)
    ax.text(
        0.03,
        0.95,
        "ideal synthetic\n(matched SNR)",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        color=MUTE,
    )
    ax.text(
        0.03,
        0.12,
        "flat: constant $f$",
        transform=ax.transAxes,
        fontsize=12,
        color=GREEN,
        fontweight="bold",
    )

    # Real: f vs A
    ax = axes[1]
    for key, colour in (("pre", BLUE), ("post", ORANGE)):
        t, f, a = _decay_segments(arr, key)
        f0 = float(f[np.argmax(a)])
        drift = 1e3 * abs(sc[key]["matched"]["drift_hz"] or 0.0)
        ax.plot(
            a,
            1e3 * (f - f0),
            "o",
            color=colour,
            markersize=4.8,
            alpha=0.8,
            label=f"{sc[key]['label']}  $+${drift:.2f} mHz",
        )
        pull = sc[key]["matched"]["f_pull_per_efold"]
        if pull:
            xs_line = np.array([a.min(), a.max()])
            ax.plot(
                xs_line,
                1e3 * pull * (np.log(xs_line) - np.log(a.max())),
                color=colour,
                linewidth=2.0,
                linestyle="--",
            )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([500, 300, 200, 100, 60])
    _log_axis_plain(ax)
    ax.set_xlabel("Amplitude (cycles), falling right")
    ax.set_ylim(-0.25, 1.75)
    ax.legend(loc="upper left", fontsize=10.5)
    ax.text(
        0.97,
        0.95,
        "ODIN EDU",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color=MUTE,
    )

    # Real: same drift vs time
    ax = axes[2]
    for key, colour in (("pre", BLUE), ("post", ORANGE)):
        t, f, a = _decay_segments(arr, key)
        f0 = float(f[np.argmax(a)])
        ax.plot(hours(t), 1e3 * (f - f0), "-", color=colour, linewidth=1.8)
    ax.set_xlabel("Time (h)")
    ax.set_ylim(-0.25, 1.75)
    ax.text(
        0.03,
        0.95,
        "same drift vs time",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        color=MUTE,
    )
    ax.text(
        0.03,
        0.72,
        "1 mHz = 1 cycle of\nphase slip per 1000 s",
        transform=ax.transAxes,
        fontsize=11.5,
        color=RED,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "drift-pull.pdf")


# ===========================================================================
# 8. Q versus amplitude (real vs ideal synthetic)
# ===========================================================================
def fig_q_vs_amplitude(arr: dict, sc: dict) -> None:
    ts, xs, p = _generate_matched_synth(duration_h=6.0, seed=3)
    st, sf, sa = _synth_segment_tone(ts, xs, p["fs"], p["f0"], seg=120.0)
    # Floor-free corrected amplitude for synthetic: amplitude itself
    ok = sa > 5.0
    st, sa = st[ok], sa[ok]
    coef_s = np.polyfit(st, np.log(sa), 1)

    fig, axes = panels(1, 3, figsize=(12.8, 4.2))

    # Synthetic residual of one-exponential fit
    ax = axes[0]
    ax.plot(hours(st), np.log(sa) - np.polyval(coef_s, st), "o", color=GREEN, markersize=3.8)
    ax.axhline(0.0, color=INK, linewidth=1.0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Log-amplitude residual")
    ax.set_ylim(-0.35, 0.35)
    ax.text(
        0.03,
        0.95,
        "ideal synthetic",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        color=MUTE,
    )
    ax.text(
        0.03,
        0.12,
        "flat: single $\\tau$",
        transform=ax.transAxes,
        fontsize=12,
        color=GREEN,
        fontweight="bold",
    )

    # Real residual
    ax = axes[1]
    for key, colour in (("pre", BLUE), ("post", ORANGE)):
        mask = arr[f"{key}_matched_decay_mask"].astype(bool)
        t = arr[f"{key}_matched_t_mid"][mask]
        a = arr[f"{key}_matched_amplitude_corrected"][mask]
        ok = np.isfinite(a) & (a > 0)
        t, a = t[ok], a[ok]
        coef = np.polyfit(t, np.log(a), 1)
        ax.plot(
            hours(t),
            np.log(a) - np.polyval(coef, t),
            "o",
            color=colour,
            markersize=3.8,
            alpha=0.75,
            label=sc[key]["label"],
        )
    ax.axhline(0.0, color=INK, linewidth=1.0)
    ax.set_xlabel("Time (h)")
    ax.set_ylim(-0.35, 0.35)
    ax.legend(loc="lower left", fontsize=10.5)
    ax.text(
        0.03,
        0.95,
        "ODIN EDU: systematic curvature",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        color=MUTE,
    )

    # Local Q vs amplitude (Pre-Vibe)
    ax = axes[2]
    _plot_band_q(ax, sc, keys=("pre",))
    lo, hi = (v / 1e3 for v in _q_band(sc, "pre"))
    ax.axhline(p["q"] / 1e3, color=GREEN, linewidth=1.6, linestyle=":", label="synthetic $Q$")
    ax.legend(loc="upper left", fontsize=10.5)
    ax.text(
        0.04,
        0.55,
        f"Pre-Vibe $Q$:\n{lo:.0f}k to {hi:.0f}k",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        color=RED,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "q-vs-amplitude.pdf")


def _plot_band_q(ax, sc: dict, keys: tuple[str, ...] = ("pre", "post")) -> None:
    """Local Q against amplitude, using the shared matched bands for both records."""
    for key in keys:
        colour = BLUE if key == "pre" else ORANGE
        bands = [b for b in sc[key]["matched"]["bands"] if b["Q"] is not None]
        mid = np.array([b["mid"] for b in bands])
        q = np.array([b["Q"] for b in bands]) / 1e3
        order = np.argsort(mid)
        ax.plot(
            mid[order],
            q[order],
            "o-",
            color=colour,
            markersize=8,
            linewidth=2.2,
            label=sc[key]["label"],
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([400, 250, 150, 80])
    _log_axis_plain(ax)
    ax.set_xlabel("Amplitude band (cycles)")
    ax.set_ylabel(r"Local $Q$   ($\times 10^{3}$)")
    if len(keys) > 1:
        ax.legend(loc="upper left")


# ===========================================================================
# 9. Plateau (real vs ideal synthetic)
# ===========================================================================
def fig_plateau(arr: dict, sc: dict) -> None:
    ts, xs, p = _generate_matched_synth(duration_h=8.0, seed=4)
    st, sf, sa = _synth_segment_tone(ts, xs, p["fs"], p["f0"], seg=120.0)

    fig, axes = panels(1, 3, figsize=(12.8, 4.2), gridspec_kw={"width_ratios": [1.1, 1.35, 1]})

    ax = axes[0]
    ax.plot(hours(st), sa, "o-", color=GREEN, markersize=2.8, linewidth=1.0, alpha=0.9)
    ax.axhline(p["sigma"], color=MUTE, linewidth=1.3, linestyle=":", label="broadband RMS")
    ax.set_yscale("log")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Segment amplitude (cycles)")
    ax.set_ylim(0.8, 1800)
    ax.legend(loc="upper right", fontsize=10)
    ax.text(
        0.03,
        0.95,
        "ideal synthetic",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        color=MUTE,
    )
    ax.text(
        0.03,
        0.12,
        "decays into noise",
        transform=ax.transAxes,
        fontsize=12,
        color=GREEN,
        fontweight="bold",
    )

    ax = axes[1]
    for key, colour in (("pre", BLUE), ("post", ORANGE)):
        t = arr[f"{key}_demod_t_mid"]
        a = arr[f"{key}_demod_amplitude"]
        ax.plot(
            hours(t),
            a,
            "o-",
            color=colour,
            markersize=2.8,
            linewidth=1.0,
            alpha=0.85,
            label=sc[key]["label"],
        )
    floor = 0.5 * (
        sc["pre"]["demod"]["plateau_amplitude"] + sc["post"]["demod"]["plateau_amplitude"]
    )
    ax.axhline(
        floor,
        color=RED,
        linewidth=1.6,
        linestyle="--",
        label=f"driven plateau $\\approx${floor:.0f} cycles",
    )
    ax.axhline(
        1.6,
        color=MUTE,
        linewidth=1.3,
        linestyle=":",
        label="broadband noise $\\approx$1.6 cycles RMS",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Time (h)")
    ax.set_ylim(0.8, 1800)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.text(
        0.03,
        0.95,
        "ODIN EDU",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        color=MUTE,
    )

    ax = axes[2]
    ax.plot(arr["plateau_spec_freq"], arr["plateau_spec_amp"], color=INK, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude spectrum")
    ax.set_xticks([7.0, 7.67, 8.5])
    ax.set_xticklabels(["7.0", "7.67", "8.5"])
    ax.text(
        0.5,
        0.95,
        "plateau-only segment",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color=MUTE,
    )
    ax.text(
        0.03,
        0.78,
        "tone still present:\nnarrowband drive",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color=RED,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "plateau.pdf")


# ===========================================================================
# 10. Injection ladder
# ===========================================================================
# Controlled experiments E1-E4 from the executed companion notebook (§5.2),
# tabulated in the investigation report §9. Base record matched to Pre-Vibe:
# fs = 149.012 Hz, f0 = 7.6699 Hz, tau = 3700 s (Q_true = 8.92e4), A0 = 600,
# white sigma = 1.5 cycles, 3 h. Ratios are estimate / truth; every coherent
# value carried status "valid".
INJECTION = [
    ("nothing injected", 1.000, 1.000),
    ("amplitude-proportional pull", 1.90, 1.000),
    ("linear frequency drift", 1.62, 1.000),
    ("measured $f(t)$, noise seed A", 1.77, 1.000),
    ("measured $f(t)$, noise seed B", 1.77, 1.000),
    ("measured $f(t)$, other $\\tau$ init, seed A", 3.80, 1.000),
    ("measured $f(t)$, other $\\tau$ init, seed B", 0.51, 1.000),
    ("driven plateau only", 1.000, 0.983),
    ("pull + plateau + wander", 1.90, 0.984),
]


def fig_injection(arr: dict, sc: dict) -> None:
    labels = [r[0] for r in INJECTION]
    coherent = np.array([r[1] for r in INJECTION])
    incoherent = np.array([r[2] for r in INJECTION])
    y = np.arange(len(labels))[::-1]

    fig, ax = panels(figsize=(12.0, 4.5))
    ax.axvline(1.0, color=INK, linewidth=1.6, zorder=1)
    ax.barh(y + 0.19, coherent, height=0.34, color=RED, zorder=2, label="coherent single-$f$ fit")
    ax.barh(
        y - 0.19,
        incoherent,
        height=0.34,
        color=GREEN,
        zorder=2,
        label="segmented demodulation",
    )
    for yy, c in zip(y, coherent, strict=True):
        ax.text(
            c + 0.06,
            yy + 0.19,
            f"{c:.2f}$\\times$",
            va="center",
            fontsize=12,
            color=RED,
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Estimated $Q$ / true $Q$")
    ax.set_xlim(0, 4.9)
    ax.set_ylim(-0.8, len(labels) - 0.35)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    ax.text(1.08, -0.72, "truth", fontsize=12.5, color=INK, fontweight="bold", ha="left", va="bottom")
    fig.tight_layout()
    save(fig, "injection-ladder.pdf")


# ===========================================================================
# 11. Coherence budget
# ===========================================================================
# Fixed-frequency-error scan on the ideal record (notebook §5.3, report §9).
# Every point reported status "valid".
DF_TAU = np.array([0.0037, 0.011, 0.037, 0.111, 0.37, 1.11])
Q_RATIO = np.array([1.000, 0.998, 0.979, 0.838, 0.395, 0.142])


def fig_coherence(arr: dict, sc: dict) -> None:
    fig, ax = panels(figsize=(11.4, 5.0))
    ax.axhspan(0.98, 1.02, color=GREEN, alpha=0.16, zorder=0)
    ax.plot(DF_TAU, Q_RATIO, "o-", color=RED, markersize=9, linewidth=2.2, zorder=3)
    ax.axhline(1.0, color=INK, linewidth=1.0, zorder=1)
    ax.axvline(0.01, color=GREEN, linewidth=1.8, linestyle="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"coherence budget   $\delta\!f \cdot \tau$")
    ax.set_ylabel("Estimated $Q$ / true $Q$")
    ax.set_ylim(0, 1.25)

    ratios = [sc[k]["canonical"]["pipeline"]["coherence_ratio"] for k in ("pre", "post")]
    ratios = [r for r in ratios if r]
    op = max(ratios) if ratios else 4.3
    ax.set_xlim(2e-3, max(6.0, 1.6 * op))
    ax.axvline(op, color=ORANGE, linewidth=2.2, zorder=2)
    ax.text(
        op * 0.92,
        0.62,
        f"this resonator\n$\\delta\\!f\\cdot\\tau \\approx$ {op:.1f}",
        ha="right",
        fontsize=14,
        color=ORANGE,
        fontweight="bold",
    )
    ax.text(
        0.0115,
        0.10,
        "admissible\n$\\lesssim 0.01$",
        fontsize=13.5,
        color=GREEN,
        fontweight="bold",
    )
    ax.text(
        0.016,
        1.13,
        "every point still reports a finite $Q$",
        fontsize=12.5,
        color=MUTE,
    )
    fig.tight_layout()
    save(fig, "coherence-budget.pdf")


# ===========================================================================
# 12. Crop cascade
# ===========================================================================
# The two historical outcomes for the offset-9000 s specimen (report §4/§12):
# two selections of the same window differing only by float round-off gave
# tau_est = 35.6 s and 1051 s, cropping to 107 s and 3154 s of a 12 600 s
# window, while the pipeline's own envelope seed was 5698 s.
CROP_BEFORE = ((35.6, 107.0), (1051.0, 3154.0))
CROP_ENVELOPE_SEED = 5698.0


def fig_crop(arr: dict, sc: dict) -> None:
    """
    How much of the selected window each crop actually analyzed.

    The question is "how much data survived", so the lower panel draws retained
    extent directly as bars on the same time axis as the record above. Overlaying
    shaded regions on the trace hid the 107 s outcome entirely.
    """
    spec = sc["specimen"]
    pipe = spec["pipeline"]
    total_h = spec["duration"] / 3600.0
    crop_now = pipe.get("crop_duration") or spec["duration"]

    fig, axes = panels(
        2,
        1,
        figsize=(12.2, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.12},
    )

    ax = axes[0]
    ax.plot(
        hours(arr["specimen_env_t"]),
        arr["specimen_env_a"],
        ".",
        color=BLUE,
        markersize=3.0,
        alpha=0.55,
    )
    ax.set_ylabel("Amplitude\n(cycles)")
    ax.set_xlim(0, total_h)
    ax.text(
        0.985,
        0.94,
        "the selected window: 3.5 h of Post-Vibe data",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        color=MUTE,
    )

    ax = axes[1]
    rows = [
        (f"crop from $\\tau$ = {CROP_BEFORE[1][0]:.0f} s", CROP_BEFORE[1][1] / 3600.0, RED),
        (f"crop from $\\tau$ = {CROP_BEFORE[0][0]:.0f} s", CROP_BEFORE[0][1] / 3600.0, RED),
        ("3$\\times$ envelope seed", 3 * CROP_ENVELOPE_SEED / 3600.0, MUTE),
        ("what it keeps now", crop_now / 3600.0, GREEN),
    ]
    for i, (_label, extent, colour) in enumerate(rows):
        ax.barh(
            i,
            min(extent, total_h),
            height=0.55,
            color=colour,
            alpha=0.85 if colour != MUTE else 0.45,
            zorder=2,
        )
        if extent > total_h:
            ax.annotate(
                "",
                xy=(total_h * 1.005, i),
                xytext=(total_h * 0.96, i),
                arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.6),
            )
        shown = min(extent, total_h) * 3600.0
        text = f"{shown:,.0f} s".replace(",", "\u2009")
        if extent > total_h:
            text = "whole window"
        ax.text(
            min(extent, total_h) - 0.05 if extent > 1.5 else min(extent, total_h) + 0.05,
            i,
            text,
            va="center",
            fontsize=13,
            color="white" if extent > 1.5 else colour,
            fontweight="bold",
            ha="right" if extent > 1.5 else "left",
            zorder=3,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlabel("Data retained for the fit (h)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.text(
        0.985,
        0.06,
        "same analysis window; $\\tau$ seed differs at float round-off",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12.5,
        color=RED,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "crop-cascade.pdf")


# ===========================================================================
# 13. Method diagram over real segments
# ===========================================================================
def fig_method(arr: dict, sc: dict) -> None:
    fig, axes = panels(1, 2, figsize=(12.4, 4.7))
    t = arr["pre_demod_t_mid"]
    a = arr["pre_demod_amplitude"]
    ac = arr["pre_demod_amplitude_corrected"]
    mask = arr["pre_demod_decay_mask"].astype(bool)
    floor = sc["pre"]["demod"]["plateau_amplitude"]
    seg = sc["pre"]["demod"]["seg_duration"]

    # --- left: segmentation
    ax = axes[0]
    # The demodulated envelope drawn symmetrically: the raw min/max band read as a
    # trumpet opening the wrong way once the plateau dominated it.
    band(ax, t, -a, a, color="#E4E4E4")
    for edge in np.arange(0, hours(t)[-1], hours(np.array([seg]))[0])[::4]:
        ax.axvline(edge, color=MUTE, linewidth=0.5, alpha=0.7)
    ax.plot(hours(t), a, "o", color=BLUE, markersize=4.5)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Signal (cycles)")
    ax.text(
        0.97,
        0.95,
        f"{sc['pre']['demod']['n_segments']} segments of {seg:.0f} s\n"
        r"each gives $(t_i, f_i, A_i, \sigma_i)$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13.5,
        color=MUTE,
    )

    # --- right: floor correction and the fitted region
    ax = axes[1]
    ax.plot(hours(t), a, "o", color=FAINT, markersize=4.5, label="measured $A_i$")
    ok = np.isfinite(ac) & (ac > 0)
    ax.plot(
        hours(t[ok & mask]),
        ac[ok & mask],
        "o",
        color=BLUE,
        markersize=5.5,
        label=r"floor-corrected $\sqrt{A_i^2 - A_{\rm floor}^2}$",
        zorder=3,
    )
    slope = sc["pre"]["demod"]["log_slope"]
    intercept = sc["pre"]["demod"]["log_intercept"]
    if slope and intercept:
        tt = t[mask]
        ax.plot(
            hours(tt),
            np.exp(intercept + slope * tt),
            color=GREEN,
            linewidth=3.0,
            zorder=4,
            label="robust log-linear fit",
        )
    ax.axhline(floor, color=RED, linewidth=1.8, linestyle="--", zorder=2)
    ax.axhline(3 * floor, color=GREEN, linewidth=1.6, linestyle=":", zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Amplitude (cycles)")
    ax.set_ylim(3, 1200)
    # Right-hand labels: the decay occupies the left of the panel, so the plateau
    # tail is the only place these can sit without covering data.
    ax.text(
        0.02,
        floor * 0.62,
        f"$A_{{\\rm floor}} = {floor:.0f}$ cycles",
        transform=ax.get_yaxis_transform(),
        color=RED,
        fontsize=13,
        va="top",
        ha="left",
        fontweight="bold",
    )
    ax.text(
        0.985,
        3 * floor * 1.25,
        r"fit region: $A > 3 A_{\rm floor}$",
        transform=ax.get_yaxis_transform(),
        color=GREEN,
        fontsize=13,
        va="bottom",
        ha="right",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=11.5)
    fig.tight_layout()
    save(fig, "demod-method.pdf")


# ===========================================================================
# 14. Gate decision flow
# ===========================================================================
def fig_gates(arr: dict, sc: dict) -> None:
    fig, ax = panels(figsize=(12.6, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 13.85)
    ax.set_ylim(-0.45, 4.0)

    def box(x, y, w, h, text, colour, fontsize=13.5, bold=False, fill="white"):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.10,rounding_size=0.12",
                linewidth=1.8,
                edgecolor=colour,
                facecolor=fill,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=colour,
            fontweight="bold" if bold else "normal",
        )

    def arrow(x0, y0, x1, y1, colour=MUTE, label=None, label_dy=0.16):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.8),
        )
        if label:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + label_dy,
                label,
                ha="center",
                fontsize=12,
                color=colour,
            )

    # The checks are applied in sequence, so the diagram is one chain with
    # refusals dropping out of it, not three independent branches off one node.
    y, h, fs = 2.40, 1.15, 11.5
    box(0.05, y, 2.05, h, "measured\ndiagnostics", MUTE, bold=True, fill="#F4F4F4", fontsize=fs)
    box(2.55, y, 2.45, h, "usable\ndynamic\nrange?", BLUE, fontsize=fs)
    box(5.45, y, 2.55, h, "coherent decay?\n" r"$\delta\!f\cdot\tau < 0.01$", BLUE, fontsize=fs)
    box(8.45, y, 2.60, h, "estimators\nagree to 1.5$\\times$?", BLUE, fontsize=fs)
    box(11.45, y, 2.35, h, "report $Q$\n$+$ CI", GREEN, bold=True, fontsize=fs)

    mid = y + h / 2
    for x0, x1 in ((2.10, 2.55), (5.00, 5.45), (8.00, 8.45), (11.05, 11.45)):
        arrow(x0, mid, x1, mid, colour=GREEN if x0 > 10 else BLUE)

    box(
        2.30,
        0.40,
        2.95,
        0.95,
        "no finite $Q$:\nplateau-\ndominated",
        MUTE,
        bold=True,
        fill="#F0F0F0",
        fontsize=fs,
    )
    box(8.30, 0.40, 2.95, 0.95, "no finite $Q$:\nthey\ndisagree", RED, bold=True, fontsize=fs)
    arrow(3.78, y, 3.78, 1.40, colour=MUTE)
    ax.text(3.88, 1.80, "no", fontsize=11.5, color=MUTE)
    arrow(9.75, y, 9.75, 1.40, colour=RED)
    ax.text(9.85, 1.80, "no", fontsize=11.5, color=RED)

    ax.text(
        6.72,
        y - 0.16,
        "yes $\\rightarrow$ coherent $Q$\nno $\\rightarrow$ demodulated $Q$",
        ha="center",
        va="top",
        fontsize=fs,
        color=MUTE,
    )

    ax.text(
        6.9,
        -0.40,
        "each check can refuse: the result is a number, a limit, or no $Q$ at all",
        ha="center",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
    )
    save(fig, "gates-flow.pdf")


# ===========================================================================
# 16. Pre versus Post
# ===========================================================================
def fig_pre_post(arr: dict, sc: dict) -> None:
    fig, axes = panels(1, 2, figsize=(12.4, 4.7), gridspec_kw={"width_ratios": [1.35, 1]})

    # --- left: local Q at matched amplitude
    ax = axes[0]
    _plot_band_q(ax, sc)
    gains = []
    for pre_b, post_b in zip(
        sc["pre"]["matched"]["bands"], sc["post"]["matched"]["bands"], strict=True
    ):
        if pre_b["Q"] and post_b["Q"]:
            gains.append(100.0 * (post_b["Q"] / pre_b["Q"] - 1.0))
    if gains:
        ax.text(
            0.06,
            0.74,
            f"Post is higher in every band:\n$+${min(gains):.0f}% to $+${max(gains):.0f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color=GREEN,
            fontweight="bold",
        )

    # --- right: the frequency-pull coefficient
    ax = axes[1]
    pulls = [1e3 * (sc[key]["matched"]["f_pull_per_efold"] or 0.0) for key in ("pre", "post")]
    ax.bar([0, 1], pulls, width=0.45, color=[BLUE, ORANGE], zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([sc["pre"]["label"], sc["post"]["label"]], fontsize=13)
    ax.set_ylabel("Pull (mHz per amplitude e-fold)")
    ax.axhline(0.0, color=INK, linewidth=1.2, zorder=3)
    ax.set_xlim(-0.6, 1.6)
    span = max(abs(p) for p in pulls) or 1.0
    ax.set_ylim(-1.45 * span, 0.42 * span)
    for x, v in zip([0, 1], pulls, strict=True):
        ax.text(
            x,
            v - 0.05 * span,
            f"{v:.2f}",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )
    if pulls[1]:
        ax.text(
            0.5,
            0.055,
            f"weaker by $\\approx${abs(pulls[0] / pulls[1]):.0f}$\\times$",
            transform=ax.transAxes,
            ha="center",
            fontsize=14.5,
            color=GREEN,
            fontweight="bold",
        )
    fig.tight_layout()
    save(fig, "pre-post.pdf")


# ===========================================================================
# B2. Backup — spectra
# ===========================================================================
def fig_backup_spectra(arr: dict, sc: dict) -> None:
    if "decay_spec_freq" not in arr:
        print("skipping backup-spectra.pdf (no decay spectrum in cache)")
        return
    fig, ax = panels(figsize=(11.4, 4.8))
    ax.plot(
        arr["decay_spec_freq"],
        arr["decay_spec_amp"],
        color=BLUE,
        linewidth=1.2,
        label="early (decay) segment",
    )
    ax.plot(
        arr["plateau_spec_freq"],
        arr["plateau_spec_amp"],
        color=ORANGE,
        linewidth=1.2,
        label="late (plateau) segment",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude spectrum (cycles)")
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.06,
        "one dominant peak; nearest features $\\sim$$10^{5}\\times$ down",
        transform=ax.transAxes,
        fontsize=13.5,
        color=MUTE,
    )
    fig.tight_layout()
    save(fig, "backup-spectra.pdf")


def main() -> None:
    arr, sc = load_cache()
    fig_title_strip(arr, sc)
    fig_symptom(arr, sc)
    fig_overlay(arr, sc)
    fig_numbers_traces(arr, sc)
    fig_window_sweep(arr, sc, after=False)
    fig_window_sweep(arr, sc, after=True)
    fig_assumptions(arr, sc, late=False)
    fig_assumptions(arr, sc, late=True)
    fig_drift(arr, sc)
    fig_q_vs_amplitude(arr, sc)
    fig_plateau(arr, sc)
    fig_injection(arr, sc)
    fig_coherence(arr, sc)
    fig_crop(arr, sc)
    fig_method(arr, sc)
    fig_gates(arr, sc)
    fig_pre_post(arr, sc)
    fig_backup_spectra(arr, sc)


if __name__ == "__main__":
    main()
