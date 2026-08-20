"""
Estimator selection for user-facing Q values (P3).

The analysis pipeline runs several Q estimators with different assumptions:

- ``Q_profile`` (and NLS/DFT): globally phase-coherent fits — exact on short,
  drift-free records, catastrophically biased under frequency drift.
- ``Q_demod``: incoherent segmented demodulation — drift-immune, models the
  driven plateau, needs a record long enough for ~8+ segments.
- ``Q_envelope``: incoherent envelope slope — robust but plateau-biased on
  long windows.

This module implements the lightweight regime classifier recommended by the
2026-08-18 investigation: the record's own demod diagnostics decide which
estimator is trustworthy, and cross-estimator agreement within tolerance is a
hard condition for a valid finite Q. Plateau-dominated windows never yield a
finite Q.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Cross-estimator agreement tolerance: two finite valid Q values must agree
#: within this max/min ratio for the selected Q to remain valid.
AGREEMENT_RATIO_TOLERANCE = 1.5

#: A record is "short" (coherent-fit friendly) when its duration is below
#: this multiple of the decay time.
SHORT_RECORD_TAU_MULTIPLE = 1.5


@dataclass(frozen=True)
class QSelection:
    """Selected user-facing Q estimate with regime and agreement metadata."""

    #: Selected Q value (None when no estimator is trustworthy).
    Q: float | None
    #: Which estimator supplied Q: "demod", "profile", or "none".
    source: str
    #: Record regime: "coherent_short", "long_or_drifting",
    #: "plateau_dominated", or "indeterminate".
    regime: str
    valid: bool
    status: str
    reasons: list[str]
    #: Max/min ratio between the selected Q and the other valid finite Q
    #: estimates (None when there is nothing to compare against).
    agreement_ratio: float | None


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and np.isfinite(value) and value > 0


def _classify_regime(result: dict) -> str:
    """Classify the record from the demod diagnostics."""
    if result.get("Q_demod_status") == "plateau_dominated":
        return "plateau_dominated"
    if result.get("coherence_gate_fired"):
        return "long_or_drifting"

    tau_ref = None
    for key in ("tau_demod", "tau_envelope_precrop", "tau_envelope", "tau_est"):
        value = result.get(key)
        if _finite(value):
            tau_ref = float(value)  # type: ignore[arg-type]
            break
    duration = result.get("T")
    if tau_ref is None or not _finite(duration):
        return "indeterminate"
    if float(duration) <= SHORT_RECORD_TAU_MULTIPLE * tau_ref:  # type: ignore[arg-type]
        return "coherent_short"
    return "long_or_drifting"


def select_q_estimate(
    result: dict,
    *,
    agreement_tolerance: float = AGREEMENT_RATIO_TOLERANCE,
) -> QSelection:
    """
    Select the user-facing Q estimate for one pipeline result.

    Parameters:
    -----------
    result : dict
        Result dictionary from RingDownAnalyzer (must contain the Q_demod*,
        Q_profile* and Q_envelope* fields).
    agreement_tolerance : float
        Max/min ratio within which independent valid Q estimates must agree
        for the selection to stay valid.

    Returns:
    --------
    QSelection
        Selected value with source, regime, validity and agreement metadata.

    Selection rules (investigation §19/P3):

    - plateau-dominated window: limits only, never a finite Q;
    - drifting or long record: the drift-immune demod Q;
    - short coherent record: the profile Q (falling back to demod);
    - any selected Q that disagrees with another valid finite estimate by
      more than the tolerance is demoted to a non-valid warning.
    """
    regime = _classify_regime(result)

    demod_q = result.get("Q_demod")
    demod_valid = bool(result.get("Q_demod_valid")) and _finite(demod_q)
    profile_q = result.get("Q_profile")
    profile_valid = bool(result.get("Q_profile_valid")) and _finite(profile_q)

    reasons: list[str] = []
    if regime == "plateau_dominated":
        return QSelection(
            Q=None,
            source="none",
            regime=regime,
            valid=False,
            status="limit",
            reasons=["plateau_dominated_limits_only"],
            agreement_ratio=None,
        )

    if regime == "coherent_short" and profile_valid:
        source, q_value = "profile", float(profile_q)  # type: ignore[arg-type]
    elif demod_valid:
        source, q_value = "demod", float(demod_q)  # type: ignore[arg-type]
    elif profile_valid:
        # Long record without a usable demod estimate: a valid (gated)
        # profile Q is still the best available number, flagged below by the
        # agreement check when contradicted.
        source, q_value = "profile", float(profile_q)  # type: ignore[arg-type]
    else:
        demod_status = str(result.get("Q_demod_status", "invalid"))
        return QSelection(
            Q=None,
            source="none",
            regime=regime,
            valid=False,
            status="invalid",
            reasons=["no_trustworthy_estimator", f"demod_status_{demod_status}"],
            agreement_ratio=None,
        )

    # Cross-estimator agreement: hard condition for a valid finite Q.
    others: list[float] = []
    if source != "demod" and demod_valid:
        others.append(float(demod_q))  # type: ignore[arg-type]
    if source != "profile" and profile_valid:
        others.append(float(profile_q))  # type: ignore[arg-type]
    envelope_q = result.get("Q_envelope")
    if bool(result.get("Q_envelope_valid")) and _finite(envelope_q):
        others.append(float(envelope_q))  # type: ignore[arg-type]

    agreement_ratio: float | None = None
    if others:
        ratios = [max(q_value / other, other / q_value) for other in others]
        agreement_ratio = float(max(ratios))

    valid = True
    status = "valid"
    if agreement_ratio is not None and agreement_ratio > agreement_tolerance:
        valid = False
        status = "warning"
        reasons.append("cross_estimator_disagreement")
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                "q_selection_disagreement",
                extra={
                    "event": "q_selection_disagreement",
                    "source": source,
                    "q_selected": q_value,
                    "agreement_ratio": agreement_ratio,
                },
            )

    return QSelection(
        Q=q_value if valid else None,
        source=source,
        regime=regime,
        valid=valid,
        status=status,
        reasons=reasons,
        agreement_ratio=agreement_ratio,
    )
