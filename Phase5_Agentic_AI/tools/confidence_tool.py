"""Authoritative final-decision semantics.

The system intentionally does not expose a synthetic numerical "final
confidence".  LightGBM probabilities are source-specific probabilities, not
probabilities of the case-level resolved diagnosis.  A case-level probability
would require a separately validated/calibrated model, which this prototype
does not contain.
"""
from __future__ import annotations


def calculate_decision_confidence(state) -> tuple[None, str, str]:
    """Return no synthetic final probability and an auditable explanation."""
    resolution = str(
        (state.reanalysis_result or {}).get("resolution")
        or getattr(state, "final_resolution", "")
        or ""
    )
    if not resolution:
        fusion = state.fusion_result or {}
        if fusion.get("decision") == "CONFIRMED":
            resolution = "CONFIRMED_BY_MULTI_WINDOW_AGREEMENT"

    if resolution == "NO_CONFIRMED_FAULT":
        return (
            None,
            "not_established",
            "No calibrated case-level probability is reported. The result is "
            "NO_CONFIRMED_FAULT and does not prove that the bearing is healthy."
        )

    if resolution in {"RESOLVED_BY_STRONG_PHYSICS", "CONFIRMED_BY_STRONG_PHYSICS"}:
        return (
            None,
            "physics_resolution",
            "No calibrated case-level probability is reported. The resolution "
            "is supported by strong physics evidence and multi-window consistency."
        )

    if resolution in {"CONFIRMED_BY_INDEPENDENT_AGREEMENT", "CONFIRMED_BY_MULTI_WINDOW_AGREEMENT", "STRONG_AGREEMENT"}:
        return (
            None,
            "multi_source_agreement",
            "No calibrated case-level probability is reported. The resolution "
            "is supported by independent ML/physics agreement."
        )

    return (
        None,
        "not_established",
        "No calibrated case-level probability is established by this prototype."
    )
