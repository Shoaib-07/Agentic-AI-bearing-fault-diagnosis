from __future__ import annotations

from typing import Any, Dict, List


# =============================================================================
# PHASE 5.17.5 — REASONING TRACE & DECISION EXPLANATION
# =============================================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_upper(value: Any) -> str:
    if value is None:
        return ""
    return str(value).upper()


def _build_evidence_summary(
    ml_result: Dict[str, Any],
    physics_result: Dict[str, Any],
    evidence_result: Dict[str, Any],
) -> Dict[str, Any]:

    ml = ml_result if isinstance(ml_result, dict) else {}
    physics = physics_result if isinstance(physics_result, dict) else {}
    evidence = evidence_result if isinstance(evidence_result, dict) else {}

    return {
        "ml_diagnosis": ml.get("fault"),
        "ml_confidence": _safe_float(ml.get("confidence")),
        "ml_margin": _safe_float(ml.get("margin")),
        "physics_diagnosis": physics.get("diagnosis"),
        "physics_strength": _safe_upper(physics.get("strength")),
        "ml_physics_agreement": evidence.get("ml_physics_agreement"),
        "overall_evidence_quality": _safe_float(
            evidence.get("overall_evidence_quality")
        ),
        "quality_label": evidence.get("quality_label"),
    }


def _derive_evidence_state(summary: Dict[str, Any]) -> str:

    ml = summary.get("ml_diagnosis")
    physics = summary.get("physics_diagnosis")
    strength = summary.get("physics_strength")
    confidence = summary.get("ml_confidence")
    margin = summary.get("ml_margin")
    quality = summary.get("overall_evidence_quality")

    if ml is None and physics is None:
        return "NO_EVIDENCE"

    if ml is None:
        return "PHYSICS_ONLY"

    if physics is None:
        return "ML_ONLY"

    if ml != physics and strength == "STRONG":
        return "STRONG_CONFLICT"

    if (
        ml == physics
        and strength == "STRONG"
        and confidence >= 0.70
        and margin >= 0.10
        and quality >= 0.80
    ):
        return "STRONG_AGREEMENT"

    if quality >= 0.80:
        return "HIGH_QUALITY"

    if quality >= 0.60:
        return "MODERATE_QUALITY"

    return "LOW_QUALITY"


def _generate_decision_explanation(
    evidence_state: str,
    stopping_result: Dict[str, Any],
    adaptive_result: Dict[str, Any],
    reanalysis_result: Dict[str, Any] | None,
) -> str:

    stop = bool(stopping_result.get("stop"))
    adaptive_depth = adaptive_result.get("investigation_depth")

    if reanalysis_result:
        resolution = reanalysis_result.get("resolution")

        if resolution == "RESOLVED_BY_STRONG_PHYSICS":
            return (
                "Initial evidence was conflicting, so reanalysis was performed. "
                "The conflict was resolved by strong physics evidence."
            )

        if resolution == "STRONG_AGREEMENT":
            return (
                "Reanalysis confirmed agreement between ML and physics evidence."
            )

        if resolution == "NO_CONFIRMED_FAULT":
            return (
                "Reanalysis did not provide sufficient evidence for a "
                "confirmed fault."
            )

    if evidence_state == "STRONG_AGREEMENT" and stop:
        return (
            "ML and strong physics evidence agree with sufficient confidence "
            "and evidence quality, allowing safe stopping."
        )

    if evidence_state == "STRONG_CONFLICT":
        return (
            "ML and strong physics evidence disagree; investigation must "
            "continue until the conflict is resolved."
        )

    if evidence_state in ("LOW_QUALITY", "MODERATE_QUALITY"):
        return (
            f"Evidence quality is insufficient for immediate stopping; "
            f"{adaptive_depth.lower()} investigation is required."
        )

    if not stop:
        return (
            "Stopping conditions are not satisfied, so the investigation "
            "continues."
        )

    return (
        "The available evidence satisfies the defined stopping policy."
    )


def reasoning_trace_tool(
    ml_result: Dict[str, Any] | None,
    physics_result: Dict[str, Any] | None,
    evidence_result: Dict[str, Any] | None,
    adaptive_result: Dict[str, Any] | None,
    stopping_result: Dict[str, Any] | None,
    reanalysis_result: Dict[str, Any] | None = None,
    action_history: List[str] | None = None,
) -> Dict[str, Any]:

    try:

        ml = ml_result if isinstance(ml_result, dict) else {}
        physics = (
            physics_result
            if isinstance(physics_result, dict)
            else {}
        )
        evidence = (
            evidence_result
            if isinstance(evidence_result, dict)
            else {}
        )
        adaptive = (
            adaptive_result
            if isinstance(adaptive_result, dict)
            else {}
        )
        stopping = (
            stopping_result
            if isinstance(stopping_result, dict)
            else {}
        )

        summary = _build_evidence_summary(
            ml,
            physics,
            evidence,
        )

        evidence_state = _derive_evidence_state(summary)

        explanation = _generate_decision_explanation(
            evidence_state,
            stopping,
            adaptive,
            reanalysis_result,
        )

        trace = {
            "evidence_state": evidence_state,
            "evidence_summary": summary,
            "adaptive_investigation": {
                "required": adaptive.get("investigation_required"),
                "depth": adaptive.get("investigation_depth"),
                "recommended_windows": adaptive.get(
                    "recommended_windows"
                ),
                "reason": adaptive.get("reason"),
            },
            "stopping_decision": {
                "stop": stopping.get("stop"),
                "decision": stopping.get("decision"),
                "reason": stopping.get("reason"),
                "safety_status": stopping.get("safety_status"),
            },
            "reanalysis": {
                "performed": bool(reanalysis_result),
                "resolution": (
                    reanalysis_result.get("resolution")
                    if isinstance(reanalysis_result, dict)
                    else None
                ),
            },
            "action_history": (
                list(action_history)
                if isinstance(action_history, list)
                else []
            ),
            "decision_explanation": explanation,
            "safety_constraints": [
                "No unsupported fault confirmation.",
                "Unresolved strong conflict requires continued investigation.",
                "Missing evidence does not permit stopping.",
                "Stopping logic failure defaults to continued investigation.",
                "The system is decision support only and does not control plant equipment.",
            ],
        }

        return {
            "status": "success",
            "trace": trace,
        }

    except Exception as exc:

        return {
            "status": "error",
            "trace": {
                "evidence_state": "UNKNOWN",
                "decision_explanation": (
                    "Reasoning trace generation failed."
                ),
                "safety_constraints": [
                    "Continue investigation when trace generation fails.",
                    "Do not make unsupported autonomous decisions.",
                ],
            },
            "error": str(exc),
        }