from __future__ import annotations

from typing import Any, Dict


# =============================================================================
# PHASE 5.17.3 — ADAPTIVE INVESTIGATION DEPTH
# =============================================================================

DEFAULT_CONFIG = {
    # Evidence quality thresholds
    "high_quality": 0.80,
    "moderate_quality": 0.60,

    # ML thresholds
    "strong_ml_confidence": 0.70,
    "strong_ml_margin": 0.10,

    # Physics
    "strong_physics": "STRONG",

    # Investigation limits
    "minimum_windows": 1,
    "standard_windows": 3,
    "maximum_windows": 5,
}


# =============================================================================
# SAFE HELPERS
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


# =============================================================================
# MAIN ADAPTIVE DECISION
# =============================================================================

def adaptive_investigation_tool(
    ml_result: Dict[str, Any] | None,
    physics_result: Dict[str, Any] | None,
    evidence_quality_result: Dict[str, Any] | None,
    reanalysis_result: Dict[str, Any] | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

    cfg = dict(DEFAULT_CONFIG)

    if config:
        cfg.update(config)

    try:
        ml = ml_result if isinstance(ml_result, dict) else {}
        physics = physics_result if isinstance(physics_result, dict) else {}
        evidence = (
            evidence_quality_result
            if isinstance(evidence_quality_result, dict)
            else {}
        )
        reanalysis = (
            reanalysis_result
            if isinstance(reanalysis_result, dict)
            else {}
        )

        # ---------------------------------------------------------------------
        # Extract evidence
        # ---------------------------------------------------------------------

        overall_quality = _safe_float(
            evidence.get("overall_evidence_quality"), 0.0
        )

        quality_label = str(
            evidence.get("quality_label", "VERY_LOW")
        ).upper()

        ml_confidence = _safe_float(
            ml.get("confidence"), 0.0
        )

        ml_margin = _safe_float(
            ml.get("margin"), 0.0
        )

        physics_strength = _safe_upper(
            physics.get("strength")
        )

        ml_diagnosis = ml.get("fault")
        physics_diagnosis = physics.get("diagnosis")

        agreement = (
            ml_diagnosis is not None
            and physics_diagnosis is not None
            and ml_diagnosis == physics_diagnosis
        )

        conflict = (
            ml_diagnosis is not None
            and physics_diagnosis is not None
            and ml_diagnosis != physics_diagnosis
        )

        # ---------------------------------------------------------------------
        # Case 1 — Missing evidence
        # ---------------------------------------------------------------------

        if not ml or ml.get("status") != "success":
            return {
                "status": "success",
                "investigation_required": True,
                "investigation_depth": "MINIMUM",
                "recommended_windows": 1,
                "reason": "ML evidence is missing or invalid.",
                "stop_allowed": False,
                "safety_status": "SAFE",
            }

        if not physics or physics.get("status") != "success":
            return {
                "status": "success",
                "investigation_required": True,
                "investigation_depth": "MINIMUM",
                "recommended_windows": 1,
                "reason": "Physics evidence is missing or invalid.",
                "stop_allowed": False,
                "safety_status": "SAFE",
            }

        # ---------------------------------------------------------------------
        # Case 2 — Strong conflict
        # Never reduce investigation for unresolved conflict.
        # ---------------------------------------------------------------------

        if conflict and physics_strength == cfg["strong_physics"]:

            return {
                "status": "success",
                "investigation_required": True,
                "investigation_depth": "MAXIMUM",
                "recommended_windows": cfg["maximum_windows"],
                "reason": (
                    "Strong ML/physics conflict requires maximum "
                    "multi-window investigation."
                ),
                "stop_allowed": False,
                "safety_status": "SAFE",
            }

        # ---------------------------------------------------------------------
        # Case 3 — Strong completed reanalysis
        # ---------------------------------------------------------------------

        if reanalysis.get("status") == "success":

            resolution = reanalysis.get("resolution")

            if resolution in (
                "STRONG_AGREEMENT",
                "RESOLVED_BY_STRONG_PHYSICS",
            ):
                return {
                    "status": "success",
                    "investigation_required": False,
                    "investigation_depth": "MINIMUM",
                    "recommended_windows": cfg["minimum_windows"],
                    "reason": (
                        "Reanalysis resolved the case with strong evidence."
                    ),
                    "stop_allowed": True,
                    "safety_status": "SAFE",
                }

            if resolution == "NO_CONFIRMED_FAULT":
                return {
                    "status": "success",
                    "investigation_required": False,
                    "investigation_depth": "MINIMUM",
                    "recommended_windows": cfg["minimum_windows"],
                    "reason": (
                        "Reanalysis found insufficient evidence for "
                        "a confirmed fault."
                    ),
                    "stop_allowed": True,
                    "safety_status": "SAFE",
                }

        # ---------------------------------------------------------------------
        # Case 4 — Strong agreement
        # ---------------------------------------------------------------------

        if (
            agreement
            and physics_strength == cfg["strong_physics"]
            and ml_confidence >= cfg["strong_ml_confidence"]
            and ml_margin >= cfg["strong_ml_margin"]
            and overall_quality >= cfg["high_quality"]
        ):

            return {
                "status": "success",
                "investigation_required": False,
                "investigation_depth": "MINIMUM",
                "recommended_windows": cfg["minimum_windows"],
                "reason": (
                    "High-quality ML and strong physics agree; "
                    "additional investigation is unnecessary."
                ),
                "stop_allowed": True,
                "safety_status": "SAFE",
            }

        # ---------------------------------------------------------------------
        # Case 5 — High evidence quality but not enough for immediate stop
        # ---------------------------------------------------------------------

        if overall_quality >= cfg["high_quality"]:

            return {
                "status": "success",
                "investigation_required": True,
                "investigation_depth": "STANDARD",
                "recommended_windows": cfg["standard_windows"],
                "reason": (
                    "Evidence quality is high, but additional "
                    "multi-window confirmation is recommended."
                ),
                "stop_allowed": False,
                "safety_status": "SAFE",
            }

        # ---------------------------------------------------------------------
        # Case 6 — Moderate evidence
        # ---------------------------------------------------------------------

        if overall_quality >= cfg["moderate_quality"]:

            return {
                "status": "success",
                "investigation_required": True,
                "investigation_depth": "STANDARD",
                "recommended_windows": cfg["standard_windows"],
                "reason": (
                    "Evidence is moderate; perform standard "
                    "multi-window investigation."
                ),
                "stop_allowed": False,
                "safety_status": "SAFE",
            }

        # ---------------------------------------------------------------------
        # Case 7 — Low evidence
        # ---------------------------------------------------------------------

        return {
            "status": "success",
            "investigation_required": True,
            "investigation_depth": "MAXIMUM",
            "recommended_windows": cfg["maximum_windows"],
            "reason": (
                "Evidence quality is low; maximum investigation "
                "is required before a decision."
            ),
            "stop_allowed": False,
            "safety_status": "SAFE",
        }

    except Exception as exc:

        # Fail-safe behavior
        return {
            "status": "error",
            "investigation_required": True,
            "investigation_depth": "MAXIMUM",
            "recommended_windows": cfg["maximum_windows"],
            "reason": (
                "Adaptive investigation failed; "
                "falling back to maximum investigation."
            ),
            "stop_allowed": False,
            "safety_status": "SAFE_FAIL",
            "error": str(exc),
        }