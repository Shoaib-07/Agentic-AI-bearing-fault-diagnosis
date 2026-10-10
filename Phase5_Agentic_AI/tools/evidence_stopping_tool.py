from __future__ import annotations

from typing import Any, Dict


# =============================================================================
# PHASE 5.17.4 — EVIDENCE-AWARE STOPPING
# =============================================================================

HIGH_QUALITY_THRESHOLD = 0.80
MODERATE_QUALITY_THRESHOLD = 0.60

STRONG_CONFIDENCE = 0.70
STRONG_MARGIN = 0.10

STRONG_PHYSICS = "STRONG"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _upper(value: Any) -> str:
    if value is None:
        return ""
    return str(value).upper()


def evidence_stopping_tool(
    ml_result: Dict[str, Any] | None,
    physics_result: Dict[str, Any] | None,
    evidence_quality_result: Dict[str, Any] | None,
    reanalysis_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

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
        # Extract values
        # ---------------------------------------------------------------------

        overall_quality = _safe_float(
            evidence.get("overall_evidence_quality"), 0.0
        )

        ml_status = ml.get("status")
        physics_status = physics.get("status")

        ml_diagnosis = ml.get("fault")
        physics_diagnosis = physics.get("diagnosis")

        confidence = _safe_float(
            ml.get("confidence"), 0.0
        )

        margin = _safe_float(
            ml.get("margin"), 0.0
        )

        physics_strength = _upper(
            physics.get("strength")
        )

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

        # =====================================================================
        # RULE 1 — Completed reanalysis has highest priority
        # =====================================================================

        if reanalysis.get("status") == "success":

            resolution = reanalysis.get("resolution")

            if resolution in (
                "STRONG_AGREEMENT",
                "RESOLVED_BY_STRONG_PHYSICS",
            ):
                return {
                    "status": "success",
                    "stop": True,
                    "decision": "STOP",
                    "reason": (
                        "Reanalysis produced a strong resolution; "
                        "additional investigation is unnecessary."
                    ),
                    "safety_status": "SAFE",
                }

            if resolution == "NO_CONFIRMED_FAULT":
                return {
                    "status": "success",
                    "stop": True,
                    "decision": "STOP",
                    "reason": (
                        "Reanalysis found insufficient evidence for a "
                        "confirmed fault; safely stop without confirmation."
                    ),
                    "safety_status": "SAFE",
                }

        # =====================================================================
        # RULE 2 — Missing evidence
        # =====================================================================

        if ml_status != "success":
            return {
                "status": "success",
                "stop": False,
                "decision": "CONTINUE",
                "reason": "ML evidence is missing or invalid.",
                "safety_status": "SAFE",
            }

        if physics_status != "success":
            return {
                "status": "success",
                "stop": False,
                "decision": "CONTINUE",
                "reason": "Physics evidence is missing or invalid.",
                "safety_status": "SAFE",
            }

        # =====================================================================
        # RULE 3 — Unresolved strong conflict
        # =====================================================================

        if conflict and physics_strength == STRONG_PHYSICS:
            return {
                "status": "success",
                "stop": False,
                "decision": "CONTINUE",
                "reason": (
                    "Strong ML/physics conflict remains unresolved; "
                    "investigation must continue."
                ),
                "safety_status": "SAFE",
            }

        # =====================================================================
        # RULE 4 — Strong agreement + high evidence quality
        # =====================================================================

        if (
            agreement
            and physics_strength == STRONG_PHYSICS
            and confidence >= STRONG_CONFIDENCE
            and margin >= STRONG_MARGIN
            and overall_quality >= HIGH_QUALITY_THRESHOLD
        ):
            return {
                "status": "success",
                "stop": True,
                "decision": "STOP",
                "reason": (
                    "ML and strong physics agree with high-quality evidence."
                ),
                "safety_status": "SAFE",
            }

        # =====================================================================
        # RULE 5 — High quality but stopping conditions incomplete
        # =====================================================================

        if overall_quality >= HIGH_QUALITY_THRESHOLD:
            return {
                "status": "success",
                "stop": False,
                "decision": "CONTINUE",
                "reason": (
                    "Evidence quality is high, but the stopping conditions "
                    "for a safe confirmed decision are not all satisfied."
                ),
                "safety_status": "SAFE",
            }

        # =====================================================================
        # RULE 6 — Moderate / low evidence
        # =====================================================================

        return {
            "status": "success",
            "stop": False,
            "decision": "CONTINUE",
            "reason": (
                "Evidence is insufficient for safe stopping; "
                "investigation must continue."
            ),
            "safety_status": "SAFE",
        }

    except Exception as exc:

        # =====================================================================
        # FAIL-SAFE
        # =====================================================================

        return {
            "status": "error",
            "stop": False,
            "decision": "CONTINUE",
            "reason": (
                "Stopping evaluation failed; investigation must continue."
            ),
            "safety_status": "SAFE_FAIL",
            "error": str(exc),
        }