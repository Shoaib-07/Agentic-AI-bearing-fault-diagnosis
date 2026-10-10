from typing import Any, Dict


# =============================================================================
# PHASE 5.17.2 - EVIDENCE QUALITY SCORING
# =============================================================================

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _ml_quality(ml: Dict[str, Any]) -> float:
    """
    Score ML evidence using confidence and prediction margin.

    This is an evidence-quality score, NOT classifier probability.
    """

    if not isinstance(ml, dict):
        return 0.0

    if ml.get("status") != "success":
        return 0.0

    confidence = float(
        ml.get("confidence", 0.0)
    )

    margin = float(
        ml.get("margin", 0.0)
    )

    # Confidence contribution.
    confidence_score = _clamp(
        confidence / 0.90
    )

    # Margin contribution.
    margin_score = _clamp(
        margin / 0.20
    )

    return _clamp(
        0.60 * confidence_score
        +
        0.40 * margin_score
    )


def _physics_quality(physics: Dict[str, Any]) -> float:
    """
    Score physics evidence.

    STRONG evidence receives the highest score.
    """

    if not isinstance(physics, dict):
        return 0.0

    if physics.get("status") != "success":
        return 0.0

    strength = str(
        physics.get("strength", "")
    ).upper()

    if strength == "STRONG":
        return 1.0

    if strength == "MODERATE":
        return 0.60

    if strength == "LOW":
        return 0.30

    return 0.0


def _multi_window_quality(reanalysis: Dict[str, Any]) -> float | None:
    """Score within-recording multi-window consistency.

    This is deliberately NOT called temporal quality: several windows from one
    six-second recording do not establish a health/degradation trend over time.
    """
    if not isinstance(reanalysis, dict) or reanalysis.get("status") != "success":
        return None
    consistency = _clamp(float(reanalysis.get("diagnosis_consistency", 0.0)))
    windows = max(0, int(reanalysis.get("windows_analyzed", 0)))
    coverage = _clamp(windows / 5.0)
    return _clamp(0.70 * consistency + 0.30 * coverage)

def _cross_source_quality(
    ml: Dict[str, Any],
    physics: Dict[str, Any]
) -> float:
    """
    Score agreement between ML and physics evidence.
    """

    if not isinstance(ml, dict):
        return 0.0

    if not isinstance(physics, dict):
        return 0.0

    if (
        ml.get("status") != "success"
        or
        physics.get("status") != "success"
    ):
        return 0.0

    ml_diagnosis = ml.get(
        "fault"
    )

    physics_diagnosis = physics.get(
        "diagnosis"
    )

    if not ml_diagnosis or not physics_diagnosis:
        return 0.0

    if ml_diagnosis == physics_diagnosis:
        if str(physics.get("strength", "")).upper() == "STRONG":
            return 1.0
        return 0.75

    # ``Healthy_or_Insufficient`` means that physics did not find sufficient
    # fault-specific evidence.  It is compatible with an ML Healthy result;
    # it is not a fault-class conflict.  Because the physics strength is LOW,
    # retain partial rather than full cross-source support.
    if (
        ml_diagnosis == "Healthy"
        and physics_diagnosis == "Healthy_or_Insufficient"
    ):
        return 0.75

    # A genuine fault-class disagreement is meaningful evidence, but is not
    # agreement and therefore receives only limited cross-source support.
    return 0.25


def _quality_label(score: float) -> str:

    if score >= 0.80:
        return "HIGH"

    if score >= 0.60:
        return "MODERATE"

    if score >= 0.40:
        return "LOW"

    return "VERY_LOW"


def _recommended_action(
    ml: Dict[str, Any],
    physics: Dict[str, Any],
    reanalysis: Dict[str, Any],
    overall_score: float
) -> str:

    # =========================================================================
    # 1. USE COMPLETED REANALYSIS FIRST
    # =========================================================================

    if isinstance(reanalysis, dict):

        if reanalysis.get("status") == "success":

            resolution = reanalysis.get(
                "resolution"
            )

            if resolution == "STRONG_AGREEMENT":
                return "ACCEPT_AND_EXPLAIN"

            if resolution == "RESOLVED_BY_STRONG_PHYSICS":
                return "ACCEPT_AND_EXPLAIN"

            if resolution == "NO_CONFIRMED_FAULT":
                return "NO_CONFIRMED_FAULT"

    # =========================================================================
    # 2. MISSING ML EVIDENCE
    # =========================================================================

    if not isinstance(ml, dict):
        return "COLLECT_ML_EVIDENCE"

    if ml.get("status") != "success":
        return "COLLECT_ML_EVIDENCE"

    # =========================================================================
    # 3. MISSING PHYSICS EVIDENCE
    # =========================================================================

    if not isinstance(physics, dict):
        return "COLLECT_PHYSICS_EVIDENCE"

    if physics.get("status") != "success":
        return "COLLECT_PHYSICS_EVIDENCE"

    # =========================================================================
    # 4. EXTRACT CURRENT EVIDENCE
    # =========================================================================

    ml_diagnosis = ml.get(
        "fault"
    )

    physics_diagnosis = physics.get(
        "diagnosis"
    )

    physics_strength = str(
        physics.get("strength", "")
    ).upper()

    confidence = float(
        ml.get("confidence", 0.0)
    )

    margin = float(
        ml.get("margin", 0.0)
    )

    # =========================================================================
    # 5. STRONG PHYSICS CONFLICT
    # =========================================================================

    if (
        ml_diagnosis != physics_diagnosis
        and
        physics_strength == "STRONG"
    ):
        return "INVESTIGATE_CONFLICT"

    # =========================================================================
    # 6. STRONG SINGLE-WINDOW AGREEMENT
    # =========================================================================

    if (
        ml_diagnosis == physics_diagnosis
        and
        physics_strength == "STRONG"
        and
        confidence >= 0.70
        and
        margin >= 0.10
    ):
        return "ACCEPT_AND_EXPLAIN"

    # =========================================================================
    # 7. WEAK ML EVIDENCE
    # =========================================================================

    if (
        confidence < 0.70
        or
        margin < 0.10
    ):
        return "REANALYZE_SIGNAL"

    # =========================================================================
    # 8. HIGH OVERALL EVIDENCE
    # =========================================================================

    if overall_score >= 0.80:
        return "ACCEPT_AND_EXPLAIN"

    # =========================================================================
    # 9. DEFAULT
    # =========================================================================

    return "REANALYZE_SIGNAL"


def evidence_quality_tool(
    ml_result=None,
    physics_result=None,
    reanalysis_result=None
):
    """
    Calculate evidence quality for the current diagnostic state.

    Returns:
        ML evidence quality
        Physics evidence quality
        Temporal evidence quality
        Cross-source evidence quality
        Overall evidence quality
        Recommended next action
    """

    try:

        ml_score = _ml_quality(
            ml_result
        )

        physics_score = _physics_quality(
            physics_result
        )

        multi_window_score = _multi_window_quality(reanalysis_result)

        cross_source_score = _cross_source_quality(
            ml_result,
            physics_result
        )

        # Average only dimensions that actually exist. A missing temporal
        # history must not penalize an otherwise strong single-recording case.
        dimensions = [ml_score, physics_score, cross_source_score]
        if multi_window_score is not None:
            dimensions.append(multi_window_score)
        overall_score = _clamp(sum(dimensions) / len(dimensions))

        label = _quality_label(
            overall_score
        )

        action = _recommended_action(
            ml_result,
            physics_result,
            reanalysis_result,
            overall_score
        )

        ml_diagnosis = (
            ml_result.get("fault")
            if isinstance(ml_result, dict)
            else None
        )

        physics_diagnosis = (
            physics_result.get("diagnosis")
            if isinstance(physics_result, dict)
            else None
        )

        agreement = None

        if (
            ml_diagnosis is not None
            and
            physics_diagnosis is not None
        ):
            agreement = (
                ml_diagnosis
                ==
                physics_diagnosis
            )

        return {
            "status": "success",

            "ml_evidence_quality": round(
                ml_score,
                4
            ),

            "physics_evidence_quality": round(
                physics_score,
                4
            ),

            "temporal_evidence_quality": None,

            "temporal_evidence_status": "NOT_AVAILABLE",

            "multi_window_evidence_quality": (
                round(multi_window_score, 4) if multi_window_score is not None else None
            ),

            "cross_source_quality": round(
                cross_source_score,
                4
            ),

            "overall_evidence_quality": round(
                overall_score,
                4
            ),

            "quality_label": label,

            "ml_diagnosis": ml_diagnosis,

            "physics_diagnosis": physics_diagnosis,

            "ml_physics_agreement": agreement,

            "ml_physics_compatible": (
                ml_diagnosis == "Healthy"
                and physics_diagnosis == "Healthy_or_Insufficient"
            ) or bool(agreement),

            "cross_source_relation": (
                "AGREE" if agreement else
                "COMPATIBLE_NO_FAULT_EVIDENCE" if (
                    ml_diagnosis == "Healthy"
                    and physics_diagnosis == "Healthy_or_Insufficient"
                ) else
                "CONFLICT"
            ),

            "recommended_action": action,
        }

    except Exception as e:

        return {
            "status": "failed",
            "message": (
                "Evidence quality scoring failed safely: "
                f"{type(e).__name__}: {e}"
            )
        }