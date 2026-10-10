def fusion_tool(ml_result, physics_result):

    ml_fault = ml_result.get("fault")
    ml_confidence = ml_result.get("confidence", 0.0)
    ml_margin = ml_result.get("margin", 0.0)

    physics_fault = physics_result.get("diagnosis")
    physics_strength = physics_result.get("strength")

    # --------------------------------------------------------
    # 1. Agreement + STRONG physics + high ML confidence
    # --------------------------------------------------------
    if (
        ml_fault == physics_fault
        and physics_strength == "STRONG"
        and ml_confidence >= 0.70
    ):
        return {
            "diagnosis": ml_fault,
            "decision": "CONFIRMED",
            "action": "ACCEPT_AND_EXPLAIN",
            "confidence": None,
            "ml_source_confidence": ml_confidence,
            "consistency": "high",
            "evidence": [
                "ML and physics diagnostics agree",
                "Physics evidence is STRONG",
                "ML confidence is >= 0.70"
            ],
            "status": "success"
        }

    # --------------------------------------------------------
    # 2. Healthy agreement
    # --------------------------------------------------------
    if (
        ml_fault == "Healthy"
        and physics_fault == "Healthy"
        and ml_confidence >= 0.70
    ):
        return {
            "diagnosis": "Healthy",
            "decision": "CONFIRMED",
            "action": "ACCEPT_AND_EXPLAIN",
            "confidence": None,
            "ml_source_confidence": ml_confidence,
            "consistency": "high",
            "evidence": [
                "ML and physics both indicate Healthy"
            ],
            "status": "success"
        }

    # --------------------------------------------------------
    # 3. Strong physics conflicts with ML
    # --------------------------------------------------------
    if (
        physics_strength == "STRONG"
        and ml_fault != physics_fault
    ):
        return {
            "diagnosis": ml_fault,
            "decision": "CONFLICT",
            "action": "INVESTIGATE_CONFLICT",
            "confidence": None,
            "ml_source_confidence": ml_confidence,
            "consistency": "low",
            "evidence": [
                "Strong physics evidence conflicts with ML diagnosis"
            ],
            "status": "success"
        }

    # --------------------------------------------------------
    # 4. Moderate agreement
    # --------------------------------------------------------
    if (
        ml_fault == physics_fault
        and physics_strength == "MODERATE"
        and ml_confidence >= 0.70
        and ml_margin >= 0.10
    ):
        return {
            "diagnosis": ml_fault,
            "decision": "PROVISIONAL",
            "action": "REPORT_WITH_CAUTION",
            "confidence": None,
            "ml_source_confidence": ml_confidence,
            "consistency": "high",
            "evidence": [
                "ML and physics diagnostics agree",
                "Physics evidence is MODERATE"
            ],
            "status": "success"
        }

    # --------------------------------------------------------
    # 5. High-confidence ML with weak physics
    # --------------------------------------------------------
    if (
        ml_confidence >= 0.90
        and ml_margin >= 0.20
        and physics_strength == "LOW"
    ):
        return {
            "diagnosis": ml_fault,
            "decision": "ML_SUPPORTED",
            "action": "ACCEPT_WITH_LIMITED_PHYSICS_EVIDENCE",
            "confidence": None,
            "ml_source_confidence": ml_confidence,
            "consistency": "moderate",
            "evidence": [
                "ML confidence is high",
                "Physics evidence is LOW"
            ],
            "status": "success"
        }

    # --------------------------------------------------------
    # 6. Low-confidence ML
    # --------------------------------------------------------
    if (
        ml_confidence < 0.70
        or ml_margin < 0.10
    ):
        return {
            "diagnosis": ml_fault,
            "decision": "UNCERTAIN",
            "action": "REANALYZE_SIGNAL",
            "confidence": None,
            "ml_source_confidence": ml_confidence,
            "consistency": "low",
            "evidence": [
                "ML confidence or prediction margin is insufficient"
            ],
            "status": "success"
        }

    # --------------------------------------------------------
    # 7. Default
    # --------------------------------------------------------
    return {
        "diagnosis": ml_fault,
        "decision": "PROVISIONAL",
        "action": "REPORT_WITH_CAUTION",
        "confidence": None,
        "ml_source_confidence": ml_confidence,
        "consistency": "moderate",
        "evidence": [
            "Evidence is available but does not meet confirmation criteria"
        ],
        "status": "success"
    }