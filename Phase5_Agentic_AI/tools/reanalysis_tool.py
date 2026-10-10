from collections import Counter


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_diagnosis_from_ml(result):
    if not isinstance(result, dict):
        return None
    return result.get("fault")


def _get_diagnosis_from_physics(result):
    if not isinstance(result, dict):
        return None

    return result.get("diagnosis")


def _get_diagnosis_from_fusion(result):
    if not isinstance(result, dict):
        return None

    return result.get("diagnosis")


def _majority(values):
    values = [v for v in values if v is not None]

    if not values:
        return None

    counts = Counter(values)
    return counts.most_common(1)[0][0]


def _consistency(values):
    values = [v for v in values if v is not None]

    if not values:
        return 0.0

    counts = Counter(values)
    return counts.most_common(1)[0][1] / len(values)


def reanalysis_tool(window_results):
    """
    Analyze multiple diagnostic windows and summarize the evidence.

    Phase 5 agentic reanalysis policy.

    Safety principle:
    Low ML confidence alone must neither confirm a fault nor
    declare the bearing Healthy.

    A low-confidence ML fault can be retained only when there
    is sufficiently strong and consistent independent physics
    evidence. High-confidence, consistent ML agreement may also
    support a diagnosis.

    This tool does not modify the frozen Phase 4 ML model,
    physics thresholds, or fusion rules.
    """

    if not window_results:
        return {
            "status": "failed",
            "message": "No window results available for reanalysis."
        }

    ml_diagnoses = []
    physics_diagnoses = []
    fusion_diagnoses = []

    ml_confidences = []

    strong_physics_windows = 0

    window_diagnoses = []
    physics_window_diagnoses = []

    # =========================================================
    # EXTRACT EVIDENCE FROM EVERY ANALYZED WINDOW
    # =========================================================

    for result in window_results:

        if not isinstance(result, dict):
            continue

        ml = result.get("ml_result")
        physics = result.get("physics_result")
        fusion = result.get("fusion_result")

        # -------------------------
        # ML
        # -------------------------

        ml_diagnosis = _get_diagnosis_from_ml(ml)

        if ml_diagnosis is not None:
            ml_diagnoses.append(ml_diagnosis)
            window_diagnoses.append(ml_diagnosis)

        if isinstance(ml, dict):

            confidence = ml.get("confidence")

            if confidence is not None:
                ml_confidences.append(
                    _safe_float(confidence)
                )

        # -------------------------
        # Physics
        # -------------------------

        physics_diagnosis = _get_diagnosis_from_physics(
            physics
        )

        if physics_diagnosis is not None:

            physics_diagnoses.append(
                physics_diagnosis
            )

            physics_window_diagnoses.append(
                physics_diagnosis
            )

        if isinstance(physics, dict):

            if physics.get("strength") == "STRONG":
                strong_physics_windows += 1

        # -------------------------
        # Fusion
        # -------------------------

        fusion_diagnosis = _get_diagnosis_from_fusion(
            fusion
        )

        if fusion_diagnosis is not None:
            fusion_diagnoses.append(
                fusion_diagnosis
            )

    # =========================================================
    # MAJORITY AND CONSISTENCY
    # =========================================================

    ml_majority_diagnosis = _majority(
        ml_diagnoses
    )

    ml_consistency = _consistency(
        ml_diagnoses
    )

    physics_majority_diagnosis = _majority(
        physics_diagnoses
    )

    physics_consistency = _consistency(
        physics_diagnoses
    )

    fusion_majority_diagnosis = _majority(
        fusion_diagnoses
    )

    fusion_consistency = _consistency(
        fusion_diagnoses
    )

    average_ml_confidence = (
        sum(ml_confidences) / len(ml_confidences)
        if ml_confidences
        else 0.0
    )

    # =========================================================
    # CONFLICT DETECTION
    # =========================================================

    # Healthy + Healthy_or_Insufficient is compatible evidence: the ML source
    # predicts no fault and the physics source found no sufficient fault
    # signature. It is not a diagnostic conflict.
    no_fault_compatible = (
        ml_majority_diagnosis == "Healthy"
        and physics_majority_diagnosis in {"Healthy", "Healthy_or_Insufficient"}
    )

    conflict_detected = False

    if (
        ml_majority_diagnosis is not None
        and physics_majority_diagnosis is not None
        and ml_majority_diagnosis != physics_majority_diagnosis
        and not no_fault_compatible
    ):
        conflict_detected = True

    # =========================================================
    # DEFAULT RESULT
    # =========================================================

    majority_diagnosis = ml_majority_diagnosis

    diagnosis_consistency = ml_consistency

    final_diagnosis = majority_diagnosis

    resolution = "NO_RESOLUTION_REQUIRED"

    all_windows_have_strong_physics = (
        len(window_results) > 0
        and strong_physics_windows == len(window_results)
    )

    # =========================================================
    # RULE 1 — STRONG PHYSICS + CONSISTENT FAULT
    # =========================================================
    #
    # This is the primary protection against low-confidence ML
    # false positives.
    #
    # If physics repeatedly and strongly identifies a fault,
    # preserve that fault even when ML confidence is below 0.70.
    # =========================================================

    strong_physics_fault = (
        physics_majority_diagnosis is not None
        and physics_majority_diagnosis != "Healthy"
        and physics_consistency >= 0.80
        and all_windows_have_strong_physics
    )

    if (
        conflict_detected
        and strong_physics_fault
    ):

        final_diagnosis = physics_majority_diagnosis

        majority_diagnosis = physics_majority_diagnosis

        diagnosis_consistency = physics_consistency

        resolution = "RESOLVED_BY_STRONG_PHYSICS"

    elif (
        not conflict_detected
        and strong_physics_fault
        and ml_majority_diagnosis == physics_majority_diagnosis
    ):

        final_diagnosis = physics_majority_diagnosis

        majority_diagnosis = physics_majority_diagnosis

        diagnosis_consistency = min(
            ml_consistency,
            physics_consistency
        )

        resolution = (
            "CONFIRMED_BY_MULTI_WINDOW_AGREEMENT"
        )

    # =========================================================
    # RULE 2 — HIGH-CONFIDENCE ML + CONSISTENT AGREEMENT
    # =========================================================
    #
    # Strong agreement without strong physics is acceptable only
    # when ML itself has sufficiently high confidence.
    #
    # This prevents the healthy baseline cases from being
    # incorrectly confirmed as faults merely because the ML model
    # repeats a low-confidence false-positive prediction.
    # =========================================================

    elif (
        ml_majority_diagnosis is not None
        and physics_majority_diagnosis is not None
        and ml_majority_diagnosis
        == physics_majority_diagnosis
        and ml_majority_diagnosis != "Healthy"
        and ml_consistency >= 0.80
        and physics_consistency >= 0.80
        and average_ml_confidence >= 0.70
    ):

        final_diagnosis = ml_majority_diagnosis

        majority_diagnosis = ml_majority_diagnosis

        diagnosis_consistency = min(
            ml_consistency,
            physics_consistency
        )

        resolution = "STRONG_AGREEMENT"

    # =========================================================
    # RULE 3 — HEALTHY / NO CONFIRMED FAULT
    # =========================================================
    #
    # If ML suggests a fault but:
    #   - ML confidence is low, and
    #   - physics is not strong enough to confirm a fault,
    #
    # the agent must NOT confirm the fault.
    #
    # For this dataset, this is the safe Phase 5 decision for
    # the healthy baseline false-positive pattern.
    # =========================================================

    elif (
        ml_majority_diagnosis is not None
        and average_ml_confidence < 0.70
        and not strong_physics_fault
    ):

        final_diagnosis = "NO_CONFIRMED_FAULT"

        majority_diagnosis = "NO_CONFIRMED_FAULT"

        diagnosis_consistency = max(
            ml_consistency,
            physics_consistency
        )

        resolution = "NO_CONFIRMED_FAULT"

    # =========================================================
    # RULE 4 — PHYSICS-ONLY STRONG FAULT
    # =========================================================
    #
    # If ML evidence is unavailable but physics repeatedly and
    # strongly identifies a fault, retain the physics diagnosis.
    # =========================================================

    elif strong_physics_fault:

        final_diagnosis = physics_majority_diagnosis

        majority_diagnosis = physics_majority_diagnosis

        diagnosis_consistency = physics_consistency

        resolution = "CONFIRMED_BY_STRONG_PHYSICS"

    # =========================================================
    # RULE 5 — EXPLICIT HEALTHY AGREEMENT
    # =========================================================

    elif (
        ml_majority_diagnosis == "Healthy"
        and (
            physics_majority_diagnosis is None
            or physics_majority_diagnosis in {"Healthy", "Healthy_or_Insufficient"}
        )
    ):

        final_diagnosis = "NO_CONFIRMED_FAULT"

        majority_diagnosis = "NO_CONFIRMED_FAULT"

        diagnosis_consistency = max(
            ml_consistency,
            physics_consistency
        )

        resolution = "NO_CONFIRMED_FAULT"

    # =========================================================
    # RULE 6 — OTHERWISE RETAIN DOMINANT EVIDENCE
    # =========================================================

    else:

        final_diagnosis = majority_diagnosis

        resolution = "NO_STRONG_RESOLUTION"

    # =========================================================
    # FINAL OUTPUT
    # =========================================================

    return {
        "status": "success",

        "windows_analyzed":
            len(window_results),

        # ``majority_diagnosis`` is retained for backward compatibility and
        # represents the resolved case-level decision, not the ML majority.
        "majority_diagnosis":
            majority_diagnosis,

        "final_diagnosis":
            final_diagnosis,

        "final_diagnosis_source":
            ("physics" if resolution in {"RESOLVED_BY_STRONG_PHYSICS", "CONFIRMED_BY_STRONG_PHYSICS"}
             else "ml_physics_agreement" if resolution in {"CONFIRMED_BY_MULTI_WINDOW_AGREEMENT", "STRONG_AGREEMENT"}
             else "no_confirmation" if resolution == "NO_CONFIRMED_FAULT"
             else "ml" if final_diagnosis == ml_majority_diagnosis else "case_level_resolution"),

        "diagnosis_consistency":
            diagnosis_consistency,

        "average_ml_confidence":
            average_ml_confidence,

        "strong_physics_windows":
            strong_physics_windows,

        "ml_majority_diagnosis":
            ml_majority_diagnosis,

        "ml_consistency":
            ml_consistency,

        "physics_majority_diagnosis":
            physics_majority_diagnosis,

        "physics_consistency":
            physics_consistency,

        "fusion_majority_diagnosis":
            fusion_majority_diagnosis,

        "fusion_consistency":
            fusion_consistency,

        "conflict_detected":
            conflict_detected,

        "resolution":
            resolution,

        # Explicit per-source window distributions avoid the previous
        # ambiguity where ``window_diagnoses`` looked like the final decision.
        "window_diagnoses":
            dict(Counter(window_diagnoses)),

        "ml_window_diagnoses":
            dict(Counter(ml_diagnoses)),

        "physics_window_diagnoses":
            dict(Counter(physics_window_diagnoses)),

        "fusion_window_diagnoses":
            dict(Counter(fusion_diagnoses))
    }
