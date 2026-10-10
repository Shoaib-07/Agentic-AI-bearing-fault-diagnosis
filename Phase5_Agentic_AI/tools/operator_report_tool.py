# ============================================================
# PHASE 5.13.2
# OPERATOR-ORIENTED DIAGNOSTIC REPORT TOOL
# ============================================================

"""
Purpose
-------
Convert the structured diagnostic state into a concise,
operator-oriented diagnostic report.

This tool:
1. Reads existing ML evidence.
2. Reads existing physics evidence.
3. Reads multi-window reanalysis.
4. Reads root-cause analysis.
5. Produces a human-readable operator report.

Important safety rule:
This tool only summarizes existing evidence.

It does NOT:
- modify the diagnosis
- retrain the ML model
- change physics thresholds
- create new diagnostic evidence
- override fusion decisions
- control plant equipment
- claim physical causes that are not established
"""

from typing import Any


# ============================================================
# DIAGNOSIS LABELS
# ============================================================

DIAGNOSIS_LABELS = {
    "InnerRaceFault": "Inner Race Fault",
    "OuterRaceFault": "Outer Race Fault",
    "Healthy": "No Confirmed Fault",
    "NO_CONFIRMED_FAULT": "No Confirmed Fault",
}


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to integer.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_list(value: Any) -> list:
    """
    Return a list safely.
    """
    if isinstance(value, list):
        return value

    return []


# ============================================================
# DIAGNOSIS RESOLUTION
# ============================================================

def _get_diagnosis(state) -> str:
    """
    Determine diagnosis from the highest-level available result.

    Priority:
    1. final_diagnosis
    2. reanalysis_result
    3. fusion_result
    4. ml_result
    """

    # --------------------------------------------------------
    # Final diagnosis
    # --------------------------------------------------------

    if state.final_diagnosis:
        return str(state.final_diagnosis)

    # --------------------------------------------------------
    # Reanalysis diagnosis
    # --------------------------------------------------------

    if state.reanalysis_result:

        diagnosis = state.reanalysis_result.get(
            "majority_diagnosis"
        )

        if diagnosis:
            return str(diagnosis)

    # --------------------------------------------------------
    # Fusion diagnosis
    # --------------------------------------------------------

    if state.fusion_result:

        diagnosis = state.fusion_result.get(
            "diagnosis"
        )

        if diagnosis:
            return str(diagnosis)

    # --------------------------------------------------------
    # ML diagnosis
    # --------------------------------------------------------

    if state.ml_result:

        diagnosis = state.ml_result.get(
            "fault"
        )

        if diagnosis:
            return str(diagnosis)

    return "Unknown"


# ============================================================
# DIAGNOSIS DISPLAY LABEL
# ============================================================

def _get_display_diagnosis(diagnosis: str) -> str:
    """
    Convert internal diagnosis label into operator-friendly label.
    """

    return DIAGNOSIS_LABELS.get(
        diagnosis,
        diagnosis
    )


# ============================================================
# CONSISTENCY DESCRIPTION
# ============================================================

def _get_consistency_description(consistency: float) -> str:
    """
    Convert numerical diagnostic consistency into a
    human-readable description.
    """

    if consistency >= 0.90:
        return "Very high multi-window consistency"

    if consistency >= 0.80:
        return "High multi-window consistency"

    if consistency >= 0.60:
        return "Moderate multi-window consistency"

    if consistency > 0.0:
        return "Low multi-window consistency"

    return "Not established"


# ============================================================
# MAIN OPERATOR REPORT TOOL
# ============================================================

def operator_report_tool(state):
    """
    Generate an operator-oriented diagnostic report.

    Parameters
    ----------
    state : DiagnosticState

    Returns
    -------
    dict
        Structured operator report.
    """

    # ========================================================
    # 1. EXTRACT DIAGNOSIS
    # ========================================================

    diagnosis = _get_diagnosis(state)

    display_diagnosis = _get_display_diagnosis(
        diagnosis
    )

    # ========================================================
    # 2. EXTRACT ML EVIDENCE
    # ========================================================

    ml = state.ml_result or {}

    ml_confidence = _safe_float(
        ml.get("confidence"),
        0.0
    )

    ml_fault = ml.get(
        "fault",
        "Unknown"
    )

    # ========================================================
    # 3. EXTRACT PHYSICS EVIDENCE
    # ========================================================

    physics = state.physics_result or {}

    physics_strength = physics.get(
        "strength",
        "UNKNOWN"
    )

    bpfi_energy = _safe_float(
        physics.get("BPFI_Normalized_Energy"),
        0.0
    )

    bpfo_energy = _safe_float(
        physics.get("BPFO_Normalized_Energy"),
        0.0
    )

    fault_energy = _safe_float(
        physics.get("Fault_Energy"),
        0.0
    )

    bpfi_dominance = _safe_float(
        physics.get("BPFI_Dominance"),
        0.0
    )

    bpfo_dominance = _safe_float(
        physics.get("BPFO_Dominance"),
        0.0
    )

    # ========================================================
    # 4. EXTRACT REANALYSIS
    # ========================================================

    # Multi-window reanalysis is optional.  A strong-evidence case may
    # intentionally stop before reanalysis.  In that situation, zero is
    # misleading because it implies that validation was attempted and found
    # no consistent windows.  Preserve an explicit N/A state instead.
    reanalysis = state.reanalysis_result
    multi_window_assessed = isinstance(reanalysis, dict) and bool(reanalysis)

    if multi_window_assessed:
        windows_analyzed = _safe_int(
            reanalysis.get("windows_analyzed"),
            0
        )
        diagnosis_consistency = _safe_float(
            reanalysis.get("diagnosis_consistency"),
            0.0
        )
        average_ml_confidence = _safe_float(
            reanalysis.get("average_ml_confidence"),
            ml_confidence
        )
        strong_physics_windows = _safe_int(
            reanalysis.get("strong_physics_windows"),
            0
        )
        consistency_description = _get_consistency_description(diagnosis_consistency)
        multi_window_status = "ASSESSED"
        multi_window_note = "Multi-window validation was performed."
    else:
        windows_analyzed = None
        diagnosis_consistency = None
        average_ml_confidence = ml_confidence
        strong_physics_windows = None
        consistency_description = "Not assessed"
        multi_window_status = "NOT_ASSESSED"
        multi_window_note = (
            "Multi-window validation was intentionally skipped because the agent "
            "stopped early after strong, high-quality independent evidence agreed."
        )

    # ========================================================
    # 5. EXTRACT RCA
    # ========================================================

    rca = state.rca_result or {}

    root_cause_category = rca.get(
        "root_cause_category",
        "Not established"
    )

    likely_mechanism = rca.get(
        "likely_mechanism",
        "No mechanism established."
    )

    supporting_evidence = _safe_list(
        rca.get("supporting_evidence")
    )

    operator_investigation = _safe_list(
        rca.get("operator_investigation")
    )

    confidence_note = rca.get(
        "confidence_note",
        "Physical causes require operator verification."
    )

    # ========================================================
    # 6. CONSISTENCY DESCRIPTION
    # ========================================================

    # ========================================================
    # 7. BUILD REPORT
    # ========================================================

    report_lines = []

    report_lines.append(
        "FAULT DIAGNOSIS REPORT"
    )

    report_lines.append(
        "========================================"
    )

    report_lines.append(
        f"Diagnosis: {display_diagnosis}"
    )

    if diagnosis == "NO_CONFIRMED_FAULT":
        report_lines.append(
            "Important: No confirmed fault does not prove that the bearing is healthy."
        )

    report_lines.append(
        f"Multi-window consistency: {consistency_description}"
    )

    report_lines.append(
        f"Multi-window assessment: {multi_window_status}"
    )

    if not multi_window_assessed:
        report_lines.append(
            f"Note: {multi_window_note}"
        )

    supporting_ml_confidence = (
        average_ml_confidence if reanalysis else ml_confidence
    )
    final_confidence = getattr(state, "final_confidence", None)
    final_confidence_type = str(
        getattr(state, "final_confidence_type", "not_established")
    )

    report_lines.append(
        f"Case-level probability: NOT ESTABLISHED ({final_confidence_type})"
    )
    report_lines.append(
        f"ML confidence (source-specific): {supporting_ml_confidence:.2f}"
    )

    report_lines.append("")

    # ========================================================
    # EVIDENCE
    # ========================================================

    report_lines.append(
        "EVIDENCE"
    )

    report_lines.append(
        "----------------------------------------"
    )

    report_lines.append(
        f"- Initial-window ML confidence: {ml_confidence:.2f}"
    )

    report_lines.append(
        f"- Average ML confidence across analyzed windows: {average_ml_confidence:.2f}"
    )

    report_lines.append(
        f"- Physics evidence: {physics_strength}"
    )

    report_lines.append(
        f"- BPFI dominance: {bpfi_dominance:.3f}"
    )

    report_lines.append(
        f"- BPFO dominance: {bpfo_dominance:.3f}"
    )

    if multi_window_assessed:
        report_lines.append(
            f"- Windows analyzed: {windows_analyzed}"
        )
        report_lines.append(
            f"- Diagnosis consistency: {diagnosis_consistency:.3f}"
        )
        report_lines.append(
            f"- Strong physics windows: {strong_physics_windows}"
        )
    else:
        report_lines.append(
            "- Windows analyzed: N/A (multi-window validation not assessed)"
        )
        report_lines.append(
            "- Diagnosis consistency: N/A (multi-window validation not assessed)"
        )
        report_lines.append(
            "- Strong physics windows: N/A (multi-window validation not assessed)"
        )
        report_lines.append(
            f"- Multi-window note: {multi_window_note}"
        )

    report_lines.append("")

    # ========================================================
    # LIKELY MECHANISM
    # ========================================================

    report_lines.append(
        "LIKELY MECHANISM"
    )

    report_lines.append(
        "----------------------------------------"
    )

    report_lines.append(
        str(root_cause_category)
    )

    report_lines.append(
        str(likely_mechanism)
    )

    report_lines.append("")

    # ========================================================
    # SUPPORTING EVIDENCE
    # ========================================================

    report_lines.append(
        "SUPPORTING EVIDENCE"
    )

    report_lines.append(
        "----------------------------------------"
    )

    if supporting_evidence:

        for evidence in supporting_evidence:

            report_lines.append(
                f"- {evidence}"
            )

    else:

        report_lines.append(
            "- No additional supporting evidence available."
        )

    report_lines.append("")

    # ========================================================
    # OPERATOR INVESTIGATION
    # ========================================================

    report_lines.append(
        "OPERATOR INVESTIGATION"
    )

    report_lines.append(
        "----------------------------------------"
    )

    if operator_investigation:

        for index, item in enumerate(
            operator_investigation,
            start=1
        ):

            report_lines.append(
                f"{index}. {item}"
            )

    else:

        report_lines.append(
            "1. Perform additional diagnostic investigation."
        )

    report_lines.append("")

    # ========================================================
    # IMPORTANT NOTE
    # ========================================================

    report_lines.append(
        "IMPORTANT NOTE"
    )

    report_lines.append(
        "----------------------------------------"
    )

    report_lines.append(
        str(confidence_note)
    )

    report_lines.append("")

    # ========================================================
    # SAFETY STATEMENT
    # ========================================================

    report_lines.append(
        "SAFETY"
    )

    report_lines.append(
        "----------------------------------------"
    )

    report_lines.append(
        "This report is decision-support output. "
        "It does not directly control plant equipment "
        "and does not replace operator or maintenance "
        "inspection."
    )

    # ========================================================
    # FINAL REPORT STRING
    # ========================================================

    report = "\n".join(
        report_lines
    )

    # ========================================================
    # STRUCTURED RESULT
    # ========================================================

    return {
        "status": "success",

        "diagnosis": diagnosis,

        "display_diagnosis": display_diagnosis,

        "diagnostic_consistency": diagnosis_consistency,

        "consistency_description":
            consistency_description,

        "multi_window_assessed":
            multi_window_assessed,

        "multi_window_status":
            multi_window_status,

        "multi_window_note":
            multi_window_note,

        "windows_analyzed":
            windows_analyzed,

        "strong_physics_windows":
            strong_physics_windows,

        "ml_confidence":
            ml_confidence,

        "supporting_ml_confidence":
            supporting_ml_confidence,

        "final_confidence":
            final_confidence,

        "final_confidence_type":
            final_confidence_type,

        "final_confidence_explanation":
            getattr(state, "final_confidence_explanation", ""),

        "average_ml_confidence":
            average_ml_confidence,

        "physics_strength":
            physics_strength,

        "BPFI_Normalized_Energy":
            bpfi_energy,

        "BPFO_Normalized_Energy":
            bpfo_energy,

        "Fault_Energy":
            fault_energy,

        "BPFI_Dominance":
            bpfi_dominance,

        "BPFO_Dominance":
            bpfo_dominance,

        "ml_fault":
            ml_fault,

        "root_cause_category":
            root_cause_category,

        "likely_mechanism":
            likely_mechanism,

        "supporting_evidence":
            supporting_evidence,

        "operator_investigation":
            operator_investigation,

        "confidence_note":
            confidence_note,

        "report":
            report,
    }