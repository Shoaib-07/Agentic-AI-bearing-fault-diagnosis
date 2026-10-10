# ============================================================
# PHASE 5.12.1
# AGENTIC ROOT CAUSE ANALYSIS TOOL
# ============================================================

def root_cause_tool(state):
    """
    Deterministic Root Cause Analysis tool.

    The tool converts validated diagnostic evidence
    into an evidence-based fault mechanism and
    operator investigation points.

    IMPORTANT:
    This tool does NOT claim that unmeasured plant
    variables are definitely responsible.

    It identifies likely mechanisms and recommends
    what should be investigated.
    """

    # ========================================================
    # CHECK REQUIRED EVIDENCE
    # ========================================================

    ml = state.ml_result
    physics = state.physics_result
    fusion = state.fusion_result
    reanalysis = state.reanalysis_result

    if (
        ml is None
        and physics is None
        and fusion is None
        and reanalysis is None
    ):

        return {
            "status": "failed",
            "message":
                "Insufficient diagnostic evidence for RCA."
        }

    # ========================================================
    # LLM-GROUNDED RCA
    # ========================================================
    # When the reasoning agent has produced a verified diagnostic judgment,
    # preserve that judgment and its operator recommendations. Deterministic
    # RCA remains the fallback for offline/no-LLM execution.
    llm = getattr(state, "llm_diagnostic_result", None)
    if isinstance(llm, dict) and llm.get("status") == "success":
        return {
            "status": "success",
            "diagnosis": llm.get("diagnosis"),
            "root_cause_category": llm.get("fault_location") or "Diagnostic conclusion",
            "likely_mechanism": llm.get("inferred_mechanism") or "No specific mechanism inferred.",
            "supporting_evidence": llm.get("observed_evidence") or [],
            "operator_investigation": llm.get("operator_actions") or [],
            "possible_root_causes": llm.get("possible_root_causes") or [],
            "root_cause_status": llm.get("root_cause_status") or "requires_physical_verification",
            "reasoning": llm.get("reasoning") or "",
            "additional_investigation": llm.get("additional_investigation") or "",
            "confidence_note": (
                "The diagnosis is an evidence-grounded LLM interpretation of measured diagnostic evidence. "
                "Physical initiating causes remain subject to maintenance/inspection verification."
            ),
            "source": "verified_llm_diagnostic_reasoner",
        }

    # ========================================================
    # DETERMINE DIAGNOSIS
    # ========================================================

    diagnosis = None

    # Prefer the controller's resolved case-level diagnosis. This is important
    # when ML/fusion and strong multi-window physics disagree.
    if getattr(state, "final_diagnosis", None):
        diagnosis = state.final_diagnosis

    elif reanalysis is not None:

        diagnosis = reanalysis.get(
            "final_diagnosis"
        ) or reanalysis.get("majority_diagnosis")

    elif fusion is not None:

        diagnosis = fusion.get(
            "diagnosis"
        )

    elif physics is not None:

        diagnosis = physics.get(
            "diagnosis"
        )

    elif ml is not None:

        diagnosis = ml.get(
            "fault"
        )

    # ========================================================
    # NO CONFIRMED FAULT
    # ========================================================

    if diagnosis == "NO_CONFIRMED_FAULT":

        return {
            "status": "success",
            "diagnosis": "NO_CONFIRMED_FAULT",
            "root_cause_category": "No confirmed fault",
            "likely_mechanism": "No fault mechanism was confirmed by the available evidence.",
            "supporting_evidence": [
                "The available evidence did not establish a confirmed bearing fault.",
                "Multi-window assessment did not provide sufficient fault-specific evidence."
            ],
            "operator_investigation": [
                "Continue routine condition monitoring.",
                "Collect repeated timestamped measurements for trend and early-warning analysis.",
                "Investigate any persistent or newly emerging vibration anomaly."
            ],
            "confidence_note": (
                "No confirmed fault is not proof that the bearing is healthy. "
                "A physical fault mechanism requires additional evidence and inspection."
            )
        }

    # ========================================================
    # HEALTHY CASE (legacy/pre-canonical state)
    # ========================================================

    if diagnosis == "Healthy":

        return {

            "status":
                "success",

            "diagnosis":
                "Healthy",

            "root_cause_category":
                "No fault detected",

            "likely_mechanism":
                "No significant bearing fault evidence "
                "was identified.",

            "supporting_evidence": [
                "No fault-specific evidence was identified in the available recording."
            ],

            "operator_investigation": [
                "Continue routine condition monitoring.",
                "No fault-specific intervention is "
                "recommended from the available evidence."
            ],

            "confidence_note":
                "RCA is limited because no fault "
                "mechanism was identified."
        }

    # ========================================================
    # INNER RACE FAULT
    # ========================================================

    if diagnosis == "InnerRaceFault":

        supporting_evidence = []

        operator_investigation = []

        # ----------------------------------------------------
        # Physics evidence
        # ----------------------------------------------------

        if physics is not None:

            bpfi_energy = physics.get(
                "BPFI_Normalized_Energy"
            )

            bpfi_dominance = physics.get(
                "BPFI_Dominance"
            )

            fault_energy = physics.get(
                "Fault_Energy"
            )

            strength = physics.get(
                "strength"
            )

            if bpfi_energy is not None:

                supporting_evidence.append(
                    "Elevated BPFI-related spectral "
                    "energy was detected."
                )

            if bpfi_dominance is not None:

                if bpfi_dominance >= 0.80:

                    supporting_evidence.append(
                        "BPFI strongly dominates the "
                        "fault-related spectral energy."
                    )

            if fault_energy is not None:

                supporting_evidence.append(
                    "Elevated total fault-related "
                    "energy was detected."
                )

            if strength == "STRONG":

                supporting_evidence.append(
                    "Physics-based diagnosis is STRONG."
                )

        # ----------------------------------------------------
        # ML evidence
        # ----------------------------------------------------

        if ml is not None:

            ml_fault = ml.get(
                "fault"
            )

            ml_confidence = ml.get(
                "confidence"
            )

            if ml_fault == "InnerRaceFault":

                supporting_evidence.append(
                    "ML independently classified the "
                    "signal as InnerRaceFault."
                )

            if ml_confidence is not None:

                supporting_evidence.append(
                    f"ML confidence was "
                    f"{ml_confidence:.2f}."
                )

        # ----------------------------------------------------
        # Multi-window evidence
        # ----------------------------------------------------

        if reanalysis is not None:

            windows = reanalysis.get(
                "windows_analyzed",
                0
            )

            consistency = reanalysis.get(
                "diagnosis_consistency",
                0.0
            )

            strong_physics = reanalysis.get(
                "strong_physics_windows",
                0
            )

            if windows >= 1:

                supporting_evidence.append(
                    f"The diagnosis was observed "
                    f"consistently across "
                    f"{windows} analyzed windows."
                )

            if consistency >= 0.80:

                supporting_evidence.append(
                    f"Diagnostic consistency was "
                    f"{consistency:.2f}."
                )

            if strong_physics >= 1:

                supporting_evidence.append(
                    f"Strong physics evidence was "
                    f"observed in {strong_physics} "
                    f"window(s)."
                )

        # ----------------------------------------------------
        # Operator investigation
        # ----------------------------------------------------

        operator_investigation = [

            "Inspect the bearing inner race and "
            "rolling-contact surfaces for localized "
            "damage, pitting, spalling, or wear.",

            "Check bearing lubrication condition "
            "and contamination indicators.",

            "Check shaft alignment and operating "
            "load conditions.",

            "Inspect bearing installation, fit, "
            "clearance, and preload conditions.",

            "Compare current vibration behavior "
            "with historical operating conditions."
        ]

        # ====================================================
        # RETURN INNER-RACE RCA
        # ====================================================

        return {

            "status":
                "success",

            "diagnosis":
                "InnerRaceFault",

            "root_cause_category":
                "Inner-race bearing defect",

            "likely_mechanism":
                (
                    "The diagnostic evidence is consistent "
                    "with an inner-race-related bearing "
                    "defect, characterized by strong "
                    "BPFI-related vibration response."
                ),

            "supporting_evidence":
                supporting_evidence,

            "operator_investigation":
                operator_investigation,

            "confidence_note":
                (
                    "The fault mechanism is supported by "
                    "current vibration evidence. Physical "
                    "causes such as lubrication, alignment, "
                    "load, or installation conditions "
                    "require operator inspection and are "
                    "not directly established by the "
                    "current signal alone."
                )
        }

    # ========================================================
    # OUTER RACE FAULT
    # ========================================================

    if diagnosis == "OuterRaceFault":

        supporting_evidence = []

        # ----------------------------------------------------
        # Physics evidence
        # ----------------------------------------------------

        if physics is not None:

            bpfo_energy = physics.get(
                "BPFO_Normalized_Energy"
            )

            bpfo_dominance = physics.get(
                "BPFO_Dominance"
            )

            fault_energy = physics.get(
                "Fault_Energy"
            )

            strength = physics.get(
                "strength"
            )

            if bpfo_energy is not None:

                supporting_evidence.append(
                    "Elevated BPFO-related spectral "
                    "energy was detected."
                )

            if bpfo_dominance is not None:

                if bpfo_dominance >= 0.80:

                    supporting_evidence.append(
                        "BPFO strongly dominates the "
                        "fault-related spectral energy."
                    )

            if fault_energy is not None:

                supporting_evidence.append(
                    "Elevated total fault-related "
                    "energy was detected."
                )

            if strength == "STRONG":

                supporting_evidence.append(
                    "Physics-based diagnosis is STRONG."
                )

        # ----------------------------------------------------
        # ML evidence
        # ----------------------------------------------------

        if ml is not None:

            if ml.get("fault") == "OuterRaceFault":

                supporting_evidence.append(
                    "ML independently classified the "
                    "signal as OuterRaceFault."
                )

        # ----------------------------------------------------
        # Multi-window evidence
        # ----------------------------------------------------

        if reanalysis is not None:

            windows = reanalysis.get(
                "windows_analyzed",
                0
            )

            consistency = reanalysis.get(
                "diagnosis_consistency",
                0.0
            )

            if windows >= 1:

                supporting_evidence.append(
                    f"The diagnosis was evaluated across "
                    f"{windows} signal window(s)."
                )

            if consistency >= 0.80:

                supporting_evidence.append(
                    f"Diagnostic consistency was "
                    f"{consistency:.2f}."
                )

        # ----------------------------------------------------
        # Operator investigation
        # ----------------------------------------------------

        operator_investigation = [

            "Inspect the bearing outer race and "
            "rolling-contact surfaces for localized "
            "damage, pitting, spalling, or wear.",

            "Inspect the bearing housing and seating "
            "condition.",

            "Check lubrication condition and possible "
            "contamination.",

            "Check operating load and mechanical "
            "alignment.",

            "Compare the current vibration signature "
            "with historical operating conditions."
        ]

        return {

            "status":
                "success",

            "diagnosis":
                "OuterRaceFault",

            "root_cause_category":
                "Outer-race bearing defect",

            "likely_mechanism":
                (
                    "The diagnostic evidence is consistent "
                    "with an outer-race-related bearing "
                    "defect, characterized by BPFO-related "
                    "vibration response."
                ),

            "supporting_evidence":
                supporting_evidence,

            "operator_investigation":
                operator_investigation,

            "confidence_note":
                (
                    "The identified mechanism is supported "
                    "by vibration evidence. Specific physical "
                    "causes require inspection and are not "
                    "directly established by vibration alone."
                )
        }

    # ========================================================
    # UNKNOWN / UNRESOLVED DIAGNOSIS
    # ========================================================

    return {

        "status":
            "success",

        "diagnosis":
            diagnosis,

        "root_cause_category":
            "Unresolved",

        "likely_mechanism":
            (
                "The available diagnostic evidence does "
                "not support a sufficiently specific "
                "root-cause mechanism."
            ),

        "supporting_evidence":
            [
                "Available evidence is insufficient "
                "for a specific RCA conclusion."
            ],

        "operator_investigation":
            [
                "Collect additional vibration data.",
                "Inspect machine operating conditions.",
                "Review historical diagnostic cases.",
                "Investigate conflicting diagnostic evidence."
            ],

        "confidence_note":
            (
                "Additional evidence is required before "
                "assigning a specific root cause."
            )
    }