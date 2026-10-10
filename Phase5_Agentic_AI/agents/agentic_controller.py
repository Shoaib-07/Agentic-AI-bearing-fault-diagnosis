"""Genuinely agentic, safety-constrained diagnostic controller.

Design principle
----------------
The local LLM (Ollama) is the planner: it observes the current evidence and
selects the next diagnostic action from a dynamically generated set of safe,
eligible actions.  The deterministic policy is a *guard*, not a fixed workflow.
Numerical diagnosis is always produced by deterministic diagnostic tools.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Dict

import numpy as np

from agents.state import DiagnosticState
from tools.ml_tool import ml_diagnosis_tool
from tools.physics_tool import physics_diagnosis_tool
from tools.fusion_tool import fusion_tool
from tools.reanalysis_tool import reanalysis_tool
from tools.root_cause_tool import root_cause_tool
from tools.operator_report_tool import operator_report_tool
from tools.evidence_quality_tool import evidence_quality_tool
from tools.adaptive_investigation_tool import adaptive_investigation_tool
from tools.evidence_stopping_tool import evidence_stopping_tool
from tools.reasoning_trace_tool import reasoning_trace_tool
from tools.confidence_tool import calculate_decision_confidence
from tools.raw_feature_pipeline import number_of_windows
from tools.llm_diagnostic_reasoner import reason_about_case, _extract_json_object
from knowledge.knowledge_graph import build_case_graph, compact_reasoning_context

try:
    from ollama import chat
except Exception:
    chat = None

ALLOWED_ACTIONS = [
    "ml_diagnosis", "physics_diagnosis", "fusion", "evidence_quality",
    "adaptive_investigation", "evidence_stopping", "reanalyze_signal",
    "diagnostic_reasoning", "root_cause_analysis", "operator_report", "finish",
]

ACTION_ALIASES = {
    "ml": "ml_diagnosis", "mldiagnosis": "ml_diagnosis", "ml_diagnosis": "ml_diagnosis",
    "physics": "physics_diagnosis", "physicsdiagnosis": "physics_diagnosis", "physics_diagnosis": "physics_diagnosis", "collect_physics_evidence": "physics_diagnosis", "collect_physics": "physics_diagnosis",
    "fusion": "fusion",
    "evidencequality": "evidence_quality", "evidence_quality": "evidence_quality", "collect_evidence_quality": "evidence_quality", "assess_evidence_quality": "evidence_quality",
    "adaptiveinvestigation": "adaptive_investigation", "adaptive_investigation": "adaptive_investigation",
    "evidencestopping": "evidence_stopping", "evidence_stopping": "evidence_stopping",
    "reanalyze": "reanalyze_signal", "reanalyzesignal": "reanalyze_signal", "reanalyze_signal": "reanalyze_signal",
    "rootcause": "root_cause_analysis", "rootcauseanalysis": "root_cause_analysis", "root_cause_analysis": "root_cause_analysis",
    "operatorreport": "operator_report", "operator_report": "operator_report",
    "finish": "finish",
}

PLANNER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "reason"],
    "properties": {
        "action": {"type": "string"},
        "reason": {"type": "string"}
    }
}



def _compact(value: Any, limit: int = 7000) -> str:
    text = json.dumps(value, default=str, indent=2)
    return text if len(text) <= limit else text[:limit] + "..."


def _normalize_action(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    key = text.replace("_", "")
    return ACTION_ALIASES.get(text, ACTION_ALIASES.get(key, text))


def _factual_summary(state: DiagnosticState) -> Dict[str, Any]:
    """Small, factual state summary supplied to the LLM; no filename labels."""
    ml = state.ml_result if isinstance(state.ml_result, dict) else None
    ph = state.physics_result if isinstance(state.physics_result, dict) else None
    fu = state.fusion_result if isinstance(state.fusion_result, dict) else None
    eq = state.evidence_quality_result if isinstance(state.evidence_quality_result, dict) else None
    ad = state.adaptive_investigation_result if isinstance(state.adaptive_investigation_result, dict) else None
    stop = state.evidence_stopping_result if isinstance(state.evidence_stopping_result, dict) else None
    ra = state.reanalysis_result if isinstance(state.reanalysis_result, dict) else None
    return {
        "ml": None if ml is None else {
            "status": ml.get("status"), "fault": ml.get("fault"),
            "confidence": ml.get("confidence"), "margin": ml.get("margin"),
            "probabilities": ml.get("probabilities"),
        },
        "physics": None if ph is None else {
            "status": ph.get("status"), "diagnosis": ph.get("diagnosis"),
            "strength": ph.get("strength"),
            "BPFO_Normalized_Energy": ph.get("BPFO_Normalized_Energy"),
            "BPFI_Normalized_Energy": ph.get("BPFI_Normalized_Energy"),
            "BPFO_Dominance": ph.get("BPFO_Dominance"),
            "BPFI_Dominance": ph.get("BPFI_Dominance"),
        },
        "fusion": None if fu is None else {
            "status": fu.get("status"), "diagnosis": fu.get("diagnosis"),
            "decision": fu.get("decision"), "action": fu.get("action"),
            "confidence": fu.get("confidence"), "consistency": fu.get("consistency"),
        },
        "evidence_quality": None if eq is None else {
            "status": eq.get("status"), "overall_evidence_quality": eq.get("overall_evidence_quality"),
            "quality_label": eq.get("quality_label"), "recommended_action": eq.get("recommended_action"),
        },
        "adaptive_investigation": None if ad is None else {
            "investigation_required": ad.get("investigation_required"),
            "investigation_depth": ad.get("investigation_depth"),
            "recommended_windows": ad.get("recommended_windows"),
            "stop_allowed": ad.get("stop_allowed"),
            "reason": ad.get("reason"),
        },
        "evidence_stopping": None if stop is None else {
            "stop": stop.get("stop"), "decision": stop.get("decision"),
            "reason": stop.get("reason"), "safety_status": stop.get("safety_status"),
        },
        "llm_diagnostic": None if not isinstance(state.llm_diagnostic_result, dict) else {
            "diagnosis": state.llm_diagnostic_result.get("diagnosis"),
            "fault_location": state.llm_diagnostic_result.get("fault_location"),
            "diagnostic_confidence": state.llm_diagnostic_result.get("diagnostic_confidence"),
            "additional_investigation": state.llm_diagnostic_result.get("additional_investigation"),
        },
        "reanalysis": None if ra is None else {
            "status": ra.get("status"), "windows_analyzed": ra.get("windows_analyzed"),
            "final_diagnosis": ra.get("final_diagnosis"), "diagnosis_consistency": ra.get("diagnosis_consistency"),
            "average_ml_confidence": ra.get("average_ml_confidence"),
            "strong_physics_windows": ra.get("strong_physics_windows"),
            "conflict_detected": ra.get("conflict_detected"), "resolution": ra.get("resolution"),
        },
        "completed_actions": [x.get("action") for x in state.tool_history if x.get("action")],
        "llm_decision_count": sum(1 for x in state.tool_history if x.get("decision_source") == "ollama"),
        "policy_decision_count": sum(1 for x in state.tool_history if x.get("decision_source") == "policy_required"),
        "safety_fallback_count": sum(1 for x in state.tool_history if x.get("decision_source") in {"safety_fallback", "ollama_rationale_replaced"}),
        "investigation_budget_remaining": max(0, int(state.max_steps or 0) - len(state.tool_history)),
        "uncertainty": {
            "ml_physics_conflict": bool(
                ml and ph and ml.get("fault") is not None and ph.get("diagnosis") is not None
                and ml.get("fault") != ph.get("diagnosis")
                and ph.get("diagnosis") != "Healthy_or_Insufficient"
            ),
            "ml_confidence": ml.get("confidence") if ml else None,
            "evidence_quality": eq.get("quality_label") if eq else None,
            "stopping_required": stop is not None and not bool(stop.get("stop")),
        },
    }


def _eligible_actions(state: DiagnosticState) -> list[str]:
    """Return only actions that are scientifically valid at the current state.

    The controller is deliberately *evidence-generation aware*.  Meta-actions
    (quality/stopping/adaptive planning) are tied to the current evidence
    revision and cannot be repeated until new signal evidence is produced.
    This prevents an LLM from creating an investigation loop by repeatedly
    selecting ``reanalyze_signal``.
    """
    def ok(value):
        return isinstance(value, dict) and value.get("status") == "success"

    ml_ok, ph_ok, fu_ok = ok(state.ml_result), ok(state.physics_result), ok(state.fusion_result)
    eq_ok, ad_ok, stop_ok, re_ok = (
        ok(state.evidence_quality_result),
        ok(state.adaptive_investigation_result),
        ok(state.evidence_stopping_result),
        ok(state.reanalysis_result),
    )

    # Numerical prerequisites are deterministic safety constraints.
    if not ml_ok:
        return ["ml_diagnosis"]
    if not ph_ok:
        return ["physics_diagnosis"]
    if not fu_ok:
        return ["fusion"]

    # Evidence quality is mandatory before any investigation/terminal decision.
    # It is a deterministic assessment of the current evidence revision, not a
    # diagnostic action the LLM should be allowed to skip.
    if not eq_ok:
        return ["evidence_quality"]

    # Stopping must be evaluated exactly once for this evidence revision.
    # This is a safety gate: the LLM cannot bypass it by requesting more data
    # after deterministic evidence already satisfies the stop criteria.
    if not stop_ok:
        return ["evidence_stopping"]

    if bool(state.evidence_stopping_result.get("stop")):
        if state.llm_diagnostic_result is None:
            return ["diagnostic_reasoning"]
        if state.rca_result is None:
            return ["root_cause_analysis"]
        if state.operator_report is None:
            return ["operator_report"]
        return ["finish"]

    # Stopping says CONTINUE.  At this point adaptive investigation and new
    # evidence collection are the only meaningful next decisions.  Adaptive
    # planning is allowed once per evidence revision; after it runs, execute
    # its recommendation rather than planning forever.
    if not ad_ok:
        return ["adaptive_investigation", "reanalyze_signal"]

    if bool(state.adaptive_investigation_result.get("investigation_required", True)):
        reanalysis_count = sum(1 for x in state.tool_history if x.get("action") == "reanalyze_signal")
        if reanalysis_count < 2:
            return ["reanalyze_signal"]
        return ["finish"]

    # A conservative planner result says no more evidence is required, but the
    # stopping tool did not permit stopping. Do not invent a diagnosis: close
    # as unresolved and let the report explain that more operator investigation
    # is required.
    return ["finish"]

def _guard_reason(state: DiagnosticState, action: str) -> str:
    action = _normalize_action(action)
    if action == "reanalyze_signal":
        ad = state.adaptive_investigation_result or {}
        return f"Additional evidence is warranted: {ad.get('reason', 'current evidence has not met the stopping criteria.')}"
    if action == "diagnostic_reasoning":
        return "Validated evidence is available; the LLM must now make the case-level diagnostic judgment and explain it."
    if action == "root_cause_analysis":
        return "A safe terminal diagnosis is available; RCA can now translate the validated evidence into a likely mechanism and operator checks."
    if action == "operator_report":
        return "The diagnostic evidence and RCA are available; the operator report can now summarize the decision without adding new numerical claims."
    if action == "finish":
        return "Required evidence, RCA, and operator reporting are complete."
    if action == "evidence_stopping":
        return "Stopping criteria must be evaluated before deciding whether further investigation is necessary."
    if action == "adaptive_investigation":
        return "Evidence quality is available; adaptive investigation determines whether more evidence is needed and how much."
    if action == "evidence_quality":
        return "Current evidence is assessed for quality and cross-source consistency."
    if action == "fusion":
        return "Independent ML and physics evidence are available and can now be reconciled."
    if action == "physics_diagnosis":
        return "The ML result is available; an independent bearing-frequency analysis is useful before accepting the diagnosis."
    return "Initial numerical evidence is required."


def _rationale_is_grounded(state: DiagnosticState, action: str, reason: str) -> bool:
    """Reject common small-LLM contradictions before exposing the rationale."""
    r = str(reason or "").lower()
    ml = state.ml_result if isinstance(state.ml_result, dict) else {}
    ph = state.physics_result if isinstance(state.physics_result, dict) else {}
    eq = state.evidence_quality_result if isinstance(state.evidence_quality_result, dict) else {}
    ra = state.reanalysis_result if isinstance(state.reanalysis_result, dict) else {}

    if ml.get("status") == "success" and any(x in r for x in ("ml tool is not available", "ml evidence is not available", "no ml evidence is available")):
        return False
    if ph.get("status") == "success" and any(x in r for x in ("physics tool is not available", "physics evidence is not available", "no physics evidence is available")):
        return False

    ml_diag = str(ml.get("fault", "")).lower()
    ph_diag = str(ph.get("diagnosis", "")).lower()
    if any(x in r for x in ("disagree", "disagreement", "conflict")) and ml_diag and ph_diag:
        normalized_ph = ph_diag.replace("_or_insufficient", "")
        if ml_diag == normalized_ph:
            return False

    try:
        conf = float(ml.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if "low probability" in r and conf >= 0.70:
        return False

    quality = str(eq.get("quality_label", "")).lower()
    if any(x in r for x in ("very low", "very_low")) and quality in {"high", "moderate"}:
        return False

    if ra.get("status") == "success" and "missing" in r and "reanalysis" in r:
        return False
    return True

def _llm_choose_action(state: DiagnosticState, model: str) -> tuple[str, str, str, str | None]:
    eligible = _eligible_actions(state)
    if not eligible:
        return "finish", "No eligible action remained.", "safety_guard", None

    # If policy leaves exactly one scientifically valid action, do not pretend
    # that an LLM choice occurred. This is an honest safety/prerequisite
    # decision and prevents unnecessary Ollama fallbacks.
    if len(eligible) == 1:
        action = eligible[0]
        return action, _guard_reason(state, action), "policy_required", action

    if chat is None:
        action = eligible[0]
        return action, _guard_reason(state, action), "safety_fallback", None

    summary = _factual_summary(state)
    action_guidance = {
        "ml_diagnosis": "Generate initial ML evidence from the raw vibration windows.",
        "physics_diagnosis": "Generate independent bearing characteristic-frequency evidence.",
        "fusion": "Reconcile completed ML and physics results.",
        "evidence_quality": "Assess the strength and consistency of evidence; this action does not generate new signal data.",
        "adaptive_investigation": "Plan whether more evidence is justified and the appropriate investigation depth.",
        "evidence_stopping": "Apply conservative deterministic stopping criteria.",
        "reanalyze_signal": "Generate genuinely new numerical evidence from additional windows across the recording.",
        "diagnostic_reasoning": "Make the final evidence-grounded diagnostic judgment, fault location, mechanism and operator recommendation.",
        "root_cause_analysis": "Translate the validated LLM diagnosis into a safe structured RCA record.",
        "operator_report": "Summarize the validated diagnosis, evidence, uncertainty and operator recommendation.",
        "finish": "Close the case only when the terminal requirements are already satisfied.",
    }

    def call_planner(extra_instruction: str = "", num_predict: int = 96):
        prompt = f"""You are the planning brain of a safety-constrained industrial bearing diagnostic agent.

Your job is ONLY to choose the next action. Numerical diagnosis is produced by deterministic tools.

CURRENT EVIDENCE:
{_compact(summary)}

ELIGIBLE ACTIONS (you MUST choose exactly one of these):
{json.dumps({a: action_guidance[a] for a in eligible}, indent=2)}

DECISION RULES:
1. Choose only an eligible action.
2. Choose additional signal evidence only when the CURRENT uncertainty justifies it.
3. Do not request reanalysis when the current evidence already supports safe stopping.
4. If a safety/assessment action is the only eligible action, it is policy-required and not an LLM choice.
4. Never repeat an action merely because it was previously executed; choose it only if the state says it is eligible.
5. Do not invent measurements, diagnoses, conflicts, or missing evidence.
6. A terminal action such as finish is valid only if it is explicitly in the eligible list.
7. Return JSON only.

{extra_instruction}

JSON:
{{"action":"<eligible action>","reason":"one short factual sentence based only on CURRENT EVIDENCE"}}
"""
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": "You are an industrial diagnostic planner. Follow the eligible-action constraint exactly. Never invent evidence."},
                {"role": "user", "content": prompt},
            ],
            format=PLANNER_SCHEMA,
            options={"temperature": 0, "num_ctx": 1536, "num_predict": int(num_predict)},
            keep_alive="10m",
        )
        try:
            response = chat(**kwargs, think=False)
        except TypeError:
            # Compatibility with older Ollama Python clients and unit-test
            # doubles that do not expose `think` or `keep_alive`.
            try:
                response = chat(**kwargs)
            except TypeError:
                kwargs.pop("keep_alive", None)
                response = chat(**kwargs)
        content = response.get("message", {}).get("content", "") if isinstance(response, dict) else getattr(getattr(response, "message", None), "content", "")
        return _extract_json_object(content)

    try:
        try:
            data = call_planner()
        except Exception as first_parse_error:
            # Qwen may occasionally truncate a tiny JSON response at the
            # generation boundary. Retry once with a larger bounded budget;
            # never synthesize a planner action from malformed text.
            data = call_planner(
                "IMPORTANT: The previous response was incomplete or invalid JSON. "
                "Return one complete compact JSON object now, with no markdown or prose.",
                num_predict=160,
            )
        proposed = data.get("action")
        action = _normalize_action(proposed)
        reason = str(data.get("reason") or "").strip()

        if action not in eligible:
            # One corrective retry is part of the planner protocol. It is not a
            # diagnostic fallback: the same evidence and eligible set are used.
            data = call_planner(
                f"CORRECTION: Your previous action {proposed!r} was invalid. "
                f"Select exactly one of {eligible!r}; do not output any other string."
            )
            proposed = data.get("action")
            action = _normalize_action(proposed)
            reason = str(data.get("reason") or "").strip()

        if action not in eligible:
            safe = eligible[0]
            return safe, _guard_reason(state, safe), "safety_fallback", str(proposed)

        if not _rationale_is_grounded(state, action, reason):
            return action, _guard_reason(state, action), "ollama_rationale_replaced", str(proposed)

        return action, reason, "ollama", str(proposed)

    except Exception as exc:
        action = eligible[0]
        return action, f"Ollama planning failed ({type(exc).__name__}); a deterministic safe action was selected.", "safety_fallback", None

def _window_count_for_case(input_data: Dict[str, Any]) -> int:
    from tools.raw_mat_loader import load_bearing_mat
    case = load_bearing_mat(input_data["mat_file"], validate=True)
    return number_of_windows(len(case["signal"]), case["sampling_rate"])


def _run_reanalysis(input_data: Dict[str, Any], recommended: int = 10):
    total = _window_count_for_case(input_data)
    n = max(1, min(int(recommended), total))
    if n == total:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, n, dtype=int).tolist()
        indices = list(dict.fromkeys(indices))
    windows = []
    for idx in indices:
        c = {"mat_file": input_data["mat_file"], "window": idx}
        ml = ml_diagnosis_tool(c)
        ph = physics_diagnosis_tool(c)
        fu = fusion_tool(ml, ph) if ml.get("status") == "success" and ph.get("status") == "success" else {"status": "failed"}
        windows.append({"window": idx, "ml_result": ml, "physics_result": ph, "fusion_result": fu})
    return windows


def _finalize_from_evidence(state: DiagnosticState) -> None:
    if isinstance(state.llm_diagnostic_result, dict) and state.llm_diagnostic_result.get("status") == "success":
        state.final_diagnosis = state.llm_diagnostic_result.get("diagnosis")
        state.final_consistency = (state.reanalysis_result or {}).get("diagnosis_consistency", (state.fusion_result or {}).get("consistency"))
    elif state.reanalysis_result:
        resolution = str(state.reanalysis_result.get("resolution") or "")
        if resolution == "NO_CONFIRMED_FAULT":
            state.final_diagnosis = "NO_CONFIRMED_FAULT"
        else:
            state.final_diagnosis = (state.reanalysis_result.get("final_diagnosis") or state.reanalysis_result.get("majority_diagnosis"))
        state.final_consistency = state.reanalysis_result.get("diagnosis_consistency", 0.0)
    elif state.fusion_result:
        state.final_diagnosis = state.fusion_result.get("diagnosis")
    elif state.ml_result:
        state.final_diagnosis = state.ml_result.get("fault")

    # No synthetic case-level probability is exposed. Source-specific ML
    # probabilities remain available in the evidence section.
    score, score_type, explanation = calculate_decision_confidence(state)
    state.final_confidence = score
    state.final_confidence_type = score_type
    state.final_confidence_explanation = explanation


def _terminal_ready(state: DiagnosticState) -> bool:
    """Return True only when the case has a safe terminal decision and report."""
    stop = state.evidence_stopping_result or {}
    return (
        isinstance(stop, dict)
        and stop.get("status") == "success"
        and bool(stop.get("stop"))
        and state.rca_result is not None
        and state.operator_report is not None
        and state.final_diagnosis is not None
    )


def _derive_final_resolution(state: DiagnosticState) -> str:
    """Return a semantic case-level resolution, not a generic STOP flag."""
    if state.reanalysis_result:
        resolution = (state.reanalysis_result or {}).get("resolution")
        if resolution:
            return str(resolution)

    fusion = state.fusion_result or {}
    decision = str(fusion.get("decision") or "")
    if decision == "CONFIRMED":
        return "CONFIRMED_BY_INDEPENDENT_AGREEMENT"
    if decision == "PROVISIONAL":
        return "PROVISIONAL_FUSION"
    if decision == "ML_SUPPORTED":
        return "ML_SUPPORTED_LIMITED_PHYSICS"

    return str((state.evidence_stopping_result or {}).get("decision") or "COMPLETED")


def _complete_state(state: DiagnosticState) -> DiagnosticState:
    _finalize_from_evidence(state)
    state.reasoning_trace = reasoning_trace_tool(
        state.ml_result, state.physics_result, state.evidence_quality_result,
        state.adaptive_investigation_result, state.evidence_stopping_result,
        state.reanalysis_result,
        [x.get("action") for x in state.tool_history if x.get("action")],
    )
    state.status = "COMPLETED"
    state.final_resolution = _derive_final_resolution(state)
    state.completed_at = datetime.now(timezone.utc).isoformat()
    return state


def run_agentic_case(case_id: str, input_data: Dict[str, Any], model: str | None = None, max_steps: int = 20) -> DiagnosticState:
    state = DiagnosticState(case_id=case_id, input_data=input_data)
    model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
    max_steps = max(1, int(max_steps))
    state.run_id = str(uuid4())
    state.llm_model = model
    state.llm_temperature = 0.0
    state.max_steps = max_steps
    state.started_at = datetime.now(timezone.utc).isoformat()
    # Build a case-aware KG snapshot as evidence evolves.
    state.knowledge_graph_context = compact_reasoning_context(build_case_graph(state))

    run_started = time.perf_counter()
    for step in range(max_steps):
        action, llm_reason, source, proposed = _llm_choose_action(state, model)
        action_started = time.perf_counter()
        guard = _guard_reason(state, action)
        state.tool_history.append({
            "type": "LLM_DECISION" if source == "ollama" else "POLICY_DECISION" if source == "policy_required" else "SAFETY_DECISION",
            "step": step + 1,
            "action": action,
            "reason": llm_reason or guard,
            "llm_proposal": proposed or action,
            "decision_source": source,
            "safety_guard": guard,
            "counts_toward_budget": True,
        })

        if action == "ml_diagnosis":
            state.ml_result = ml_diagnosis_tool(input_data)
        elif action == "physics_diagnosis":
            state.physics_result = physics_diagnosis_tool(input_data)
        elif action == "fusion":
            state.fusion_result = fusion_tool(state.ml_result, state.physics_result)
        elif action == "evidence_quality":
            state.evidence_quality_result = evidence_quality_tool(state.ml_result, state.physics_result, state.reanalysis_result)
        elif action == "adaptive_investigation":
            state.adaptive_investigation_result = adaptive_investigation_tool(state.ml_result, state.physics_result, state.evidence_quality_result, state.reanalysis_result)
        elif action == "evidence_stopping":
            state.evidence_stopping_result = evidence_stopping_tool(state.ml_result, state.physics_result, state.evidence_quality_result, state.reanalysis_result)
        elif action == "diagnostic_reasoning":
            state.knowledge_graph_context = compact_reasoning_context(build_case_graph(state))
            state.llm_diagnostic_result = reason_about_case(state, model)
            # The action becomes an actual LLM diagnostic judgment. The policy
            # only made the action eligible; it did not supply the diagnosis.
            if state.tool_history:
                if state.llm_diagnostic_result.get("status") == "success":
                    state.tool_history[-1]["decision_source"] = "ollama"
                    state.tool_history[-1]["type"] = "LLM_DIAGNOSTIC"
                    state.tool_history[-1]["llm_proposal"] = state.llm_diagnostic_result.get("diagnosis")
                    state.tool_history[-1]["reason"] = state.llm_diagnostic_result.get("reasoning") or state.tool_history[-1]["reason"]
                else:
                    state.tool_history[-1]["decision_source"] = "safety_fallback"
                    state.tool_history[-1]["type"] = "SAFETY_DECISION"
            if state.llm_diagnostic_result.get("status") != "success":
                state.llm_diagnostic_result["fallback_diagnosis"] = None
            else:
                state.final_diagnosis = state.llm_diagnostic_result.get("diagnosis")
        elif action == "reanalyze_signal":
            recommended = 5
            if isinstance(state.adaptive_investigation_result, dict):
                try:
                    recommended = max(3, min(20, int(state.adaptive_investigation_result.get("recommended_windows", 5))))
                except Exception:
                    pass
            state.window_results = _run_reanalysis(input_data, recommended)
            state.reanalysis_result = reanalysis_tool(state.window_results)
            # Reanalysis is new evidence. Force quality/stopping to be evaluated
            # again, but let the LLM decide whether that is needed next.
            state.evidence_quality_result = None
            state.adaptive_investigation_result = None
            state.evidence_stopping_result = None
            state.knowledge_graph_context = compact_reasoning_context(build_case_graph(state))
        elif action == "root_cause_analysis":
            _finalize_from_evidence(state)
            state.rca_result = root_cause_tool(state)
        elif action == "operator_report":
            _finalize_from_evidence(state)
            state.operator_report = operator_report_tool(state)
        elif action == "finish":
            _finalize_from_evidence(state)
            state.reasoning_trace = reasoning_trace_tool(
                state.ml_result, state.physics_result, state.evidence_quality_result,
                state.adaptive_investigation_result, state.evidence_stopping_result,
                state.reanalysis_result,
                [x.get("action") for x in state.tool_history if x.get("action")],
            )
            state.status = "COMPLETED"
            state.final_resolution = _derive_final_resolution(state)
            state.completed_at = datetime.now(timezone.utc).isoformat()
            state.total_duration_ms = round((time.perf_counter() - run_started) * 1000.0, 2)
            return state

        action_elapsed_ms = round((time.perf_counter() - action_started) * 1000.0, 2)
        if state.tool_history:
            state.tool_history[-1]["duration_ms"] = action_elapsed_ms

        attr = {
            "ml_diagnosis": "ml_result", "physics_diagnosis": "physics_result", "fusion": "fusion_result",
            "evidence_quality": "evidence_quality_result", "adaptive_investigation": "adaptive_investigation_result",
            "evidence_stopping": "evidence_stopping_result", "reanalyze_signal": "reanalysis_result", "diagnostic_reasoning": "llm_diagnostic_result",
            "root_cause_analysis": "rca_result", "operator_report": "operator_report",
        }.get(action)
        if attr:
            state.evidence.append({"tool": action, "result": getattr(state, attr)})

        # Reaching a safe terminal state is completion even if the last
        # required action consumed the configured step budget.  A synthetic
        # ``finish`` step is never required just to flip the status flag.
        if _terminal_ready(state):
            state.total_duration_ms = round((time.perf_counter() - run_started) * 1000.0, 2)
            return _complete_state(state)

    # The configured budget limits investigation/planning, not the mandatory
    # terminalization of an already-safe case.  If the deterministic stopping
    # gate has approved termination, RCA and the operator report are required
    # completion artifacts and may run after the investigation budget without
    # consuming another investigation step.  This prevents a valid case from
    # becoming NEEDS_INVESTIGATION merely because the budget ended on RCA.
    stop = state.evidence_stopping_result or {}
    # Establish the deterministic fallback diagnosis before checking whether
    # terminal finalization is allowed.
    _finalize_from_evidence(state)
    if (
        isinstance(stop, dict)
        and stop.get("status") == "success"
        and bool(stop.get("stop"))
        and state.final_diagnosis is not None
    ):
        # Diagnostic reasoning is a terminal interpretation step rather than
        # an investigation-budget action. If the budget ended immediately
        # before it, still execute it so a valid case receives an LLM judgment.
        if state.llm_diagnostic_result is None:
            state.knowledge_graph_context = compact_reasoning_context(build_case_graph(state))
            state.llm_diagnostic_result = reason_about_case(state, model)
            state.tool_history.append({
                "type": "LLM_DIAGNOSTIC",
                "step": "post-budget",
                "action": "diagnostic_reasoning",
                "reason": (state.llm_diagnostic_result.get("reason") or "LLM produced a case-level diagnostic judgment.") if isinstance(state.llm_diagnostic_result, dict) else "LLM diagnostic reasoning executed.",
                "llm_proposal": state.llm_diagnostic_result.get("diagnosis") if isinstance(state.llm_diagnostic_result, dict) else None,
                "decision_source": "ollama" if isinstance(state.llm_diagnostic_result, dict) and state.llm_diagnostic_result.get("status") == "success" else "safety_fallback",
                "safety_guard": "Terminal evidence is fixed; LLM interpretation is constrained to the verified evidence and canonical diagnosis vocabulary.",
                "counts_toward_budget": False,
            })
            _finalize_from_evidence(state)
        if state.rca_result is None:
            _finalize_from_evidence(state)
            state.rca_result = root_cause_tool(state)
            state.tool_history.append({
                "type": "TERMINAL_FINALIZATION",
                "step": "post-budget",
                "action": "root_cause_analysis",
                "reason": "Terminal diagnosis was already safe; RCA is a mandatory completion artifact and does not consume investigation budget.",
                "llm_proposal": None,
                "decision_source": "terminal_required",
                "safety_guard": "Safe terminal diagnosis available.",
                "counts_toward_budget": False,
            })
        if state.operator_report is None:
            _finalize_from_evidence(state)
            state.operator_report = operator_report_tool(state)
            state.tool_history.append({
                "type": "TERMINAL_FINALIZATION",
                "step": "post-budget",
                "action": "operator_report",
                "reason": "Terminal diagnosis and RCA are available; the operator report is a mandatory completion artifact and does not consume investigation budget.",
                "llm_proposal": None,
                "decision_source": "terminal_required",
                "safety_guard": "Safe terminal diagnosis available.",
                "counts_toward_budget": False,
            })
        state.total_duration_ms = round((time.perf_counter() - run_started) * 1000.0, 2)
        return _complete_state(state)

    # Step-budget exhaustion is unresolved only when the stopping gate has not
    # established a safe terminal decision. Preserve that distinction clearly.
    _finalize_from_evidence(state)
    state.status = "NEEDS_INVESTIGATION"
    state.final_resolution = "MAX_STEPS_REACHED"
    state.completed_at = datetime.now(timezone.utc).isoformat()
    state.reasoning_trace = {"status": "max_steps_reached", "max_steps": max_steps}
    state.total_duration_ms = round((time.perf_counter() - run_started) * 1000.0, 2)
    return state
