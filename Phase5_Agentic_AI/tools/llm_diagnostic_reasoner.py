"""Evidence-grounded LLM diagnostic reasoning.

The LLM makes the case-level diagnostic judgment after deterministic evidence
has been collected. A verifier constrains the output to observed evidence and
known bearing relationships. Numerical values remain tool-owned facts.
"""
from __future__ import annotations
import json, re
from typing import Any

try:
    from ollama import chat
except Exception:
    chat = None

ALLOWED_DIAGNOSES = {"Healthy", "InnerRaceFault", "OuterRaceFault", "NO_CONFIRMED_FAULT"}

DIAGNOSTIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["diagnosis", "fault_location", "reasoning", "observed_evidence",
                  "inferred_mechanism", "possible_root_causes", "root_cause_status",
                  "operator_actions", "diagnostic_confidence", "additional_investigation"],
    "properties": {
        "diagnosis": {"type": "string", "enum": ["Healthy", "InnerRaceFault", "OuterRaceFault", "NO_CONFIRMED_FAULT"]},
        "fault_location": {"type": ["string", "null"]},
        "reasoning": {"type": "string"},
        "observed_evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "inferred_mechanism": {"type": ["string", "null"]},
        "possible_root_causes": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "root_cause_status": {"type": "string", "enum": ["inferred", "requires_physical_verification", "not_applicable"]},
        "operator_actions": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
        "diagnostic_confidence": {"type": "string", "enum": ["high", "moderate", "low", "not_established"]},
        "additional_investigation": {"type": "string"},
        "semantic_corrections": {"type": "array", "items": {"type": "string"}, "maxItems": 4}
    }
}


def _extract_json_object(content: str) -> dict[str, Any]:
    """Parse a local-LLM JSON object with conservative serialization repair.

    This function never fabricates missing semantic fields. It only removes
    markdown fences, extracts the outer JSON object, quotes simple bare keys,
    and removes trailing commas. Truncated JSON is reported as a parse failure
    so the caller can retry with a larger generation budget.
    """
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("LLM response contains no JSON object.")
    text = text[start:]

    # Find the last complete outer object while respecting quoted strings.
    depth = 0
    in_string = False
    escaped = False
    last_complete = -1
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete = i
                break
    if last_complete < 0:
        raise ValueError("LLM returned truncated JSON; the closing object was not generated.")
    text = text[:last_complete + 1]

    candidates = [text]
    repaired = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:', r'\1"\2":', text)
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    if repaired != text:
        candidates.append(repaired)

    last_error = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
            last_error = ValueError("LLM response was not a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"LLM returned invalid JSON: {last_error}")


def _compact(v: Any, limit: int = 12000) -> str:
    s = json.dumps(v, default=str, indent=2)
    return s if len(s) <= limit else s[:limit] + "..."


def _case_evidence(state) -> dict[str, Any]:
    """Build a compact, high-signal evidence packet for local LLM inference.

    The LLM does not need every raw window feature to make the final case-level
    judgment. Sending only authoritative summary fields materially reduces
    prompt processing latency while preserving the measured evidence needed for
    diagnosis, location and uncertainty reasoning.
    """
    ml = state.ml_result or {}
    ph = state.physics_result or {}
    fu = state.fusion_result or {}
    eq = state.evidence_quality_result or {}
    ra = state.reanalysis_result or {}
    ad = state.adaptive_investigation_result or {}
    stop = state.evidence_stopping_result or {}
    return {
        "ml": {
            "fault": ml.get("fault"), "confidence": ml.get("confidence"),
            "margin": ml.get("margin"),
            "probabilities": ml.get("probabilities"),
            "features": {k: ml.get("features", {}).get(k) for k in (
                "Envelope_RMS", "Kurtosis", "Dominant_Frequency_Hz",
                "Spectral_Centroid_Hz", "Peak_Magnitude"
            ) if ml.get("features", {}).get(k) is not None},
        },
        "physics": {
            "diagnosis": ph.get("diagnosis"), "strength": ph.get("strength"),
            "BPFI_Normalized_Energy": ph.get("BPFI_Normalized_Energy"),
            "BPFO_Normalized_Energy": ph.get("BPFO_Normalized_Energy"),
            "BPFI_Dominance": ph.get("BPFI_Dominance"),
            "BPFO_Dominance": ph.get("BPFO_Dominance"),
            "Fault_Energy": ph.get("Fault_Energy"),
            "BPFI_Harmonic_Normalized_Energy": ph.get("BPFI_Harmonic_Normalized_Energy"),
            "BPFO_Harmonic_Normalized_Energy": ph.get("BPFO_Harmonic_Normalized_Energy"),
        },
        "fusion": {
            "diagnosis": fu.get("diagnosis"), "decision": fu.get("decision"),
            "confidence": fu.get("confidence"), "consistency": fu.get("consistency"),
        },
        "evidence_quality": {
            "overall": eq.get("overall_evidence_quality"),
            "label": eq.get("quality_label"),
            "cross_source": eq.get("cross_source_quality"),
            "multi_window": eq.get("multi_window_evidence_quality"),
        },
        "reanalysis": {
            "windows_analyzed": ra.get("windows_analyzed"),
            "final_diagnosis": ra.get("final_diagnosis"),
            "diagnosis_consistency": ra.get("diagnosis_consistency"),
            "average_ml_confidence": ra.get("average_ml_confidence"),
            "strong_physics_windows": ra.get("strong_physics_windows"),
            "conflict_detected": ra.get("conflict_detected"),
            "resolution": ra.get("resolution"),
        },
        "adaptive_investigation": {
            "required": ad.get("investigation_required"),
            "depth": ad.get("investigation_depth"),
            "recommended_windows": ad.get("recommended_windows"),
            "reason": ad.get("reason"),
        },
        "stopping": {"stop": stop.get("stop"), "decision": stop.get("decision"), "reason": stop.get("reason")},
    }


def _allowed_diagnoses_from_evidence(state) -> set[str]:
    # The LLM can choose among the canonical case states, but a fault diagnosis
    # must be supported by at least one independent evidence source.
    allowed = {"NO_CONFIRMED_FAULT"}
    ml = state.ml_result or {}
    ph = state.physics_result or {}
    ra = state.reanalysis_result or {}
    for d in (ml.get("fault"), ph.get("diagnosis"), ra.get("final_diagnosis"), ra.get("majority_diagnosis")):
        if d == "InnerRaceFault": allowed.add(d)
        if d == "OuterRaceFault": allowed.add(d)
        if d == "Healthy": allowed.add(d)
    return allowed


def _expected_case_diagnosis(state) -> str | None:
    """Return the strongest already-validated terminal diagnosis, if one exists.

    This is a constraint on the LLM's interpretation, not a replacement for
    reasoning. Once deterministic multi-window evidence has reached a terminal
    resolution, the model is only allowed to explain that resolution.
    """
    ra = state.reanalysis_result or {}
    ra_diag = ra.get("final_diagnosis") or ra.get("majority_diagnosis")
    if ra.get("resolution") and ra_diag in {"NO_CONFIRMED_FAULT", "InnerRaceFault", "OuterRaceFault"}:
        return ra_diag
    fusion = state.fusion_result or {}
    fdiag = fusion.get("diagnosis")
    if fusion.get("decision") == "CONFIRMED" and fdiag in {"InnerRaceFault", "OuterRaceFault"}:
        return fdiag
    # A Healthy ML result with low/insufficient physics is conservatively
    # expressed to the operator as no confirmed fault.
    ph = state.physics_result or {}
    if fdiag == "Healthy" and ph.get("diagnosis") == "Healthy_or_Insufficient":
        return "NO_CONFIRMED_FAULT"
    return None


def _diagnostic_schema_for_state(state) -> dict[str, Any]:
    """Build a constrained schema so Qwen cannot select an unsupported class."""
    import copy
    schema = copy.deepcopy(DIAGNOSTIC_SCHEMA)
    expected = _expected_case_diagnosis(state)
    if expected:
        schema["properties"]["diagnosis"] = {"type": "string", "enum": [expected]}
    return schema


def _relevant_graph_relationships(state) -> list[dict[str, Any]]:
    """Expose only case-relevant KG relations to the LLM.

    Generic domain mappings such as BPFI -> Inner Race are useful knowledge but
    can bias a local LLM toward a fault that the current case does not support.
    For a no-confirmation case, only case evidence/resolution relations are
    exposed.
    """
    graph = state.knowledge_graph_context or {}
    rels = graph.get("relationships", []) or []
    expected = _expected_case_diagnosis(state)
    case_rels = [r for r in rels if r.get("scope") == "case_evidence"]
    if expected == "NO_CONFIRMED_FAULT":
        return [r for r in case_rels if r.get("relation") in {
            "contains_signal", "has_evidence", "current_resolution"
        }][:12]
    # For a confirmed fault, include case evidence plus only domain relations
    # connected to the selected fault location/characteristic frequency.
    target = "inner_race" if expected == "InnerRaceFault" else "outer_race" if expected == "OuterRaceFault" else None
    relevant = []
    for r in rels:
        if r.get("scope") == "case_evidence":
            relevant.append(r)
        elif target and (r.get("target") == target or r.get("source") == target):
            relevant.append(r)
    return relevant[:16]


def _normalize_location(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _semantic_grounding(result: dict[str, Any], state) -> tuple[dict[str, Any], str | None]:
    """Apply transparent semantic safety constraints to the LLM judgment.

    The LLM remains responsible for the case-level interpretation. This verifier
    only removes/normalizes fields that would contradict the selected diagnosis
    or the authoritative multi-window resolution. Every normalization is logged
    in ``semantic_corrections`` for research traceability.
    """
    out = dict(result)
    corrections = list(out.get("semantic_corrections") or [])
    diagnosis = out.get("diagnosis")
    ra = state.reanalysis_result or {}
    ra_diag = ra.get("final_diagnosis") or ra.get("majority_diagnosis")

    if ra_diag == "NO_CONFIRMED_FAULT":
        if diagnosis != "NO_CONFIRMED_FAULT":
            return out, "LLM diagnosis conflicts with the terminal no-confirmation resolution."
        if out.get("fault_location") not in (None, "", "None", "none", "Not established", "No confirmed fault", "No confirmed fault location"):
            corrections.append("Cleared unsupported fault location because the case resolution is NO_CONFIRMED_FAULT.")
        if out.get("inferred_mechanism") not in (None, "", "None", "none", "No fault mechanism is established."):
            corrections.append("Cleared fault mechanism because the case resolution is NO_CONFIRMED_FAULT.")
        if out.get("possible_root_causes"):
            corrections.append("Cleared possible root causes because no fault was confirmed.")
        out["fault_location"] = None
        out["inferred_mechanism"] = None
        out["possible_root_causes"] = []
        out["root_cause_status"] = "not_applicable"
        out["diagnostic_confidence"] = "not_established"
        out["semantic_corrections"] = corrections
        return out, None

    if diagnosis == "NO_CONFIRMED_FAULT":
        loc = _normalize_location(out.get("fault_location"))
        if loc not in {"", "none", "not established", "no confirmed fault", "no confirmed fault location"}:
            corrections.append("Cleared specific fault location because diagnosis is NO_CONFIRMED_FAULT.")
        if out.get("inferred_mechanism") not in (None, "", "None", "none"):
            corrections.append("Cleared inferred mechanism because no fault was confirmed.")
        if out.get("possible_root_causes"):
            corrections.append("Cleared possible root causes because no fault was confirmed.")
        out["fault_location"] = None
        out["inferred_mechanism"] = None
        out["possible_root_causes"] = []
        out["root_cause_status"] = "not_applicable"
        out["diagnostic_confidence"] = "not_established"
        out["semantic_corrections"] = corrections
        return out, None

    expected = {"InnerRaceFault": "Inner Race", "OuterRaceFault": "Outer Race"}.get(diagnosis)
    if expected:
        loc = _normalize_location(out.get("fault_location"))
        if expected.lower() not in loc:
            # The canonical fault class is evidence-grounded; expose only the
            # corresponding physical location rather than an internally
            # contradictory LLM location. Record the correction explicitly.
            corrections.append(f"Normalized fault location to {expected} to match the canonical diagnosis {diagnosis}.")
            out["fault_location"] = expected
        ph = state.physics_result or {}
        ph_diag = ph.get("diagnosis")
        ph_strength = str(ph.get("strength") or "").upper()
        if ph_diag != diagnosis and ph_strength == "STRONG":
            return out, "LLM fault diagnosis conflicts with strong independent physics evidence."
        if not out.get("inferred_mechanism"):
            return out, "A supported fault diagnosis requires a stated mechanism."
        if not isinstance(out.get("possible_root_causes"), list):
            return out, "Possible root causes must be a list."
        if out.get("root_cause_status") == "inferred" and out.get("possible_root_causes"):
            out["root_cause_status"] = "requires_physical_verification"
            corrections.append("Changed root-cause status to requires_physical_verification because physical causes are not directly measured.")
    out["semantic_corrections"] = corrections
    return out, None


def _grounded(result: dict[str, Any], state) -> tuple[bool, str]:
    diagnosis = result.get("diagnosis")
    if diagnosis not in ALLOWED_DIAGNOSES:
        return False, "Diagnosis is outside the canonical diagnostic vocabulary."
    if diagnosis not in _allowed_diagnoses_from_evidence(state):
        return False, "Diagnosis is not supported by any available evidence source."

    # When multi-window reanalysis has already produced a terminal resolution,
    # the LLM must respect that stronger case-level evidence. This preserves
    # agency without allowing a language model to overturn measured evidence.
    ra = state.reanalysis_result or {}
    ra_diag = ra.get("final_diagnosis") or ra.get("majority_diagnosis")
    if ra_diag and ra.get("resolution") not in {None, ""} and diagnosis != ra_diag:
        return False, "LLM diagnosis conflicts with the validated multi-window resolution."
    fusion = state.fusion_result or {}
    fusion_diag = fusion.get("diagnosis")
    # NO_CONFIRMED_FAULT is the canonical conservative terminal state for a
    # case whose measured evidence does not establish a fault.  It is
    # semantically compatible with a deterministic ML Healthy result and
    # Healthy_or_Insufficient physics; it must not be rejected as a
    # contradiction merely because the fusion layer stores "Healthy".
    healthy_alias = {"Healthy", "Healthy_or_Insufficient", "NO_CONFIRMED_FAULT"}
    if fusion.get("decision") in {"CONFIRMED", "ML_SUPPORTED"} and fusion_diag:
        if diagnosis == "NO_CONFIRMED_FAULT" and fusion_diag in healthy_alias:
            pass
        elif diagnosis in healthy_alias and fusion_diag in healthy_alias:
            pass
        elif diagnosis != fusion_diag:
            return False, "LLM diagnosis conflicts with the validated fusion decision."
    if not str(result.get("reasoning") or "").strip():
        return False, "LLM reasoning is empty."
    actions = result.get("operator_actions")
    if not isinstance(actions, list) or not actions:
        return False, "No operator actions were provided."
    # Prevent the LLM from manufacturing measurements. Numerical evidence must
    # be cited by the tool fields, not invented in prose.
    reasoning = str(result.get("reasoning", "")).lower()
    if "bpf" in reasoning and not ((state.physics_result or {}).get("status") == "success"):
        return False, "LLM referenced bearing characteristic-frequency evidence that is unavailable."
    return True, "grounded"


def reason_about_case(state, model: str) -> dict[str, Any]:
    if chat is None:
        return {"status":"fallback", "reason":"Ollama Python client unavailable."}
    graph_context = state.knowledge_graph_context or {}
    expected_diagnosis = _expected_case_diagnosis(state)
    prompt = f"""You are the diagnostic reasoning brain of an industrial bearing condition-monitoring agent.

The deterministic tools have already measured the current case. You must now make the CASE-LEVEL diagnostic judgment.
Do not calculate or invent measurements. Use only the supplied evidence and knowledge-graph relationships.

Your responsibilities:
1. Select the most defensible diagnosis from: Healthy, InnerRaceFault, OuterRaceFault, NO_CONFIRMED_FAULT.
2. State the physical fault location when a fault is supported.
3. Explain why the evidence supports or does not support the diagnosis.
4. Separate observed evidence from inferred mechanism/cause.
5. Give likely mechanism(s), but never claim an unmeasured physical cause as confirmed.
6. Give prioritized operator/maintenance checks. These are recommendations, not automatic machine control.
7. If evidence remains insufficient, say so explicitly and recommend additional evidence/inspection.
8. SEMANTIC SAFETY: If diagnosis is NO_CONFIRMED_FAULT, fault_location MUST be null, inferred_mechanism MUST be null, possible_root_causes MUST be [], root_cause_status MUST be not_applicable, and diagnostic_confidence MUST be not_established.
9. For InnerRaceFault use fault_location Inner Race; for OuterRaceFault use fault_location Outer Race. Never report a different location.
10. If a VALIDATED CASE CONSTRAINT is provided below, you MUST use that exact diagnosis. Do not override it with generic knowledge-graph relationships.

CURRENT CASE EVIDENCE:
{_compact(_case_evidence(state))}

VALIDATED CASE CONSTRAINT:
{expected_diagnosis or "No single terminal class; choose only from the measured evidence."}

KNOWLEDGE-GRAPH RELATIONSHIPS (strictly case-relevant):
{_compact({"relationships": _relevant_graph_relationships(state)}, 3000)}

Return JSON only. Keep the response compact:
- reasoning: 1-2 short sentences, maximum 35 words
- observed_evidence: maximum 2 items
- possible_root_causes: maximum 2 items
- operator_actions: maximum 3 items
- every other text field: maximum 18 words
Use exactly these fields:
{{
  "diagnosis": "...",
  "fault_location": "<Inner Race|Outer Race|null>",
  "reasoning": "...",
  "observed_evidence": ["..."],
  "inferred_mechanism": "<mechanism|null>",
  "possible_root_causes": ["..."],
  "root_cause_status": "inferred|requires_physical_verification|not_applicable",
  "operator_actions": ["..."],
  "diagnostic_confidence": "high|moderate|low",
  "additional_investigation": "..."
}}
"""
    try:
        kwargs = dict(
            model=model,
            messages=[
                {"role":"system","content":"You are a conservative industrial diagnostic reasoner. Evidence first. Never invent sensor measurements."},
                {"role":"user","content":prompt},
            ],
            format=_diagnostic_schema_for_state(state),
            options={
                "temperature": 0,
                "num_ctx": int(__import__("os").getenv("OLLAMA_DIAGNOSTIC_CTX", "1536")),
                "num_predict": int(__import__("os").getenv("OLLAMA_DIAGNOSTIC_TOKENS", "256")),
            },
            keep_alive=__import__("os").getenv("OLLAMA_KEEP_ALIVE", "30m"),
        )
        # Qwen3 can spend substantial time in hidden thinking mode on local
        # CPU/RAM-only hardware. Structured diagnostic reasoning here is a
        # constrained evidence synthesis task, so concise generation is the
        # appropriate production mode. Older Ollama clients may not accept
        # `think`; retry without it for compatibility.
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
        content = response.get("message",{}).get("content","") if isinstance(response,dict) else getattr(getattr(response,"message",None),"content","")
        try:
            result = _extract_json_object(content)
        except Exception as first_parse_error:
            # Qwen can occasionally stop exactly at the generation limit and
            # leave an array/object unclosed. Do one bounded retry with a
            # larger output budget. We retry only serialization failure; no
            # semantic diagnosis is synthesized locally.
            retry_kwargs = dict(kwargs)
            retry_options = dict(retry_kwargs.get("options") or {})
            retry_options["num_predict"] = max(384, int(retry_options.get("num_predict", 256)))
            retry_kwargs["options"] = retry_options
            retry_messages = list(retry_kwargs["messages"])
            retry_messages[-1] = dict(retry_messages[-1])
            retry_messages[-1]["content"] += "\nIMPORTANT: The previous response was truncated or invalid JSON. Return the complete compact JSON object now. Do not use markdown or prose."
            retry_kwargs["messages"] = retry_messages
            try:
                try:
                    retry_response = chat(**retry_kwargs, think=False)
                except TypeError:
                    retry_response = chat(**retry_kwargs)
                retry_content = retry_response.get("message",{}).get("content","") if isinstance(retry_response,dict) else getattr(getattr(retry_response,"message",None),"content","")
                result = _extract_json_object(retry_content)
            except Exception as retry_error:
                return {
                    "status":"fallback",
                    "reason":f"LLM diagnostic reasoning returned invalid/truncated JSON after one retry: {type(retry_error).__name__}: {retry_error}",
                    "model":model,
                }
        ok, why = _grounded(result, state)
        if not ok:
            # A local model may still ignore the JSON-schema enum. If a
            # validated terminal diagnosis exists, give it one explicit
            # evidence-grounded correction opportunity before rejecting it.
            expected = _expected_case_diagnosis(state)
            if expected and result.get("diagnosis") != expected:
                retry_kwargs = dict(kwargs)
                retry_options = dict(retry_kwargs.get("options") or {})
                retry_options["num_predict"] = max(320, int(retry_options.get("num_predict", 256)))
                retry_kwargs["options"] = retry_options
                retry_kwargs["format"] = _diagnostic_schema_for_state(state)
                retry_messages = list(retry_kwargs["messages"])
                retry_messages[-1] = dict(retry_messages[-1])
                retry_messages[-1]["content"] += (
                    f"\nCRITICAL GROUNDING CORRECTION: The validated case resolution is {expected}. "
                    f"Your previous diagnosis '{result.get('diagnosis')}' was unsupported by the measured evidence. "
                    f"Return the complete JSON object with diagnosis exactly '{expected}'. "
                    "For NO_CONFIRMED_FAULT, use null fault_location, null inferred_mechanism, "
                    "empty possible_root_causes, root_cause_status not_applicable, and diagnostic_confidence not_established."
                )
                retry_kwargs["messages"] = retry_messages
                try:
                    try:
                        retry_response = chat(**retry_kwargs, think=False)
                    except TypeError:
                        retry_response = chat(**retry_kwargs)
                    retry_content = retry_response.get("message",{}).get("content","") if isinstance(retry_response,dict) else getattr(getattr(retry_response,"message",None),"content","")
                    retry_result = _extract_json_object(retry_content)
                    retry_ok, retry_why = _grounded(retry_result, state)
                    if retry_ok:
                        result = retry_result
                    else:
                        return {"status":"rejected", "reason":retry_why, "raw":retry_result}
                except Exception as retry_error:
                    return {"status":"rejected", "reason":f"Evidence-grounded correction failed: {type(retry_error).__name__}: {retry_error}", "raw":result}
            else:
                return {"status":"rejected", "reason":why, "raw":result}
        normalized, semantic_error = _semantic_grounding(result, state)
        if semantic_error:
            # One corrective semantic retry keeps the LLM responsible for the
            # interpretation while giving it the exact constraint that failed.
            # We do not silently overwrite a contradictory fault diagnosis.
            retry_kwargs = dict(kwargs)
            retry_options = dict(retry_kwargs.get("options") or {})
            retry_options["num_predict"] = max(320, int(retry_options.get("num_predict", 256)))
            retry_kwargs["options"] = retry_options
            retry_messages = list(retry_kwargs["messages"])
            retry_messages[-1] = dict(retry_messages[-1])
            retry_messages[-1]["content"] += (
                "\nSEMANTIC VALIDATION FAILED: " + semantic_error +
                "\nReturn a corrected complete JSON object. For NO_CONFIRMED_FAULT use null fault_location, null inferred_mechanism, an empty possible_root_causes list, root_cause_status not_applicable, and diagnostic_confidence not_established."
            )
            try:
                try:
                    retry_response = chat(**retry_kwargs, think=False)
                except TypeError:
                    retry_response = chat(**retry_kwargs)
                retry_content = retry_response.get("message",{}).get("content","") if isinstance(retry_response,dict) else getattr(getattr(retry_response,"message",None),"content","")
                retry_result = _extract_json_object(retry_content)
                retry_ok, retry_why = _grounded(retry_result, state)
                if not retry_ok:
                    return {"status":"rejected", "reason":retry_why, "raw":retry_result}
                normalized, semantic_error = _semantic_grounding(retry_result, state)
                if semantic_error:
                    return {"status":"rejected", "reason":semantic_error, "raw":retry_result}
                result = normalized
            except Exception as retry_error:
                return {"status":"rejected", "reason":f"Semantic correction failed: {type(retry_error).__name__}: {retry_error}", "raw":result}
        else:
            result = normalized
        result["status"] = "success"
        result["model"] = model
        result["source"] = "ollama"
        return result
    except Exception as exc:
        return {"status":"fallback", "reason":f"LLM diagnostic reasoning failed: {type(exc).__name__}: {exc}", "model":model}
