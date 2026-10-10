"""Operator-facing status derived only from the authoritative final state."""
from typing import Any

def _f(v, d=0.0):
    try: return float(v)
    except (TypeError, ValueError): return d

def operator_status_tool(state) -> dict[str, Any]:
    diagnosis=str(state.final_diagnosis or "UNRESOLVED")
    conf=_f(state.final_confidence)
    consistency=_f(getattr(state,"final_consistency",None))
    resolution=str(getattr(state,"final_resolution","") or "")
    ph=state.physics_result or {}
    physics=str(ph.get("strength","")).upper()
    stop=state.evidence_stopping_result or {}
    completed=(getattr(state,"status","") == "COMPLETED")
    terminal=bool(stop.get("stop")) and completed

    if diagnosis == "NO_CONFIRMED_FAULT":
        status="NO CONFIRMED FAULT" if terminal else "WARNING"
        severity="LOW" if terminal else "MODERATE"
    elif diagnosis == "Healthy":
        # Backward-compatible handling for pre-reanalysis/legacy states. A
        # completed no-confirmation case is canonicalized before reaching this.
        status="NO CONFIRMED FAULT" if terminal else "WARNING"
        severity="LOW" if terminal else "MODERATE"
    elif diagnosis in {"InnerRaceFault","OuterRaceFault"}:
        status="CRITICAL" if conf >= .85 and (consistency >= .90 or physics=="STRONG") and terminal else "FAULT"
        severity="CRITICAL" if status=="CRITICAL" else "HIGH"
    else:
        status="WARNING"; severity="MODERATE"

    why=[]; ml=state.ml_result or {}; fu=state.fusion_result or {}; ra=state.reanalysis_result or {}
    if ml.get("fault"):
        why.append(f"ML evidence: {ml.get('fault')} (source confidence {_f(ml.get('confidence')):.1%})")
    if ph.get("strength"): why.append(f"Physics evidence: {ph.get('strength')}")
    if ra.get("windows_analyzed"):
        why.append(f"{ra.get('windows_analyzed')} windows validated; final-diagnosis consistency {_f(ra.get('diagnosis_consistency')):.1%}")
        if ra.get("final_diagnosis_source"):
            why.append(f"Final resolution: {diagnosis} via {str(ra.get('final_diagnosis_source')).replace('_', ' ')}")
    elif state.window_results:
        why.append(f"{len(state.window_results)} additional windows analyzed")
    if fu.get("decision"): why.append(f"Fusion decision: {fu.get('decision')}")
    if stop.get("reason") and len(why)<2: why.append(str(stop.get("reason")))
    if not why: why=["Insufficient completed evidence for a concise diagnostic explanation."]

    if diagnosis == "NO_CONFIRMED_FAULT": action="Continue routine condition monitoring; investigate a persistent trend or new anomaly."
    elif diagnosis == "Healthy": action="Continue routine condition monitoring; investigate a persistent trend or new anomaly."
    elif diagnosis == "InnerRaceFault": action="Inspect the bearing inner race, lubrication, seating, load and alignment; schedule maintenance inspection."
    elif diagnosis == "OuterRaceFault": action="Inspect the bearing outer race, lubrication, seating, load and alignment; schedule maintenance inspection."
    else: action="Do not treat the result as a confirmed fault; collect additional diagnostic evidence."

    temporal_status=(state.evidence_quality_result or {}).get("temporal_evidence_status","NOT_AVAILABLE")
    trend={"available":False,"label":"NOT AVAILABLE","message":"A single uploaded recording does not establish a temporal health/degradation trend. Use repeated timestamped measurements for trend and early-warning analysis.","status":temporal_status}
    if diagnosis == "NO_CONFIRMED_FAULT":
        limitation=(
            "A no-confirmation result does not prove the bearing is healthy; "
            "physical fault mechanisms require additional evidence and inspection. "
            "This is decision-support output only."
        )
    elif diagnosis in {"InnerRaceFault", "OuterRaceFault"}:
        limitation=(
            "The identified fault mechanism is supported by vibration evidence, "
            "but the specific physical root cause requires maintenance inspection. "
            "This is decision-support output only."
        )
    else:
        limitation="This is decision-support output only and does not directly control plant equipment."
    return {"status":status,"diagnosis":diagnosis,"confidence":None,"confidence_available":False,"severity":severity,"why":why[:4],"trend":trend,"recommended_action":action,"limitation":limitation}
