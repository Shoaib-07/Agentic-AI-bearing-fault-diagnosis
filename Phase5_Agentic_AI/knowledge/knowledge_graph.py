"""Case-aware bearing diagnostic knowledge graph.

The graph uses a stable node/edge schema that can be exported to Neo4j later.
The current implementation is local-first for research reproducibility.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def load_schema() -> dict[str, Any]:
    return json.loads((ROOT / "diagnostic_knowledge.json").read_text(encoding="utf-8"))


def build_case_graph(state) -> dict[str, Any]:
    schema = load_schema()
    nodes = {n["id"]: dict(n) for n in schema["nodes"]}
    edges = [dict(e, scope="domain_knowledge") for e in schema["edges"]]

    def add_case_node(node_id: str, label: str, node_type: str, data: Any = None):
        nodes[node_id] = {"id": node_id, "type": node_type, "label": label, "case_data": data}

    case_id = str(state.case_id)
    add_case_node(f"case:{case_id}", f"Case {case_id}", "case")
    add_case_node("evidence:ml", "Current ML Evidence", "case_evidence", state.ml_result)
    add_case_node("evidence:physics", "Current Physics Evidence", "case_evidence", state.physics_result)
    add_case_node("evidence:fusion", "Current Fusion Evidence", "case_evidence", state.fusion_result)
    add_case_node("evidence:reanalysis", "Multi-window Evidence", "case_evidence", state.reanalysis_result)

    edges.extend([
        {"source": f"case:{case_id}", "target": "vibration", "relation": "contains_signal", "scope": "case_evidence"},
        {"source": f"case:{case_id}", "target": "evidence:ml", "relation": "has_evidence", "scope": "case_evidence"},
        {"source": f"case:{case_id}", "target": "evidence:physics", "relation": "has_evidence", "scope": "case_evidence"},
        {"source": f"case:{case_id}", "target": "evidence:fusion", "relation": "has_evidence", "scope": "case_evidence"},
    ])
    if state.reanalysis_result:
        edges.append({"source": f"case:{case_id}", "target": "evidence:reanalysis", "relation": "has_evidence", "scope": "case_evidence"})

    ml = state.ml_result or {}
    ph = state.physics_result or {}
    diag = state.final_diagnosis or (state.fusion_result or {}).get("diagnosis")
    if ml.get("fault") == "InnerRaceFault":
        edges.append({"source":"evidence:ml","target":"inner_race","relation":"supports_fault_evidence", "scope":"case_evidence"})
    elif ml.get("fault") == "OuterRaceFault":
        edges.append({"source":"evidence:ml","target":"outer_race","relation":"supports_fault_evidence", "scope":"case_evidence"})
    # A dominance ratio alone is NOT a fault finding. The case graph only
    # labels a characteristic-frequency feature as fault-supporting when the
    # physics tool itself classifies the evidence as at least MODERATE and the
    # corresponding fault location agrees. Otherwise the generic schema
    # relation (physics -> measures -> feature) remains informational only.
    ph_strength = str(ph.get("strength") or "").upper()
    ph_diag = str(ph.get("diagnosis") or "")
    if ph_strength in {"MODERATE", "STRONG"} and ph_diag == "InnerRaceFault" and float(ph.get("BPFI_Dominance", 0) or 0) > 0.5:
        edges.append({"source":"evidence:physics","target":"bpfi","relation":"supports_fault_evidence", "scope":"case_evidence"})
    if ph_strength in {"MODERATE", "STRONG"} and ph_diag == "OuterRaceFault" and float(ph.get("BPFO_Dominance", 0) or 0) > 0.5:
        edges.append({"source":"evidence:physics","target":"bpfo","relation":"supports_fault_evidence", "scope":"case_evidence"})
    if diag == "InnerRaceFault":
        edges.append({"source":f"case:{case_id}","target":"inner_race","relation":"current_hypothesis", "scope":"case_evidence"})
    elif diag == "OuterRaceFault":
        edges.append({"source":f"case:{case_id}","target":"outer_race","relation":"current_hypothesis", "scope":"case_evidence"})
    elif diag == "NO_CONFIRMED_FAULT":
        edges.append({"source":f"case:{case_id}","target":"no_confirmed_fault","relation":"current_resolution", "scope":"case_evidence"})

    return {"schema_version": schema["schema_version"], "nodes": list(nodes.values()), "edges": edges}


def compact_reasoning_context(graph: dict[str, Any], max_edges: int = 24) -> dict[str, Any]:
    return {
        "nodes": [{"id":n["id"], "type":n["type"], "label":n["label"]} for n in graph["nodes"]],
        "relationships": graph["edges"][:max_edges],
    }
