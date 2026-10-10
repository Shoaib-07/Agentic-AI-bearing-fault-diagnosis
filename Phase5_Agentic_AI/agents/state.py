from dataclasses import dataclass, field
from typing import Any
from typing import Any, Dict, List, Optional

@dataclass
class DiagnosticState:
    case_id: str
    input_data: Any = None

    ml_result: Any = None
    physics_result: Any = None
    fusion_result: Any = None

    window_results: list = field(default_factory=list)
    reanalysis_result: Any = None

    evidence: list = field(default_factory=list)
    tool_history: list = field(default_factory=list)

    rca_result: Any = None
    operator_report: Any = None

    status: str = "NEW"

    final_diagnosis: Any = None
    final_confidence: Any = None
    final_confidence_type: str = "not_established"
    final_confidence_explanation: str = ""
    final_consistency: Any = None

    # =========================================================
    # PHASE 5.17 — ADAPTIVE REASONING STATE
    # =========================================================

    evidence_quality_result: Optional[Dict[str, Any]] = None
    adaptive_investigation_result: Optional[Dict[str, Any]] = None
    evidence_stopping_result: Optional[Dict[str, Any]] = None
    reasoning_trace: Optional[Dict[str, Any]] = None

    # Reproducibility / authoritative-run metadata
    run_id: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.0
    max_steps: int = 20
    started_at: str = ""
    completed_at: str = ""
    total_duration_ms: float = 0.0
    final_resolution: str = ""

    # LLM-owned diagnostic judgment (verified against deterministic evidence)
    llm_diagnostic_result: Optional[Dict[str, Any]] = None
    knowledge_graph_context: Optional[Dict[str, Any]] = None
