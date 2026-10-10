"""Research evaluation harness for the held-out bearing test recordings.

Ground-truth labels are derived from filenames ONLY for scoring. Labels are
never supplied to ML, physics, fusion, or the agent during diagnosis.

Outputs:
- research_evaluation_results/window_level_results.csv
- research_evaluation_results/case_level_results.csv
- research_evaluation_results/summary.json
- research_evaluation_results/confusion_matrix_ml.csv
- research_evaluation_results/confusion_matrix_physics.csv
- research_evaluation_results/confusion_matrix_fusion.csv

This is a held-out evaluation on six recordings, not a claim of real-plant
or incipient-fault generalization.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "Phase5_Agentic_AI"
sys.path.insert(0, str(AGENT_DIR))

from agents.agentic_controller import run_agentic_case
from tools.ml_tool import ml_diagnosis_tool
from tools.physics_tool import physics_diagnosis_tool
from tools.fusion_tool import fusion_tool
from tools.raw_feature_pipeline import number_of_windows
from tools.raw_mat_loader import load_bearing_mat

TEST_DIR = ROOT / "test_data"
OUT_DIR = ROOT / "research_evaluation_results"
CLASSES = ["Healthy", "InnerRaceFault", "OuterRaceFault"]


def label_from_filename(name: str) -> str:
    n = name.lower()
    if "baseline" in n:
        return "Healthy"
    if "inner" in n:
        return "InnerRaceFault"
    if "outer" in n:
        return "OuterRaceFault"
    return "Unknown"


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
    }


def save_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(OUT_DIR / f"confusion_matrix_{name}.csv")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    files = sorted(TEST_DIR.glob("*.mat"))
    if not files:
        raise SystemExit("No .mat files found in test_data/")

    window_rows = []
    case_rows = []
    agent_rows = []

    for path in files:
        true_label = label_from_filename(path.name)
        case = load_bearing_mat(path, validate=True)
        total = number_of_windows(len(case["signal"]), case["sampling_rate"])

        ml_preds, ph_preds, fu_preds = [], [], []
        for idx in range(total):
            inp = {"mat_file": str(path), "window": idx}
            ml = ml_diagnosis_tool(inp)
            ph = physics_diagnosis_tool(inp)
            fu = fusion_tool(ml, ph) if ml.get("status") == "success" and ph.get("status") == "success" else {"diagnosis": "UNRESOLVED"}
            ml_pred = ml.get("fault", "UNRESOLVED")
            ph_pred = "Healthy" if ph.get("diagnosis") == "Healthy_or_Insufficient" else ph.get("diagnosis", "UNRESOLVED")
            fu_pred = fu.get("diagnosis", "UNRESOLVED")
            ml_preds.append(ml_pred); ph_preds.append(ph_pred); fu_preds.append(fu_pred)
            window_rows.append({
                "file": path.name, "window": idx + 1, "true_label": true_label,
                "ml_prediction": ml_pred, "ml_confidence": ml.get("confidence"),
                "physics_prediction": ph_pred, "physics_strength": ph.get("strength"),
                "fusion_prediction": fu_pred,
            })

        case_rows.append({
            "file": path.name, "true_label": true_label,
            "ml_majority": Counter(ml_preds).most_common(1)[0][0],
            "physics_majority": Counter(ph_preds).most_common(1)[0][0],
            "fusion_majority": Counter(fu_preds).most_common(1)[0][0],
            "windows": total,
        })

        try:
            state = run_agentic_case(
                f"RESEARCH_{path.stem}", {"mat_file": str(path)},
                model=os.getenv("OLLAMA_MODEL", "qwen3:4b"), max_steps=20,
            )
            agent_rows.append({
                "file": path.name, "true_label": true_label,
                "agent_prediction": state.final_diagnosis,
                "agent_status": state.status,
                "final_confidence": state.final_confidence,
                "final_confidence_type": getattr(state, "final_confidence_type", "not_established"),
                "final_confidence_explanation": getattr(state, "final_confidence_explanation", ""),
                "final_consistency": getattr(state, "final_consistency", None),
                "steps": len(state.tool_history),
                "fallback_steps": sum("safety policy" in str(x.get("reason", "")).lower() for x in state.tool_history),
            })
        except Exception as exc:
            agent_rows.append({"file": path.name, "true_label": true_label, "agent_prediction": "ERROR", "agent_status": type(exc).__name__, "error": str(exc)})

    wdf = pd.DataFrame(window_rows)
    cdf = pd.DataFrame(case_rows)
    adf = pd.DataFrame(agent_rows)
    wdf.to_csv(OUT_DIR / "window_level_results.csv", index=False)
    cdf.to_csv(OUT_DIR / "case_level_results.csv", index=False)
    adf.to_csv(OUT_DIR / "agent_case_results.csv", index=False)

    summary = {
        "evaluation_scope": "six held-out recordings in test_data",
        "warning": "This is not sufficient to claim real-plant or incipient-fault generalization.",
        "window_level": {}, "case_level_majority": {}, "agent_case_level": {},
    }
    for col, key in [("ml_prediction", "ml"), ("physics_prediction", "physics"), ("fusion_prediction", "fusion")]:
        y_true, y_pred = wdf["true_label"].tolist(), wdf[col].tolist()
        summary["window_level"][key] = metrics(y_true, y_pred)
        save_confusion(y_true, y_pred, key)
    for col, key in [("ml_majority", "ml"), ("physics_majority", "physics"), ("fusion_majority", "fusion")]:
        summary["case_level_majority"][key] = metrics(cdf["true_label"], cdf[col])
    valid_agent = adf[adf["agent_prediction"].isin(CLASSES)]
    if not valid_agent.empty:
        summary["agent_case_level"] = metrics(valid_agent["true_label"], valid_agent["agent_prediction"])
        summary["agent_case_level"]["cases_evaluated"] = int(len(valid_agent))
        summary["agent_case_level"]["mean_steps"] = float(valid_agent["steps"].mean())
        summary["agent_case_level"]["mean_fallback_steps"] = float(valid_agent["fallback_steps"].mean())
    else:
        summary["agent_case_level"] = {"cases_evaluated": 0}

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
