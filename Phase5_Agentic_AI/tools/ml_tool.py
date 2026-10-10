"""Frozen ML diagnostic tool using the canonical raw feature pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Dict

from .raw_feature_pipeline import predict_window, extract_case_features


def ml_diagnosis_tool(case: Dict) -> Dict:
    try:
        if not isinstance(case, dict):
            return {"status": "failed", "message": "Case input must be a dictionary."}
        mat_file = case.get("mat_file")
        if not mat_file:
            return {"status": "failed", "message": "MAT file path is missing."}
        window = int(case.get("window", 0))
        return predict_window(mat_file, window)
    except Exception as exc:
        return {"status": "failed", "message": f"{type(exc).__name__}: {exc}"}


def extract_all_ml_windows(mat_file: str | Path):
    """Return canonical raw features for every diagnostic window."""
    return extract_case_features(mat_file)
