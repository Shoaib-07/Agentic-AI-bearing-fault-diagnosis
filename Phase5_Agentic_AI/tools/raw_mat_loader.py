"""Universal raw MATLAB input interface for the bearing diagnostic agent.

Phase 5.18.2
-------------
Loads a raw bearing .mat file and converts the dataset-specific MATLAB
structure into a stable Python dictionary used by downstream agent tools.

The loader is intentionally diagnosis-agnostic: it never reads the filename
as a fault label and it does not perform feature extraction or prediction.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

import numpy as np
from scipy.io import loadmat

from .raw_signal_validator import validate_signal


REQUIRED_BEARING_FIELDS = ("gs", "sr")
OPTIONAL_BEARING_FIELDS = ("rate", "load")
OPTIONAL_FAULT_FREQUENCIES = ("FTF", "BPFO", "BPFI", "BSF")


def _scalar(value: Any, name: str) -> float:
    """Convert a MATLAB scalar-like value to a Python float."""
    arr = np.asarray(value).squeeze()
    if arr.size != 1:
        raise ValueError(f"'{name}' must contain exactly one value; got shape {arr.shape}.")
    try:
        result = float(arr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' is not numeric.") from exc
    if not np.isfinite(result):
        raise ValueError(f"'{name}' is not finite.")
    return result


def _bearing_field(bearing: np.ndarray, name: str) -> Any:
    """Read a field from the 1x1 MATLAB structured bearing object."""
    if not isinstance(bearing, np.ndarray) or bearing.dtype.names is None:
        raise ValueError("'bearing' is not a MATLAB structured array.")
    if name not in bearing.dtype.names:
        raise ValueError(f"Required field '{name}' is missing from 'bearing'.")
    return bearing[name][0, 0]


def _extract_signal(bearing: np.ndarray) -> np.ndarray:
    raw = _bearing_field(bearing, "gs")
    signal = np.asarray(raw, dtype=np.float64).squeeze()
    if signal.size == 0:
        raise ValueError("Vibration signal 'gs' is empty.")
    signal = signal.reshape(-1)
    if not np.all(np.isfinite(signal)):
        raise ValueError("Vibration signal 'gs' contains NaN or Inf values.")
    return signal


def _optional_scalar(bearing: np.ndarray, name: str) -> Optional[float]:
    if not isinstance(bearing, np.ndarray) or bearing.dtype.names is None:
        return None
    if name not in bearing.dtype.names:
        return None
    return _scalar(bearing[name][0, 0], name)


def _top_level_scalar(mat: Dict[str, Any], name: str) -> Optional[float]:
    if name not in mat:
        return None
    return _scalar(mat[name], name)


def validate_raw_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Phase 5.18.3 raw-signal validation layer."""
    return validate_signal(case["signal"], case["sampling_rate"])


@lru_cache(maxsize=16)
def _load_bearing_mat_cached(path_str: str, validate: bool) -> Dict[str, Any]:
    """Cached MAT loader used by repeated diagnostic tools in one case.

    Loading a MATLAB file repeatedly was a major avoidable latency source during
    multi-window investigation. The cache is keyed by the absolute path and
    validation flag and is safe for this read-only diagnostic workflow.
    """
    path = Path(path_str)
    mat = loadmat(path, squeeze_me=False, struct_as_record=True)
    if "bearing" not in mat:
        raise ValueError("MAT file does not contain the required 'bearing' structure.")

    bearing = mat["bearing"]
    if bearing.shape != (1, 1):
        raise ValueError(f"Expected a 1x1 'bearing' structure, got shape {bearing.shape}.")

    missing = [name for name in REQUIRED_BEARING_FIELDS if name not in bearing.dtype.names]
    if missing:
        raise ValueError(f"Missing required bearing fields: {missing}")

    signal = _extract_signal(bearing)
    sampling_rate = _scalar(_bearing_field(bearing, "sr"), "sr")

    case: Dict[str, Any] = {
        "status": "loaded",
        "mat_file": str(path),
        "file_name": path.name,
        "signal": signal,
        "sampling_rate": sampling_rate,
        "num_samples": int(signal.size),
        "duration_sec": float(signal.size / sampling_rate),
        "rate": _optional_scalar(bearing, "rate"),
        "load": _optional_scalar(bearing, "load"),
        "FTF": _top_level_scalar(mat, "FTF"),
        "BPFO": _top_level_scalar(mat, "BPFO"),
        "BPFI": _top_level_scalar(mat, "BPFI"),
        "BSF": _top_level_scalar(mat, "BSF"),
    }

    if validate:
        validation = validate_raw_case(case)
        case["validation"] = validation
        if validation["status"] != "valid":
            case["status"] = "invalid"
    return case


def load_bearing_mat(mat_file: str | Path, *, validate: bool = True) -> Dict[str, Any]:
    path = Path(mat_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Expected a .mat file, got: {path.suffix or '<no extension>'}")
    return _load_bearing_mat_cached(str(path), bool(validate))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a raw bearing .mat file")
    parser.add_argument("mat_file")
    args = parser.parse_args()

    result = load_bearing_mat(args.mat_file)
    print("Status       :", result["status"])
    print("File         :", result["file_name"])
    print("Samples      :", result["num_samples"])
    print("Sampling Hz  :", result["sampling_rate"])
    print("Duration sec :", round(result["duration_sec"], 4))
    print("Rate         :", result["rate"])
    print("Load         :", result["load"])
    print("FTF/BPFO/BPFI/BSF:", result["FTF"], result["BPFO"], result["BPFI"], result["BSF"])
    print("Validation   :", result["validation"])
