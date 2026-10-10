"""Canonical raw-signal feature pipeline for the final bearing agent.

The feature definitions here intentionally reproduce the Phase 3.13 training
pipeline exactly. Inference uses the frozen StandardScaler fitted only on the
training split and the frozen LightGBM classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis

from .raw_mat_loader import load_bearing_mat

WINDOW_DURATION_SEC = 0.1
OVERLAP = 0.50
LOW_CUT_HZ = 1000.0
HIGH_CUT_HZ = 5000.0

FEATURES = [
    "Envelope_RMS",
    "Dominant_Frequency_Hz",
    "Kurtosis",
    "Spectral_Centroid_Hz",
    "Peak_Magnitude",
]

TOOLS_DIR = Path(__file__).resolve().parent
MODEL_PATH = TOOLS_DIR / "lightgbm_frozen.joblib"
SCALER_PATH = TOOLS_DIR / "feature_scaler.joblib"


def window_parameters(fs: float) -> Tuple[int, int]:
    """Return the exact Phase 3 window and hop sizes for a sampling rate."""
    window_size = int(fs * WINDOW_DURATION_SEC)
    step_size = int(window_size * (1.0 - OVERLAP))
    if window_size < 8 or step_size < 1:
        raise ValueError(f"Invalid window configuration for fs={fs}.")
    return window_size, step_size


def number_of_windows(signal_length: int, fs: float) -> int:
    window_size, step_size = window_parameters(fs)
    if signal_length < window_size:
        return 0
    return 1 + (signal_length - window_size) // step_size


def get_window(signal: np.ndarray, fs: float, window_index: int) -> np.ndarray:
    """Return one 0-based, 100-ms window using the training convention."""
    n_windows = number_of_windows(len(signal), fs)
    if window_index < 0 or window_index >= n_windows:
        raise ValueError(
            f"Window {window_index} is out of range. "
            f"Available 0..{n_windows - 1}."
        )
    window_size, step_size = window_parameters(fs)
    start = window_index * step_size
    return np.asarray(signal[start:start + window_size], dtype=np.float64)


def extract_selected_features(window: np.ndarray, fs: float) -> Dict[str, float]:
    """Reproduce the five Phase 3.13 features exactly."""
    window = np.asarray(window, dtype=np.float64).reshape(-1)
    if window.size < 8 or not np.all(np.isfinite(window)):
        raise ValueError("Window is invalid.")

    # Kurtosis: Phase 3.13 uses Fisher=False, bias=True.
    kurtosis_value = float(kurtosis(window, fisher=False, bias=True))

    # FFT of raw window (after the same DC handling used in Phase 3.13).
    centered = window - np.mean(window)
    N = len(centered)
    fft_values = np.fft.rfft(centered)
    magnitude = np.abs(fft_values) / N
    frequencies = np.fft.rfftfreq(N, d=1.0 / fs)

    if len(magnitude) > 1:
        dominant_index = int(np.argmax(magnitude[1:]) + 1)
    else:
        dominant_index = 0

    dominant_frequency = float(frequencies[dominant_index])
    peak_magnitude = float(magnitude[dominant_index])

    magnitude_sum = float(np.sum(magnitude))
    spectral_centroid = (
        float(np.sum(frequencies * magnitude) / magnitude_sum)
        if magnitude_sum > 0 else 0.0
    )

    # Envelope feature: exact Phase 3.13 1-5 kHz Butterworth + Hilbert.
    nyquist = fs / 2.0
    if HIGH_CUT_HZ >= nyquist:
        raise ValueError(
            f"Sampling rate {fs} Hz is too low for {HIGH_CUT_HZ} Hz high-cut."
        )
    b, a = butter(
        4,
        [LOW_CUT_HZ / nyquist, HIGH_CUT_HZ / nyquist],
        btype="band",
    )
    filtered = filtfilt(b, a, centered)
    envelope = np.abs(hilbert(filtered))
    envelope_rms = float(np.sqrt(np.mean(envelope ** 2)))

    return {
        "Envelope_RMS": envelope_rms,
        "Dominant_Frequency_Hz": dominant_frequency,
        "Kurtosis": kurtosis_value,
        "Spectral_Centroid_Hz": spectral_centroid,
        "Peak_Magnitude": peak_magnitude,
    }


def extract_case_features(mat_file: str | Path) -> pd.DataFrame:
    """Extract all valid diagnostic windows from a raw .mat file."""
    case = load_bearing_mat(mat_file, validate=True)
    if case["status"] != "loaded":
        raise ValueError(f"Input validation failed: {case.get('validation')}")

    signal = case["signal"]
    fs = float(case["sampling_rate"])
    n = number_of_windows(len(signal), fs)
    rows = []
    window_size, step_size = window_parameters(fs)

    for idx in range(n):
        window = get_window(signal, fs, idx)
        feats = extract_selected_features(window, fs)
        rows.append({
            "File": case["file_name"],
            "Window": idx + 1,  # public/reporting convention
            "Start_Time_sec": float(idx * step_size / fs),
            **feats,
        })

    if not rows:
        raise ValueError("Signal is shorter than one complete 100-ms window.")
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_frozen_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_window(mat_file: str | Path, window_index: int = 0) -> Dict:
    case = load_bearing_mat(mat_file, validate=True)
    if case["status"] != "loaded":
        return {"status": "failed", "message": str(case.get("validation"))}

    signal = case["signal"]
    fs = float(case["sampling_rate"])
    window = get_window(signal, fs, int(window_index))
    features = extract_selected_features(window, fs)

    model, scaler = load_frozen_artifacts()
    X_raw = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
    X_scaled_array = scaler.transform(X_raw)
    # Preserve the exact feature names used when the frozen classifier was fit.
    # This avoids sklearn feature-name warnings and improves reproducibility.
    X_scaled = pd.DataFrame(X_scaled_array, columns=FEATURES, index=X_raw.index)
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    classes = list(model.classes_)
    prob_map = {str(c): float(p) for c, p in zip(classes, probabilities)}
    ordered = np.sort(probabilities)[::-1]
    confidence = float(ordered[0])
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else confidence

    return {
        "fault": str(prediction),
        "confidence": confidence,
        "margin": margin,
        "probabilities": prob_map,
        "features": features,
        "sampling_rate": fs,
        "window": int(window_index),
        "window_number": int(window_index) + 1,
        "start_time_sec": float(window_index * window_parameters(fs)[1] / fs),
        "status": "success",
    }
