"""Physics-based bearing fault evidence tool.

Uses the same 100-ms / 50%-overlap windows as the canonical ML pipeline.
Characteristic-frequency energy is computed from the Hilbert envelope after
1-5 kHz band-pass filtering. Thresholds were calibrated from the original
training recordings only and are kept fixed during inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from .raw_mat_loader import load_bearing_mat
from .raw_feature_pipeline import get_window, window_parameters

LOW_CUT = 1000.0
HIGH_CUT = 5000.0
BASE_TOLERANCE_HZ = 3.0

# Training-derived fixed thresholds using the canonical 100-ms window.
# They are deliberately separated from the ML model.
STRONG_BPFI_THRESHOLD = 0.060
STRONG_BPFO_THRESHOLD = 0.080
MODERATE_TARGET_THRESHOLD = 0.020
STRONG_DOMINANCE = 0.70


def _band_energy(freqs, magnitude, target, tolerance):
    mask = np.abs(freqs - target) <= tolerance
    return float(np.sum(magnitude[mask] ** 2)) if np.any(mask) else 0.0


def _envelope_spectrum(signal, fs):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    centered = signal - np.mean(signal)
    nyquist = fs / 2.0
    if HIGH_CUT >= nyquist:
        raise ValueError(f"Sampling rate {fs} Hz is too low for physics filtering.")
    sos = butter(
        4, [LOW_CUT, HIGH_CUT], btype="bandpass", fs=fs, output="sos"
    )
    filtered = sosfiltfilt(sos, centered)
    envelope = np.abs(hilbert(filtered))
    envelope -= np.mean(envelope)

    N = len(envelope)
    magnitude = np.abs(np.fft.rfft(envelope)) / N
    if len(magnitude) > 2:
        magnitude[1:-1] *= 2.0
    if len(magnitude):
        magnitude[0] = 0.0
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, magnitude


def extract_physics_features(signal, fs, FTF, BPFO, BPFI, BSF):
    freqs, magnitude = _envelope_spectrum(signal, fs)
    resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else fs / len(signal)
    tolerance = max(BASE_TOLERANCE_HZ, 0.60 * resolution)
    total_energy = float(np.sum(magnitude ** 2)) + 1e-15

    def energy(target):
        return _band_energy(freqs, magnitude, float(target), tolerance)

    def harmonic_energy(target, max_harmonic=3):
        return sum(energy(float(target) * h) for h in range(1, max_harmonic + 1))

    bpfo = energy(BPFO)
    bpfi = energy(BPFI)
    bsf = energy(BSF)
    ftf = energy(FTF)
    bpfo_h = harmonic_energy(BPFO)
    bpfi_h = harmonic_energy(BPFI)
    bsf_h = harmonic_energy(BSF, 2)
    ftf_h = harmonic_energy(FTF, 2)

    norm = lambda x: float(x / total_energy)
    bpfo_n, bpfi_n = norm(bpfo), norm(bpfi)
    fault_energy = bpfo_n + bpfi_n
    bpfo_dom = float(bpfo_n / (fault_energy + 1e-15))
    bpfi_dom = float(bpfi_n / (fault_energy + 1e-15))

    if bpfi_n >= STRONG_BPFI_THRESHOLD and bpfi_dom >= STRONG_DOMINANCE:
        diagnosis, strength = "InnerRaceFault", "STRONG"
    elif bpfo_n >= STRONG_BPFO_THRESHOLD and bpfo_dom >= STRONG_DOMINANCE:
        diagnosis, strength = "OuterRaceFault", "STRONG"
    elif bpfi_n >= MODERATE_TARGET_THRESHOLD and bpfi_dom >= 0.60:
        diagnosis, strength = "InnerRaceFault", "MODERATE"
    elif bpfo_n >= MODERATE_TARGET_THRESHOLD and bpfo_dom >= 0.60:
        diagnosis, strength = "OuterRaceFault", "MODERATE"
    elif bpfi_n < MODERATE_TARGET_THRESHOLD and bpfo_n < MODERATE_TARGET_THRESHOLD:
        diagnosis, strength = "Healthy_or_Insufficient", "LOW"
    else:
        diagnosis, strength = "Healthy_or_Insufficient", "LOW"

    return {
        "diagnosis": diagnosis,
        "strength": strength,
        "BPFO_Normalized_Energy": bpfo_n,
        "BPFI_Normalized_Energy": bpfi_n,
        "Fault_Energy": fault_energy,
        "BPFO_Dominance": bpfo_dom,
        "BPFI_Dominance": bpfi_dom,
        "BPFO_Harmonic_Normalized_Energy": norm(bpfo_h),
        "BPFI_Harmonic_Normalized_Energy": norm(bpfi_h),
        "BSF_Normalized_Energy": norm(bsf),
        "FTF_Normalized_Energy": norm(ftf),
        "sampling_rate": float(fs),
        "FTF": float(FTF),
        "BPFO": float(BPFO),
        "BPFI": float(BPFI),
        "BSF": float(BSF),
        "frequency_resolution_hz": resolution,
        "tolerance_hz": tolerance,
        "status": "success",
    }


def physics_diagnosis_tool(case: Dict) -> Dict:
    try:
        if not isinstance(case, dict):
            return {"status": "failed", "message": "Case input must be a dictionary."}
        mat_file = case.get("mat_file")
        if not mat_file:
            return {"status": "failed", "message": "MAT file path is missing."}

        loaded = load_bearing_mat(mat_file, validate=True)
        if loaded["status"] != "loaded":
            return {"status": "failed", "message": str(loaded.get("validation"))}

        required = ("FTF", "BPFO", "BPFI", "BSF")
        missing = [x for x in required if loaded.get(x) is None]
        if missing:
            return {"status": "failed", "message": f"Missing characteristic frequencies: {missing}"}

        idx = int(case.get("window", 0))
        window = get_window(loaded["signal"], loaded["sampling_rate"], idx)
        result = extract_physics_features(
            window, loaded["sampling_rate"],
            loaded["FTF"], loaded["BPFO"], loaded["BPFI"], loaded["BSF"]
        )
        result["load"] = loaded.get("load")
        result["machine_rate"] = loaded.get("rate")
        result["window"] = idx
        result["window_number"] = idx + 1
        return result
    except Exception as exc:
        return {"status": "failed", "message": f"{type(exc).__name__}: {exc}"}
