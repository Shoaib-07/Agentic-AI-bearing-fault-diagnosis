"""Diagnosis-independent validation for raw bearing vibration signals.

Phase 5.18.3
-------------
Checks whether a loaded raw vibration signal is technically suitable for the
next diagnostic stages. This module does NOT infer or use the fault label.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np


def validate_signal(signal: Any, sampling_rate: float, *,
                    min_duration_sec: float = 0.1,
                    required_max_frequency_hz: float = 5000.0,
                    min_std: float = 1e-12) -> Dict[str, Any]:
    """Validate a raw 1-D vibration signal and acquisition rate.

    The checks are deliberately diagnosis-independent. ``required_max_frequency_hz``
    represents the highest frequency required by the current feature pipeline,
    so the signal must satisfy Nyquist for that pipeline.
    """
    x = np.asarray(signal, dtype=np.float64).squeeze()
    checks: Dict[str, bool] = {}
    warnings: list[str] = []
    failures: list[str] = []

    checks["signal_is_1d"] = bool(x.ndim == 1)
    if not checks["signal_is_1d"]:
        failures.append("signal_is_1d")
        return _result(checks, failures, warnings, x, sampling_rate)

    checks["signal_nonempty"] = bool(x.size > 0)
    checks["signal_finite"] = bool(np.all(np.isfinite(x)))
    checks["sampling_rate_finite"] = bool(np.isfinite(sampling_rate))
    checks["sampling_rate_positive"] = bool(np.isfinite(sampling_rate) and sampling_rate > 0)

    if not checks["signal_nonempty"]:
        failures.append("signal_nonempty")
    if not checks["signal_finite"]:
        failures.append("signal_finite")
    if not checks["sampling_rate_finite"]:
        failures.append("sampling_rate_finite")
    if not checks["sampling_rate_positive"]:
        failures.append("sampling_rate_positive")

    if checks["sampling_rate_positive"]:
        duration = x.size / float(sampling_rate)
        nyquist = float(sampling_rate) / 2.0
        checks["minimum_duration"] = bool(duration >= min_duration_sec)
        checks["nyquist_support"] = bool(nyquist > required_max_frequency_hz)
        if not checks["minimum_duration"]:
            failures.append("minimum_duration")
        if not checks["nyquist_support"]:
            failures.append("nyquist_support")
    else:
        duration = float("nan")
        nyquist = float("nan")
        checks["minimum_duration"] = False
        checks["nyquist_support"] = False

    if x.size:
        std = float(np.std(x))
        checks["nonzero_variance"] = bool(std > min_std)
        checks["dynamic_range_nonzero"] = bool(float(np.ptp(x)) > min_std)
        zero_fraction = float(np.mean(x == 0.0))
        checks["not_predominantly_zero"] = bool(zero_fraction < 0.99)
        if not checks["nonzero_variance"]:
            failures.append("nonzero_variance")
        if not checks["dynamic_range_nonzero"]:
            failures.append("dynamic_range_nonzero")
        if not checks["not_predominantly_zero"]:
            failures.append("not_predominantly_zero")
        if zero_fraction > 0.5:
            warnings.append("more_than_50_percent_samples_are_exact_zero")
    else:
        std = float("nan")
        zero_fraction = float("nan")
        checks["nonzero_variance"] = False
        checks["dynamic_range_nonzero"] = False
        checks["not_predominantly_zero"] = False

    # Non-fatal diagnostics: useful to the agent/operator but not a diagnosis.
    if x.size:
        mean = float(np.mean(x))
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        crest = peak / rms if rms > 0 else float("inf")
    else:
        mean = rms = peak = crest = float("nan")

    return _result(checks, failures, warnings, x, sampling_rate,
                   duration=duration, nyquist=nyquist, mean=mean,
                   std=std, rms=rms, peak=peak, crest_factor=crest,
                   zero_fraction=zero_fraction)


def _result(checks, failures, warnings, x, sampling_rate, **metrics):
    return {
        "status": "valid" if not failures else "invalid",
        "checks": checks,
        "failed_checks": failures,
        "warnings": warnings,
        "metrics": {
            "num_samples": int(x.size),
            "sampling_rate_hz": float(sampling_rate),
            **metrics,
        },
    }
