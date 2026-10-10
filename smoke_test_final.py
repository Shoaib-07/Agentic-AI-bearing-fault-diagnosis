"""Final package smoke test: raw loader + validator on one supplied MAT file."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGENT = ROOT / "Phase5_Agentic_AI"
sys.path.insert(0, str(AGENT))

from tools.raw_mat_loader import load_bearing_mat

p = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test_data" / "baseline_3.mat"
r = load_bearing_mat(p)
print("FILE:", r["file_name"])
print("SAMPLES:", r["num_samples"])
print("FS:", r["sampling_rate"])
print("DURATION_SEC:", round(r["duration_sec"], 4))
print("VALIDATION:", r["validation"]["status"])
