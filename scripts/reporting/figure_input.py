"""Recalibrated test predictions for calibration_dca.py, full cohort and HIV-positive subgroup.

Usage: python scripts/reporting/figure_input.py ANALYSIS_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    a = Path(sys.argv[1]).expanduser()
    d = pd.read_csv(a / "test_predictions.csv")
    out = pd.DataFrame({"patient_id": d["patient_id"], "y_true": d["y"]})
    for m in ("ehr", "cxr", "fusion", "stack"):
        out[f"p_{m}"] = d[f"{m}_recal"]
    for tag, frame in (("full", out), ("hiv", out[d["hiv"] == 1])):
        (a / "figure_input" / tag).mkdir(parents=True, exist_ok=True)
        frame.to_csv(a / "figure_input" / tag / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
