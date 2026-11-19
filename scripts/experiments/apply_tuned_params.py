"""Write the tuned XGBoost settings into the EHR configs.

Usage: python scripts/experiments/apply_tuned_params.py BEST_PARAMS_JSON CONFIG [CONFIG ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> None:
    best = json.loads(Path(sys.argv[1]).read_text())["best_params"]
    p = {k.rsplit(".", 1)[-1]: v for k, v in best.items()}
    text = (
        f"  params: {{n_estimators: {int(p['n_estimators'])}, max_depth: {int(p['max_depth'])}, "
        f"learning_rate: {p['learning_rate']:.4f}, subsample: {p['subsample']:.3f}, "
        f"colsample_bytree: {p['colsample_bytree']:.3f},\n"
        f"           min_child_weight: {p['min_child_weight']:.3f}, "
        f"reg_lambda: {p['reg_lambda']:.3f}}}   "
        "# 30-trial search on this cohort's validation patients "
        "(configs/pneumonia/ehr_xgb_tune.yaml)\n"
    )
    for name in sys.argv[2:]:
        f = Path(name)
        new, n = re.subn(
            r"  params: \{n_estimators:.*?\n.*?\n", text, f.read_text(), count=1, flags=re.S
        )
        assert n == 1, name
        f.write_text(new)
        print("updated", name)


if __name__ == "__main__":
    main()
