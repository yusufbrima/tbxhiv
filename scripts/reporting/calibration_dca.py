"""Calibration and clinical net benefit of the EHR, CXR, fusion and stacked models.

Reads a test_predictions.csv per cohort (columns y_true, patient_id, p_ehr, p_cxr, p_fusion,
p_stack; recalibrated probabilities from scripts/analysis/main_analysis.py) and writes, per cohort:
  calibration.csv  ten bins of predicted probability (deciles, so a rare outcome still fills every
                   bin): mean predicted, observed rate, n
  summary.csv      Brier score, expected calibration error (ECE, 10 bins), calibration intercept
                   and slope (logistic recalibration of logit p), with patient-level bootstrap CIs
  dca.csv          net benefit across threshold probabilities DCA_TMIN-DCA_TMAX (environment;
                   default 0.05-0.60), with treat-all / none
and a two-panel figure (calibration, decision curve) as PDF + PNG.

Usage: python scripts/reporting/calibration_dca.py OUT_DIR TAG=STACKING_DIR [TAG=STACKING_DIR ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODELS = {
    "p_ehr": "EHR XGBoost",
    "p_cxr": "CXR only",
    "p_fusion": "End-to-end fusion",
    "p_stack": "Stacked EHR + CXR",
}
COLORS = {"p_ehr": "#2a78d6", "p_cxr": "#eb6834", "p_fusion": "#1baf7a", "p_stack": "#6250d6"}
BINS = np.linspace(0, 1, 11)
TMIN, TMAX = float(os.environ.get("DCA_TMIN", 0.05)), float(os.environ.get("DCA_TMAX", 0.60))
THRESHOLDS = np.round(np.arange(TMIN, TMAX + 1e-9, 0.005 if TMAX <= 0.3 else 0.01), 3)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    idx = np.clip(np.digitize(p, BINS[1:-1]), 0, 9)
    ece = sum(
        abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).sum()
        for b in range(10)
        if (idx == b).any()
    ) / len(y)
    lr = LogisticRegression(C=1e6).fit(logit(p)[:, None], y)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
        "intercept": float(lr.intercept_[0]),
        "slope": float(lr.coef_[0, 0]),
    }


def net_benefit(y: np.ndarray, p: np.ndarray, t: float) -> float:
    pred = p >= t
    n = len(y)
    return float(np.sum(pred & (y == 1)) / n - np.sum(pred & (y == 0)) / n * t / (1 - t))


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cohorts = [a.split("=", 1) for a in sys.argv[2:]]
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(6.3, 2.6 * len(cohorts)), squeeze=False)
    rng = np.random.default_rng(0)
    for row, (tag, folder) in enumerate(cohorts):
        d = pd.read_csv(Path(folder).expanduser() / "test_predictions.csv")
        y = d["y_true"].to_numpy(int)
        pats = d["patient_id"].to_numpy()
        groups = [np.flatnonzero(pats == u) for u in np.unique(pats)]
        cal_rows, sum_rows, dca_rows = [], [], []
        for col, name in MODELS.items():
            p = d[col].to_numpy(float)
            edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
            idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                if (idx == b).sum() >= 10:
                    cal_rows.append(
                        {
                            "model": name,
                            "bin": b,
                            "predicted": p[idx == b].mean(),
                            "observed": y[idx == b].mean(),
                            "n": int((idx == b).sum()),
                        }
                    )
            point = cal_stats(y, p)
            boots = []
            for _ in range(2000):
                take = np.concatenate(
                    [groups[i] for i in rng.integers(0, len(groups), len(groups))]
                )
                boots.append(cal_stats(y[take], p[take]))
            b = pd.DataFrame(boots)
            sum_rows.append(
                {
                    "model": name,
                    **point,
                    **{f"{k}_lower": float(b[k].quantile(0.025)) for k in point},
                    **{f"{k}_upper": float(b[k].quantile(0.975)) for k in point},
                }
            )
            for t in THRESHOLDS:
                dca_rows.append(
                    {"model": name, "threshold": t, "net_benefit": net_benefit(y, p, t)}
                )
        prev = y.mean()
        for t in THRESHOLDS:
            dca_rows.append(
                {
                    "model": "Treat all",
                    "threshold": t,
                    "net_benefit": prev - (1 - prev) * t / (1 - t),
                }
            )
        sub = out / tag
        sub.mkdir(exist_ok=True)
        cal, summ, dca = pd.DataFrame(cal_rows), pd.DataFrame(sum_rows), pd.DataFrame(dca_rows)
        cal.to_csv(sub / "calibration.csv", index=False)
        summ.to_csv(sub / "summary.csv", index=False)
        dca.to_csv(sub / "dca.csv", index=False)
        print(
            f"== {tag}\n"
            + summ[["model", "brier", "ece", "intercept", "slope"]].round(3).to_string(index=False)
        )
        ax_c, ax_d = axes[row]
        ax_c.plot([0, 1], [0, 1], color="#52514e", lw=0.6, ls=(0, (2, 2)))
        for col, name in MODELS.items():
            g = cal[cal["model"] == name]
            ax_c.plot(
                g["predicted"],
                g["observed"],
                marker="o",
                ms=3,
                lw=1.2,
                color=COLORS[col],
                label=name,
            )
            g = dca[dca["model"] == name]
            ax_d.plot(g["threshold"], g["net_benefit"], lw=1.2, color=COLORS[col], label=name)
        g = dca[dca["model"] == "Treat all"]
        ax_d.plot(
            g["threshold"],
            g["net_benefit"],
            lw=0.8,
            color="#52514e",
            ls=(0, (4, 3)),
            label="Treat all",
        )
        ax_d.axhline(0, color="#52514e", lw=0.6)
        ax_d.set_ylim(-0.02, max(0.05, dca["net_benefit"].max() * 1.1))
        ax_c.set(
            xlabel="Predicted probability",
            ylabel="Observed pneumonia rate",
            title=tag,
            xlim=(0, lim := max(0.05, 1.1 * max(cal["predicted"].max(), cal["observed"].max()))),
            ylim=(0, lim),
        )
        ax_d.set(xlabel="Threshold probability", ylabel="Net benefit", title=tag)
        for ax in (ax_c, ax_d):
            ax.grid(color="#e4e3df", lw=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, 1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "calibration_dca.pdf", bbox_inches="tight")
    fig.savefig(out / "calibration_dca.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
