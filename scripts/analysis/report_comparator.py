"""EHR-plus-report comparator.

The radiology report is the radiologist's reading of the same index radiograph, written after it,
so it is not available at prediction time; it bounds how much a reading of the film can add to the
EHR, and how much the label depends on that reading.

  * report model: TF-IDF (word 1-2 grams) on the report's findings and impression, logistic
    regression; trained on TRAINING encounters, regularisation chosen by 5-fold cross-validation
    grouped by patient on training encounters only (validation and test untouched);
  * stacked EHR + report: logistic regression on [logit EHR, logit report] fitted on validation
    encounters, exactly as the stacked EHR + CXR model, and both stackers refitted inside every
    bootstrap resample (validation and test patients resampled);
  * contrasts on test encounters: report alone, EHR + report minus EHR, EHR + report minus
    EHR + CXR, and EHR + CXR + report minus EHR + report (does the film add beyond its report?).

Writes OUT/report_overall.csv, OUT/report_comparisons.csv, OUT/report_info.json.
Usage: python scripts/analysis/report_comparator.py OUT DATASET EHR_RUN CXR_RUN
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from main_analysis import auc, draw, logit, patient_index, predictions  # noqa: E402

B = int(os.environ.get("PAPERB_BOOT", 2000))
CS = (0.1, 0.3, 1.0, 3.0, 10.0)


def fit_lr(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1.0).fit(x, y)


def cols(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    return np.column_stack([logit(df[n]) for n in names])


def report_model(lab: pd.DataFrame, rep: pd.Series) -> tuple[pd.Series, dict]:
    tr = lab[lab["dev_split"] == "train"]
    text_tr, y_tr = rep.reindex(tr.index).fillna(""), tr["y"].to_numpy()
    groups = tr["patient_id"].astype(str).to_numpy()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True)
    x_tr = vec.fit_transform(text_tr)
    cv = {}
    for c in CS:
        scores = []
        for a, b in GroupKFold(5).split(x_tr, y_tr, groups):
            m = LogisticRegression(C=c, max_iter=2000).fit(x_tr[a], y_tr[a])
            scores.append(roc_auc_score(y_tr[b], m.predict_proba(x_tr[b])[:, 1]))
        cv[c] = float(np.mean(scores))
        print(f"C={c}: cv AUROC {cv[c]:.4f}", flush=True)
    best = max(cv, key=cv.get)
    m = LogisticRegression(C=best, max_iter=2000).fit(x_tr, y_tr)
    rest = lab[lab["dev_split"].isin(["val", "test"])]
    p = m.predict_proba(vec.transform(rep.reindex(rest.index).fillna("")))[:, 1]
    return pd.Series(p, index=rest.index), {"C": best, "cv_auroc": cv, "n_train": int(len(tr))}


def main() -> None:
    out, ds = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    ehr_run, cxr_run = Path(sys.argv[3]).expanduser(), Path(sys.argv[4]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(ds / "labels.parquet").set_index("study_id")
    lab = lab[lab["label_cap"].notna()].copy()
    lab["y"] = lab["label_cap"].astype(int)
    rep = pd.read_parquet(ds / "reports.parquet").set_index("study_id")["report_text"]
    p_rep, info = report_model(lab, rep)

    frames = {}
    for split in ("val", "test"):
        f = pd.DataFrame({"ehr": predictions(ehr_run, split), "cxr": predictions(cxr_run, split)})
        f = f.join(p_rep.rename("report"), how="inner").dropna()
        f = f.join(lab[["y", "patient_id"]], how="inner")
        frames[split] = f
    va, te = frames["val"], frames["test"]
    stacks = {"stack": ["ehr", "cxr"], "stack_rep": ["ehr", "report"],
              "stack_all": ["ehr", "cxr", "report"]}  # fmt: skip

    def score(v: pd.DataFrame, t: pd.DataFrame) -> dict[str, float]:
        s = {m: auc(t["y"].to_numpy(), t[m].to_numpy()) for m in ("ehr", "cxr", "report")}
        for name, inputs in stacks.items():
            lr = fit_lr(cols(v, inputs), v["y"].to_numpy())
            s[name] = auc(t["y"].to_numpy(), lr.predict_proba(cols(t, inputs))[:, 1])
        return s

    point = score(va, te)
    rng = np.random.default_rng(0)
    ite = patient_index(te["patient_id"].astype(str).to_numpy())
    iva = patient_index(va["patient_id"].astype(str).to_numpy())
    boots = []
    for i in range(B):
        boots.append(score(va.iloc[draw(iva, rng)], te.iloc[draw(ite, rng)]))
        if (i + 1) % 250 == 0:
            print(f"bootstrap {i + 1}/{B}", flush=True)
    bt = pd.DataFrame(boots)
    overall = pd.DataFrame(
        [{"model": m, "auroc": point[m], "lower": bt[m].quantile(0.025),
          "upper": bt[m].quantile(0.975)} for m in point]  # fmt: skip
    )
    pairs = [("ehr", "stack_rep"), ("stack", "stack_rep"), ("stack_rep", "stack_all"),
             ("ehr", "stack")]  # fmt: skip
    comp = []
    for a, b in pairs:
        d = bt[b] - bt[a]
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        comp.append({"a": a, "b": b, "diff": point[b] - point[a], "lower": d.quantile(0.025),
                     "upper": d.quantile(0.975), "p": max(p, 1 / B)})  # fmt: skip
    overall.to_csv(out / "report_overall.csv", index=False)
    pd.DataFrame(comp).to_csv(out / "report_comparisons.csv", index=False)
    info.update(test_encounters=int(len(te)), val_encounters=int(len(va)), bootstrap=B)
    (out / "report_info.json").write_text(json.dumps(info, indent=1))
    print(overall.round(3).to_string(index=False))
    print(pd.DataFrame(comp).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
