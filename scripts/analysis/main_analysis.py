"""Main analysis: encounter-level stacking, recalibration, interactions, HIV subgroup.

  * unit = encounter (index study): image-level predictions are averaged per study; neural runs
    with several seeds are averaged over seeds first (the split is shared, so this is valid);
  * the label is the dataset's ``label_cap`` (not the runs' own y), so every model is scored on
    the same encounters;
  * stacked model: logistic regression on [logit EHR, logit CXR], fitted on validation encounters,
    REFITTED inside every bootstrap resample (validation and test patients both resampled), so its
    intervals include the stacker's own estimation error; coefficients with intervals;
  * calibration: every model gets the same Platt recalibration fitted on validation encounters
    before calibration slope / intercept / Brier are compared;
  * strata (ED presentation, index view, ICU at the film, hours from presentation and EHR
    completeness in tertiles cut on TRAINING encounters): stacked-minus-EHR gain per level and an
    interaction test (difference of gains between levels, patient bootstrap);
  * HIV subgroup (HIV code on the encounter or before the film): the full-cohort models on its test
    encounters, contrasts B1 (stack - EHR), B2 (stack - CXR), B3 (fusion - EHR), and the minimal
    detectable difference for B1 (2.8 x its bootstrap standard error: 80 % power, alpha 0.05).

Usage: python scripts/analysis/main_analysis.py OUT DATASET EHR EHR_NOWORKUP CXR FUSION FUSION_NOWORKUP
(each model argument is a run directory or a cv directory holding seed*/ runs)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

B = int(os.environ.get("PAPERB_BOOT", 2000))
B_SUB = min(B, 1000)


def run_dirs(path: Path) -> list[Path]:
    if (path / "predictions_test.csv").exists():
        return [path]
    seeds = sorted(p for p in path.glob("seed*") if (p / "predictions_test.csv").exists())
    if seeds:
        return seeds
    sub = sorted(p for p in path.glob("*/") if (p / "predictions_test.csv").exists())
    if not sub:
        raise FileNotFoundError(f"no predictions under {path}")
    return sub[-1:]


def predictions(path: Path, split: str) -> pd.Series:
    """Per-study mean probability (over images, then over seeds)."""
    per_seed = []
    for run in run_dirs(path):
        d = pd.read_csv(run / f"predictions_{split}.csv")
        p = d[[c for c in d if c.startswith("prob_")][-1]]
        study = d["sample_id"].astype(str).str.split("|").str[1].astype("int64")
        per_seed.append(p.groupby(study).mean())
    return pd.concat(per_seed, axis=1).mean(axis=1)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def fit_stack(df: pd.DataFrame, a: str, b: str) -> LogisticRegression:
    return LogisticRegression(C=1.0).fit(np.c_[logit(df[a]), logit(df[b])], df["y"])


def apply_stack(m: LogisticRegression, df: pd.DataFrame, a: str, b: str) -> np.ndarray:
    return m.predict_proba(np.c_[logit(df[a]), logit(df[b])])[:, 1]


def patient_index(groups: np.ndarray) -> dict:
    return {g: np.flatnonzero(groups == g) for g in np.unique(groups)}


def draw(idx: dict, rng: np.random.Generator) -> np.ndarray:
    keys = list(idx)
    return (
        np.concatenate([idx[keys[k]] for k in rng.choice(len(keys), len(keys))])
        if keys
        else np.array([], int)
    )


def main() -> None:
    out, ds = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    paths = dict(zip(["ehr", "ehr_nw", "cxr", "fusion", "fusion_nw"],
                     (Path(a).expanduser() for a in sys.argv[3:8]), strict=True))  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(ds / "labels.parquet").set_index("study_id")
    lab = lab[lab["label_cap"].notna()]
    tab = pd.read_parquet(ds / "tabular.parquet").set_index("study_id")
    cols = [c for c in tab.columns if c.startswith(("lab_", "edv_", "icu_", "triage_"))]
    lab["ehr_completeness"] = tab[cols].notna().mean(axis=1).reindex(lab.index)
    frames = {}
    for split in ("val", "test"):
        f = pd.DataFrame({k: predictions(p, split) for k, p in paths.items()}).dropna()
        f = f.join(lab, how="inner")
        f["y"] = f["label_cap"].astype(int)
        frames[split] = f
    va, te = frames["val"], frames["test"]
    tr = lab[lab["dev_split"] == "train"]
    cuts = {
        c: tr[c].quantile([1 / 3, 2 / 3]).to_numpy()
        for c in ("ehr_completeness", "hours_from_presentation")
    }
    for c, (q1, q2) in cuts.items():
        te[c + "_tertile"] = np.where(te[c] <= q1, "low", np.where(te[c] <= q2, "mid", "high"))
    te["hiv"] = ((te["label_hiv"].fillna(0) == 1) | (te["hiv_prior"].fillna(0) == 1)).astype(int)

    stack, stack_nw = fit_stack(va, "ehr", "cxr"), fit_stack(va, "ehr_nw", "cxr")
    te["stack"], te["stack_nw"] = (
        apply_stack(stack, te, "ehr", "cxr"),
        apply_stack(stack_nw, te, "ehr_nw", "cxr"),
    )
    va["stack"], va["stack_nw"] = (
        apply_stack(stack, va, "ehr", "cxr"),
        apply_stack(stack_nw, va, "ehr_nw", "cxr"),
    )
    models = ["ehr", "cxr", "fusion", "stack", "ehr_nw", "fusion_nw", "stack_nw"]
    pairs = [("ehr", "stack"), ("cxr", "stack"), ("fusion", "stack"), ("ehr", "fusion"),
             ("ehr_nw", "stack_nw"), ("ehr_nw", "fusion_nw"), ("ehr", "ehr_nw")]  # fmt: skip
    strata = ["ed_presentation", "index_view", "icu_at_film", "hours_from_presentation_tertile",
              "ehr_completeness_tertile"]  # fmt: skip

    rng = np.random.default_rng(0)
    gte, gva = te["patient_id"].astype(str).to_numpy(), va["patient_id"].astype(str).to_numpy()
    ite, iva = patient_index(gte), patient_index(gva)
    boot_auc = {m: [] for m in models}
    boot_coef = []
    boot_sub: dict = {}
    hiv_mask = te["hiv"].to_numpy() == 1
    for b in range(B):
        vb, tb = va.iloc[draw(iva, rng)], te.iloc[draw(ite, rng)].copy()
        s1, s2 = fit_stack(vb, "ehr", "cxr"), fit_stack(vb, "ehr_nw", "cxr")
        boot_coef.append([*s1.coef_[0], s1.intercept_[0]])
        tb["stack"], tb["stack_nw"] = (
            apply_stack(s1, tb, "ehr", "cxr"),
            apply_stack(s2, tb, "ehr_nw", "cxr"),
        )
        y = tb["y"].to_numpy()
        for m in models:
            boot_auc[m].append(auc(y, tb[m].to_numpy()))
        if b < B_SUB:
            for col in strata:
                for lvl, d in tb.groupby(col):
                    key = (col, str(lvl))
                    boot_sub.setdefault(key, []).append(auc(d["y"].to_numpy(), d["stack"].to_numpy())
                                                        - auc(d["y"].to_numpy(), d["ehr"].to_numpy()))  # fmt: skip
            h = tb[tb["hiv"] == 1]
            for name, (a, c) in {
                "B1": ("ehr", "stack"),
                "B2": ("cxr", "stack"),
                "B3": ("ehr", "fusion"),
            }.items():
                boot_sub.setdefault(("hiv", name), []).append(auc(h["y"].to_numpy(), h[c].to_numpy())
                                                              - auc(h["y"].to_numpy(), h[a].to_numpy()))  # fmt: skip
    ba = {m: np.asarray(v) for m, v in boot_auc.items()}
    y = te["y"].to_numpy()
    overall = pd.DataFrame([{"model": m, "auroc": auc(y, te[m].to_numpy()), "lower": np.nanquantile(ba[m], 0.025),
                             "upper": np.nanquantile(ba[m], 0.975)} for m in models])  # fmt: skip

    def summarise(point: float, draws: np.ndarray) -> dict:
        d = draws[~np.isnan(draws)]
        if len(d) < 2:
            return {"diff": point, "lower": np.nan, "upper": np.nan, "p": np.nan, "se": np.nan}
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan
        return {"diff": point, "lower": float(np.quantile(d, 0.025)), "upper": float(np.quantile(d, 0.975)),
                "p": max(p, 1 / max(len(d), 1)), "se": float(d.std(ddof=1))}  # fmt: skip

    comps = pd.DataFrame([{"a": a, "b": c, **summarise(auc(y, te[c].to_numpy()) - auc(y, te[a].to_numpy()),
                                                         ba[c] - ba[a])} for a, c in pairs])  # fmt: skip
    coef = np.asarray(boot_coef)
    coefs = pd.DataFrame({"term": ["logit_ehr", "logit_cxr", "intercept"],
                          "value": [*stack.coef_[0], stack.intercept_[0]],
                          "lower": np.quantile(coef, 0.025, axis=0), "upper": np.quantile(coef, 0.975, axis=0)})  # fmt: skip

    # strata: gain per level, interaction = difference between levels (binary) or high - low
    sub, inter = [], []
    for col in strata:
        levels = sorted(te[col].astype(str).unique())
        gains = {}
        for lvl in levels:
            d = te[te[col].astype(str) == lvl]
            g = auc(d["y"].to_numpy(), d["stack"].to_numpy()) - auc(
                d["y"].to_numpy(), d["ehr"].to_numpy()
            )
            gains[lvl] = g
            sub.append({"stratum": col, "level": lvl, "n": len(d), "positives": int(d["y"].sum()),
                        "ehr": auc(d["y"].to_numpy(), d["ehr"].to_numpy()),
                        "stack": auc(d["y"].to_numpy(), d["stack"].to_numpy()),
                        **summarise(g, np.asarray(boot_sub[(col, lvl)]))})  # fmt: skip
        lo, hi = ("low", "high") if "tertile" in col else (levels[0], levels[-1])
        if lo in gains and hi in gains:
            draws = np.asarray(boot_sub[(col, hi)]) - np.asarray(boot_sub[(col, lo)])
            inter.append(
                {
                    "stratum": col,
                    "contrast": f"{hi} - {lo}",
                    **summarise(gains[hi] - gains[lo], draws),
                }
            )
    sub, inter = pd.DataFrame(sub), pd.DataFrame(inter)

    # HIV subgroup
    h = te[hiv_mask]
    hiv_rows = []
    for name, (a, c) in {
        "B1": ("ehr", "stack"),
        "B2": ("cxr", "stack"),
        "B3": ("ehr", "fusion"),
    }.items():
        s = summarise(auc(h["y"].to_numpy(), h[c].to_numpy()) - auc(h["y"].to_numpy(), h[a].to_numpy()),
                      np.asarray(boot_sub[("hiv", name)]))  # fmt: skip
        hiv_rows.append({"contrast": name, "a": a, "b": c, **s, "mdd_80pct_power": 2.8 * s["se"]})
    hiv = pd.DataFrame(hiv_rows)
    hiv_info = {"encounters": int(len(h)), "patients": int(h["patient_id"].nunique()),
                "positives": int(h["y"].sum()), **{m: auc(h["y"].to_numpy(), h[m].to_numpy()) for m in models}}  # fmt: skip

    # calibration after the same validation recalibration
    cal = []
    for m in models:
        r = LogisticRegression(C=1e6).fit(logit(va[m]).reshape(-1, 1), va["y"])
        q = r.predict_proba(logit(te[m]).reshape(-1, 1))[:, 1]
        fit = LogisticRegression(C=1e6).fit(logit(q).reshape(-1, 1), y)
        cal.append({"model": m, "slope": float(fit.coef_[0, 0]), "intercept": float(fit.intercept_[0]),
                    "brier": float(np.mean((q - y) ** 2)), "brier_raw": float(np.mean((te[m] - y) ** 2))})  # fmt: skip
        te[m + "_recal"] = q
    cal = pd.DataFrame(cal)

    for name, frame in {"overall": overall, "comparisons": comps, "stacker": coefs, "strata": sub,
                        "interactions": inter, "hiv": hiv, "calibration": cal}.items():  # fmt: skip
        frame.to_csv(out / f"{name}.csv", index=False)
    te.reset_index().to_csv(out / "test_predictions.csv", index=False)
    info = {"test_encounters": int(len(te)), "test_patients": int(te["patient_id"].nunique()),
            "test_positives": int(y.sum()), "val_encounters": int(len(va)), "hiv": hiv_info,
            "tertile_cuts_train": {k: v.tolist() for k, v in cuts.items()}}  # fmt: skip
    (out / "info.json").write_text(json.dumps(info, indent=2, default=float))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        for name, frame in (("overall", overall), ("comparisons", comps), ("stacker", coefs), ("interactions", inter),
                            ("hiv", hiv), ("calibration", cal)):  # fmt: skip
            print(f"== {name}\n{frame.round(4).to_string(index=False)}")
    print(json.dumps(info, indent=2, default=float))


if __name__ == "__main__":
    main()
