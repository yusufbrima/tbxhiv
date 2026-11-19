"""LaTeX tables for the paper from the analysis outputs (never typed by hand).

Reads ~/tbmxai_runs/analysis/pneumonia (scripts/analysis/main_analysis.py) and its calibration_dca/ folder
(calibration_dca.py on the recalibrated test predictions) and writes performance.tex, gains.tex
and subgroups.tex into OUT_DIR.

Usage: python scripts/reporting/make_tables.py OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

A = Path("~/tbmxai_runs/analysis/pneumonia").expanduser()
NAMES = {
    "ehr": "EHR XGBoost",
    "cxr": "CXR only",
    "fusion": "End-to-end fusion",
    "stack": "Stacked EHR + CXR",
    "ehr_nw": "EHR XGBoost, no workup",
    "fusion_nw": "End-to-end fusion, no workup",
    "stack_nw": "Stacked EHR + CXR, no workup",
}
STRATA = {
    "ed_presentation": ("Route", {"1": "ED", "0": "direct"}),
    "index_view": ("View", {"PA": "PA", "AP": "AP"}),
    "icu_at_film": ("ICU", {"1": "yes", "0": "no"}),
    "hours_from_presentation_tertile": ("Hours", {"low": "low", "mid": "mid", "high": "high"}),
    "ehr_completeness_tertile": ("EHR completeness", {"low": "low", "mid": "mid", "high": "high"}),
}  # fmt: skip


def ci(d: float, lo: float, hi: float, digits: int = 3) -> str:
    f = f"{{:+.{digits}f}}"
    return f"{f.format(d)} [{f.format(lo)}, {f.format(hi)}]".replace("-", "$-$")


def signed(x: float, digits: int) -> str:
    return f"{x:+.{digits}f}".replace("-", "$-$")


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def table(caption: str, label: str, cols: str, header: str, rows: list[str]) -> str:
    return (
        "\\begin{table*}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{{\\small\\setlength{{\\tabcolsep}}{{4pt}}\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


def performance(info: dict) -> str:
    ov = pd.read_csv(A / "overall.csv").set_index("model")
    rows = []
    for tag, folder in (("Full cohort", "Full cohort"), ("HIV-positive", "HIV-positive")):
        cal = pd.read_csv(A / "calibration_dca" / folder / "summary.csv").set_index("model")
        for i, key in enumerate(("ehr", "cxr", "fusion", "stack")):
            c = cal.loc[NAMES[key]]
            if tag == "Full cohort":
                r = ov.loc[key]
                auc = f"{num(r.auroc)} [{num(r.lower)}, {num(r.upper)}]"
            else:
                auc = num(info["hiv"][key])
            slope = f"{c.slope:.2f} [{c.slope_lower:.2f}, {c.slope_upper:.2f}]"
            rows.append(
                f"  {tag if i == 0 else ''} & {NAMES[key]} & {auc} & {c.brier:.4f} & "
                f"{signed(c.intercept, 2)} & {slope} \\\\"
            )
        rows.append("  \\midrule")
    rows.pop()
    h = info["hiv"]
    cap = (
        f"Test performance at the encounter level ({info['test_encounters']:,} encounters, "
        f"{info['test_patients']:,} patients, {info['test_positives']} with community-onset pneumonia; "
        f"HIV-positive: {h['encounters']} encounters, {h['patients']} patients, {h['positives']} positive). "
        "AUROC with a patient-level bootstrap 95\\% CI (2,000 resamples; HIV-positive: point estimates, "
        "see Table~\\ref{tab:gains} for its contrasts). Brier score, calibration intercept and slope "
        "(1 = ideal, patient bootstrap 95\\% CI) after the same Platt recalibration on validation "
        "encounters for every model."
    )
    return table(
        cap,
        "tab:performance",
        "llcccc",
        "Cohort & Model & AUROC & Brier & Cal.\\ intercept & Cal.\\ slope",
        rows,
    )


SHORT = {"ehr": "EHR", "cxr": "CXR", "fusion": "Fusion", "stack": "Stacked",
         "ehr_nw": "EHR", "fusion_nw": "Fusion", "stack_nw": "Stacked"}  # fmt: skip


def holm_text(hiv: pd.DataFrame) -> str:
    """Holm-adjusted p values of the HIV contrasts, in contrast order."""
    ordered = hiv.sort_values("p")
    adjusted, running = {}, 0.0
    for i, (_, r) in enumerate(ordered.iterrows()):
        running = max(running, min(1.0, (len(ordered) - i) * r.p))
        adjusted[r.contrast] = running
    return ", ".join(f"{c} {pval(adjusted[c])}" for c in hiv.contrast)


def gains(info: dict) -> str:
    comp = pd.read_csv(A / "comparisons.csv")
    hiv = pd.read_csv(A / "hiv.csv")
    rows = []
    groups = (
        (
            "Full cohort, all features",
            comp[~comp.a.str.endswith("_nw") & ~comp.b.str.endswith("_nw")],
        ),
        ("Full cohort, no workup", comp[comp.a.str.endswith("_nw") & comp.b.str.endswith("_nw")]),
    )
    for label, g in groups:
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append(
                f"  {label if i == 0 else ''} & {SHORT[r.b]} $-$ {SHORT[r.a]} & "
                f"{ci(r['diff'], r.lower, r.upper)} & {pval(r.p)} & -- \\\\"
            )
    r = comp[(comp.a == "ehr") & (comp.b == "ehr_nw")].iloc[0]
    rows.append(
        f"   & EHR, no workup $-$ EHR, all & {ci(r['diff'], r.lower, r.upper)} & {pval(r.p)} & -- \\\\"
    )
    rows.append("  \\midrule")
    for i, (_, r) in enumerate(hiv.iterrows()):
        rows.append(
            f"  {'HIV-positive' if i == 0 else ''} & {r.contrast}: {SHORT[r.b]} $-$ {SHORT[r.a]} & "
            f"{ci(r['diff'], r.lower, r.upper)} & {pval(r.p)} & {r.mdd_80pct_power:.3f} \\\\"
        )
    cap = (
        "Paired differences in test AUROC, with patient-level bootstrap 95\\% CIs and two-sided $p$ "
        "(2,000 resamples; the stacker is refitted in every resample). EHR: XGBoost; CXR: image model; "
        "Fusion: end-to-end fusion; Stacked: stacked EHR + CXR. ``No workup'': EHR features without "
        "microbiology orders, medication orders, laboratory-order counts and clinical service (327 of "
        "2,253 features), for the EHR, fusion and stacked models alike. MDD: minimal detectable "
        "difference at 80\\% power ($2.8 \\times$ the bootstrap standard error). HIV-positive contrasts "
        "B1--B3 are exploratory (Section~\\ref{sec:methods}); a difference counts if it is at least 0.02 "
        f"with a CI excluding zero after Holm correction (Holm-adjusted $p$: {holm_text(hiv)})."
    )
    return table(
        cap,
        "tab:gains",
        "llccc",
        "Cohort & Difference & $\\Delta$ AUROC [95\\% CI] & $p$ & MDD",
        rows,
    )


def subgroups() -> str:
    st = pd.read_csv(A / "strata.csv", dtype={"level": str})
    it = pd.read_csv(A / "interactions.csv").set_index("stratum")
    rows = []
    for key, (label, levels) in STRATA.items():
        g = st[st.stratum == key].set_index("level")
        for j, (lv, name) in enumerate(levels.items()):
            r = g.loc[lv]
            inter = ""
            if j == 0:
                x = it.loc[key]
                inter = ci(x["diff"], x.lower, x.upper)
            rows.append(
                f"  {label if j == 0 else ''} & {name} & {int(r.n):,} & {int(r.positives)} & "
                f"{num(r.ehr)} & {ci(r['diff'], r.lower, r.upper)} & {inter} \\\\"
            )
        rows.append("  \\midrule")
    rows.pop()
    cap = (
        "Where the radiograph adds: stacked minus EHR XGBoost AUROC by pre-specified stratum (full "
        "cohort, test encounters; route: through the ED, or direct admission without an ED registration; ICU: radiograph taken in intensive "
        "care; hours: from presentation to the radiograph; EHR: the EHR model's AUROC). Interaction: "
        "difference of gains between levels with a patient bootstrap 95\\% CI (ED minus direct "
        "admission, PA minus AP, ICU minus not, highest minus lowest tertile). Tertiles were cut on "
        "training encounters (hours: 1.3 and 2.7; EHR completeness: share of laboratory, triage, ED "
        "vital-sign and intensive-care chart feature values present before the radiograph, cuts 0.24 "
        "and 0.28)."
    )
    return table(
        cap,
        "tab:subgroups",
        "llrrccc",
        "Stratum & Level & Encounters & Pneumonia & EHR & Gain [95\\% CI] & Interaction [95\\% CI]",
        rows,
    )


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((A / "info.json").read_text())
    for name, text in (
        ("performance", performance(info)),
        ("gains", gains(info)),
        ("subgroups", subgroups()),
    ):
        (out / f"{name}.tex").write_text(text)
    print(f"wrote performance, gains, subgroups into {out}")


if __name__ == "__main__":
    main()
