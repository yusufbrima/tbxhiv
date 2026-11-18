"""Self-contained HTML report (figures embedded; opens offline, shareable as one file)."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

from tbmxai.data.quality import ASSUMED_IDS_WARNING
from tbmxai.reporting.base import REPORTERS, ReportWriter, RunResult
from tbmxai.reporting.figures import render_figures
from tbmxai.reporting.writers import _calibration_text, _threshold_text

_CSS = """
:root { --fg:#1d1d1f; --muted:#5f6368; --bg:#ffffff; --card:#f6f7f9; --line:#e2e4e8; --accent:#0072B2; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8eaed; --muted:#9aa0a6; --bg:#16181c; --card:#1f2228; --line:#2f333a; --accent:#56B4E9; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; }
.muted { color: var(--muted); }
.grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; margin: 16px 0; }
.kpi { background: var(--card); border:1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.kpi b { display:block; font-size: 20px; color: var(--accent); font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 10px; text-align: left; white-space: nowrap; }
th { color: var(--muted); font-weight: 600; }
figure { margin: 12px 0; } figure img { max-width: 100%; height: auto; background: #fff; border-radius: 6px; }
.figs { display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.warn { border-left: 4px solid #D55E00; background: var(--card); padding: 8px 12px; border-radius: 4px; }
code { background: var(--card); padding: 1px 4px; border-radius: 4px; }
"""


def _table(frame: pd.DataFrame, max_rows: int = 200) -> str:
    return (
        '<div class="scroll">'
        + frame.head(max_rows).to_html(
            index=False, border=0, na_rep="–", float_format=lambda v: f"{v:.3f}", escape=True
        )
        + "</div>"
    )


def _ci(v: float, lo: float, hi: float) -> str:
    if pd.isna(v):
        return "–"
    return f"{v:.3f}" if pd.isna(lo) else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def _img(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}"></figure>'


@REPORTERS.register("html")
class HTMLReportWriter(ReportWriter):
    """Write ``report.html`` (plus PNGs in ``figures/``)."""

    def write(self, result: RunResult, out_dir: Path) -> Path:
        cfg, cohort = result.config, result.cohort_summary
        figures = render_figures(result, out_dir)
        parts = [
            f"<h1>{html.escape(cfg.name)}</h1>",
            f'<p class="muted">Model <code>{html.escape(cfg.model.name)}</code> · modalities '
            f"{html.escape(', '.join(map(str, cfg.view.modalities)))} · split "
            f"<code>{html.escape(cfg.split.name)}</code> (patient-level) · seed {cfg.seed}"
            f" · threshold {html.escape(_threshold_text(result))}"
            f" · calibration {html.escape(_calibration_text(result))}</p>",
        ]
        if cohort.get("patient_ids_assumed"):
            parts.append(f'<p class="warn"><b>Warning:</b> {html.escape(ASSUMED_IDS_WARNING)}</p>')
        test = result.evaluations.get("test")
        kpis = [("Samples", cohort.get("n_samples")), ("Patients", cohort.get("n_patients"))]
        if test is not None:
            for m in ("auroc", "sensitivity", "specificity", "brier"):
                if m in test.metrics:
                    r = test.metrics[m]
                    kpis.append((f"Test {m}", f"{r.value:.3f}"))
        parts.append(
            '<div class="grid">'
            + "".join(
                f'<div class="kpi"><span class="muted">{html.escape(str(k))}</span><b>{html.escape(str(v))}</b></div>'
                for k, v in kpis
            )
            + "</div>"
        )

        for split, ev in result.evaluations.items():
            unit = f", {ev.ci_unit}-level bootstrap" if ev.ci_unit else ""
            conf = f" ({ev.confidence:.0%} CI{unit})" if ev.confidence else ""
            frame = pd.DataFrame(
                {
                    "metric": list(ev.metrics),
                    "value": [_ci(m.value, m.lower, m.upper) for m in ev.metrics.values()],
                }
            )
            parts += [
                f"<h2>{split.capitalize()} performance{conf}</h2>",
                f'<p class="muted">n = {ev.n_samples} samples, {ev.n_clusters or "?"} patients; '
                f"class counts {html.escape(json.dumps(ev.class_counts))}</p>",
                _table(frame),
            ]

        figs = [f for f in ("roc", "calibration") if f in figures]
        if figs:
            parts += [
                "<h2>Discrimination and calibration</h2>",
                '<div class="figs">' + "".join(_img(figures[f], f) for f in figs) + "</div>",
            ]
        if result.modality_ablation is not None and not result.modality_ablation.empty:
            ab = result.modality_ablation.assign(
                value=[
                    _ci(v, lo, hi)
                    for v, lo, hi in result.modality_ablation[
                        ["value", "lower", "upper"]
                    ].itertuples(index=False)
                ]
            )
            wide = ab.pivot_table(
                index=["available", "withheld", "n"],
                columns="metric",
                values="value",
                aggfunc="first",
            ).reset_index()
            parts += ["<h2>Missing-modality robustness</h2>", _table(wide)]
            if "modality_ablation_auroc" in figures:
                parts.append(_img(figures["modality_ablation_auroc"], "ablation"))
        if result.subgroups is not None and not result.subgroups.empty:
            sg = result.subgroups.assign(
                value=[
                    _ci(v, lo, hi)
                    for v, lo, hi in result.subgroups[["value", "lower", "upper"]].itertuples(
                        index=False
                    )
                ]
            )
            wide = sg.pivot_table(
                index=["column", "group", "n"], columns="metric", values="value", aggfunc="first"
            ).reset_index()
            parts += ["<h2>Subgroup analysis (test)</h2>"]
            if "subgroups_auroc" in figures:
                parts.append(_img(figures["subgroups_auroc"], "subgroups"))
            parts.append(_table(wide))
        expl = [k for k in figures if k.startswith(("importance_", "saliency_"))]
        if expl:
            parts += [
                "<h2>Explanations</h2>",
                '<div class="figs">' + "".join(_img(figures[k], k) for k in expl) + "</div>",
            ]
        if "training_history" in figures:
            parts += ["<h2>Training</h2>", _img(figures["training_history"], "history")]
        if result.tuning_best is not None:
            best = result.tuning_best
            parts += [
                "<h2>Hyper-parameter search</h2>",
                f'<p class="muted">Best validation value {best["value"]:.4f} (trial {best["trial"]}, {best["direction"]}).</p>',
                _table(
                    pd.DataFrame(
                        {
                            "parameter": list(best["params"]),
                            "value": [str(v) for v in best["params"].values()],
                        }
                    )
                ),
            ]
            if "tuning" in figures:
                parts.append(_img(figures["tuning"], "tuning"))
        if result.cohort_table is not None:
            parts += ["<h2>Cohort</h2>", _table(result.cohort_table)]
        quality = out_dir / "tables" / "data_quality_summary.csv"
        if quality.exists():
            parts += ["<h2>Data quality</h2>", _table(pd.read_csv(quality))]
        parts.append(
            f'<p class="muted">Full machine-readable outputs: <code>tables/</code>, '
            f"<code>metrics.json</code>, <code>run_info.json</code> in {html.escape(str(out_dir))}</p>"
        )
        doc = (
            f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(cfg.name)}</title><style>{_CSS}</style></head>"
            f"<body><main>{''.join(parts)}</main></body></html>"
        )
        path = out_dir / "report.html"
        path.write_text(doc)
        return path
