"""Publication-style figures built from a :class:`RunResult` (matplotlib, headless).

Every figure is saved as PNG under ``<run>/figures/`` and embedded in the HTML
report. All figures are computed from the same tidy tables/predictions that are
written to disk, so they can be regenerated or restyled elsewhere.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from tbmxai.core.types import Predictions

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from tbmxai.explain.base import Explanation
    from tbmxai.reporting.base import RunResult

# Colour-blind-safe palette (Okabe–Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)


def to_base64(fig: Figure) -> str:
    """PNG bytes of ``fig`` as a base64 string (for HTML embedding)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def roc_figure(predictions: dict[str, Predictions]) -> Figure | None:
    """ROC curves of the positive class (binary) per split."""
    binary = {k: p for k, p in predictions.items() if p.is_binary and len(np.unique(p.y_true)) == 2}
    if not binary:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for color, (split, p) in zip(PALETTE, binary.items(), strict=False):
        fpr, tpr, _ = roc_curve(p.y_true, p.y_prob[:, 1])
        from sklearn.metrics import roc_auc_score

        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=1.8,
            label=f"{split} (AUROC {roc_auc_score(p.y_true, p.y_prob[:, 1]):.3f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax.set(xlabel="1 − specificity", ylabel="Sensitivity", title="ROC", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    return fig


def calibration_figure(predictions: dict[str, Predictions], n_bins: int = 10) -> Figure | None:
    """Reliability diagram (quantile bins): predicted vs observed positive rate."""
    usable = {k: p for k, p in predictions.items() if len(p) >= 2 * n_bins}
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for color, (split, p) in zip(PALETTE, usable.items(), strict=False):
        if p.is_binary:
            conf, hit = p.y_prob[:, 1], (p.y_true == 1).astype(float)
            xlabel, ylabel = "Predicted probability", "Observed frequency"
        else:
            conf, hit = p.y_prob.max(1), (p.y_prob.argmax(1) == p.y_true).astype(float)
            xlabel, ylabel = "Confidence (top label)", "Accuracy"
        frame = pd.DataFrame({"c": conf, "h": hit})
        frame["bin"] = pd.qcut(frame["c"], q=n_bins, duplicates="drop")
        agg = frame.groupby("bin", observed=True).agg(c=("c", "mean"), h=("h", "mean"))
        ax.plot(agg["c"], agg["h"], marker="o", ms=3.5, color=color, lw=1.4, label=split)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax.set(xlabel=xlabel, ylabel=ylabel, title="Calibration", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="upper left")
    return fig


def forest_figure(
    table: pd.DataFrame,
    metric: str,
    label_cols: list[str],
    title: str,
    reference: float | None = None,
) -> Figure | None:
    """Point estimate + CI per row (subgroups, ablation …)."""
    rows = table[table["metric"] == metric].copy()
    if rows.empty:
        return None
    rows["label"] = (
        rows[label_cols].astype(str).agg(" = ".join, axis=1) + "  (n=" + rows["n"].astype(str) + ")"
    )
    rows = rows.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(rows) + 1.0))
    y = np.arange(len(rows))
    lo = (rows["value"] - rows["lower"]).clip(lower=0).fillna(0)
    hi = (rows["upper"] - rows["value"]).clip(lower=0).fillna(0)
    ax.errorbar(
        rows["value"],
        y,
        xerr=[lo, hi],
        fmt="o",
        ms=4,
        color=PALETTE[0],
        ecolor="#7f7f7f",
        capsize=2,
    )
    if reference is not None and np.isfinite(reference):
        ax.axvline(reference, color=PALETTE[1], ls="--", lw=0.9)
    ax.set_yticks(y, rows["label"])
    ax.set(xlabel=metric, title=title)
    ax.grid(axis="y", alpha=0)
    return fig


def importance_figure(table: pd.DataFrame, title: str, top: int = 20) -> Figure | None:
    """Horizontal bar chart of the ``top`` features by ``importance_mean``."""
    if table is None or table.empty or "importance_mean" not in table:
        return None
    t = table.sort_values("importance_mean", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(t) + 1.0))
    err = t.get("importance_std")
    ax.barh(
        [str(i)[:45] for i in t.index],
        t["importance_mean"],
        xerr=err,
        color=PALETTE[0],
        ecolor="#9a9a9a",
        capsize=1.5,
    )
    ax.set(xlabel="importance", title=title)
    ax.grid(axis="y", alpha=0)
    return fig


def history_figure(history: pd.DataFrame) -> Figure | None:
    """Training curves: losses (left) and validation metrics (right)."""
    if history is None or history.empty:
        return None
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for i, col in enumerate(c for c in ("train_loss", "val_loss") if c in history):
        a.plot(history["epoch"], history[col], color=PALETTE[i], label=col)
    a.set(xlabel="epoch", title="Loss")
    a.legend()
    metrics = [c for c in history if c.startswith("val_") and c != "val_loss"]
    for i, col in enumerate(metrics):
        b.plot(history["epoch"], history[col], color=PALETTE[i % len(PALETTE)], label=col)
    b.set(xlabel="epoch", title="Validation metrics")
    b.legend(fontsize=7)
    return fig


def tuning_figure(trials: pd.DataFrame, direction: str) -> Figure | None:
    """Trial values and best-so-far curve of an Optuna study."""
    if trials is None or trials.empty or "value" not in trials:
        return None
    done = trials[trials.get("state", "COMPLETE") == "COMPLETE"].sort_values("number")
    if done.empty:
        return None
    best = done["value"].cummax() if direction == "maximize" else done["value"].cummin()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.scatter(done["number"], done["value"], s=12, color=PALETTE[5], label="trial")
    ax.plot(done["number"], best, color=PALETTE[1], lw=1.6, label="best so far")
    ax.set(
        xlabel="trial",
        ylabel=str(done["metric"].iat[0]) if "metric" in done else "value",
        title="Hyper-parameter search",
    )
    ax.legend()
    return fig


def saliency_grid(exp: Explanation, max_items: int = 16) -> Figure | None:
    """One row per sample: the image, then a saliency overlay per explained class.

    Rows are grouped by true class, so every class is visible in the figure.
    """
    maps = np.asarray(exp.attributions)
    images = exp.extra.get("images")
    if maps.ndim == 3:
        maps = maps[:, None]
    if maps.ndim != 4 or images is None or len(maps) == 0:
        return None
    names = exp.extra.get("class_names") or [str(i) for i in range(maps.shape[1])]
    targets = exp.extra.get("target_classes") or list(range(maps.shape[1]))
    y = np.asarray(exp.extra.get("y_true", np.zeros(len(maps), int)))
    order = np.argsort(y, kind="stable")[:max_items]
    cols = 1 + maps.shape[1]
    fig, axes = plt.subplots(
        len(order), cols, figsize=(2.0 * cols, 2.1 * len(order)), squeeze=False
    )
    for r, i in enumerate(order):
        cat = exp.extra.get("category", [""] * len(maps))[i]
        prob = exp.extra.get("prob", [np.nan] * len(maps))[i]
        axes[r, 0].imshow(images[i], cmap="gray")
        axes[r, 0].set_title(f"true: {names[y[i]]}\n{cat} p={prob:.2f}", fontsize=7)
        for c in range(maps.shape[1]):
            axes[r, c + 1].imshow(images[i], cmap="gray")
            axes[r, c + 1].imshow(maps[i, c], cmap="jet", alpha=0.35, vmin=0, vmax=1)
            if r == 0:
                axes[r, c + 1].set_title(f"→ {names[targets[c]]}", fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(exp.method.replace("_", " ").title() + " (columns: explained class)", fontsize=9)
    fig.tight_layout()
    return fig


def render_figures(result: RunResult, out_dir: Path) -> dict[str, Path]:
    """Create every applicable figure under ``out_dir/figures`` and return their paths."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Figure | None] = {
        "roc": roc_figure(result.predictions),
        "calibration": calibration_figure(result.predictions),
        "training_history": history_figure(result.training_history),
    }
    if result.subgroups is not None and not result.subgroups.empty:
        overall = result.subgroups[
            (result.subgroups["column"] == "__overall__") & (result.subgroups["metric"] == "auroc")
        ]
        ref = float(overall["value"].iat[0]) if len(overall) else None
        sub = result.subgroups[result.subgroups["column"] != "__overall__"]
        figs["subgroups_auroc"] = forest_figure(
            sub, "auroc", ["column", "group"], "Subgroup AUROC (test)", ref
        )
    if result.modality_ablation is not None and not result.modality_ablation.empty:
        figs["modality_ablation_auroc"] = forest_figure(
            result.modality_ablation, "auroc", ["available"], "Test AUROC by available modalities"
        )
    if result.tuning_trials is not None and result.tuning_best is not None:
        figs["tuning"] = tuning_figure(
            result.tuning_trials, result.tuning_best.get("direction", "maximize")
        )
    for exp in result.explanations:
        if isinstance(exp.attributions, pd.DataFrame) and exp.scope == "global":
            figs[f"importance_{exp.method}"] = importance_figure(
                exp.attributions, exp.method.replace("_", " ")
            )
        elif isinstance(exp.extra.get("global_long"), pd.DataFrame):
            long = exp.extra["global_long"]
            for cls in long["target_class"].unique():
                part = long[(long["target_class"] == cls) & (long["sample_class"] == "all")]
                figs[f"importance_{exp.method}_{cls}"] = importance_figure(
                    part.set_index("feature"), f"{exp.method}: output '{cls}' (mean |SHAP|)"
                )
        elif exp.modality.value == "imaging":
            figs[f"saliency_{exp.method}"] = saliency_grid(exp)
    paths = {}
    for name, fig in figs.items():
        if fig is None:
            continue
        path = fig_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths[name] = path
    return paths
