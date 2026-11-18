"""Statistical comparison of models evaluated on the **same** samples.

For every pair of models and every metric:

* **Paired cluster bootstrap** (primary): patients are resampled with
  replacement (stratified by patient label), the *same* resample is applied to
  both models, and the difference ``metric(A) − metric(B)`` is recomputed. This
  yields a percentile CI of the difference and a two-sided p-value
  ``2·min(P(Δ ≤ 0), P(Δ ≥ 0))``. Valid for any metric and for patients that
  contribute several samples.
* **DeLong test** (AUROC, binary): asymptotic test for two correlated ROC
  curves (DeLong et al., 1988; fast algorithm of Sun & Xu, 2014).
* **McNemar exact test** (accuracy, sensitivity, specificity at the decision
  threshold): exact binomial test on discordant pairs, restricted to positives
  for sensitivity and negatives for specificity.

DeLong and McNemar assume independent samples; when patients contribute more
than one sample the result table flags ``clustered=True`` and the cluster
bootstrap should be used as the primary evidence (or compare on one index sample
per patient via ``view.index_policy``).

p-values are adjusted for multiple comparisons within each metric across all
model pairs (Holm by default; Benjamini–Hochberg optional).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tbmxai.core.types import Predictions
from tbmxai.core.utils import get_logger
from tbmxai.evaluation.bootstrap import BootstrapCI
from tbmxai.evaluation.evaluator import Evaluator
from tbmxai.evaluation.metrics import METRICS

logger = get_logger(__name__)

DEFAULT_COMPARISON_METRICS = (
    "auroc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "spec_at_sens",
    "sens_at_spec",
    "balanced_accuracy",
    "f1_macro",
    "brier",
    "ece",
)


# ------------------------------------------------------------------ DeLong


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    """Two-sided DeLong test for paired AUROCs (binary labels, 1 = positive).

    Returns:
        ``auc_a, auc_b, diff, se, z, p``.
    """
    y = np.asarray(y_true).astype(int)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        return {
            "auc_a": np.nan,
            "auc_b": np.nan,
            "diff": np.nan,
            "se": np.nan,
            "z": np.nan,
            "p": np.nan,
        }
    preds = np.vstack([score_a, score_b]).astype(float)
    x, z = preds[:, pos], preds[:, neg]
    v01, v10, aucs = np.empty((2, m)), np.empty((2, n)), np.empty(2)
    for r in range(2):
        tx, tz = _midrank(x[r]), _midrank(z[r])
        tall = _midrank(np.concatenate([x[r], z[r]]))
        aucs[r] = (tall[:m].sum() / m - (m + 1) / 2) / n
        v01[r] = (tall[:m] - tx) / n  # per-positive structural components
        v10[r] = 1.0 - (tall[m:] - tz) / m  # per-negative structural components
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var <= 0:
        return {
            "auc_a": aucs[0],
            "auc_b": aucs[1],
            "diff": diff,
            "se": 0.0,
            "z": np.nan,
            "p": 1.0 if diff == 0 else 0.0,
        }
    zstat = diff / np.sqrt(var)
    return {
        "auc_a": aucs[0],
        "auc_b": aucs[1],
        "diff": diff,
        "se": float(np.sqrt(var)),
        "z": float(zstat),
        "p": float(2 * stats.norm.sf(abs(zstat))),
    }


# ----------------------------------------------------------------- McNemar


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact (binomial) McNemar test on paired correctness indicators."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_a, n, 0.5).pvalue)
    return {"a_only_correct": only_a, "b_only_correct": only_b, "p": p}


# -------------------------------------------------------------- corrections


def adjust_pvalues(p: Sequence[float], method: str = "holm") -> np.ndarray:
    """Holm–Bonferroni or Benjamini–Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q, k = p[ok], int(ok.sum())
    if k == 0:
        return out
    order = np.argsort(q)
    if method == "holm":
        adj = np.maximum.accumulate((k - np.arange(k)) * q[order])
    elif method in ("bh", "fdr_bh"):
        adj = q[order] * k / np.arange(1, k + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
    elif method == "none":
        adj = q[order]
    else:
        raise ValueError(f"Unknown correction '{method}'")
    res = np.empty(k)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


# -------------------------------------------------------------- comparator


@dataclass
class ModelComparison:
    """Per-model metrics and pairwise test results."""

    models: pd.DataFrame
    pairwise: pd.DataFrame
    info: dict[str, object]

    def write(self, out_dir: str | Path) -> Path:
        """Write ``comparison_{models,pairwise}.csv``, ``comparison.md`` and a figure."""
        from tbmxai.reporting.writers import markdown_table

        out = Path(out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        self.models.to_csv(out / "comparison_models.csv", index=False)
        self.pairwise.to_csv(out / "comparison_pairwise.csv", index=False)
        lines = [
            "# Model comparison",
            "",
            f"- Samples: {self.info['n_samples']} · patients: {self.info['n_patients']} · "
            f"clustered (patients with >1 sample): {self.info['clustered']}",
            f"- Paired cluster bootstrap: {self.info['n_bootstrap']} resamples · "
            f"{self.info['confidence']:.0%} CI · correction: {self.info['correction']} "
            f"(within metric, across {self.info['n_pairs']} pairs) · α = {self.info['alpha']}",
            "",
        ]
        m = self.models.assign(
            cell=[
                f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else f"{v:.3f}"
                for v, lo, hi in self.models[["value", "lower", "upper"]].itertuples(index=False)
            ]
        )
        lines += [
            "## Per-model performance",
            "",
            markdown_table(m.pivot(index="metric", columns="model", values="cell").reset_index()),
            "",
        ]
        cols = [
            "metric",
            "model_a",
            "model_b",
            "diff",
            "diff_lower",
            "diff_upper",
            "p_bootstrap",
            "p_bootstrap_adj",
            "p_delong",
            "p_delong_adj",
            "p_mcnemar",
            "p_mcnemar_adj",
            "significant",
        ]
        shown = self.pairwise[[c for c in cols if c in self.pairwise]].copy()
        for c in shown.columns:
            if shown[c].dtype.kind == "f":
                shown[c] = shown[c].map(
                    lambda v: (
                        "–"
                        if not np.isfinite(v)
                        else (f"{v:.4f}" if abs(v) >= 1e-4 else f"{v:.1e}")
                    )
                )
        lines += [
            "## Pairwise differences (A − B)",
            "",
            markdown_table(shown),
            "",
            "Positive `diff` means A > B (for brier / ece lower is better). `significant` uses "
            "the adjusted bootstrap p-value.",
            "",
        ]
        (out / "comparison.md").write_text("\n".join(lines))
        try:
            self._figure(out)
        except Exception as exc:
            logger.warning("Comparison figure failed: %s", exc)
        return out

    def _figure(self, out: Path) -> None:
        from tbmxai.reporting.figures import PALETTE, plt

        pw = self.pairwise
        metrics = list(dict.fromkeys(pw["metric"]))
        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(5.5, 1.0 + 0.45 * len(pw)), squeeze=False
        )
        for ax, metric in zip(axes[:, 0], metrics, strict=True):
            part = pw[pw["metric"] == metric].iloc[::-1]
            y = np.arange(len(part))
            ax.errorbar(
                part["diff"],
                y,
                xerr=[
                    (part["diff"] - part["diff_lower"]).clip(lower=0),
                    (part["diff_upper"] - part["diff"]).clip(lower=0),
                ],
                fmt="o",
                ms=4,
                capsize=2,
                color=PALETTE[0],
                ecolor="#888888",
            )
            for yy, (_, row) in zip(y, part.iterrows(), strict=True):
                if row["significant"]:
                    ax.annotate(
                        "*",
                        (row["diff_upper"], yy),
                        xytext=(4, -3),
                        textcoords="offset points",
                        color=PALETTE[1],
                    )
            ax.axvline(0, color="grey", lw=0.8, ls="--")
            ax.set_yticks(
                y, [f"{a} − {b}" for a, b in zip(part["model_a"], part["model_b"], strict=True)]
            )
            ax.set_title(f"Δ {metric} (paired cluster bootstrap CI; * adjusted p < α)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "comparison_differences.png", bbox_inches="tight")
        plt.close(fig)


class ModelComparator:
    """Pairwise statistical comparison of models on identical samples.

    Args:
        metrics: Metric names (default :data:`DEFAULT_COMPARISON_METRICS`).
        n_bootstrap: Paired bootstrap resamples.
        confidence: CI level for metrics and differences.
        threshold: Binary decision threshold (sensitivity, specificity, McNemar).
        correction: ``holm`` (default), ``bh`` or ``none``.
        alpha: Significance level for the ``significant`` column.
        cluster: Resample patients (requires patient ids) rather than samples.
        seed: Bootstrap seed.
    """

    def __init__(
        self,
        metrics: Sequence[str] | None = None,
        n_bootstrap: int = 2000,
        confidence: float = 0.95,
        threshold: float = 0.5,
        correction: str = "holm",
        alpha: float = 0.05,
        cluster: bool = True,
        seed: int = 0,
    ) -> None:
        self.metrics = list(metrics or DEFAULT_COMPARISON_METRICS)
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.threshold = threshold
        self.correction = correction
        self.alpha = alpha
        self.cluster = cluster
        self.seed = seed

    @staticmethod
    def align(
        predictions: Mapping[str, Predictions], allow_subset: bool = False
    ) -> dict[str, Predictions]:
        """Align predictions by sample id; require identical samples and labels.

        Raises:
            ValueError: If sample sets differ (unless ``allow_subset``) or labels disagree.
        """
        if len(predictions) < 2:
            raise ValueError("Need at least two models to compare.")
        for name, p in predictions.items():
            if p.sample_ids is None:
                raise ValueError(f"Predictions of '{name}' have no sample ids.")
        sets = {k: set(map(str, p.sample_ids)) for k, p in predictions.items()}
        common = set.intersection(*sets.values())
        if not allow_subset and any(s != common for s in sets.values()):
            sizes = {k: len(s) for k, s in sets.items()}
            raise ValueError(
                f"Models were evaluated on different samples {sizes} (common {len(common)}). "
                "Use a shared protocol (tbmxai protocol) or pass allow_subset=True to compare "
                "on the intersection."
            )
        order = sorted(common)
        aligned = {}
        for name, p in predictions.items():
            pos = pd.Series(np.arange(len(p)), index=p.sample_ids.astype(str))
            aligned[name] = p.subset(pos[order].to_numpy())
        ref = next(iter(aligned.values()))
        for name, p in aligned.items():
            if not np.array_equal(p.y_true, ref.y_true):
                raise ValueError(f"Labels of '{name}' differ from the other models.")
        return aligned

    def compare(
        self,
        predictions: Mapping[str, Predictions],
        allow_subset: bool = False,
        thresholds: Mapping[str, float] | None = None,
    ) -> ModelComparison:
        """Per-model metrics with CIs + all pairwise tests.

        Args:
            predictions: Model name → predictions on the same samples.
            allow_subset: Compare on the common samples if the sets differ.
            thresholds: Per-model binary decision thresholds (e.g. chosen on each
                model's validation data); models not listed use ``self.threshold``.
        """
        preds = self.align(predictions, allow_subset)
        thr = {k: (thresholds or {}).get(k, self.threshold) for k in preds}
        names = list(preds)
        ref = preds[names[0]]
        y = ref.y_true
        groups = ref.group_ids if self.cluster else None
        clustered = bool(groups is not None and pd.Series(groups).value_counts().max() > 1)
        built = [METRICS.build(name) for name in self.metrics]
        metric_objs = {obj.name: obj for obj in built}
        hard = {k: p.predicted_labels(thr[k]) for k, p in preds.items()}

        boot = BootstrapCI(self.n_bootstrap, self.confidence, seed=self.seed, cluster=self.cluster)
        rows = []
        for name, p in preds.items():
            evaluator = Evaluator(list(metric_objs.values()), threshold=thr[name], bootstrap=boot)
            res = evaluator.evaluate(p)
            for r in res.metrics.values():
                rows.append(
                    {
                        "model": name,
                        "metric": r.name,
                        "value": r.value,
                        "lower": r.lower,
                        "upper": r.upper,
                        "n": res.n_samples,
                        "ci_unit": res.ci_unit,
                    }
                )
        models = pd.DataFrame(rows)

        # Bootstrap distributions of every metric for every model on the SAME resamples.
        dist = {k: {m: np.empty(self.n_bootstrap) for m in metric_objs} for k in names}
        for b, idx in enumerate(boot.resample_indices(y, groups)):
            yt = y[idx]
            for k, p in preds.items():
                yp, yd = p.y_prob[idx], hard[k][idx]
                for m, obj in metric_objs.items():
                    dist[k][m][b] = obj(yt, yp, yd)

        alpha_ci = (1 - self.confidence) / 2
        pair_rows = []
        for a, b in itertools.combinations(names, 2):
            pa, pb = preds[a], preds[b]
            for m, obj in metric_objs.items():
                point = obj(y, pa.y_prob, hard[a]) - obj(y, pb.y_prob, hard[b])
                d = dist[a][m] - dist[b][m]
                d = d[np.isfinite(d)]
                if len(d):
                    lo, hi = np.quantile(d, [alpha_ci, 1 - alpha_ci])
                    p_boot = min(
                        1.0,
                        2
                        * min(
                            (np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)
                        ),
                    )
                else:
                    lo = hi = p_boot = np.nan
                row = {
                    "metric": m,
                    "model_a": a,
                    "model_b": b,
                    "value_a": obj(y, pa.y_prob, hard[a]),
                    "value_b": obj(y, pb.y_prob, hard[b]),
                    "diff": point,
                    "diff_lower": lo,
                    "diff_upper": hi,
                    "p_bootstrap": p_boot,
                    "p_delong": np.nan,
                    "p_mcnemar": np.nan,
                    "higher_is_better": obj.higher_is_better,
                    "n": len(y),
                    "clustered": clustered,
                }
                if m == "auroc" and pa.is_binary:
                    row["p_delong"] = delong_test(y, pa.y_prob[:, 1], pb.y_prob[:, 1])["p"]
                if (
                    m in ("accuracy", "sensitivity", "specificity", "balanced_accuracy")
                    and pa.is_binary
                ):
                    mask = {"sensitivity": y == 1, "specificity": y == 0}.get(
                        m, np.ones(len(y), bool)
                    )
                    if m != "balanced_accuracy":
                        row["p_mcnemar"] = mcnemar_exact(
                            (hard[a] == y)[mask], (hard[b] == y)[mask]
                        )["p"]
                pair_rows.append(row)
        pairwise = pd.DataFrame(pair_rows)
        for col in ("p_bootstrap", "p_delong", "p_mcnemar"):
            pairwise[f"{col}_adj"] = np.nan
            for m in pairwise["metric"].unique():
                sel = pairwise["metric"] == m
                pairwise.loc[sel, f"{col}_adj"] = adjust_pvalues(
                    pairwise.loc[sel, col].to_numpy(), self.correction
                )
        pairwise["significant"] = pairwise["p_bootstrap_adj"] < self.alpha
        info = {
            "models": names,
            "n_samples": len(y),
            "n_patients": int(pd.unique(groups).size) if groups is not None else None,
            "clustered": clustered,
            "n_bootstrap": self.n_bootstrap,
            "confidence": self.confidence,
            "correction": self.correction,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "thresholds": thr,
            "n_pairs": len(names) * (len(names) - 1) // 2,
        }
        if clustered:
            logger.info(
                "Patients contribute several samples: DeLong/McNemar assume independence; "
                "use the cluster-bootstrap p-values as primary evidence."
            )
        return ModelComparison(models, pairwise, info)


def load_run_predictions(
    path: str | Path, split: str = "test"
) -> tuple[Predictions, dict[str, object]]:
    """Predictions from a run dir (``predictions_<split>.csv``), a CV dir (pooled
    ``oof_predictions_seed*.csv``, first seed) or a predictions CSV; plus run metadata."""
    p = Path(path).expanduser()
    meta: dict[str, object] = {"source": str(p)}
    if p.is_dir():
        csv = p / f"predictions_{split}.csv"
        if not csv.exists():
            oof = sorted(p.glob("oof_predictions_seed*.csv"))
            if not oof:
                raise FileNotFoundError(
                    f"No predictions_{split}.csv or oof_predictions_seed*.csv in {p}"
                )
            csv = oof[0]
            meta["kind"] = "out_of_fold"
        import json

        proto = p / "protocol.json"
        if proto.exists():
            meta["protocol_fingerprint"] = json.loads(proto.read_text()).get("fingerprint")
        thr_file = p / "threshold.json"
        if thr_file.exists():  # operating point chosen on this run's validation data
            meta["threshold"] = float(json.loads(thr_file.read_text())["threshold"])
    else:
        csv = p
    frame = pd.read_csv(csv, dtype={"sample_id": str, "patient_id": str})
    return Predictions.from_frame(frame), meta
