# Evaluation, comparison and explainability

This document covers how tbmxai measures and compares models, and why.

- [Metrics](#metrics)
- [Confidence intervals](#confidence-intervals)
- [Subgroup analysis](#subgroup-analysis)
- [Missing-modality robustness](#missing-modality-robustness)
- [Cross-validation and repeated seeds](#cross-validation-and-repeated-seeds)
- [Statistical comparison of models](#statistical-comparison-of-models)
- [Explainability](#explainability)
- [Protocols: experimental consistency](#protocols-experimental-consistency)

## Metrics

| Metric | Definition |
|---|---|
| `accuracy`, `balanced_accuracy` | Fraction correct; mean per-class recall |
| `f1_micro`, `f1_macro` | Pooled and unweighted per-class F1 |
| `auroc` | Area under the ROC curve (macro one-vs-rest for multiclass) |
| `sensitivity`, `specificity`, `ppv`, `npv` | Binary: computed for the positive class (index 1). Multiclass: macro one-vs-rest. |
| `brier` | Binary: `mean((p₁ − y)²)`. Multiclass: mean squared error over all classes |
| `ece` | Expected calibration error: top label, 15 equal-width bins |
| `spec_at_sens` | Binary: highest specificity with sensitivity ≥ target (default 0.90), read off the ROC curve. Reported as `spec_at_sens90`. |
| `sens_at_spec` | Binary: highest sensitivity with specificity ≥ target (default 0.70). Reported as `sens_at_spec70`. |

- Undefined values (e.g. AUROC when only one class is present, or the operating-point
  metrics on a multiclass task) are reported as `NaN`.
- Binary hard predictions use `evaluation.threshold`, or the threshold chosen on validation
  (`threshold_selection`).
- The operating-point defaults follow the WHO target product profile for TB triage tests
  (sensitivity ≥ 90 %, specificity ≥ 70 %). Other targets: `spec_at_sens@0.95`,
  `sens_at_spec@0.8`.

### Fixed operating points: two different questions

- **`spec_at_sens90` / `sens_at_spec70`** pick the operating point on the *evaluated* data. They
  answer "how good is the best trade-off this model's scores allow?" and are independent of
  any threshold, but optimistic: in practice the threshold must be fixed in advance.
- **`threshold_selection: sensitivity | specificity`** fixes the threshold on *validation* to
  reach the target there, then reports `sensitivity` / `specificity` on test. This is what a
  deployed test would achieve. Under a site shift (e.g. Shenzhen → Montgomery) the target is
  often missed on test; the gap between the two numbers measures that shift.

## Post-hoc calibration

`evaluation.calibration` fits a calibrator on the **validation** predictions and applies it to
every split before the threshold is chosen. Calibrated probabilities then flow into all
metrics, subgroups, modality ablation, saved predictions and `tbmxai compare`. The fit
(parameters, validation log loss / Brier / ECE before and after) is saved to
`calibration.json` and summarised in the report.

| Method | Tasks | Parameters | Ranking (AUROC) |
|---|---|---|---|
| `temperature` | binary, multiclass | one temperature `T` dividing the logits | unchanged |
| `platt` | binary | slope and intercept on the positive-class logit | unchanged |
| `isotonic` | binary | monotone step function | can change slightly (ties); needs a few hundred validation samples |

Caveats:
- The validation set is also used for early stopping and threshold selection; the
  before/after numbers in `calibration.json` are in-sample and so optimistic. Judge calibration
  by `brier` and `ece` on test.
- All three methods are monotone, so with a validation-chosen threshold the hard test
  predictions are (almost) unchanged: calibration fixes probabilities, not decisions.
- Calibration learned on one site does not remove a probability shift at another site.
- New metrics can be registered in `METRICS`.

Any model's predictions can be evaluated with
`tbmxai evaluate predictions.csv -m auroc -m ece -g sex`. The CSV needs `y_true` and
`prob_<class>` columns; a `patient_id` column enables patient-level CIs.

## Confidence intervals

CIs are percentile bootstrap intervals (`n_bootstrap`, `confidence`).

- **The patient is the resampling unit** (cluster bootstrap): patients are drawn with
  replacement and keep all their images or encounters. When patients contribute several
  samples, per-sample CIs are far too narrow, because correlated samples would count as
  independent.
- Resampling is stratified by patient label, so AUROC stays defined in every resample.
- `bootstrap_unit: sample` switches to per-sample resampling. Every result records its
  `ci_unit`.

## Subgroup analysis

`subgroups.columns` takes any metadata column, such as sex, age, HIV status or
`modality_pattern`.

- Categorical columns are used as they are.
- Numeric columns use `bins`, or quartiles if no bins are given.
- Groups smaller than `min_size` are skipped.
- Each group gets the full metric suite with patient-level CIs. An `__overall__` row is
  always included.
- Results go to `tables/subgroups.csv` and a forest plot in the report.

## Missing-modality robustness

With `evaluation.modality_ablation: true`, a model that supports missing modalities is
evaluated on the test split once per subset of modalities: CXR only, EHR only, both, and so
on. The modalities left out are withheld for every sample. Results go to
`tables/modality_ablation.csv` and the report.

To report performance by *natural* availability instead, add `modality_pattern` to
`subgroups.columns`.

## Cross-validation and repeated seeds

```bash
tbmxai cv CONFIG --folds 5 --seed 0 --seed 1 --seed 2
```

Each fold × seed is a complete run in its own folder. The CV folder adds:

| File | Content |
|---|---|
| `cv_summary.csv` | Mean, SD, min and max of every metric across runs |
| `oof_metrics.csv` | **Out-of-fold** metrics. Each seed's K test folds are pooled, so every patient is predicted once by a model that never saw them. CIs are patient-level. |
| `oof_subgroups.csv`, `oof_predictions_seed<s>.csv` | Subgroups and predictions on the pooled out-of-fold set |
| `cv_report.md` | Summary |

- With a protocol, CV keeps the protocol's sample set and takes the fold assignment from the
  seed. Models compared on the same sample set therefore get the same folds.
- With `tuning.enabled`, each fold tunes on its own data (nested CV).

## Statistical comparison of models

Overlapping or non-overlapping CIs of two models are not a test of their difference. Use
`tbmxai compare` on models evaluated on **the same samples** (ideally under one protocol):

```bash
tbmxai compare RUN_EHR RUN_CXR RUN_MM -n ehr -n cxr -n multimodal -o comparisons/tb \
    [-m auroc -m sensitivity ...] [--n-bootstrap 2000] [--correction holm] [--alpha 0.05]
```

Every pair of models is tested on every metric:

| Test | Applies to | Description |
|---|---|---|
| **Paired cluster bootstrap** (primary) | any metric | Patients are resampled, and the *same* resample is applied to both models. Gives the CI of the difference and a two-sided p-value. Valid when patients contribute several samples. |
| **DeLong** | AUROC (binary) | Test for two correlated ROC curves (DeLong et al. 1988; fast algorithm of Sun & Xu 2014) |
| **McNemar (exact)** | accuracy, sensitivity, specificity | Exact binomial test on discordant pairs at the decision threshold. It uses positives only for sensitivity and negatives only for specificity. |

**Multiple testing.** p-values are adjusted within each metric across all model pairs
(`--correction holm` by default, or `bh` / `none`). The `significant` column uses the
adjusted bootstrap p-value at `--alpha`.

**Safeguards:**
- The command refuses runs evaluated on different samples or labels (`--allow-subset`
  compares on the intersection).
- It warns when runs don't share a protocol fingerprint.
- DeLong and McNemar assume independent samples. When patients contribute several samples
  the output flags `clustered=True`; treat the cluster bootstrap as the primary evidence in
  that case, or compare on one index sample per patient.

**Inputs** can be run folders (`predictions_test.csv`), CV folders (pooled out-of-fold
predictions) or prediction CSVs.

**Outputs:**
- `comparison_models.csv`: per-model metrics with CIs;
- `comparison_pairwise.csv`: differences, CIs, raw and adjusted p-values;
- `comparison.md`;
- `comparison_differences.png`: forest plot of the differences.

## Explainability

| Method | Models | Samples explained | Output |
|---|---|---|---|
| `grad_cam` | `neural` with an imaging encoder (CNN, hybrid, ViT, Swin; unimodal or fused) | shared explanation set | Maps `(sample, class, H, W)` in `explanations/grad_cam.npz`, plus a gallery figure. Layer: last timm `feature_info` stage; non-timm encoders name it via `saliency_layer()` (TorchXRayVision: dense `features` block; RAD-DINO: last block's `norm1`, since patch tokens get no gradient after it) |
| `score_cam` | same | shared explanation set | Gradient-free CAM. Each channel is weighted by the gain in target-class probability when only its region is visible. `max_channels` bounds the cost. |
| `integrated_gradients` | same | shared explanation set | Pixel attributions (captum), smoothed for display |
| `shap` | `logistic_regression`, `random_forest`, `xgboost` | whole evaluation split (class-stratified subsample if large) | Per-sample values per raw EHR feature for every class output; mean \|SHAP\| per class output × true class |
| `permutation_importance` | any model with tabular input | whole evaluation split | AUROC drop per raw feature |

**How samples are chosen.** The explanation set never depends on a model.
- It is drawn from ground truth: `explain.samples.n_per_class` samples of **every** class.
- Each patient contributes at most one sample, and only samples with all modalities present
  are eligible.
- Patients are ordered by a seeded hash of their id.

Every model and modality is therefore explained on the same patients.
`explain.samples.file` pins an explicit list; a protocol pins it across experiments. The
chosen samples are written to `xai_samples.csv`.

**Every class output is explained** (`explain.target_classes: all`). Saliency galleries show
one column per class, with rows grouped by true class. The predicted outcome (TP/FN/FP/TN) is
only an annotation.

**Fusion models.** Saliency runs through the full network with the patient's other
modalities held fixed. The maps therefore show which image regions drove *this* prediction
given this patient's EHR.

## Protocols: experimental consistency

Models are only comparable when they are trained, tested, analysed by subgroup and explained
on the same samples. Different configs can produce different sample sets. For example, a
CXR experiment drops patients without an image, and an EHR experiment may predict per study
rather than per image. A protocol fixes this:

```bash
tbmxai protocol ehr.yaml cxr.yaml multimodal.yaml --out protocols/tb
```

1. Every config is ingested and harmonised, and samples are matched by stable sample id.
2. Only samples present in **every** config are kept. Labels must agree, or the build fails.
3. The reference config's splitter assigns **patients** to train/val/test, with a leakage
   check.
4. The explanation set is chosen from the test split as described above, among samples that
   are complete in every config.

The protocol directory contains:
- `samples.csv`: `sample_id, patient_id, split, y, xai, complete_in_all`;
- `protocol.json`: counts and a SHA-256 fingerprint.

Experiments use it via `protocol.path` (or `TBMXAI_PROTOCOL`). They then use exactly these
samples, splits and explanation samples, and fail if any protocol sample is missing. Each
run records the protocol fingerprint, and `tbmxai compare` checks it.

Without a protocol, a single run still uses one deterministic split and one explanation set
for all of its own analyses.
