# Outputs and reproducibility

- [Run folder](#run-folder)
- [Tables](#tables)
- [Loading results](#loading-results)
- [Reproducibility checklist](#reproducibility-checklist)

## Run folder

Each run writes `<output_root>/<name>/<timestamp>/` (or the folder given with `-o`):

```text
report.html                  self-contained report with embedded figures
report.md                    Markdown summary
metrics.json                 config, cohort, metrics with CIs, subgroups (nested JSON)
tables/                      tidy long-format tables (CSV and/or Parquet), see below
figures/                     PNG figures: ROC, calibration, subgroups, modality ablation,
                             feature importance, saliency galleries, training, tuning
predictions_{val,test}.csv   sample_id, patient_id, y_true, prob_<class>..., metadata...
splits.csv                   sample_id, patient_id, split
xai_samples.csv              samples used for explanations (every class)
explanations/                <method>.npz + <method>_index.csv (saliency maps),
                             shap_values.csv, shap_feature_values.csv
config.yaml                  fully resolved config (re-runnable)
config_best.yaml             best config after tuning (if tuned)
tuning/                      Optuna trials, best parameters, study database (if tuned)
run_info.json                provenance: versions, git commit, CUDA, hashes, command, seed
protocol.json                protocol used and its fingerprint (if any)
cohort_summary.json          cohort counts
model.pkl                    fitted model (pickle; load only trusted files)
history.csv, checkpoints/    neural models: per-epoch log, best and last checkpoints
manifest.json                every file in the folder with its size
```

## Tables

`tables/` holds every result as a tidy long-format table, ready for pandas, R, seaborn or
plotly. `tables/schema.json` describes every column.

| Table | Columns |
|---|---|
| `metrics` | experiment, split, metric, value, lower, upper, confidence, ci_unit, n, n_patients |
| `subgroups` | experiment, split, column, group, n, metric, value, lower, upper |
| `modality_ablation` | experiment, available, withheld, n, metric, value, lower, upper |
| `cohort` | experiment, split, class, n_samples, n_patients |
| `explanations` | experiment, method, modality, feature, target_class, sample_class, n, importance_mean, importance_std, mean_shap |
| `training_history` | epoch, train_loss, val_loss, val_\<metric\>..., lr, epoch_time_s |
| `tuning_trials` | experiment, metric, number, value, state, duration, param:\<path\>... |
| `data_quality_summary` / `_features` / `_groups` / `_modalities` | Data checks: missingness, flagged features, class balance, availability patterns |

Tables from several runs can be stacked (`pd.concat`) to compare models or folds.

## Loading results

```python
import seaborn as sns
import tbmxai

run = tbmxai.load_run("~/tbmxai_runs/tb_ehr_xgb/20261115-120000")
run["config"]           # ExperimentConfig
run["run_info"]         # provenance
run["tables"]           # name -> DataFrame
run["predictions"]      # split -> DataFrame
run["splits"]           # sample/patient split assignment

auroc = run["tables"]["subgroups"].query("metric == 'auroc'")
sns.pointplot(data=auroc, x="value", y="group", hue="column", linestyle="none")
```

## Reproducibility checklist

1. **Environment.** `uv sync` installs the exact versions in `uv.lock`; `run_info.json`
   records the versions actually used.
2. **Code.** Commit before running. `run_info.json` stores the git commit and whether the
   working tree had uncommitted changes.
3. **Config.** The resolved `config.yaml` is saved with each run and identified by
   `run_info.json` → `config_sha256`. Re-run with `tbmxai run <run>/config.yaml`.
4. **Data.** Input files are fingerprinted (SHA-256) in `run_info.json`. `splits.csv` fixes
   the patient partition, and a protocol fixes samples, split and explanation set across
   experiments (`protocol.json` fingerprint).
5. **Randomness.** `seed` seeds Python, NumPy and PyTorch, the splits and the bootstraps.
   For bit-exact GPU training set `training.deterministic: true` (slower, and some
   operations may still be non-deterministic).
6. **Pretrained weights.** timm and Hugging Face cache downloads. For long-term
   reproducibility, keep a local copy and point `checkpoint:` at it.
7. **Reporting.** Report point estimates **with** patient-level CIs, subgroup results, and
   statistical tests for model comparisons (`tbmxai compare`).
