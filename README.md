# When does the chest X-ray add to the EHR? Leakage-safe multimodal pneumonia prediction

Code and configurations for the paper. Everything in its tables and figures is regenerated from
credentialed MIMIC data by the scripts below. No patient-level data are included in this repository,
and the manuscript is not part of it.

The study asks where a presentation chest radiograph adds to a strong EHR model for community-onset
pneumonia: one index radiograph per encounter from every eligible MIMIC-IV patient linked to
MIMIC-CXR, a label based on the principal diagnosis and antibiotic treatment, EHR features limited
to what was available when each radiograph was taken, and a patient-level split whose test patients
were never seen by image pretraining. It contrasts a tuned XGBoost EHR model, a MIMIC-pretrained ConvNeXt-T image model, an
end-to-end fusion network and a stacked model, with pre-specified interaction tests, a workup-feature
ablation and an HIV-positive subgroup analysis.

## Install

```bash
uv sync                                    # exact environment from uv.lock (Python 3.12)
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run pytest --basetemp ~/.cache/tbmxai-pytest
```

The `tbmxai` package (`src/`) is a configuration-driven framework: every model, splitter, metric and
explainer is a registered component named in a YAML config (`docs/CONFIGURATION.md`,
`docs/EVALUATION.md`).

## Data (not redistributed)

MIMIC-IV, MIMIC-IV-ED, MIMIC-CXR-JPG and MIMIC-CXR (reports) from PhysioNet, credentialed access.
Set your PhysioNet username and data locations in `scripts/mimic/env.sh`; `docs/MIMIC.md` describes
every step and the EHR feature groups. Data default to `~/data`, runs to `~/tbmxai_runs`.

## Pipeline

| Step | Command |
|---|---|
| 1. Download the tables | `scripts/mimic/01_download_tables.sh` |
| 2. Development sample (20,000 patients; fixes the patient split) | `scripts/mimic/02_select.sh` → `03b_download_images_gcs.sh` (or `03_download_images.sh`) → `04_build.sh` |
| 3. Image pretraining on CheXpert findings (ConvNeXt-T) | `scripts/mimic/06_pretrain.sh` (config `configs/pretrain/`) |
| 4. Full cohort, inheriting the sample's split | `scripts/mimic/full.sh` (config `configs/mimic/build_full.yaml`) |
| 5. Index-radiograph cohort and label | `uv run python scripts/cohort/build_index_cohort.py ~/data/tbmxai/mimic_full ~/data/tbmxai/mimic_full_index` |
| 6. EHR features available at film time | `uv run python scripts/cohort/build_available_features.py ~/data/tbmxai/mimic_full_index ~/data/tbmxai/mimic_full ~/data/tbmxai/mimic_full_index_available` |
| 7. Image cache (recommended) | `scripts/mimic/04b_cache_images.sh` |
| 8. EHR search (30 trials, validation patients), then the models with and without workup features | `scripts/experiments/train_ehr.sh` |
| 9. Image model, fusion, fusion without workup (3 seeds each) | `scripts/experiments/train_cxr_fusion.sh` |
| 10. Main analysis | `uv run python scripts/analysis/main_analysis.py ~/tbmxai_runs/analysis/pneumonia ~/data/tbmxai/mimic_full_index_available ~/tbmxai_runs/pneumonia/{ehr_xgb,ehr_xgb_noworkup,cxr,fusion,fusion_noworkup}` |

Model settings are in `configs/pneumonia/` (EHR XGBoost, image model, fusion; the `_noworkup`
variants drop the pre-specified workup features `micro_*`, `med_*`, `lab_*_count`, `svc_*`).
`scripts/tmux_job.sh NAME CMD...` runs a long step in a detached tmux session with a log file.

## Analyses → paper

| Paper | Source |
|---|---|
| Performance, paired differences, strata and interactions, HIV contrasts B1–B3 | `scripts/analysis/main_analysis.py` (`overall.csv`, `comparisons.csv`, `strata.csv`, `interactions.csv`, `hiv.csv`, `stacker.csv`, `calibration.csv`) |
| Tables (LaTeX) | `scripts/reporting/make_tables.py OUT_DIR` |
| Calibration and decision-curve figure | `scripts/reporting/figure_input.py ANALYSIS_DIR`, then `scripts/reporting/calibration_dca.py` on its output (`DCA_TMIN=0.01 DCA_TMAX=0.25`) |

## Notes

- A laboratory or intensive-care value is used only if it was charted and its result stored before
  the radiograph (`storetime`, or `charttime` when that is missing). Admission attributes are used
  only once the patient has been admitted, and race and language come from earlier admissions.
- The image model uses no EHR features. The paper's image-model runs were trained on the first
  build of the cohort (step 5), which has the same radiographs, labels and split as step 6.
- Models are fitted on training patients; stacking, recalibration and every threshold are fitted on
  validation patients; test patients are used only for evaluation. Bootstrap intervals resample
  patients, and the stacker is refitted inside every resample.
- The earlier composite target "HIV and pneumonia" can be reproduced with
  `TARGET=label_hiv_pneumonia uv run tbmxai run configs/mimic/tabular_xgb.yaml` on the development
  sample.

## License

Apache License 2.0 (`LICENSE`).
