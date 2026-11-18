# MIMIC-IV + MIMIC-CXR development dataset

This guide builds a **~20,000-patient subset** of MIMIC-IV, MIMIC-IV-ED, MIMIC-CXR-JPG and the
MIMIC-CXR reports, then trains EHR-only, CXR-only and multimodal models on it.

Every step is a short script in `scripts/mimic/`. Settings live in **one file** you edit once,
so you never need to copy long commands.

## 0. Prerequisites

1. A **credentialed PhysioNet account** that has signed the data use agreements for:
   [MIMIC-IV](https://physionet.org/content/mimiciv/),
   [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/),
   [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) and
   [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) (for the reports).
2. `wget`, `uv`, and about **50 GB** of free disk space: roughly 10 GB of tables and 20–30 GB of
   images for 5k patients.
3. Open `scripts/mimic/env.sh` in an editor and set:
   - `PHYSIONET_USER`: your PhysioNet username
   - `MIMIC_DIR`: where the raw downloads go (default `~/data/physionet`)
   - `TBMXAI_MIMIC_OUT`: where the built subset goes (default `~/data/tbmxai/mimic_subset`)
   - `TBMXAI_OUTPUT_ROOT`: where experiment runs go (default `~/tbmxai_runs`)
   - optionally `MIMIC_N_PATIENTS` (default 20000) and the dataset versions
   - `MIMIC_MAX_SIDE` (default 1024): each CXR is shrunk to this longest side right after it
     is downloaded (chunks of `MIMIC_DOWNLOAD_CHUNK` = 2000 images), so ~82k images take
     ~20 GB instead of ~150 GB. Set it empty to keep the originals.

None of these locations need to be inside the repository.

Optional: store your PhysioNet login once so download scripts don't ask for a password
(saved in `~/.netrc`, readable only by you; delete the `physionet.org` line to revoke):

```bash
bash scripts/mimic/setup_login.sh
```

## 1. Dry run (optional, no credentials, ~1 minute)

```bash
bash scripts/mimic/00_dry_run.sh
```

This generates a *fake* MIMIC tree with the real file layout and runs every step on it.

## 2. Download the tables (~10 GB; asks for your PhysioNet password once)

```bash
bash scripts/mimic/01_download_tables.sh
bash scripts/mimic/status.sh          # shows which tables are present
```

| Dataset | Files |
|---|---|
| MIMIC-IV `hosp/` | patients, admissions, diagnoses_icd, labevents, d_labitems, omr, prescriptions, microbiologyevents, transfers, services, procedures_icd |
| MIMIC-IV `icu/` | icustays, chartevents, d_items |
| MIMIC-IV-ED `ed/` | edstays, triage, vitalsign, diagnosis |
| MIMIC-CXR-JPG | mimic-cxr-2.0.0-{metadata, split, chexpert, negbio}.csv.gz |
| MIMIC-CXR | mimic-cxr-reports.zip |

Downloads are resumable: re-run the script if it is interrupted.

## 3. Select the patient subset

`02_select.sh` needs only 8 small files (under ~100 MB): `patients`, `admissions`,
`diagnoses_icd`, ED `edstays` + `diagnosis`, and the CXR-JPG `metadata`, `split` and
`chexpert` CSVs. To run selection while the large tables are still downloading, fetch just
these in a second terminal (both scripts skip files that are already complete):

```bash
bash scripts/mimic/01a_download_select_inputs.sh
```


```bash
bash scripts/mimic/02_select.sh
```

What it does:
- keeps frontal CXRs (PA/AP);
- links each study to the hospital admission or ED stay during which it was taken;
- derives all labels;
- samples patients (seeded): every patient positive for the rare targets (`tb_episode`, `tb`,
  `hiv_pneumonia`)
  is included first, then the rest is filled randomly.

It writes `selection/image_urls.txt`, which lists only the subset's JPGs.

Enrichment raises prevalence compared with the source population. Both prevalences are recorded
in `selection/selection_info.json`.

## 4. Download only the selected images (~20–30 GB)

```bash
bash scripts/mimic/03_download_images.sh
```

## 5. Build tbmxai-ready tables

```bash
bash scripts/mimic/04_build.sh
```

Output in `$TBMXAI_MIMIC_OUT` is a [standard dataset](DATASETS.md): one file per
modality, all linked by `patient_id` + `study_id`. The MIMIC configs use it with
`data: {dataset: ...}`.

| File | Grain | Content |
|---|---|---|
| `dataset.yaml` | – | Label columns and classes, prediction time (`study_time`), image root |
| `labels.parquet` | study | `label_*` targets + subgroup metadata (sex, age, race, insurance, prior HIV, encounter type, official split) |
| `tabular.parquet` | study | All EHR features (see below) |
| `images.parquet` | image | `image_path` (relative to the MIMIC-CXR-JPG root), view position, `present` flag |
| `reports.parquet` | study | `report_text` (FINDINGS + IMPRESSION) and `report_full` |
| `feature_dictionary.csv` | feature | Group, type, coverage and description of every feature |
| `build_info.json` | – | Counts, label prevalence, the full build config |

## 5b. Cache the images (recommended)

```bash
bash scripts/mimic/04b_cache_images.sh
```

Decodes every selected JPG once and resizes it (shorter side 512 px) into
`$TBMXAI_IMAGE_CACHE`. This VM has only 4 CPU cores, so without the cache, decoding
full-resolution JPGs every epoch would starve the GPU.

## 5c. Fix the experimental protocol (one per target)

```bash
bash scripts/mimic/04c_protocol.sh                  # label_pneumonia
bash scripts/mimic/04c_protocol.sh label_tb
```

This fixes, for one target:
- **one sample set**: frontal images present in all three experiments;
- **one patient-level train/val/test split**;
- **one explanation set**: 8 test patients per class, complete in every modality.

The EHR-only, CXR-only and multimodal models are therefore trained, tested, analysed by
subgroup and explained on exactly the same samples. `05_train.sh` and `06_cv.sh` use the
protocol automatically when it exists.

## 6. Train and evaluate

```bash
bash scripts/mimic/05_train.sh                                  # EHR-only XGBoost, pneumonia
bash scripts/mimic/05_train.sh cxr_densenet                     # CXR-only CNN
bash scripts/mimic/05_train.sh multimodal_xattn                 # CXR + EHR, cross-attention fusion
bash scripts/mimic/05_train.sh tabular_xgb label_tb             # another target
```

Cross-validation (patient-level, out-of-fold pooled metrics with patient-level CIs):

```bash
bash scripts/mimic/06_cv.sh tabular_xgb label_pneumonia 5
```

Statistical comparison of the latest EHR-only, CXR-only and multimodal runs of a target (same
protocol test set; paired patient-cluster bootstrap, DeLong, McNemar, Holm-adjusted):

```bash
bash scripts/mimic/07_compare.sh label_pneumonia
```

Hyper-parameter search: `uv run tbmxai tune configs/mimic/tabular_xgb.yaml`
(this uses the `tuning:` section of the config).

Each run writes `report.html` with ROC, calibration, subgroup forest plots, modality ablation,
SHAP or Grad-CAM/IG galleries (one column per class), training curves and the data-quality
report. `tables/` holds the machine-readable data behind every figure.

## Targets

| Label | Definition |
|---|---|
| `label_pneumonia` | Encounter ICD-9 480–486, 487.0, 136.3 / ICD-10 J12–J18, B59 |
| `label_tb` | ICD-9 010–018 / ICD-10 A15–A19 |
| `label_tb_episode` | Active-TB episode: CXR within ±30 days of any TB-coded admission / ED stay (whole stay), an *M. tuberculosis* culture or a first pyrazinamide prescription. `<NA>` for the patient's other CXRs; works without an encounter |
| `label_pulmonary_tb` | ICD-9 010, 011, 012, 018 / ICD-10 A15, A19 |
| `label_hiv` | ICD-9 042, V08 / ICD-10 B20, Z21, O98.7 |
| `label_hiv_pneumonia` | `hiv` AND `pneumonia` in the same encounter |
| `label_chexpert_pneumonia`, `label_chexpert_consolidation` | Report-derived CheXpert labels; uncertain (−1) dropped |

ICD labels come from the diagnoses of the **encounter during which the CXR was taken**
(hospital admission; for ED-only visits, the ED diagnoses). Add or override targets under
`labels:` in `configs/mimic/build.yaml`. Expect very few TB positives in MIMIC, because it is
a Boston ICU/ED cohort; read TB metrics together with their CIs.

## EHR features (all groups enabled)

**Leakage rule:** every feature uses only information recorded **before the CXR time**. The
current encounter's discharge diagnoses are never features, because they define the labels.

| Group | Prefix | Source | Window |
|---|---|---|---|
| demographics | `demo_` | patients, admissions | static (sex, age at study, race group, language) |
| encounter | `enc_` | admissions, edstays, icustays | at study time (type, admission type/location, insurance, hours since arrival, in ICU) |
| history | `hist_` | prior admissions + diagnoses, prior ED visits | everything that ended before the study (counts, ICD-chapter flags, prior TB/HIV/pneumonia) |
| labs | `lab_` | labevents (all numeric items above a 2% coverage threshold) | 48 h: last/min/max/mean/count |
| ED triage | `triage_` | triage | most recent ED arrival within 72 h (vitals, pain, acuity, chief-complaint flags) |
| ED vitals | `edv_` | vitalsign | 24 h: last/min/max/mean/count |
| ICU charts | `icu_` | chartevents (all numeric items above 5% coverage among ICU studies) | 24 h |
| OMR | `omr_` | omr (BMI, weight, height, blood pressure, eGFR …) | previous days, ≤ 365 d |
| medications | `med_` | prescriptions | started within 72 h (top 100 drugs + groups: antitubercular, antiretroviral, PCP agents, broad antibiotics, steroids, antifungals, anticoagulants) |
| microbiology orders | `micro_` | microbiologyevents | ordered within 72 h; **orders only, never results** (incl. `micro_afb_ordered`) |
| care unit | `ward_` | transfers | care unit at CXR time, hours in it, moves so far in the encounter, ICU / ED flags |
| service | `svc_` | services | clinical service at CXR time (MED, SURG, CMED …), service changes so far |
| procedure history | `proc_` | procedures_icd | procedures from **previous** admissions only (current-admission codes are assigned at discharge) |

Microbiology orders and anti-TB drugs reflect clinical suspicion. They are legitimately known
at CXR time, but they make TB look easier to detect than in a pure CXR-triage setting. For
that setting, remove `microbiology_orders` (and optionally `medications`) from `features:` in
`build.yaml`.

Windows, aggregations and coverage thresholds can be set per group in `build.yaml`. New groups
can be added by subclassing `tbmxai.datasets.mimic.FeatureGroup` and registering it in
`FEATURE_GROUPS`.

## Splits, timepoints and CIs

- Splits are always **patient-level**. Options: `holdout`, stratified by patient label;
  **temporal**; or the official MIMIC-CXR split
  (`{name: predefined, params: {column: official_split, val_value: validate}}`).
- **Temporal validation:** MIMIC dates are shifted per patient, so calendar dates cannot be
  compared across patients. Use the de-identified `year_group` instead:
  `split: {name: temporal, params: {column: year_group, val_start: "2014 - 2016", test_start: "2017 - 2019"}}`.
  The exact group labels are in `labels.parquet`.
- **Prediction timepoint:** the CXR time (`study_time`). All features precede it.
  `view.index_policy: first` (with `index_time_column: study_time`) keeps one index CXR per
  patient.
- **CIs:** bootstrap CIs resample **patients** (cluster bootstrap), because patients contribute
  several CXRs.
- **Missing modalities:** the multimodal config keeps samples with missing modalities, trains
  with modality dropout, and reports CXR-only / EHR-only / both performance
  (`tables/modality_ablation.csv`).

## Notes

- One sample = one frontal image in **all** MIMIC configs, including the EHR-only one, so
  every model is evaluated on identical samples. EHR features are joined per study, so a study
  with PA and AP images contributes two samples. Set `selection.one_image_per_study: true` to keep one.
- Full-resolution JPGs are decoded on every epoch. An image cache is on the roadmap; until
  then, use `num_workers` ≥ 8.
- Each run's `run_info.json` records SHA-256 fingerprints of the input files, so you can verify
  which subset a result came from.

## How many TB cases MIMIC has (v3.1, CXR-JPG 2.1.0)

| TB case definition | Patients with a frontal CXR |
|---|---|
| `tb`: CXR during a TB-coded encounter | 44 |
| `tb_episode`: ±30 days of TB-coded encounter, *M. tuberculosis* culture or pyrazinamide | 65 |
| Any CXR at any time in a patient ever TB-coded (ceiling; mostly unrelated images) | 77 |

185 patients have an active-TB code, but only 77 have any chest X-ray in MIMIC-CXR. MIMIC is
therefore too small to *train* TB models; use it as an external TB test site and train on
public TB sets. Pneumonia (6,971 patients) and HIV-associated pneumonia (220) are trainable.

Leakage note for `tb_episode`: the `medications` and `microbiology_orders` feature groups can
contain the label-defining evidence (pyrazinamide, AFB culture orders) when it precedes the
CXR. Drop those groups for TB experiments with EHR features.

## Faster image download on Google Cloud

PhysioNet mirrors MIMIC-CXR-JPG in a requester-pays GCS bucket. From a GCP VM this is far
faster than HTTPS (~50 images/min over HTTPS here). One-time setup: add your Google account at
physionet.org/settings/cloud, request GCS access on the MIMIC-CXR-JPG page, and run
`gcloud auth login` with that account. Then:

```bash
GCP_BILLING_PROJECT=my-project bash scripts/mimic/03b_download_images_gcs.sh   # instead of 03_*.sh
```

It copies only the selected images (same folder layout as wget), shrinks each 2,000-image chunk
(`MIMIC_MAX_SIDE`) and is resumable. Transfer is billed to the billing project.

## One patient split for every stage (`dev_split`)

`02_select.sh` writes `selection/dev_split.csv`: a fixed, stratified patient-level split
(70 % train / 10 % val / 20 % test; `selection.dev_split` in `configs/mimic/build.yaml`;
`tbmxai mimic dev-split` creates it for an older selection). The build copies it into
`labels.csv` as `dev_split`, and every MIMIC config uses it
(`split: {name: predefined, params: {column: dev_split}}`). Supervised pretraining trains on
`train`, early-stops on `val` and never sees `test`, so downstream test patients are unseen by
every stage. (MIMIC-CXR's official split is unsuitable here: its validate/test sets hold ~1 %
of patients.) 20,000-patient selection: 14,000 / 2,000 / 4,000 patients.

**Extending a cohort.** `selection.dev_split.inherit_from` points at an earlier `dev_split.csv`: its
patients keep their assignment and only the new patients are split (same fractions and seed). The full
cohort (`configs/mimic/build_full.yaml`, `scripts/mimic/full.sh`) uses it to keep the 20,000-patient
development sample's split, so the image encoder pretrained on that split never meets its training
patients in the full cohort's test set.

## Supervised pretraining (MIMIC as the learner)

```bash
bash scripts/mimic/06_pretrain.sh                 # densenet121 and convnext_tiny
```

Trains an ImageNet-initialised backbone on every frontal image of the selection to predict the
14 CheXpert findings (multi-label; uncertain labels masked, unmentioned = negative), on the
shared `dev_split` (train; early stopping on val mean AUROC; test never used). Patients
positive for `tb_episode` are excluded so MIMIC can later serve as an external TB test site.
Needs only the selection and the images (not `04_build`). Output per backbone in
`$TBMXAI_OUTPUT_ROOT/pretrain/<name>/`: `backbone.pt` (use as `checkpoint:` of a `timm_vision`
encoder with `pretrained: false`), `metrics.json` (per-finding validation AUROC),
`history.csv`, `manifest.csv`, `last.pt` (resume with `--resume`).
