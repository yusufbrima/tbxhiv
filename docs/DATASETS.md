# Bringing your own data: the standard dataset format

Any dataset can be used in tbmxai **without writing code**. Build one table per modality,
all linkable by a key, and add a short `dataset.yaml`. Dataset adapters, such as the MIMIC
builder, write this same format.

- [Layout](#layout)
- [Keys and linking](#keys-and-linking)
- [What goes in each file](#what-goes-in-each-file)
- [dataset.yaml](#datasetyaml)
- [Image folders without a table](#image-folders-without-a-table)
- [Recipes: raw data → standard dataset without code](#recipes-raw-data--standard-dataset-without-code)
- [Pooling datasets (multi-site)](#pooling-datasets-multi-site)
- [Datasets without patient IDs](#datasets-without-patient-ids)
- [Validate before use](#validate-before-use)
- [Using a dataset in an experiment](#using-a-dataset-in-an-experiment)
- [Avoiding leakage: your responsibility](#avoiding-leakage-your-responsibility)

## Layout

```text
my_dataset/
├── dataset.yaml   # name, label columns and their classes
├── labels.csv     # patient_id, [study_id], label column(s), metadata (sex, age, site, …)
├── tabular.csv    # patient_id, [study_id], EHR / demographic features        (optional)
├── images.csv     # patient_id, [study_id], image_path, [image metadata]      (optional)
└── reports.csv    # patient_id, [study_id], report_text                       (optional)
```

- The file names are fixed. `labels.csv` is required, plus at least one modality file.
- Any combination of modalities works: EHR only, images only, reports only, or all three.
- Each file may be `.csv`, or `.parquet` for large tables (e.g. `tabular.parquet`).
- Image files can live anywhere (PNG, JPEG, TIFF or DICOM). `image_path` is relative to the
  dataset folder, or to `image_root` in `dataset.yaml`, or absolute.

Start from a template that has the right headers:

```bash
tbmxai dataset init my_dataset            # --modality tabular|images|reports to choose
```

## Keys and linking

Two key columns, with fixed names, link the files:

| Column | Required | Meaning |
|---|---|---|
| `patient_id` | in every file | The person. It is the unit of splitting: a patient is never in two splits. |
| `study_id` | optional | A finer key: study, visit or encounter. Use it when labels or features differ per study. |

Two files are joined on `patient_id` + `study_id` when both have `study_id`, otherwise on
`patient_id` alone. For example:

- **One row per patient everywhere:** use `patient_id` only.
- **Several X-rays per patient, one label per patient:** `images.csv` has several rows per
  `patient_id`; each image becomes a sample with the patient's label and EHR.
- **Several visits per patient, a label per visit:** put `study_id` in `labels.csv` and in
  the files that are per visit (`images.csv`, `tabular.csv`). A file without `study_id`
  (e.g. patient-level demographics) applies to all of the patient's visits.

Each row of the finest file you load (usually `images.csv`) becomes one **sample**. The
validator rejects links that would multiply rows (both files having several rows per key).

## What goes in each file

A column's role is decided by the file it is in. There are no lists of columns to maintain.

| File | Columns besides the keys |
|---|---|
| `labels.csv` | The label columns declared in `dataset.yaml`. **Every other column is metadata**: available for subgroups, splits and the prediction time, never a model input. |
| `tabular.csv` | **Every column is a model input.** Leave out anything that must not be an input. Optional `feature_time`: when the features were recorded (checked against the prediction time; not an input). |
| `images.csv` | `image_path`. Other columns (e.g. `view`) are image metadata. |
| `reports.csv` | `report_text`. Other columns are ignored. |

- A column can be in both `tabular.csv` and `labels.csv`, e.g. `age`: it is loaded once and
  used both as an input and for subgroups.
- Label values may be text or numbers. Integer values read back as floats (`1.0`) still
  match a class `'1'`.
- Label columns may appear only in `labels.csv`; the validator reports an error otherwise.

## dataset.yaml

```yaml
format_version: '2.0'
name: my_tb_cohort
description: Adults screened for TB at site X, 2019–2023
labels:                            # label columns of labels.csv
  tb_status: {classes: [negative, positive], description: Culture-confirmed TB}
  hiv:       {classes: ['0', '1']}
time_column: visit_date            # optional: prediction time in labels.csv
patient_ids: provided              # 'assumed' when patient_id is not a real patient (see below)
image_root: /data/cxr              # optional: base for relative image_path (default: this folder)
```

- **Classes** are listed in encoding order, with the positive class **last** for binary
  targets.
- `time_column` enables temporal splits, index-encounter selection and the `feature_time`
  check.
- `~` and `${ENV_VAR}` are expanded in `image_root`.

## Datasets without patient IDs

Some public collections give no patient information. They can still be used:

```bash
tbmxai dataset index-images /raw/tb_kaggle tb_kaggle --patient-id-from none --label-from-folder label
```

This writes `images.csv`, `labels.csv` and a `dataset.yaml` with `patient_ids: assumed`.
Each image is then treated as its own patient, which means **patient-level splitting cannot
be guaranteed**: the same person may appear in both training and test data. tbmxai never
hides this:
- the validator reports it;
- the cohort is flagged (`patient_ids_assumed`);
- the data-quality report, `report.md` and `report.html` all carry a warning.

The validator also looks for near-identical images belonging to different "patients"
(possible duplicates).

## Validate before use

```bash
tbmxai dataset validate my_dataset                  # decodes a random sample of 500 images
tbmxai dataset validate my_dataset --check-images 0 # decode every image
```

| Checks | Level |
|---|---|
| `dataset.yaml` schema; `labels.csv` and one modality file present and readable | error |
| `patient_id` in every file; `image_path` / `report_text` present | error |
| Duplicate keys in `labels.csv` or `tabular.csv`; links that would multiply rows | error |
| Label values match the declared classes; label columns outside `labels.csv` | error |
| Features recorded after the prediction time (`feature_time` > `time_column`) | error |
| Image files missing, or sampled images that cannot be decoded | error / warning |
| Rows that don't link to a `labels.csv` row | warning |
| Features that alone separate a label almost perfectly (possible leakage) | warning |
| ID-like or entirely empty features; tiny classes | warning |
| Near-identical images belonging to different patients | warning |
| Report-derived labels combined with report text; patient IDs assumed | warning |

The command exits with an error code if any errors are found, so it can gate a pipeline.

## Using a dataset in an experiment

```yaml
data:
  dataset: /data/my_tb_cohort        # folder with dataset.yaml
  label_column: tb_status            # optional if the dataset has a single label
  use: [tabular, images]             # optional: which files to load (default: all present)
view:
  modalities: [imaging, tabular]
  metadata_columns: [sex, age, site]
```

`use` controls the sample unit: loading `images` gives one sample per image, even for an
EHR-only model. That keeps EHR, imaging and multimodal experiments on identical samples.
Each run fingerprints `dataset.yaml` and every data file in `run_info.json`.

## Avoiding leakage: your responsibility

The format can carry times and the validator flags obvious problems, but only you can
decide what was **known at the moment of prediction**:

- `tabular.csv` must contain only data recorded **before** the prediction time: no
  discharge diagnoses, results that came back later, or treatment given because of the
  outcome.
- Labels derived from a report (e.g. NLP labels) must not be predicted from that same
  report text.
- If one patient has several samples, give them the same `patient_id`. Otherwise a patient
  can land in both training and test.
