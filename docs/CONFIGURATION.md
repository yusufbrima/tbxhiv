# Configuration reference

An experiment is one YAML file. The file is validated on load, so unknown keys and wrong
types are errors. The same file drives `tbmxai run` and `tbmxai.Pipeline.from_yaml(...)`.

- [Overview](#overview)
- [Paths and interpolation](#paths-and-interpolation)
- [`data`: sources and linking](#data-sources-and-linking)
- [`view`: modalities, missing data, timepoints](#view-modalities-missing-data-timepoints)
- [`split`: patient-level splitting](#split-patient-level-splitting)
- [`model`: classical and neural models](#model-classical-and-neural-models)
- [`training`: neural-network training](#training-neural-network-training)
- [`tuning`: hyperparameter optimisation](#tuning-hyperparameter-optimisation)
- [`evaluation` and `subgroups`](#evaluation-and-subgroups)
- [`explain`: explainability](#explain-explainability)
- [`protocol`: shared experimental protocol](#protocol-shared-experimental-protocol)
- [`reporting`](#reporting)

Example configs: [`configs/`](../configs) (synthetic data) and [`configs/mimic/`](../configs/mimic).

## Overview

```yaml
name: my_experiment          # used in output folder names; no slashes
seed: 42                     # seeds Python/NumPy/PyTorch, splits and bootstraps
paths: {...}                 # named roots usable as ${name}
output_dir: ${output_root}
data: {...}                  # sources and how to link them
view: {...}                  # modalities used as model input
split: {...}                 # patient-level splitter
model: {...}                 # classical or neural model
training: {...}              # neural models only
tuning: {...}                # optional Optuna search
evaluation: {...}            # metrics, CIs, modality ablation
subgroups: {...}             # subgroup analysis on the test split
explain: {...}               # explainability methods and samples
protocol: {...}              # optional shared protocol
reporting: {...}             # output writers
```

## Paths and interpolation

Data, configs and outputs can live anywhere. Any string value can reference variables:

| Syntax | Resolves to |
|---|---|
| `${NAME}` | An entry of `paths:`, otherwise an environment variable. An unresolved name is an error. |
| `${NAME:-default}` | Same as above, falling back to `default` |
| `~/...` | Home directory |

```yaml
paths:
  data_root: ${TBMXAI_DATA_ROOT:-~/data/tb}
  output_root: ${TBMXAI_OUTPUT_ROOT:-~/tbmxai_runs}
output_dir: ${output_root}
data:
  sources:
    - {name: tabular_file, params: {path: "${data_root}/ehr.csv"}}
```

- Relative paths resolve against the current working directory.
- Inside YAML flow mappings (`{...}`), quote interpolated values: `"${data_root}/x.csv"`.
- Override `paths:` entries from the CLI with `--data-root`, `--output-root` or
  `-p KEY=VALUE`, or in Python with `Pipeline.from_yaml(cfg, path_overrides={...})`.

## `data`: sources and linking

There are two ways to specify data:
- reference a dataset in the **standard format** (recommended; see
  [DATASETS.md](DATASETS.md));
- list the **sources** explicitly.

```yaml
data:
  dataset: /data/my_tb_cohort      # folder with dataset.yaml
  label_column: tb_status          # optional if the dataset has one label
  use: [tabular, images]           # optional: files to load (default: all present)
```

With `dataset`, the files, the `patient_id` / `study_id` keys and the class order come
from the dataset folder. Explicit `sources` (below) may be added on top.

```yaml
data:
  patient_column: patient_id     # canonical patient id; each source's key_column is renamed to it
  label_column: tb_status
  class_names: [non-TB, TB]      # encoding order; positive class second for binary tasks
  link_columns: [study_id]       # optional extra join keys (per-study / per-visit tables)
  join: inner                    # inner | left | outer
  sources:
    - name: tabular_file
      params: {path: "${data_root}/ehr.csv", key_column: pid}
```

### Source types

| `name` | Modality | Parameters |
|---|---|---|
| `tabular_file` | tabular | `path` (CSV/TSV/Parquet/Excel), `key_column`, `feature_columns` (default: all; `[]` = label/metadata only), `metadata_columns`, `rename`, `read_kwargs` |
| `cxr_manifest` | imaging | `path` (table listing images), `key_column`, `path_column`, `image_root` (base for relative paths), `check_exists`, `metadata_columns` |
| `image_folder` (alias `cxr_directory`) | imaging | `root`, `patient_id_from` (`filename` / `path` / `dicom` / `table` / `none`), `id_pattern`, `study_pattern`, `metadata` (+ column names), `label_from_folder`, `suffixes`. For standard datasets, use `tbmxai dataset index-images` instead ([DATASETS.md](DATASETS.md#image-folders-without-a-table)) |
| `report_file` | text | `path`, `key_column`, `text_column` |

Images may be PNG, JPEG, TIFF or DICOM. Minimal file layouts:

```text
ehr.csv           pid,age,sex,hiv_status,cough_weeks,crp,cd4,tb_status
cxr_manifest.csv  pid,path,view            # path relative to image_root, or absolute
reports.csv       pid,report
```

### Linking rules

- Sources are joined on the patient column plus the `link_columns` that both tables contain.
- A join where **both** tables have duplicate keys (many-to-many) is rejected. Add a link
  column or aggregate one table first.
- The label column may sit in any source. Link columns are never model inputs.
- Each sample gets a stable id built from the patient, the link columns and the image file
  name. The id is identical across experiments that share the underlying samples.

## `view`: modalities, missing data, timepoints

```yaml
view:
  modalities: [imaging, tabular]   # any of: tabular, imaging, text
  tabular_columns: null            # restrict tabular features (default: all)
  tabular_exclude: []              # glob patterns to leave out, e.g. [micro_*, med_*] (ablations)
  metadata_columns: [sex, age]     # carried along for subgroups / splits (not inputs)
  missing_modalities: drop         # drop | keep
  index_policy: all                # all | first | last | random
  index_time_column: null          # orders a patient's samples (needed for first / last)
```

### Missing modalities

tbmxai distinguishes four situations:

| Situation | Setting |
|---|---|
| **Modality selection** (experiment design) | `view.modalities` |
| **Naturally missing modalities** (e.g. EHR but no CXR) | Join with `data.join: left` or `outer`, then `view.missing_modalities: drop` (complete cases) or `keep`. With `keep`, each sample has an availability mask and a `modality_pattern` metadata value such as `imaging+tabular`. Samples with none of the selected modalities are always dropped. |
| **Modality dropout during training** | `model.params.modality_dropout: {imaging: 0.3, tabular: 0.2}` (neural models); at least one available modality is always kept |
| **Inference with missing modalities** | Neural models replace missing inputs with learned "missing" embeddings. Fusion re-weights over the available modalities. Measured with `evaluation.modality_ablation`. |

- Classical tabular models impute fully missing rows.
- Models without missing-modality support refuse incomplete views with an explicit error.

### Prediction timepoints

The prediction timepoint is the moment the prediction is made; inputs must only use
information available then.

- `index_policy: all` keeps every sample.
- `first`, `last` or `random` keeps one index sample per patient, ordered by
  `index_time_column`.

One sample per patient removes within-patient correlation.

## `split`: patient-level splitting

Every splitter assigns **patients**, not samples. After splitting, the pipeline verifies
that no patient appears in two splits (`PatientLeakageError`). The assignment is saved to
`splits.csv`.

| `name` | Behaviour and parameters |
|---|---|
| `holdout` | Random split stratified by patient label (max over the patient's samples). `val_size`, `test_size`, `stratify` |
| `kfold` | `StratifiedGroupKFold` over patients; fold `fold` is the test set and validation is carved from the rest. `n_splits`, `fold`, `val_size`, `stratify`. Use `tbmxai cv` to run all folds. |
| `temporal` | Patients assigned by the time of their **first** sample: earlier patients train, later ones validate and test. Explicit cut-offs (`test_start`, `val_start`) or recent fractions (`test_size`, `val_size`). `strict: true` drops training patients' samples from later periods. `column`: a metadata time column. |
| `predefined` | An existing split column. `column`, `train_value`, `val_value`, `test_value` |
| `external` | External validation: test on one site, train/validate on the others (patient-level). `column`, `test_values` (one value or a list), `val_size`, `stratify`, `internal_test_size` (share of source patients held out as an in-distribution test set, reported alongside the external site), `split_seed` (fix the split across run seeds, so seeds vary only initialisation and data order) |

```yaml
split: {name: temporal, params: {column: year_group, val_start: "2014 - 2016", test_start: "2017 - 2019"}}
```

Clinical data drift over time, so report a temporal split alongside the random one.

## `model`: classical and neural models

### Classical models

```yaml
model: {name: xgboost, params: {n_estimators: 400, max_depth: 4}}
```

The models are `logistic_regression`, `random_forest` and `xgboost`. `params` are passed to
the estimator. Preprocessing (median/mode imputation, standardisation, one-hot encoding) is
fitted on the training split inside the model.

### Neural model

```yaml
model:
  name: neural
  params:
    encoders:                      # keys = modalities; one key gives a unimodal network
      imaging: {name: timm_vision,    params: {backbone: convnext_tiny, pretrained: true}}
      tabular: {name: ft_transformer, params: {d_token: 64, n_layers: 2}}
      text:    {name: hf_text,        params: {model_name: emilyalsentzer/Bio_ClinicalBERT, freeze: true}}
    fusion: {name: cross_attention, params: {query: tabular, dim: 256}}
    head:   {hidden_dims: [128], dropout: 0.2}
    image:  {size: 320, augment: true, in_chans: 1, mean: 0.5, std: 0.25,
             cache_dir: ~/data/cxr_cache, cache_size: 512}
    modality_dropout: {imaging: 0.2, tabular: 0.2}
```

**Encoders** (`tbmxai backbones` lists the curated names):

| Encoder | Options |
|---|---|
| `timm_vision` | `backbone`: any timm model. Curated: **CNN** (`resnet18/50`, `densenet121`, `efficientnet_b0`, `convnext_tiny`), **hybrid** (`coatnet_0_rw_224`, `maxvit_tiny_tf_224`, `levit_128`), **transformer** (`vit_*`, `deit_small_patch16_224`, `swin_tiny_patch4_window7_224`). `pretrained` or scratch; `in_chans` (1 = grayscale); `image_size` for ViT/Swin/MaxViT; timm kwargs such as `drop_path_rate` |
| `mlp_tabular` | `hidden_dims`, `dropout` |
| `ft_transformer` | `d_token`, `n_layers`, `n_heads`, `dropout` (per-feature tokens for attention fusion) |
| `hf_text` | `model_name` (Hugging Face id or local path), `pretrained` or scratch (`config_overrides`), `max_length`, `pooling` (`cls` / `mean`) |

Every encoder also accepts `freeze: true` (linear probing) and `checkpoint: <path>` (your own
weights, e.g. from domain pretraining).

**Fusion operators:**

| Name | Input | Description |
|---|---|---|
| `concat` | pooled | Concatenation with optional projection (`dim`) |
| `gated` | pooled | Softmax gates per modality (`fusion.last_gates`) |
| `bilinear` | pooled, 2 modalities | Full (`rank: null`) or low-rank factorised (`rank: k`) bilinear pooling |
| `attention` | tokens | Joint Transformer over all modality tokens with a `[CLS]` token |
| `cross_attention` | tokens | The `query` modality attends to the others (`fusion.last_attention`) |
| `late` | pooled | Per-modality heads with learnable weighted averaging of logits |

**Image settings:**
- `image.size` is the model input size.
- `image.cache_dir` / `image.cache_size` enable the decode-once image cache (shorter side in
  pixels). Pre-fill it with `tbmxai cache-images CONFIG`.

## `training`: neural-network training

```yaml
training:
  epochs: 50
  batch_size: 32
  eval_batch_size: null
  num_workers: 4
  device: auto                     # auto | cpu | cuda | cuda:1 | mps
  precision: "32"                  # "32" | 16-mixed | bf16-mixed
  accumulate_grad_batches: 1
  grad_clip_norm: 1.0              # null disables
  class_weights: balanced          # balanced | null | [w0, w1, ...]
  label_smoothing: 0.0
  optimizer:
    name: adamw                    # adamw | adam | sgd
    lr: 1.0e-4
    weight_decay: 0.01             # not applied to biases / norm parameters
    encoder_lr_mult: 1.0           # e.g. 0.1 for pretrained encoders
  scheduler:
    name: cosine                   # cosine | linear | step | plateau | constant
    warmup_epochs: 2               # linear warm-up; fractional epochs allowed
    warmup_start_factor: 0.01
    min_lr_ratio: 0.0              # cosine / linear floor
    step_size: 10                  # step
    gamma: 0.1                     # step / plateau
    patience: 3                    # plateau
  early_stopping: {enabled: true, monitor: val_auroc, mode: auto, patience: 10, min_delta: 0.0}
  checkpoint: {enabled: true, monitor: null, save_top_k: 1, save_last: true, restore_best: true}
  resume_from: null                # e.g. <run>/checkpoints/last.pt
  deterministic: false
```

- **Monitors:** `val_loss`, `val_auroc`, `val_accuracy`, `val_balanced_accuracy`,
  `val_f1_macro`, `val_ece`.
- **`mode: auto`** minimises loss, ECE and Brier, and maximises everything else.
- **Checkpoints** store the model, optimiser, scheduler and early-stopping state plus the
  training history, so `resume_from` continues exactly where training stopped.

## `tuning`: hyperparameter optimisation

`tbmxai run` ignores this section while `enabled: false`; `tbmxai tune CONFIG` switches it on.

```yaml
tuning:
  enabled: false
  n_trials: 40
  timeout_minutes: null
  metric: auroc                    # any registered metric, on the validation split
  direction: auto                  # inferred from the metric
  sampler: tpe                     # tpe | random | qmc | gp | cmaes  (+ sampler_params)
  pruner: median                   # none | median | hyperband | successive_halving (+ pruner_params)
  storage: null                    # default: sqlite in <run>/tuning/ (resumable); a DB URL for parallel workers
  n_jobs: 1
  search_space:                    # dotted paths into this config
    model.params.max_depth:   {type: int, low: 2, high: 8}
    training.optimizer.lr:    {type: float, low: 1.0e-5, high: 1.0e-3, log: true}
    model.params.fusion.name: {type: categorical, choices: [concat, gated, attention]}
```

How tuning runs:
1. Trials train on **train** and are scored on **validation**. Neural trials report every
   epoch, so weak trials are pruned early.
2. The best configuration is refitted and evaluated on **test** once.

Results go to `tuning/` (`trials.csv`, `best_params.json`, `best_config.yaml`, `optuna.db`)
and to `tables/tuning_trials.csv`. Resume a study with `tbmxai tune CONFIG -o <same run dir>`.
Inside `tbmxai cv`, each fold tunes separately (nested cross-validation).

## `evaluation` and `subgroups`

```yaml
evaluation:
  metrics: null                    # null = full suite, or a list of metric names;
                                   # operating-point metrics take a target: spec_at_sens@0.95
  threshold: 0.5                   # binary decision threshold
  threshold_selection: fixed       # fixed | youden | sensitivity | specificity: choose the threshold
                                   # on the validation predictions, apply it to test (saved to
                                   # threshold.json; `tbmxai compare` uses each run's own threshold)
  target_sensitivity: 0.9          # for threshold_selection: sensitivity
  target_specificity: 0.7          # for threshold_selection: specificity
  calibration: none                # none | temperature | platt | isotonic: post-hoc calibration
                                   # fitted on validation, applied to every split before the
                                   # threshold is chosen (saved to calibration.json)
  n_bootstrap: 1000                # 0 disables CIs
  confidence: 0.95
  bootstrap_unit: patient          # patient (cluster bootstrap) | sample
  modality_ablation: false         # also evaluate test with each modality subset withheld
  splits: [val, test]

subgroups:
  columns: [sex, age, hiv_status, modality_pattern]
  min_size: 20                     # smaller groups are skipped
  bins: {age: [0, 30, 45, 120]}    # numeric columns without bins use quartiles
```

Definitions and methods are in [EVALUATION.md](EVALUATION.md).

## `explain`: explainability

```yaml
explain:
  methods:
    - {name: grad_cam}
    - {name: score_cam, params: {max_channels: 256}}
    - {name: integrated_gradients, params: {n_steps: 32}}
    - {name: shap}
    - {name: permutation_importance, params: {n_repeats: 5}}
  samples:
    split: test
    n_per_class: 8                 # explanation samples per class (every class included)
    seed: null                     # default: experiment seed
    file: null                     # CSV with sample_id to pin the samples explicitly
  target_classes: all              # explain every class output | positive
```

See [EVALUATION.md](EVALUATION.md#explainability) for how samples are chosen and what each
method produces.

## `protocol`: shared experimental protocol

```yaml
protocol:
  path: "${TBMXAI_PROTOCOL:-}"     # directory written by `tbmxai protocol`; empty = none
  use_split: true                  # use the protocol's train/val/test assignment
  use_xai_samples: true            # use the protocol's explanation samples
```

When set, the run restricts itself to the protocol's samples, using its split and its
explanation set. It fails if any protocol sample is missing. See
[EVALUATION.md](EVALUATION.md#protocols-experimental-consistency).

## `reporting`

```yaml
reporting:
  writers: [json, tables, markdown, html]
  table_format: csv                # csv | parquet | both
  save_predictions: true
```

The files written are described in [OUTPUTS.md](OUTPUTS.md).
