"""Command-line interface (``tbmxai --help``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

import tbmxai
from tbmxai.core.config import ExperimentConfig
from tbmxai.core.registry import Registry
from tbmxai.core.types import Predictions
from tbmxai.core.utils import configure_logging

app = typer.Typer(
    name="tbmxai",
    help="Modular multimodal TB detection: ingest, harmonise, train, evaluate, explain, report.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Experiment YAML.")]
DataRootOpt = Annotated[
    str | None,
    typer.Option("--data-root", help="Override paths.data_root (data may live anywhere)."),
]
OutputRootOpt = Annotated[
    str | None, typer.Option("--output-root", help="Override paths.output_root.")
]
PathOpt = Annotated[
    list[str] | None,
    typer.Option("--path", "-p", help="Override any paths: entry, KEY=VALUE (repeatable)."),
]


def _path_overrides(
    data_root: str | None, output_root: str | None, extra: list[str] | None
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in extra or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--path expects KEY=VALUE, got '{item}'")
        overrides[key.strip()] = value.strip()
    if data_root:
        overrides["data_root"] = data_root
    if output_root:
        overrides["output_root"] = output_root
    return overrides


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


@app.command()
def run(
    config: ConfigArg,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Exact run directory (overrides naming)."),
    ] = None,
    data_root: DataRootOpt = None,
    output_root: OutputRootOpt = None,
    path: PathOpt = None,
) -> None:
    """Run the full pipeline described by CONFIG."""
    overrides = _path_overrides(data_root, output_root, path)
    result = tbmxai.Pipeline.from_yaml(config, overrides).run(output_dir)
    for split, evaluation in result.evaluations.items():
        _print_metrics(evaluation.to_frame(), title=f"{split} metrics")
    console.print(f"[green]Outputs:[/green] {result.output_dir}")


@app.command()
def tune(
    config: ConfigArg,
    n_trials: Annotated[int | None, typer.Option(help="Override tuning.n_trials.")] = None,
    timeout_minutes: Annotated[
        float | None, typer.Option(help="Override tuning.timeout_minutes.")
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Exact run directory (resume a study)."),
    ] = None,
    data_root: DataRootOpt = None,
    output_root: OutputRootOpt = None,
    path: PathOpt = None,
) -> None:
    """Optimise hyper-parameters with Optuna (tuning: section), then refit and test the best."""
    cfg = ExperimentConfig.from_yaml(config, _path_overrides(data_root, output_root, path))
    updates: dict[str, object] = {"enabled": True}
    if n_trials is not None:
        updates["n_trials"] = n_trials
    if timeout_minutes is not None:
        updates["timeout_minutes"] = timeout_minutes
    cfg = cfg.model_copy(update={"tuning": cfg.tuning.model_copy(update=updates)})
    if not cfg.tuning.search_space:
        raise typer.BadParameter("The config has no tuning.search_space.")
    result = tbmxai.Pipeline(cfg).run(output_dir)
    if result.tuning_best:
        console.print_json(json.dumps(result.tuning_best, default=str))
    for split, evaluation in result.evaluations.items():
        _print_metrics(evaluation.to_frame(), title=f"{split} metrics (best params)")
    console.print(f"[green]Outputs:[/green] {result.output_dir}")


@app.command()
def cv(
    config: ConfigArg,
    folds: Annotated[int | None, typer.Option(help="Patient-level folds (omit: seeds only).")] = 5,
    seeds: Annotated[
        list[int] | None, typer.Option("--seed", help="Repeatable; default: config seed.")
    ] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
    data_root: DataRootOpt = None,
    output_root: OutputRootOpt = None,
    path: PathOpt = None,
) -> None:
    """Patient-level K-fold CV and/or repeated seeds with out-of-fold aggregation."""
    from tbmxai.experiments import RepeatedExperiment

    cfg = ExperimentConfig.from_yaml(config, _path_overrides(data_root, output_root, path))
    result = RepeatedExperiment(cfg, folds=folds or None, seeds=seeds).run(output_dir)
    if result.oof_metrics is not None and not result.oof_metrics.empty:
        _print_metrics(
            result.oof_metrics[["seed", "metric", "value", "lower", "upper"]],
            title="Out-of-fold metrics (patient-level bootstrap CI)",
        )
    _print_metrics(
        result.summary[result.summary["split"] == "test"], title="Test metrics across runs"
    )
    console.print(f"[green]Outputs:[/green] {result.output_dir}")


@app.command()
def protocol(
    configs: Annotated[
        list[Path], typer.Argument(exists=True, dir_okay=False, help="Configs to be compared (≥1).")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Protocol directory to write.")],
    reference: Annotated[int, typer.Option(help="Index of the config defining split/seed.")] = 0,
    n_per_class: Annotated[int | None, typer.Option(help="Explanation samples per class.")] = None,
    data_root: DataRootOpt = None,
    path: PathOpt = None,
) -> None:
    """Fix one sample set, patient-level split and XAI set shared by all CONFIGS."""
    from tbmxai.protocol import build_protocol

    overrides = _path_overrides(data_root, None, path)
    cfgs = [ExperimentConfig.from_yaml(c, overrides) for c in configs]
    proto = build_protocol(cfgs, reference=reference, n_per_class=n_per_class)
    proto.save(out)
    console.print_json(json.dumps(proto.info, default=str))
    console.print(
        f"[green]Protocol →[/green] {out}  (set protocol.path or TBMXAI_PROTOCOL to use it)"
    )


@app.command()
def compare(
    runs: Annotated[
        list[Path], typer.Argument(exists=True, help="Run dirs, CV dirs or prediction CSVs (≥2).")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Directory for comparison outputs.")],
    names: Annotated[
        list[str] | None,
        typer.Option("--name", "-n", help="Model names, in RUNS order (repeatable)."),
    ] = None,
    metrics: Annotated[list[str] | None, typer.Option("--metric", "-m", help="Repeatable.")] = None,
    split: Annotated[str, typer.Option(help="Which predictions_<split>.csv to use.")] = "test",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    threshold: Annotated[
        float | None,
        typer.Option(
            help="Binary threshold for every model. Default: each run's threshold.json "
            "(validation-chosen) if present, else 0.5."
        ),
    ] = None,
    correction: Annotated[str, typer.Option(help="holm | bh | none")] = "holm",
    alpha: float = 0.05,
    allow_subset: Annotated[
        bool, typer.Option(help="Compare on the common samples if sample sets differ.")
    ] = False,
    seed: int = 0,
) -> None:
    """Statistical comparison of models on the same samples (paired bootstrap, DeLong, McNemar)."""
    from tbmxai.evaluation.compare import ModelComparator, load_run_predictions

    if len(runs) < 2:
        raise typer.BadParameter("Give at least two runs.")
    labels = names or [Path(r).name if Path(r).is_file() else Path(r).parent.name for r in runs]
    if len(labels) != len(runs) or len(set(labels)) != len(labels):
        raise typer.BadParameter("Model names must be unique, one per run (use --name).")
    preds, fingerprints, run_thresholds = {}, {}, {}
    for label, run in zip(labels, runs, strict=True):
        preds[label], meta = load_run_predictions(run, split)
        fingerprints[label] = meta.get("protocol_fingerprint")
        if "threshold" in meta:
            run_thresholds[label] = meta["threshold"]
    if len({f for f in fingerprints.values()}) > 1 or None in fingerprints.values():
        console.print(
            f"[yellow]Warning:[/yellow] runs do not share one protocol fingerprint: {fingerprints}"
        )
    if threshold is None and run_thresholds:
        console.print(f"Using each run's validation-chosen threshold: {run_thresholds}")
    result = ModelComparator(
        metrics,
        n_bootstrap,
        confidence,
        0.5 if threshold is None else threshold,
        correction,
        alpha,
        seed=seed,
    ).compare(
        preds,
        allow_subset=allow_subset,
        thresholds=run_thresholds if threshold is None else None,
    )
    result.info["protocol_fingerprints"] = fingerprints
    result.write(out)
    shown = result.pairwise[
        [
            "metric",
            "model_a",
            "model_b",
            "diff",
            "diff_lower",
            "diff_upper",
            "p_bootstrap_adj",
            "p_delong_adj",
            "significant",
        ]
    ]
    _print_metrics(shown, title=f"Pairwise differences ({correction}-adjusted p-values)")
    console.print(f"[green]Comparison →[/green] {out}")


@app.command()
def ingest(
    config: ConfigArg,
    out: Annotated[
        Path | None, typer.Option("--out", help="Save the linked cohort (.parquet or .csv).")
    ] = None,
    data_root: DataRootOpt = None,
    path: PathOpt = None,
) -> None:
    """Ingest and link the data sources in CONFIG; print a cohort summary."""
    overrides = _path_overrides(data_root, None, path)
    cohort = tbmxai.Pipeline(ExperimentConfig.from_yaml(config, overrides)).ingest()
    console.print_json(json.dumps(cohort.summary()))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".parquet":
            cohort.frame.to_parquet(out, index=False)
        else:
            cohort.frame.to_csv(out, index=False)
        console.print(f"[green]Saved cohort →[/green] {out}")


@app.command()
def evaluate(
    predictions: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="CSV with y_true and prob_* columns."),
    ],
    metrics: Annotated[
        list[str] | None, typer.Option("--metric", "-m", help="Repeatable; default = full suite.")
    ] = None,
    n_bootstrap: Annotated[int, typer.Option(help="Bootstrap resamples (0 = no CI).")] = 1000,
    confidence: Annotated[float, typer.Option(help="CI level.")] = 0.95,
    threshold: Annotated[float, typer.Option(help="Binary decision threshold.")] = 0.5,
    label_column: Annotated[str, typer.Option(help="Ground-truth column.")] = "y_true",
    patient_column: Annotated[
        str, typer.Option(help="Patient id column for cluster (patient-level) bootstrap CIs.")
    ] = "patient_id",
    subgroup: Annotated[
        list[str] | None, typer.Option("--subgroup", "-g", help="Repeatable metadata column.")
    ] = None,
    min_size: Annotated[int, typer.Option(help="Minimum subgroup size.")] = 20,
    out: Annotated[Path | None, typer.Option(help="Write metrics CSV here.")] = None,
    seed: int = 0,
) -> None:
    """Evaluate an existing PREDICTIONS file (from any model) with CIs."""
    preds = Predictions.from_frame(
        pd.read_csv(predictions), label_column=label_column, group_column=patient_column
    )
    evaluator = tbmxai.Evaluator(
        metrics,
        threshold=threshold,
        bootstrap=tbmxai.BootstrapCI(n_bootstrap, confidence, seed=seed) if n_bootstrap else None,
    )
    frame = evaluator.evaluate(preds).to_frame()
    _print_metrics(frame, title=f"{predictions.name} (n={len(preds)})")
    if subgroup:
        frame = tbmxai.SubgroupAnalyzer(evaluator, subgroup, min_size=min_size).analyze(preds)
        console.print(frame.round(3).to_string(index=False))
    if out is not None:
        frame.to_csv(out, index=False)
        console.print(f"[green]Saved →[/green] {out}")


@app.command("cache-images")
def cache_images(
    config: ConfigArg,
    cache_dir: Annotated[
        Path | None, typer.Option(help="Cache root (default: model.params.image.cache_dir).")
    ] = None,
    size: Annotated[
        int | None, typer.Option(help="Shorter side in px (default: image.cache_size).")
    ] = None,
    workers: Annotated[int | None, typer.Option(help="Parallel decode processes.")] = None,
    data_root: DataRootOpt = None,
    path: PathOpt = None,
) -> None:
    """Decode + resize every image of CONFIG's cohort once into the image cache."""
    from tbmxai.data.image_cache import ImageCache
    from tbmxai.data.sources import IMAGE_PATH_COLUMN

    cfg = ExperimentConfig.from_yaml(config, _path_overrides(data_root, None, path))
    image = dict(cfg.model.params.get("image") or {})
    root = cache_dir or image.get("cache_dir")
    if not root:
        raise typer.BadParameter(
            "No cache dir: pass --cache-dir or set model.params.image.cache_dir"
        )
    side = size or int(image.get("cache_size") or max(512, int(image.get("size", 224))))
    cohort = tbmxai.Pipeline(cfg).ingest()
    if IMAGE_PATH_COLUMN not in cohort.frame:
        raise typer.BadParameter("The config has no imaging source.")
    paths = cohort.frame[IMAGE_PATH_COLUMN].dropna().astype(str).unique()
    stats = ImageCache(root, side).build(paths, workers=workers)
    console.print(f"[green]Image cache[/green] {root} (short side {side}px): {stats}")


@app.command()
def components() -> None:
    """List every registered (swappable) component."""
    table = Table(title="Registered components")
    table.add_column("kind", style="cyan")
    table.add_column("names")
    for kind, registry in sorted(Registry.all().items()):
        table.add_row(kind, ", ".join(registry.names()))
    console.print(table)


@app.command()
def backbones() -> None:
    """List curated vision backbones (by family) and text encoders."""
    from tbmxai.models.nn import TEXT_BACKBONES, VISION_BACKBONES

    table = Table(title="Encoder backbones (any timm / Hugging Face id also works)")
    table.add_column("family", style="cyan")
    table.add_column("names")
    for family, names in VISION_BACKBONES.items():
        table.add_row(f"vision/{family}", ", ".join(names))
    table.add_row("text", ", ".join(TEXT_BACKBONES))
    console.print(table)


@app.command()
def synth(
    out_dir: Annotated[Path, typer.Argument(help="Where to write the synthetic dataset.")],
    n_patients: int = 300,
    image_size: int = 64,
    seed: int = 0,
) -> None:
    """Generate a synthetic multimodal dataset (EHR + CXR + reports) for smoke tests."""
    from tbmxai.data.synthetic import make_synthetic_dataset

    paths = make_synthetic_dataset(
        out_dir.expanduser(), n_patients=n_patients, image_size=image_size, seed=seed
    )
    console.print(f"[green]Synthetic dataset →[/green] {paths.root}")


@app.command("pretrain")
def pretrain(
    config: Annotated[Path, typer.Argument(exists=True, help="Pretraining config (YAML).")],
    resume: Annotated[bool, typer.Option(help="Continue from last.pt if present.")] = False,
    cache_only: Annotated[
        bool, typer.Option(help="Only build the image cache for the manifest, then exit.")
    ] = False,
    path: PathOpt = None,
) -> None:
    """Supervised multi-label CXR pretraining; exports backbone.pt for `checkpoint:`."""
    from tbmxai.data.image_cache import ImageCache
    from tbmxai.pretrain import PretrainConfig, Pretrainer, build_manifest

    cfg = PretrainConfig.from_yaml(config, _path_overrides(None, None, path))
    if cache_only:
        if cfg.image.cache_dir is None:
            raise typer.BadParameter("image.cache_dir is not set in the config")
        manifest = build_manifest(cfg.data)
        stats = ImageCache(cfg.image.cache_dir, cfg.image.cache_size).build(manifest["image_path"])
        console.print(f"Image cache: {stats}")
        return
    summary = Pretrainer(cfg).run(resume=resume)
    console.print_json(json.dumps(summary, default=str))


# --------------------------------------------------------------- datasets

dataset_app = typer.Typer(
    help="Standard dataset format: template, image indexing, validation.",
    no_args_is_help=True,
)
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("init")
def dataset_init(
    folder: Annotated[
        Path, typer.Argument(help="Dataset folder to create (existing files are kept).")
    ],
    name: Annotated[str | None, typer.Option(help="Dataset name (default: folder name).")] = None,
    modality: Annotated[
        list[str] | None,
        typer.Option("--modality", "-m", help="tabular | images | reports (repeatable)."),
    ] = None,
) -> None:
    """Write a template dataset.yaml and example file headers."""
    from tbmxai.data.dataset_format import init_dataset

    written = init_dataset(
        folder, name=name, modalities=modality or ("tabular", "images", "reports")
    )
    for p in written:
        console.print(f"  wrote {p}")
    console.print(
        f"[green]Template ready.[/green] Fill the files, edit dataset.yaml, then: "
        f"tbmxai dataset validate {folder}"
    )


@dataset_app.command("index-images")
def dataset_index_images(
    image_dir: Annotated[
        Path, typer.Argument(exists=True, help="Image folder (scanned recursively).")
    ],
    dataset: Annotated[Path, typer.Argument(help="Dataset folder to write images.csv into.")],
    patient_id_from: Annotated[
        str, typer.Option(help="filename | path | dicom | table | none")
    ] = "filename",
    id_pattern: Annotated[
        str, typer.Option(help="Regex with group 'id' (filename / path).")
    ] = r"(?P<id>.+)",
    study_pattern: Annotated[
        str | None, typer.Option(help="Regex with group 'id' on the relative path → study_id.")
    ] = None,
    lookup: Annotated[
        Path | None,
        typer.Option(help="Table mapping image file → patient (patient-id-from table)."),
    ] = None,
    lookup_image_column: Annotated[
        str, typer.Option(help="Image file/stem column in --lookup.")
    ] = "image",
    lookup_patient_column: Annotated[
        str, typer.Option(help="Patient column in --lookup.")
    ] = "patient_id",
    label_from_folder: Annotated[
        str | None,
        typer.Option(
            help="Label column name; the parent folder (Normal/, TB/) is the label → labels.csv."
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace existing files.")] = False,
) -> None:
    """Scan an image folder and write images.csv (patient_id, [study_id], image_path)."""
    from tbmxai.data.dataset_format import index_images

    written = index_images(
        image_dir,
        dataset,
        patient_id_from=patient_id_from,
        id_pattern=id_pattern,
        study_pattern=study_pattern,
        lookup=lookup,
        lookup_image_column=lookup_image_column,
        lookup_patient_column=lookup_patient_column,
        label_from_folder=label_from_folder,
        overwrite=overwrite,
    )
    for p in written.values():
        console.print(f"  wrote {p}")
    console.print(f"Next: tbmxai dataset validate {dataset}")


@dataset_app.command("nlm-tb")
def dataset_nlm_tb(
    raw_dir: Annotated[
        Path, typer.Argument(exists=True, help="Folder with the downloaded NLM zip files.")
    ],
    out_dir: Annotated[Path, typer.Argument(help="Standard dataset folder to write.")],
) -> None:
    """Convert the Shenzhen + Montgomery TB sets (NLM) into a standard dataset."""
    from tbmxai.datasets.nlm_tb import build_nlm_tb

    info = build_nlm_tb(raw_dir, out_dir)
    console.print_json(json.dumps(info, default=str))
    console.print(f"Next: tbmxai dataset validate {out_dir}")


@dataset_app.command("build")
def dataset_build(
    source: Annotated[
        str, typer.Argument(help="Recipe file (*.yaml) or adapter name (see `dataset adapters`).")
    ],
    raw_dir: Annotated[Path, typer.Argument(help="Folder with the raw download.")],
    out_dir: Annotated[Path, typer.Argument(help="Standard dataset folder to write.")],
    param: Annotated[
        list[str] | None,
        typer.Option("--param", "-p", help="Adapter argument KEY=VALUE (repeatable)."),
    ] = None,
    validate: Annotated[bool, typer.Option(help="Validate the result.")] = True,
    check_images: Annotated[
        int, typer.Option(help="Images to decode during validation (0 = all).")
    ] = 500,
) -> None:
    """Build a standard dataset from raw data with a recipe or a registered adapter."""
    from tbmxai.data.dataset_format import validate_dataset
    from tbmxai.datasets.adapters import resolve_adapter

    params = _path_overrides(None, None, param)
    adapter = resolve_adapter(source, **params)
    info = adapter.build(raw_dir, out_dir)
    console.print_json(json.dumps(info, default=str))
    if validate:
        report = validate_dataset(out_dir, check_images=None if check_images == 0 else check_images)
        report.to_frame().to_csv(Path(out_dir) / "validation.csv", index=False)
        for issue in report.errors + report.warnings:
            console.print(f"  [{issue['level']}] {issue['check']}: {issue['message']}")
        if not report.ok:
            raise typer.Exit(1)
        console.print(f"[green]Valid.[/green] Use it with `data: {{dataset: {out_dir}}}`.")


@dataset_app.command("merge")
def dataset_merge(
    out_dir: Annotated[Path, typer.Argument(help="Merged dataset folder to write.")],
    sources: Annotated[list[Path], typer.Argument(help="Dataset folders to pool (2 or more).")],
    label: Annotated[
        list[str],
        typer.Option("--label", "-l", help="LABEL=class1,class2 (positive last; repeatable)."),
    ],
    rename: Annotated[
        list[str] | None,
        typer.Option("--rename", "-r", help="LABEL:old=new value harmonisation (repeatable)."),
    ] = None,
    name: Annotated[str | None, typer.Option(help="Merged dataset name.")] = None,
    validate: Annotated[bool, typer.Option(help="Validate the result.")] = True,
    check_images: Annotated[
        int, typer.Option(help="Images to fingerprint for cross-dataset duplicates (0 = all).")
    ] = 0,
) -> None:
    """Pool datasets into one with a `source` column (multi-site / external validation)."""
    from tbmxai.data.dataset_format import validate_dataset
    from tbmxai.datasets.merge import merge_datasets

    labels: dict[str, list[str]] = {}
    for item in label:
        key, sep, classes = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--label expects LABEL=a,b, got '{item}'")
        labels[key.strip()] = [c.strip() for c in classes.split(",") if c.strip()]
    renames: dict[str, dict[str, str]] = {}
    for item in rename or []:
        col, sep, pair = item.partition(":")
        old, sep2, new = pair.partition("=")
        if not (sep and sep2):
            raise typer.BadParameter(f"--rename expects LABEL:old=new, got '{item}'")
        renames.setdefault(col.strip(), {})[old.strip()] = new.strip()
    info = merge_datasets(sources, out_dir, labels, renames, name)
    console.print_json(json.dumps(info, default=str))
    if validate:
        report = validate_dataset(out_dir, check_images=None if check_images == 0 else check_images)
        report.to_frame().to_csv(Path(out_dir) / "validation.csv", index=False)
        for issue in report.errors + report.warnings:
            console.print(f"  [{issue['level']}] {issue['check']}: {issue['message']}")
        if not report.ok:
            raise typer.Exit(1)
        console.print(f"[green]Valid.[/green] Use it with `data: {{dataset: {out_dir}}}`.")


@dataset_app.command("adapters")
def dataset_adapters() -> None:
    """List registered dataset adapters and recipe parsers."""
    from tbmxai.datasets.adapters import ADAPTERS, PARSERS

    table = Table(title="Dataset adapters (tbmxai dataset build NAME RAW OUT)")
    table.add_column("name")
    table.add_column("description")
    for name in ADAPTERS.names():
        table.add_row(name, ADAPTERS.get(name).description)
    console.print(table)
    console.print("Recipe parsers (parse: NAME): " + ", ".join(sorted(PARSERS)))
    console.print("Recipes: any recipe.yaml, e.g. configs/datasets/*.yaml")


@dataset_app.command("validate")
def dataset_validate(
    folder: Annotated[
        Path, typer.Argument(exists=True, help="Dataset folder (with dataset.yaml).")
    ],
    check_images: Annotated[
        int, typer.Option(help="Images to decode and fingerprint (random sample; 0 = all).")
    ] = 500,
    out: Annotated[Path | None, typer.Option(help="Write the issues as CSV here.")] = None,
) -> None:
    """Check a dataset folder: ids, links, labels, features, images, leakage hints."""
    from tbmxai.data.dataset_format import validate_dataset

    report = validate_dataset(folder, check_images=None if check_images == 0 else check_images)
    style = {"error": "red", "warning": "yellow", "info": "cyan"}
    table = Table(title=f"Dataset validation: {folder}")
    for col in ("level", "check", "message"):
        table.add_column(col)
    for issue in report.issues:
        table.add_row(
            f"[{style[issue['level']]}]{issue['level']}[/]", issue["check"], issue["message"]
        )
    if report.issues:
        console.print(table)
    console.print_json(json.dumps(report.info, default=str))
    if out is not None:
        report.to_frame().to_csv(out, index=False)
    if report.ok:
        console.print(
            f"[green]Valid[/green] ({len(report.warnings)} warnings). Use it with "
            f"`data: {{dataset: {folder}, label_column: ...}}`."
        )
    else:
        console.print(f"[red]{len(report.errors)} errors[/red] — fix them before use.")
        raise typer.Exit(code=1)


# ------------------------------------------------------------------ MIMIC

mimic_app = typer.Typer(
    help="MIMIC-IV + MIMIC-CXR adapter: status → select → (download images) → build.",
    no_args_is_help=True,
)
app.add_typer(mimic_app, name="mimic")
MimicConfigArg = Annotated[
    Path,
    typer.Argument(
        exists=True, dir_okay=False, help="MIMIC build YAML (configs/mimic/build.yaml)."
    ),
]


def _mimic_builder(config: Path, memory_limit: str | None):  # type: ignore[no-untyped-def]
    from tbmxai.datasets.mimic import MimicBuildConfig, MimicBuilder

    return MimicBuilder(MimicBuildConfig.from_yaml(config), memory_limit=memory_limit)


@mimic_app.command("status")
def mimic_status(config: MimicConfigArg) -> None:
    """Show which raw MIMIC tables are present and selection/build progress."""
    builder = _mimic_builder(config, None)
    status = builder.tables.status()
    table = Table(title="Raw MIMIC tables")
    for col in ("table", "required", "present", "size_mb", "path"):
        table.add_column(col)
    for row in status.itertuples(index=False):
        mark = (
            "[green]yes[/green]"
            if row.present
            else ("[red]NO[/red]" if row.required else "[yellow]no[/yellow]")
        )
        table.add_row(row.table, str(row.required), mark, str(row.size_mb or ""), row.path)
    console.print(table)
    sel = builder.config.selection_dir
    if (sel / "images.parquet").exists():
        images = pd.read_parquet(sel / "images.parquet")
        root = builder.config.sources.mimic_cxr_jpg.expanduser()
        present = sum((root / p).exists() for p in images["path"])
        console.print(
            f"Selection: {images['subject_id'].nunique()} patients, {len(images)} images "
            f"({present} downloaded) in {sel}"
        )
    else:
        console.print("Selection: not run yet (tbmxai mimic select CONFIG)")
    built = builder.out / "build_info.json"
    console.print(f"Build: {'done → ' + str(builder.out) if built.exists() else 'not run yet'}")


@mimic_app.command("select")
def mimic_select(
    config: MimicConfigArg,
    memory_limit: Annotated[
        str | None, typer.Option(help="DuckDB memory limit, e.g. 24GB.")
    ] = None,
) -> None:
    """Sample the patient subset and write the image download list."""
    info = _mimic_builder(config, memory_limit).select()
    console.print_json(
        json.dumps(
            {k: info[k] for k in ("n_patients", "n_studies", "n_images", "prevalence_subset")},
            default=str,
        )
    )


@mimic_app.command("dev-split")
def mimic_dev_split(
    config: MimicConfigArg,
    overwrite: Annotated[bool, typer.Option(help="Recreate an existing split.")] = False,
) -> None:
    """Write selection/dev_split.csv (shared patient-level train/val/test) if missing."""
    dev = _mimic_builder(config, None).ensure_dev_split(overwrite=overwrite)
    console.print(dev["dev_split"].value_counts().to_dict())


@mimic_app.command("build")
def mimic_build(
    config: MimicConfigArg,
    memory_limit: Annotated[
        str | None, typer.Option(help="DuckDB memory limit, e.g. 24GB.")
    ] = None,
) -> None:
    """Compute features, labels, image manifest and reports for the selected subset."""
    info = _mimic_builder(config, memory_limit).build()
    console.print_json(
        json.dumps(
            {
                k: info[k]
                for k in (
                    "n_patients",
                    "n_studies",
                    "n_images",
                    "n_images_present",
                    "n_features",
                    "features_per_group",
                    "label_prevalence",
                )
            },
            default=str,
        )
    )


@mimic_app.command("shrink-images")
def mimic_shrink_images(
    url_list: Annotated[Path, typer.Argument(exists=True, help="File of PhysioNet image URLs.")],
    mimic_dir: Annotated[Path, typer.Argument(help="Download root (MIMIC_DIR).")],
    max_side: Annotated[int, typer.Option(help="Longest side after shrinking (px).")] = 1024,
    quality: Annotated[int, typer.Option(help="JPEG quality.")] = 95,
    workers: Annotated[int, typer.Option(help="Parallel processes.")] = 4,
) -> None:
    """Shrink downloaded CXR JPGs in place (idempotent; used by 03_download_images.sh)."""
    from tbmxai.datasets.mimic.images import shrink_many, url_to_path

    root = mimic_dir.expanduser()
    urls = [u for u in url_list.read_text().split() if u]
    counts = shrink_many((url_to_path(u, root) for u in urls), max_side, quality, workers)
    console.print(f"Shrink to {max_side}px: {counts}")


@mimic_app.command("fetch-gcs")
def mimic_fetch_gcs(
    url_list: Annotated[Path, typer.Argument(exists=True, help="File of PhysioNet image URLs.")],
    mimic_dir: Annotated[Path, typer.Argument(help="Download root (MIMIC_DIR).")],
    bucket: Annotated[str, typer.Option(help="GCS mirror bucket.")] = (
        "mimic-cxr-jpg-2.1.0.physionet.org"
    ),
    project: Annotated[
        str | None, typer.Option(help="Billing project (requester-pays buckets).")
    ] = None,
    workers: Annotated[int, typer.Option(help="Parallel downloads.")] = 32,
) -> None:
    """Copy MIMIC-CXR-JPG images from PhysioNet's GCS mirror (needs `gcloud auth login`)."""
    from tbmxai.datasets.mimic.images import fetch_from_gcs

    urls = [u for u in url_list.read_text().split() if u]
    counts = fetch_from_gcs(urls, mimic_dir.expanduser(), bucket, project, workers)
    console.print(f"GCS fetch: {counts}")
    if counts.get("failed") or counts.get("not_found"):
        raise typer.Exit(1)


@mimic_app.command("fake")
def mimic_fake(
    out_dir: Annotated[
        Path, typer.Argument(help="Where to write a fake MIMIC tree (no real data).")
    ],
    n_patients: int = 300,
    seed: int = 0,
) -> None:
    """Generate a fake MIMIC tree with the real layout, for dry runs without credentials."""
    from tbmxai.datasets.mimic.fake import make_fake_mimic

    roots = make_fake_mimic(out_dir.expanduser(), n_patients=n_patients, seed=seed)
    console.print(f"[green]Fake MIMIC tree →[/green] {roots.root}")


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(tbmxai.__version__)


def _print_metrics(frame: pd.DataFrame, title: str) -> None:
    table = Table(title=title)
    for col in frame.columns:
        table.add_column(str(col))
    for row in frame.itertuples(index=False):
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
