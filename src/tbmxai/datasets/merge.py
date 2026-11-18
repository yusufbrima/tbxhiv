"""Pool several standard datasets into one (multi-site studies).

Each input keeps its own images in place: image paths are written as absolute
paths. Every table gains a ``source`` column (the input dataset's name) so the
``external`` splitter and subgroup analysis can work per site. Label values can
be harmonised (e.g. NLM ``normal`` and TBX11K ``non_tb`` → ``non_tb``).

Columns missing from an input are left empty (e.g. TBX11K has no age/sex); a
modality missing from an input simply has no rows for its patients.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tbmxai.core.utils import get_logger
from tbmxai.data.dataset_format import (
    IMAGE_PATH,
    PATIENT_ID,
    DatasetSpec,
    LabelSpec,
    _read,
)

logger = get_logger(__name__)

ROLES = ("labels", "tabular", "images", "reports")


def merge_datasets(
    sources: Sequence[str | Path],
    out_dir: str | Path,
    labels: Mapping[str, Sequence[str]],
    rename: Mapping[str, Mapping[str, str]] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Concatenate standard datasets into ``out_dir``.

    Args:
        sources: Dataset folders (each with ``dataset.yaml``).
        out_dir: Output dataset folder.
        labels: Label column → classes (encoding order, positive last) of the merged
            dataset. Every input must have these label columns.
        rename: Label column → {old value: new value}, applied before checking classes.
        name: Name of the merged dataset (default: input names joined by ``+``).

    Returns:
        Build summary (also written to ``build_info.json``).
    """
    if len(sources) < 2:
        raise ValueError("merge needs at least two datasets")
    rename = {k: dict(v) for k, v in (rename or {}).items()}
    specs = [DatasetSpec.load(s) for s in sources]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"datasets must have distinct names, got {names}")

    merged: dict[str, list[pd.DataFrame]] = {r: [] for r in ROLES}
    for spec in specs:
        files = spec.files()
        for col in labels:
            if col not in spec.labels:
                raise ValueError(f"dataset '{spec.name}' has no label '{col}'")
        for role in ROLES:
            if role not in files:
                continue
            frame = _read(files[role])
            if role == "labels":
                for col, mapping in rename.items():
                    if col in frame:
                        frame[col] = frame[col].map(lambda v, m=mapping: m.get(v, v))
                for col, classes in labels.items():
                    bad = sorted(set(frame[col].dropna().astype(str)) - set(classes))
                    if bad:
                        raise ValueError(
                            f"'{spec.name}': label '{col}' values {bad} not in {list(classes)}; "
                            "use rename to harmonise them"
                        )
            if role == "images":
                root = spec.images_root
                frame[IMAGE_PATH] = (
                    frame[IMAGE_PATH]
                    .astype(str)
                    .map(lambda p, r=root: p if Path(p).is_absolute() else str((r / p).resolve()))
                )
            frame.insert(1, "source", spec.name)
            merged[role].append(frame)

    patients = [set(f[PATIENT_ID].astype(str)) for f in merged["labels"]]
    for i in range(len(patients)):
        for j in range(i + 1, len(patients)):
            if clash := patients[i] & patients[j]:
                raise ValueError(
                    f"patient_id collision between '{names[i]}' and '{names[j]}' "
                    f"({len(clash)} ids, e.g. {sorted(clash)[:3]}); prefix the ids in one recipe"
                )

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for role, frames in merged.items():
        if not frames:
            continue
        table = pd.concat(frames, ignore_index=True, sort=False)
        if role != "labels":
            table = table.drop(columns="source")  # metadata lives in labels.csv
        table.to_csv(out / f"{role}.csv", index=False)
        rows[role] = len(table)

    merged_name = name or "+".join(names)
    DatasetSpec(
        name=merged_name,
        description="Merged: " + "; ".join(f"{s.name} ({s.description})" for s in specs),
        patient_ids="assumed" if any(s.patient_ids == "assumed" for s in specs) else "provided",
        labels={
            col: LabelSpec(classes=list(classes), description=specs[0].labels[col].description)
            for col, classes in labels.items()
        },
    ).save(out)
    label_frame = pd.concat(merged["labels"], ignore_index=True, sort=False)
    info = {
        "created_utc": datetime.now(UTC).isoformat(),
        "name": merged_name,
        "sources": {s.name: str(s.root) for s in specs},
        "rename": rename,
        "rows": rows,
        "label_counts_per_source": {
            col: label_frame.groupby("source")[col]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict(orient="index")
            for col in labels
        },
    }
    (out / "build_info.json").write_text(json.dumps(info, indent=2, default=str))
    logger.info("Merged %s → %s (%s)", names, out, rows)
    return info
