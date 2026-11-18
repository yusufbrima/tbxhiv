"""Dataset recipes, the adapter registry and dataset merging."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from tbmxai.data.dataset_format import DatasetSpec, validate_dataset
from tbmxai.datasets.adapters import PARSERS, register_parser
from tbmxai.datasets.recipe import Recipe

REPO = Path(__file__).resolve().parents[1]


def _png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(value)
    Image.fromarray(rng.integers(0, 255, (32, 32)).astype(np.uint8)).save(path)


def _fake_tbx(raw: Path) -> None:
    root = raw / "TBX" / "imgs"
    for folder, prefix, n in (("health", "h", 6), ("sick", "s", 4), ("tb", "tb", 4)):
        for i in range(n):
            _png(root / folder / f"{prefix}{i:04d}.png", hash((folder, i)) % 10_000)
    # an exact duplicate (same label) and a conflicting duplicate (different label)
    Image.open(root / "sick" / "s0000.png").save(root / "sick" / "s0099.png")
    Image.open(root / "health" / "h0000.png").save(root / "tb" / "tb0099.png")
    (raw / "TBX" / "extra").mkdir()
    _png(raw / "TBX" / "extra" / "x.png", 1)  # must not be picked up
    (raw / "TBX" / ".hidden.png").write_bytes(b"")
    lists = raw / "TBX" / "lists"
    lists.mkdir()
    (lists / "train.txt").write_text("health/h0000.png\nhealth/h0001.png\nsick/s0000.png\n")
    (lists / "val.txt").write_text("tb/tb0000.png\n")
    pd.DataFrame({"image": [f"h{i:04d}" for i in range(6)], "age": range(30, 36)}).to_csv(
        raw / "TBX" / "meta.csv", index=False
    )


TBX_RECIPE = {
    "name": "fake_tbx",
    "patient_ids": "assumed",
    "records": {
        "files": ["**/imgs/health/*.png", "**/imgs/sick/*.png", "**/imgs/tb/*.png"],
        "sort_by": "patient_id",
        "drop_duplicate_images": True,
    },
    "fields": {
        "patient_id": {"from": "stem", "format": "tbx_{}"},
        "cls": {"from": "parent", "map": {"health": "healthy", "sick": "sick", "tb": "tb"}},
        "tb": {"from": "field:cls", "map": {"tb": "tb"}, "unmapped": "missing", "default": "no"},
        "split": {
            "from": "list",
            "lists": {"train": "TBX/lists/train.txt", "val": "TBX/lists/val.txt"},
        },
        "age": {"from": "table", "path": "TBX/meta.csv", "key": "image", "column": "age"},
        "number": {"from": "stem", "regex": r"(\d+)$", "cast": "int"},
        "age_group": {"from": "field:age", "bins": [0, 33, 200], "labels": ["<33", "33+"]},
    },
    "tables": {"labels": ["tb", "cls", "split", "age_group"], "tabular": ["age", "number"]},
    "labels": {"tb": {"classes": ["no", "tb"]}, "cls": {"classes": ["healthy", "sick", "tb"]}},
}


def test_recipe_sources_transforms_and_duplicates(tmp_path) -> None:
    _fake_tbx(tmp_path / "raw")
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(TBX_RECIPE, sort_keys=False))
    info = Recipe.load(tmp_path / "r.yaml").build(tmp_path / "raw", tmp_path / "out")
    labels = pd.read_csv(tmp_path / "out" / "labels.csv").set_index("patient_id")
    tab = pd.read_csv(tmp_path / "out" / "tabular.csv").set_index("patient_id")
    images = pd.read_csv(tmp_path / "out" / "images.csv")

    # 14 images + 2 duplicates; s0099 dropped (duplicate), h0000 + tb0099 dropped (label conflict)
    assert len(labels) == 13 and "tbx_s0099" not in labels.index
    assert {"tbx_h0000", "tbx_tb0099"}.isdisjoint(labels.index)
    reasons = sorted(d["reason"] for d in info["duplicate_images_dropped"])
    assert reasons == ["duplicate", "label conflict"]
    assert not images["image_path"].str.contains("extra|hidden").any()

    assert labels.loc["tbx_tb0001", "tb"] == "tb" and labels.loc["tbx_h0001", "tb"] == "no"
    assert (
        labels.loc["tbx_h0001", "split"] == "train" and labels.loc["tbx_tb0000", "split"] == "val"
    )
    assert pd.isna(labels.loc["tbx_tb0001", "split"])
    assert tab.loc["tbx_h0001", "age"] == 31 and pd.isna(tab.loc["tbx_s0001", "age"])
    assert tab.loc["tbx_tb0003", "number"] == 3
    assert labels.loc["tbx_h0005", "age_group"] == "33+"
    assert list(labels.index) == sorted(labels.index)

    spec = DatasetSpec.load(tmp_path / "out")
    assert spec.patient_ids == "assumed" and spec.images_root == (tmp_path / "raw").resolve()
    assert validate_dataset(tmp_path / "out", check_images=None).ok


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fields": {"cls": {"from": "parent", "map": {"health": "healthy"}}}}, "not in map"),
        ({"labels": {"tb": {"classes": ["no", "yes"]}}}, "not in classes"),
        ({"fields": {"x": {"from": "nowhere"}}}, "unknown source"),
        ({"fields": {"x": {"from": "file"}}}, "needs `glob`"),
        ({"fields": {"x": {"from": "field:later"}, "later": {"from": "stem"}}}, "before it"),
        ({"fields": {"x": {"from": "stem", "parse": "nope"}}}, "unknown parser"),
        ({"fields": {"x": {"from": "stem", "cast": "float"}}}, "cannot cast"),
        ({"records": {"files": "*.png", "table": "t.csv"}}, "exactly one"),
    ],
)
def test_recipe_errors(tmp_path, change, message) -> None:
    _fake_tbx(tmp_path / "raw")
    recipe = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TBX_RECIPE.items()}
    for key, value in change.items():
        if key == "fields":
            recipe["fields"] = {**recipe["fields"], **value}
        else:
            recipe[key] = {**recipe[key], **value} if key == "labels" else value
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    with pytest.raises(ValueError, match=message):
        Recipe.load(tmp_path / "r.yaml").build(tmp_path / "raw", tmp_path / "out")


def test_table_records_and_custom_parser(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {"pid": ["a", "b", "c"], "dx": ["TB", "none", "TB"], "note": ["x;61", "y;45", "z;30"]}
    ).to_csv(raw / "meta.csv", index=False)

    @register_parser("semicolon_age")
    def _age(text: str) -> dict:
        return {"age": float(text.split(";")[1])}

    try:
        (tmp_path / "r.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "tab",
                    "records": {"table": "meta.csv"},
                    "fields": {
                        "patient_id": {"from": "column:pid"},
                        "tb": {"from": "column:dx", "map": {"TB": "tb", "none": "no"}},
                        "parsed": {"from": "column:note", "parse": "semicolon_age"},
                    },
                    "tables": {"labels": ["tb"], "tabular": ["age"]},
                    "labels": {"tb": {"classes": ["no", "tb"]}},
                }
            )
        )
        Recipe.load(tmp_path / "r.yaml").build(raw, tmp_path / "out")
    finally:
        PARSERS.pop("semicolon_age")
    assert pd.read_csv(tmp_path / "out" / "tabular.csv")["age"].tolist() == [61.0, 45.0, 30.0]
    assert not (tmp_path / "out" / "images.csv").exists()
