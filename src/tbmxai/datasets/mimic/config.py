"""Configuration for building a tbmxai-ready MIMIC-IV + MIMIC-CXR subset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from tbmxai.core.config import InterpolatedConfig, _Strict


class MimicSourcesConfig(_Strict):
    """Roots of the PhysioNet downloads (each may live anywhere on disk).

    Attributes:
        mimic_iv: MIMIC-IV root (contains ``hosp/`` and ``icu/``), e.g. ``.../mimiciv/3.1``.
        mimic_iv_ed: MIMIC-IV-ED root (contains ``ed/``). Optional.
        mimic_cxr_jpg: MIMIC-CXR-JPG root (metadata CSVs + ``files/``).
        mimic_cxr_reports: MIMIC-CXR root with ``files/pXX/pY/sZ.txt`` reports, or the
            path to ``mimic-cxr-reports.zip``. Optional.
    """

    mimic_iv: Path
    mimic_iv_ed: Path | None = None
    mimic_cxr_jpg: Path
    mimic_cxr_reports: Path | None = None


class DevSplitConfig(_Strict):
    """One patient-level train / val / test split shared by every MIMIC experiment.

    Written once per selection to ``selection/dev_split.csv`` and copied into the
    dataset (``labels.csv: dev_split``). Supervised pretraining trains on ``train``,
    early-stops on ``val`` and never sees ``test``; downstream experiments use the same
    split (``split: {name: predefined, params: {column: dev_split}}``), so the test
    patients are unseen by every stage. Stratified by the listed labels (any positive
    study per patient).
    """

    val: float = Field(default=0.1, gt=0, lt=0.5)
    test: float = Field(default=0.2, gt=0, lt=0.5)
    seed: int = 42
    stratify: list[str] = Field(
        default_factory=lambda: ["tb_episode", "hiv_pneumonia", "pneumonia"]
    )
    inherit_from: Path | None = Field(
        default=None,
        description="An existing dev_split.csv (subject_id, dev_split): its patients keep their "
        "assignment and only the remaining patients are split (stratified, same fractions and "
        "seed). Extends a cohort without moving any patient between train and test, so models "
        "trained on the earlier split stay leakage-free on the new test set.",
    )


class SelectionConfig(_Strict):
    """How the development subset is sampled (always by patient)."""

    n_patients: int = Field(default=5000, ge=1)
    seed: int = 42
    views: list[str] = Field(default_factory=lambda: ["PA", "AP"])
    one_image_per_study: bool = Field(
        default=False, description="Keep one frontal image per study (PA preferred)."
    )
    max_studies_per_patient: int | None = Field(default=None, ge=1)
    require_encounter: bool = Field(
        default=True,
        description="Keep only studies taken during a hospital admission or ED stay "
        "(needed for ICD-based labels and encounter features). Studies positive for an "
        "enriched `episode` label are kept even without an encounter.",
    )
    dev_split: DevSplitConfig = Field(default_factory=DevSplitConfig)
    enrich_labels: list[str] = Field(
        default_factory=lambda: ["tb_episode", "tb", "hiv_pneumonia"],
        description="Include every patient positive for these labels before random filling. "
        "Changes prevalence vs. the source population; reported in build_info.json.",
    )


class LabelDefinition(_Strict):
    """A binary target defined from ICD codes, CheXpert findings or clinical episodes.

    * ``icd``: positive when any diagnosis of the study's encounter (admission, else ED
      stay) starts with one of the prefixes; ``<NA>`` without an encounter.
    * ``chexpert``: a CheXpert finding of the study's report.
    * ``episode``: positive when the study falls within ``window_days`` of any evidence
      event of the patient, from ANY encounter: an admission / ED stay coded with the
      ICD prefixes (its whole stay), a culture growing one of ``organisms`` (specimen
      time) or a first prescription of one of ``drugs``. Patients with evidence but
      studies outside every window are ``<NA>`` (``outside_window: missing``, avoids
      labelling images years away from the episode) or 0; patients without any
      evidence are 0. Works for studies without an encounter.

    ``all_of`` additionally requires other labels to be positive (e.g. HIV ∧ pneumonia).
    """

    source: Literal["icd", "chexpert", "episode"] = "icd"
    icd9: list[str] = Field(default_factory=list)
    icd10: list[str] = Field(default_factory=list)
    organisms: list[str] = Field(
        default_factory=list, description="episode: microbiology org_name substrings."
    )
    drugs: list[str] = Field(
        default_factory=list, description="episode: prescription drug-name substrings."
    )
    window_days: int = Field(default=30, ge=0, description="episode: ± days around evidence.")
    outside_window: Literal["missing", "negative"] = Field(
        default="missing", description="episode: label of an evidence patient's other studies."
    )
    all_of: list[str] = Field(default_factory=list)
    chexpert_finding: str | None = None
    uncertain: Literal["positive", "negative", "drop"] = Field(
        default="drop", description="CheXpert -1 handling (U-ones / U-zeros / ignore)."
    )
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> LabelDefinition:
        if self.source == "chexpert" and not self.chexpert_finding:
            raise ValueError("chexpert labels need `chexpert_finding`")
        if self.source == "icd" and not (self.icd9 or self.icd10 or self.all_of):
            raise ValueError("icd labels need icd9/icd10 prefixes or all_of")
        if self.source == "episode" and not (
            self.icd9 or self.icd10 or self.organisms or self.drugs
        ):
            raise ValueError("episode labels need ICD prefixes, organisms or drugs")
        return self


def default_labels() -> dict[str, LabelDefinition]:
    """Built-in targets (codes are MIMIC style: no dots)."""
    return {
        "pneumonia": LabelDefinition(
            icd9=["480", "481", "482", "483", "484", "485", "486", "4870", "1363"],
            icd10=["J12", "J13", "J14", "J15", "J16", "J17", "J18", "B59"],
            description="Pneumonia incl. influenza pneumonia and PCP",
        ),
        "tb": LabelDefinition(
            icd9=["010", "011", "012", "013", "014", "015", "016", "017", "018"],
            icd10=["A15", "A16", "A17", "A18", "A19"],
            description="Active tuberculosis (any site)",
        ),
        "tb_episode": LabelDefinition(
            source="episode",
            icd9=["010", "011", "012", "013", "014", "015", "016", "017", "018"],
            icd10=["A15", "A16", "A17", "A18", "A19"],
            organisms=["MYCOBACTERIUM TUBERCULOSIS"],
            drugs=["pyrazinamide"],
            window_days=30,
            description="Active TB episode: CXR within 30 days of a TB-coded admission/ED "
            "stay, an M. tuberculosis culture or a first pyrazinamide prescription",
        ),
        "pulmonary_tb": LabelDefinition(
            icd9=["010", "011", "012", "018"],
            icd10=["A15", "A19"],
            description="Pulmonary / respiratory / miliary TB",
        ),
        "hiv": LabelDefinition(
            icd9=["042", "V08"], icd10=["B20", "Z21", "O987"], description="HIV infection"
        ),
        "hiv_pneumonia": LabelDefinition(
            all_of=["hiv", "pneumonia"], description="Pneumonia in an HIV-positive encounter"
        ),
        "chexpert_pneumonia": LabelDefinition(
            source="chexpert", chexpert_finding="Pneumonia", uncertain="drop"
        ),
        "chexpert_consolidation": LabelDefinition(
            source="chexpert", chexpert_finding="Consolidation", uncertain="drop"
        ),
    }


def default_feature_groups() -> dict[str, dict[str, Any]]:
    """Every built-in EHR feature group, enabled with default parameters."""
    return {
        "demographics": {},
        "encounter": {},
        "history": {},
        "labs": {},
        "ed_triage": {},
        "ed_vitals": {},
        "icu_charts": {},
        "omr": {},
        "medications": {},
        "microbiology_orders": {},
        "care_unit": {},
        "service": {},
        "procedure_history": {},
    }


class MimicBuildConfig(InterpolatedConfig):
    """Top-level config for ``tbmxai mimic select`` / ``tbmxai mimic build``."""

    output_dir: Path
    sources: MimicSourcesConfig
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    labels: dict[str, LabelDefinition] = Field(default_factory=default_labels)
    features: dict[str, dict[str, Any]] = Field(
        default_factory=default_feature_groups,
        description="Feature group name → params. Remove a key to disable that group.",
    )
    physionet_base_url: str = "https://physionet.org/files"
    cxr_jpg_project: str = Field(
        default="mimic-cxr-jpg/2.1.0", description="PhysioNet project/version for image URLs."
    )

    @model_validator(mode="after")
    def _check_labels(self) -> MimicBuildConfig:
        for name, definition in self.labels.items():
            unknown = [d for d in definition.all_of if d not in self.labels]
            if unknown:
                raise ValueError(f"label '{name}': all_of refers to unknown labels {unknown}")
        unknown = [n for n in self.selection.enrich_labels if n not in self.labels]
        if unknown:
            raise ValueError(f"selection.enrich_labels has unknown labels {unknown}")
        return self

    @property
    def selection_dir(self) -> Path:
        """Where ``select`` writes its outputs."""
        return self.output_dir.expanduser() / "selection"
