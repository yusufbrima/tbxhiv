"""MIMIC-IV + MIMIC-IV-ED + MIMIC-CXR(-JPG) adapter.

Turns credentialed PhysioNet downloads into tbmxai-ready, per-study tables:
patient-level subset selection, ICD/CheXpert labels, leakage-safe EHR features
(only data recorded before each CXR), image manifest and reports.

See ``docs/MIMIC.md`` for the step-by-step workflow.
"""

from tbmxai.datasets.mimic.build import MimicBuilder, extract_sections
from tbmxai.datasets.mimic.config import (
    LabelDefinition,
    MimicBuildConfig,
    MimicSourcesConfig,
    SelectionConfig,
    default_feature_groups,
    default_labels,
)
from tbmxai.datasets.mimic.features import FEATURE_GROUPS, FeatureGroup, WindowedNumericGroup
from tbmxai.datasets.mimic.tables import TABLES, MimicTables

__all__ = [
    "FEATURE_GROUPS",
    "TABLES",
    "FeatureGroup",
    "LabelDefinition",
    "MimicBuildConfig",
    "MimicBuilder",
    "MimicSourcesConfig",
    "MimicTables",
    "SelectionConfig",
    "WindowedNumericGroup",
    "default_feature_groups",
    "default_labels",
    "extract_sections",
]
