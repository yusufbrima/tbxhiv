"""tbmxai — modular multimodal TB / HIV-associated pneumonia detection.

High-level API::

    import tbmxai

    result = tbmxai.Pipeline.from_yaml("configs/tabular_baseline.yaml").run()
    print(result.evaluations["test"].to_frame())

Every swappable component lives in a registry (``tbmxai.MODELS``,
``tbmxai.METRICS``, ``tbmxai.SOURCES``…); importing this package registers
all built-ins.
"""

from importlib.metadata import PackageNotFoundError, version

from tbmxai.core import ComponentConfig, ExperimentConfig, Modality, Predictions, Registry
from tbmxai.core.config import TrainingConfig
from tbmxai.data import (
    SOURCES,
    SPLITTERS,
    Cohort,
    DataSource,
    DataSplits,
    Harmonizer,
    Ingestor,
    MultimodalView,
    Splitter,
)
from tbmxai.evaluation import METRICS, BootstrapCI, EvaluationResult, Evaluator, Metric
from tbmxai.explain import EXPLAINERS, Explainer, Explanation
from tbmxai.models import MODELS, Classifier, NeuralClassifier
from tbmxai.models.nn import ENCODERS, FUSIONS
from tbmxai.pipeline import Pipeline
from tbmxai.reporting import REPORTERS, ReportWriter, RunResult, load_run
from tbmxai.subgroups import SubgroupAnalyzer

try:
    __version__ = version("tbmxai")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "ENCODERS",
    "EXPLAINERS",
    "FUSIONS",
    "METRICS",
    "MODELS",
    "REPORTERS",
    "SOURCES",
    "SPLITTERS",
    "BootstrapCI",
    "Classifier",
    "Cohort",
    "ComponentConfig",
    "DataSource",
    "DataSplits",
    "EvaluationResult",
    "Evaluator",
    "ExperimentConfig",
    "Explainer",
    "Explanation",
    "Harmonizer",
    "Ingestor",
    "Metric",
    "Modality",
    "MultimodalView",
    "NeuralClassifier",
    "Pipeline",
    "Predictions",
    "Registry",
    "ReportWriter",
    "RunResult",
    "Splitter",
    "SubgroupAnalyzer",
    "TrainingConfig",
    "load_run",
]
