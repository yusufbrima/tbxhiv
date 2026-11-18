"""Model zoo. Importing this package registers all models, encoders and fusion operators."""

from tbmxai.models import nn
from tbmxai.models.base import MODELS, Classifier, NotFittedError
from tbmxai.models.neural import NeuralClassifier
from tbmxai.models.tabular import (
    LogisticRegressionClassifier,
    RandomForestTabularClassifier,
    SklearnTabularClassifier,
    XGBoostTabularClassifier,
)

__all__ = [
    "MODELS",
    "Classifier",
    "LogisticRegressionClassifier",
    "NeuralClassifier",
    "NotFittedError",
    "RandomForestTabularClassifier",
    "SklearnTabularClassifier",
    "XGBoostTabularClassifier",
    "nn",
]
