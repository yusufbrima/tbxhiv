"""Modality encoders. Importing registers all built-in encoders."""

from tbmxai.models.nn.encoders.base import ENCODERS, Encoder, TokenOutput
from tbmxai.models.nn.encoders.foundation import RadDinoEncoder, XRVDenseNetEncoder
from tbmxai.models.nn.encoders.tabular import FTTransformerEncoder, MLPTabularEncoder
from tbmxai.models.nn.encoders.text import TEXT_BACKBONES, HFTextEncoder
from tbmxai.models.nn.encoders.vision import VISION_BACKBONES, TimmVisionEncoder, backbone_family

__all__ = [
    "ENCODERS",
    "TEXT_BACKBONES",
    "VISION_BACKBONES",
    "Encoder",
    "FTTransformerEncoder",
    "HFTextEncoder",
    "MLPTabularEncoder",
    "RadDinoEncoder",
    "TimmVisionEncoder",
    "TokenOutput",
    "XRVDenseNetEncoder",
    "backbone_family",
]
