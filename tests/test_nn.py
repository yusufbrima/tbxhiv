import pytest
import torch

from tbmxai.core.config import ComponentConfig
from tbmxai.models.nn import ENCODERS, FUSIONS, VISION_BACKBONES, build_network

B = 3


@pytest.mark.parametrize(
    "backbone",
    [
        VISION_BACKBONES["cnn"][0],
        "maxvit_tiny_tf_224",
        "vit_tiny_patch16_224",
        "swin_tiny_patch4_window7_224",
    ],
)
def test_vision_encoder_families_from_scratch(backbone: str) -> None:
    size = 64 if "swin" not in backbone else 224
    enc = ENCODERS.build(
        "timm_vision", backbone=backbone, pretrained=False, in_chans=1, image_size=size
    )
    x = torch.randn(B, 1, size, size)
    assert enc(x).shape == (B, enc.out_dim)
    tokens, mask = enc.forward_tokens(x)
    assert (
        tokens.ndim == 3
        and tokens.shape[0] == B
        and tokens.shape[2] == enc.out_dim
        and mask is None
    )


@pytest.mark.parametrize("name", ["mlp_tabular", "ft_transformer"])
def test_tabular_encoders(name: str) -> None:
    enc = ENCODERS.build(name, in_features=7)
    x = torch.randn(B, 7)
    assert enc(x).shape == (B, enc.out_dim)
    assert enc.forward_tokens(x)[0].shape[0] == B


def test_freeze_and_checkpoint(tmp_path) -> None:
    enc = ENCODERS.build("mlp_tabular", in_features=4)
    torch.save(enc.state_dict(), tmp_path / "w.pt")
    frozen = ENCODERS.build("mlp_tabular", in_features=4, freeze=True, checkpoint=tmp_path / "w.pt")
    assert not any(p.requires_grad for p in frozen.parameters())
    x = torch.randn(2, 4)
    enc.eval(), frozen.eval()
    assert torch.allclose(enc(x), frozen(x))


@pytest.mark.parametrize(
    "fusion", ["concat", "gated", "bilinear", "attention", "cross_attention", "late"]
)
def test_fusion_operators_end_to_end(fusion: str) -> None:
    params = {"query": "tabular"} if fusion == "cross_attention" else {}
    net = build_network(
        {
            "imaging": ComponentConfig(
                name="timm_vision", params={"backbone": "resnet18", "pretrained": False}
            ),
            "tabular": ComponentConfig(name="ft_transformer", params={"d_token": 32}),
        },
        n_classes=2,
        fusion=ComponentConfig(name=fusion, params=params),
        head={"hidden_dims": [16]},
        encoder_overrides={"tabular": {"in_features": 5}, "imaging": {"in_chans": 1}},
    )
    logits = net({"imaging": torch.randn(B, 1, 32, 32), "tabular": torch.randn(B, 5)})
    assert logits.shape == (B, 2)
    logits.sum().backward()


def test_bilinear_requires_two_modalities() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        FUSIONS.build("bilinear", input_dims={"a": 4, "b": 4, "c": 4})


def test_unimodal_network_has_no_fusion() -> None:
    net = build_network(
        {"tabular": ComponentConfig(name="mlp_tabular")},
        n_classes=3,
        encoder_overrides={"tabular": {"in_features": 6}},
    )
    assert net.fusion is None and net({"tabular": torch.randn(B, 6)}).shape == (B, 3)
