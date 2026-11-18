"""Foundation-model encoders: registration and input conversion (no weight download)."""

import torch

from tbmxai.models.nn.encoders import ENCODERS
from tbmxai.models.nn.encoders.foundation import _to_unit


def test_registered() -> None:
    assert {"xrv_densenet", "rad_dino"} <= set(ENCODERS.names())


def test_to_unit_undoes_pipeline_normalisation() -> None:
    raw = torch.rand(2, 1, 8, 8)
    x = (raw - 0.5) / 0.25  # the pipeline's default normalisation
    assert torch.allclose(_to_unit(x, 0.5, 0.25), raw, atol=1e-6)
    rgb = torch.cat([x, x, x], dim=1)
    assert torch.allclose(_to_unit(rgb, 0.5, 0.25), raw, atol=1e-6)
    assert _to_unit(torch.full((1, 1, 2, 2), 10.0), 0.5, 0.25).max() == 1.0


class _Net(torch.nn.Module):
    """Minimal fusion net: one imaging encoder + linear head (what the explainers see)."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoders = torch.nn.ModuleDict({"imaging": encoder})
        self.head = torch.nn.Linear(encoder.out_dim, 2)

    def forward(self, inputs: dict) -> torch.Tensor:
        return self.head(self.encoders["imaging"](inputs["imaging"]))


def _check_grad_cam(encoder: torch.nn.Module, side: int) -> None:
    """The CAM layer resolves, the map has the image's shape, and the layer's activations get a
    spatially varying gradient (a constant or zero gradient means a degenerate CAM)."""
    from tbmxai.explain.saliency import GradCAMExplainer, _to_nchw

    torch.manual_seed(0)
    explainer = object.__new__(GradCAMExplainer)  # skip Classifier wiring; test the CAM only
    explainer.layer, explainer.target_class = None, 1
    net = _Net(encoder).eval()
    image = torch.randn(2, 1, side, side, requires_grad=True)
    cam = explainer.saliency(net, {"imaging": image}, image)
    assert cam.shape == (2, side, side) and torch.isfinite(cam).all()

    module, channels = explainer._target(net)
    store: dict = {}

    def hook(_m, _i, out) -> None:
        store["act"] = out[0] if isinstance(out, tuple) else out
        store["act"].retain_grad()

    handle = module.register_forward_hook(hook)
    try:
        net({"imaging": image})[:, 1].sum().backward()
    finally:
        handle.remove()
    grad = _to_nchw(store["act"].grad, channels)
    assert grad.shape[1] == channels and grad.abs().sum((1,)).flatten(1).std(1).min() > 0


def test_grad_cam_on_xrv_densenet() -> None:
    from tbmxai.models.nn.encoders.foundation import XRVDenseNetEncoder

    _check_grad_cam(XRVDenseNetEncoder(weights=None), side=64)


def test_grad_cam_on_rad_dino(monkeypatch) -> None:
    import transformers
    from transformers import Dinov2Config, Dinov2Model

    from tbmxai.models.nn.encoders.foundation import RadDinoEncoder

    tiny = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=56,
        patch_size=14,
    )
    proc = type("P", (), {"image_mean": [0.5] * 3, "image_std": [0.25] * 3})()
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *_a, **_k: Dinov2Model(tiny)
    )
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *_a, **_k: proc)
    _check_grad_cam(RadDinoEncoder(size=56), side=56)
