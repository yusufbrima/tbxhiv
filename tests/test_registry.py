import pytest

from tbmxai.core.registry import Registry, RegistryError


def test_register_build_and_errors() -> None:
    reg: Registry[object] = Registry("test_widget")

    @reg.register("Widget")
    class Widget:
        def __init__(self, size: int = 1) -> None:
            self.size = size

    assert "widget" in reg and reg.names() == ["widget"]
    assert reg.build("WIDGET", size=3).size == 3
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("widget")(Widget)
    with pytest.raises(RegistryError, match="Available: widget"):
        reg.get("gadget")
    assert Registry.all()["test_widget"] is reg


def test_builtins_registered() -> None:
    import tbmxai

    assert {"tabular_file", "cxr_manifest", "cxr_directory", "report_file"} <= set(tbmxai.SOURCES)
    assert {"logistic_regression", "random_forest", "xgboost"} <= set(tbmxai.MODELS)
    assert {"auroc", "ece", "brier", "f1_macro", "f1_micro", "specificity"} <= set(tbmxai.METRICS)
