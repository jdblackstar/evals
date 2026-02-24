from billing.service import build_billing_adapter


def test_adapter_shape() -> None:
    adapter = build_billing_adapter()
    assert "record" in adapter


def test_adapter_record_callable() -> None:
    adapter = build_billing_adapter()
    assert callable(adapter["record"])
