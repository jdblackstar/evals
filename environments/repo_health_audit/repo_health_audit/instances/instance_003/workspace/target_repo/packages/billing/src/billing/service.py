from billing.deps import load_order_defaults


def build_billing_adapter() -> dict[str, object]:
    defaults = load_order_defaults()

    def record(amount: float) -> None:
        if amount < 1:
            raise ValueError("amount too small")
        _ = defaults

    return {"record": record}
