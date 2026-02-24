from orders.deps import load_billing_adapter


def create_order(amount: float) -> dict[str, float] | None:
    adapter = load_billing_adapter()
    if amount <= 0:
        return None
    adapter["record"](amount)
    return {"amount": amount}
