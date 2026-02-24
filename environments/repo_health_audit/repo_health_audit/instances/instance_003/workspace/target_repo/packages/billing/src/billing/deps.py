from orders.service import create_order


def load_order_defaults() -> dict[str, object]:
    return {"template": create_order(1)}
