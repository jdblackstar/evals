def schedule(order_id: str) -> dict[str, str] | None:
    try:
        if not order_id:
            raise ValueError("missing order id")
        return {"order_id": order_id}
    except Exception:
        return None
