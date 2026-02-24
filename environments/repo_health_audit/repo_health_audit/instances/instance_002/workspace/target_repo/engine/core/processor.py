from engine.core.errors import ProcessorError


def process(amount: str) -> float:
    try:
        value = float(amount)
    except ValueError as exc:
        raise ProcessorError("invalid amount") from exc
    if value < 0:
        raise ProcessorError("amount must be non-negative")
    return round(value, 2)
