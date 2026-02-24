from __future__ import annotations

from .service import normalize_amount


def serialize_payment(amount: str) -> dict[str, float]:
    normalized = normalize_amount(amount)
    return {"normalized_amount": normalized}
