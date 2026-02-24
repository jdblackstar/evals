from __future__ import annotations


class LedgerError(Exception):
    pass


def normalize_amount(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise LedgerError("amount must be numeric") from exc
    if value <= 0:
        raise LedgerError("amount must be positive")
    return round(value, 2)
