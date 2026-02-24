from engine.core.processor import process


def handle(raw_amount: str) -> float | None:
    try:
        return process(raw_amount)
    except Exception:
        # Inconsistent with strict typed errors in engine/core/processor.py
        return None
