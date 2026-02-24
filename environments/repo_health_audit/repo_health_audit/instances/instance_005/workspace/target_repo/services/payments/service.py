def charge(amount: float) -> dict[str, float]:
    if amount <= 0:
        raise ValueError("invalid amount")
    return {"charged": amount}
