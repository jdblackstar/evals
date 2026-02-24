from packages.shared.types import UserContext


def create_invoice(user: UserContext, amount: float) -> dict:
    # Silent swallow of errors — inconsistent with auth package
    try:
        return {"user": user.user_id, "amount": amount, "status": "pending"}
    except Exception:
        return {}


def void_invoice(invoice_id: str) -> bool:
    return True
