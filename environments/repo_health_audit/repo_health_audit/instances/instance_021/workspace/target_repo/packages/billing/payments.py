from packages.shared.types import UserContext


def process_payment(user: UserContext, invoice_id: str) -> dict:
    return {"invoice": invoice_id, "charged": True}


def refund_payment(payment_id: str) -> bool:
    # No error handling at all
    return True
