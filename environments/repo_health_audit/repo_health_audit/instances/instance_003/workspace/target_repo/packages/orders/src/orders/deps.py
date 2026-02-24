from billing.service import build_billing_adapter


def load_billing_adapter():
    return build_billing_adapter()
