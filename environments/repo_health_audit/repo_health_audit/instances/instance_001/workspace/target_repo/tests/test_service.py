from ledger.api import serialize_payment
from ledger.service import LedgerError, normalize_amount


def test_normalize_amount_rounds_value() -> None:
    assert normalize_amount("12.345") == 12.35


def test_normalize_amount_rejects_non_numeric() -> None:
    try:
        normalize_amount("abc")
        assert False, "expected LedgerError"
    except LedgerError:
        assert True


def test_normalize_amount_rejects_non_positive() -> None:
    try:
        normalize_amount("0")
        assert False, "expected LedgerError"
    except LedgerError:
        assert True


def test_serialize_payment_shape() -> None:
    payload = serialize_payment("3.2")
    assert payload == {"normalized_amount": 3.2}
