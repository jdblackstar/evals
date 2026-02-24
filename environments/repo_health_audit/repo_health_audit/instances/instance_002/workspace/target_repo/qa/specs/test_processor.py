from engine.core.processor import process
from services.payments.main import handle


def test_process_rounding() -> None:
    assert process("1.234") == 1.23


def test_process_zero() -> None:
    assert process("0") == 0


def test_process_integer() -> None:
    assert process("5") == 5


def test_handle_ok() -> None:
    assert handle("3") == 3


def test_handle_invalid_returns_none() -> None:
    assert handle("abc") is None


def test_handle_negative_returns_none() -> None:
    assert handle("-1") is None
