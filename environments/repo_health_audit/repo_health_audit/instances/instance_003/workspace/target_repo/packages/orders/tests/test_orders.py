from orders.service import create_order


def test_create_order_ok() -> None:
    assert create_order(10) == {"amount": 10}


def test_create_order_invalid_amount() -> None:
    assert create_order(-1) is None


def test_create_order_zero_amount() -> None:
    assert create_order(0) is None
