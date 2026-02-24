from webapp.models import Item, ItemError


def test_item_from_dict_valid() -> None:
    item = Item.from_dict({"name": "bolt", "quantity": 10})
    assert item.name == "bolt"
    assert item.quantity == 10


def test_item_from_dict_negative_quantity() -> None:
    try:
        Item.from_dict({"name": "bolt", "quantity": -1})
        assert False, "expected ItemError"
    except ItemError:
        assert True
