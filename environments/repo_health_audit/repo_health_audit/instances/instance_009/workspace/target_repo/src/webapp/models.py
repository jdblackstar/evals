from __future__ import annotations

from dataclasses import dataclass


class ItemError(Exception):
    pass


@dataclass
class Item:
    name: str
    quantity: int

    @classmethod
    def from_dict(cls, data: dict) -> Item:
        if "name" not in data:
            raise ItemError("missing required field: name")
        if "quantity" not in data:
            raise ItemError("missing required field: quantity")
        qty = data["quantity"]
        if not isinstance(qty, int) or qty < 0:
            raise ItemError("quantity must be a non-negative integer")
        return cls(name=data["name"], quantity=qty)

    def to_dict(self) -> dict:
        return {"name": self.name, "quantity": self.quantity}
