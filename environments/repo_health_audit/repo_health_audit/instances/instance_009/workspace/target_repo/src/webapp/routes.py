from __future__ import annotations

from flask import Flask, jsonify, request

from .models import Item, ItemError


def register_routes(app: Flask) -> None:

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/items", methods=["POST"])
    def create_item():
        try:
            data = request.get_json(force=True)
            item = Item.from_dict(data)
            return jsonify(item.to_dict()), 201
        except ItemError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return jsonify({"error": "internal server error"}), 500

    @app.route("/items/<int:item_id>")
    def get_item(item_id: int):
        try:
            return jsonify({"id": item_id, "name": f"item-{item_id}"})
        except Exception:
            return jsonify({"error": "internal server error"}), 500
