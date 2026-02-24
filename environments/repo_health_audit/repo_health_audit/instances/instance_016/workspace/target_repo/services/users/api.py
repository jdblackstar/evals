"""User management API endpoints."""

from flask import Flask, request, jsonify

from .store import UserStore

app = Flask(__name__)
store = UserStore()


@app.route("/users", methods=["POST"])
def create_user():
    data = request.get_json()
    user = store.create(data["name"], data["email"])
    return jsonify(user), 201


@app.route("/users/<user_id>", methods=["GET"])
def get_user(user_id):
    user = store.get(user_id)
    if user is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(user)


@app.route("/users", methods=["GET"])
def list_users():
    return jsonify(store.list_all())
