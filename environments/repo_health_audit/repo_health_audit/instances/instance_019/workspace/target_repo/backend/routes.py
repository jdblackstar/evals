from flask import Blueprint, jsonify, request

api = Blueprint("api", __name__)


@api.route("/users", methods=["GET"])
def list_users():
    try:
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route("/users", methods=["POST"])
def create_user():
    try:
        data = request.get_json()
        return jsonify(data), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400
