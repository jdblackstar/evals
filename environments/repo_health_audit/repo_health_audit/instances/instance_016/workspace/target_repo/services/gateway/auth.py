"""JWT-based authentication middleware."""

import jwt
from functools import wraps
from flask import request, jsonify

SECRET_KEY = "change-me-in-production"


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token."""
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])


def require_auth(f):
    """Decorator to require authentication on a route."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "missing token"}), 401
        try:
            payload = verify_token(token)
            request.user = payload
        except jwt.InvalidTokenError:
            return jsonify({"error": "invalid token"}), 401
        return f(*args, **kwargs)
    return decorated
