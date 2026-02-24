"""API gateway proxy layer."""

import requests
from flask import Flask, request, jsonify

from libs.common.config import get_service_url

app = Flask(__name__)

SERVICE_REGISTRY = {
    "users": "users",
    "notifications": "notifications",
}


def forward_request(service_name: str, path: str, method: str = "GET", **kwargs):
    """Forward a request to an internal service."""
    base_url = get_service_url(service_name)
    url = f"{base_url}/{path}"
    resp = requests.request(method, url, **kwargs)
    return resp.json(), resp.status_code


@app.route("/api/<service>/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def gateway(service, path):
    if service not in SERVICE_REGISTRY:
        return jsonify({"error": "unknown service"}), 404
    data, status = forward_request(service, path, method=request.method)
    return jsonify(data), status
