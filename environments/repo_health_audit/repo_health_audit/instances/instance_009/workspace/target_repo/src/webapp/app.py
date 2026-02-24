from __future__ import annotations

from flask import Flask

from .routes import register_routes


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///webapp.db"
    register_routes(app)
    return app
