from __future__ import annotations

import os

APP_ENV = os.getenv("APP_ENV", "dev")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ledger.db")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5"))
