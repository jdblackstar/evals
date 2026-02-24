from __future__ import annotations

import os

ORDERS_QUEUE = os.getenv("ORDERS_QUEUE", "orders-dev")
