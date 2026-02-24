from __future__ import annotations

import json

from plugins.base import BasePlugin, PluginError


class JsonPlugin(BasePlugin):
    def name(self) -> str:
        return "json"

    def process(self, data: bytes) -> dict:
        self.validate_input(data)
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PluginError(f"json parse error: {exc}") from exc
        return {"format": "json", "type": type(parsed).__name__, "size": len(data)}
