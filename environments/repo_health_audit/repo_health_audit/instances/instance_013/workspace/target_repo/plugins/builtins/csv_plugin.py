from __future__ import annotations

import csv
import io

from plugins.base import BasePlugin, PluginError


class CsvPlugin(BasePlugin):
    def name(self) -> str:
        return "csv"

    def process(self, data: bytes) -> dict:
        self.validate_input(data)
        try:
            text = data.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
        except Exception as exc:
            raise PluginError(f"csv parse error: {exc}") from exc
        return {"format": "csv", "row_count": len(rows), "columns": reader.fieldnames or []}
