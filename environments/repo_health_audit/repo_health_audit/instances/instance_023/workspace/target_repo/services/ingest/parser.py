import json


def parse_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def parse_csv_line(line: str, delimiter: str = ",") -> list[str]:
    return line.strip().split(delimiter)


def parse_document(raw: bytes, fmt: str = "json") -> dict | None:
    # No error handling for unknown format — will return None silently
    if fmt == "json":
        return parse_json(raw.decode("utf-8"))
    return None
