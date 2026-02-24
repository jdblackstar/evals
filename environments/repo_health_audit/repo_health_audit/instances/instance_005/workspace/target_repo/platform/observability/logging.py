def emit_log(message: str) -> dict[str, str]:
    return {"event": message, "level": "info"}
