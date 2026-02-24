def run_job(payload: dict[str, object]) -> dict[str, object]:
    if "id" not in payload:
        raise ValueError("missing id")
    return {"ok": True, "id": payload["id"]}
