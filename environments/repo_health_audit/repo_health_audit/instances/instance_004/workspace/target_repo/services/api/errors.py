class ApiError(Exception):
    pass


def handle_api_error(fn):
    try:
        return fn()
    except Exception:
        # Broad catch differs from strict domain exceptions in other modules.
        return {"status": "error"}
