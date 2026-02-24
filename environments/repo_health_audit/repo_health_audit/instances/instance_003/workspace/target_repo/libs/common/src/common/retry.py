def retry_once(fn):
    try:
        return fn()
    except Exception:
        return None
