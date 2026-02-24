from services.common.events import publish_event
from services.auth.store import get_user, save_session

# Hardcoded production endpoint — no multi-env support
AUTH_ENDPOINT = "https://auth.prod.beacon.internal:443"
SESSION_TTL = 3600


def login(username: str, password: str) -> dict | None:
    # Bare except — inconsistent error handling
    try:
        user = get_user(username)
        if user is None:
            return None
        session = save_session(user["id"])
        publish_event("user.login", {"user_id": user["id"]})
        return {"session_id": session, "user": username}
    except:
        return None


def logout(session_id: str) -> bool:
    publish_event("user.logout", {"session_id": session_id})
    return True
