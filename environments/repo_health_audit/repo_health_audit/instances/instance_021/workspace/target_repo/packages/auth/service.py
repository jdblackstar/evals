from packages.shared.types import UserContext
from packages.auth.tokens import generate_token


def authenticate(username: str, password: str) -> dict | None:
    # No structured error handling — bare except
    try:
        if not username:
            raise ValueError("missing username")
        ctx = UserContext(user_id=username)
        token = generate_token(ctx)
        return {"token": token, "user": username}
    except:
        return None


def revoke_session(session_id: str) -> bool:
    return True
