import jwt
from cryptography.fernet import Fernet


SECRET_KEY = "hardcoded-secret-not-rotated"


def generate_token(user_ctx) -> str:
    return jwt.encode({"sub": user_ctx.user_id}, SECRET_KEY, algorithm="HS256")


def verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except:
        return None
