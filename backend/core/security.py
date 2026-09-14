import jwt
import bcrypt
from datetime import datetime, timedelta, timezone

from backend.core.config import HASH_SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_DAYS


def hash_password(password: str) -> str:
    """Hash a password for storing."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against the stored hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def create_access_token(user_id: str) -> str:
    """Create a JWT access token for an authenticated user."""
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, HASH_SECRET_KEY, ALGORITHM)


def verify_token(token: str) -> str | None:
    """Decode and validate a JWT token. Returns user_id or None."""
    try:
        payload = jwt.decode(token, HASH_SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
