from fastapi import Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.security import verify_token
from backend.db.database import AsyncSessionFactory


async def get_current_user_id(authorization: str = Header(...)) -> str:
    """Extract and validate the user_id from the Authorization header."""
    token = authorization.replace("Bearer ", "")
    user_id = verify_token(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id


async def get_db():
    """Yield an async database session, auto-closed on exit."""
    async with AsyncSessionFactory() as session:
        yield session
