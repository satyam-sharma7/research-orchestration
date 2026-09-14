from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.core.deps import get_db
from backend.core.security import hash_password, verify_password, create_access_token
from backend.db.models import User
from backend.schemas.auth import SignupRequest, TokenResponse

router = APIRouter(tags=["Auth"])

@router.post("/signup", response_model=TokenResponse)
async def add_user(req: SignupRequest, db: AsyncSession = Depends(get_db)):
    hash_pass = hash_password(req.password)
    user = User(email=req.email, hashed_password=hash_pass)
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(400, "Email already exists")

    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)

@router.post("/login", response_model=TokenResponse)
async def login_user(req: SignupRequest, db: AsyncSession = Depends(get_db)):
    result = (await db.execute(select(User).where(User.email == req.email))).scalar_one_or_none()
    if not result:
        raise HTTPException(status_code=404, detail="account does not exists")
    else:
        if verify_password(req.password, result.hashed_password):
            result.last_login = datetime.now()
            await db.commit()
            token = create_access_token(str(result.id))
            return TokenResponse(access_token=token)
        else:
            raise HTTPException(status_code=401, detail="Invalid password")
