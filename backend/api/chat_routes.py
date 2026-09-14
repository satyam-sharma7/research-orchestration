import uuid
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.deps import get_db, get_current_user_id
from backend.core.config import FREE_LLM_API
from backend.db.models import Session, Message
from backend.schemas.chat import SessionResponse, MessageResponse, ChatRequest, ChatResponse
from backend.memory.short_term import get_short_term, add_to_short_term, clear_short_term

router = APIRouter(tags=["Chat"])

@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Session)
        .where(Session.user_id == uuid.UUID(user_id))
        .order_by(Session.updated_at.desc())
    )

    sessions = result.scalars().all()
    return [
        {
            "id": s.id,
            "title": s.title,
            "updated_at": s.updated_at
        }
        for s in sessions
    ]

@router.post("/session", response_model=dict)
async def create_session(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    session = Session(user_id=uuid.UUID(user_id))
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return {"session_id": str(session.id)}

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    # Verify ownership
    query = select(Session).where(Session.id == session_id, Session.user_id == uuid.UUID(user_id))
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=403, detail="Not your session")

    await db.delete(session)
    await db.commit()

    # Clear Redis cache
    await clear_short_term(str(session_id))

    return {"status": "deleted"}

@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    session_id: uuid.UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    # Verify ownership
    result = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == uuid.UUID(user_id)
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not your session")

    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )

    messages = result.scalars().all()
    return [
        {
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at
        } for m in messages
    ]

@router.post("/sessions/{session_id}/messages")
async def chat(
    req: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    # Verify session ownership
    result = await db.execute(
        select(Session).where(Session.id == req.session_id, Session.user_id == uuid.UUID(user_id))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=403, detail="Not your session")

    short_term_memory = await get_short_term(str(req.session_id), db)

    async def generate_response():
        full_response = ""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=FREE_LLM_API, base_url="https://freellmapi-seyc.onrender.com/v1")

            messages_paylaod = short_term_memory + [{"role": "user", "content": req.message}]
            response_stream = await client.chat.completions.create(
                model="auto",
                messages=messages_paylaod,
                stream=True
            )

            async for token in response_stream:
                content = token.choices[0].delta.content
                if content:
                    full_response += content
                    yield content

            # Once stream is done, save to DB and Memory
            db.add(Message(session_id=req.session_id, role="user", content=req.message))
            db.add(Message(session_id=req.session_id, role="assistant", content=full_response))
            await db.commit()

            await add_to_short_term(str(req.session_id), "user", req.message)
            await add_to_short_term(str(req.session_id), "assistant", full_response)

            session.updated_at = datetime.now()
            await db.commit()

        except Exception as e:
            yield f"\n[Error generating response: {str(e)}]"

    return StreamingResponse(generate_response(), media_type="text/plain")
