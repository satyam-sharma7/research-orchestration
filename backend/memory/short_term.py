import json
import uuid
import redis.asyncio as redis
from typing import List, Dict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Message
from backend.core.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

# Use asynchronous Redis client with connection pooling
redis_client = redis.Redis(
    host=REDIS_HOST,
    username="default",
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)


async def add_to_short_term(session_id: str, role: str, content: str):
    """Add a message to Redis short-term memory asynchronously."""
    key = f"chat:short_term:{session_id}"
    message = json.dumps({"role": role, "content": content})

    # Use a pipeline to ensure rpush, ptrun, and expire happen atomically
    pipeline = redis_client.pipeline()
    pipeline.rpush(key, message)
    pipeline.ltrim(key, -20, -1)  # Keep only the last 20 messages
    pipeline.expire(key, 3600)    # 1 hour TTL
    await pipeline.execute()


async def get_short_term(session_id: str, db: AsyncSession) -> List[Dict[str, str]]:
    """Get messages from Redis; if empty, hydrate from Postgres DB."""
    key = f"chat:short_term:{session_id}"

    # 1. Cache Hit: Return from Redis
    cached_messages = await redis_client.lrange(key, 0, -1)
    if cached_messages:
        return [json.loads(m) for m in cached_messages]

    # 2. Cache Miss: Fetch last 20 messages from Database
    stmt = (
        select(Message)
        .where(Message.session_id == uuid.UUID(session_id))
        .order_by(Message.created_at.desc())
        .limit(20)
    )
    result = await db.execute(stmt)
    db_messages = list(result.scalars().all())

    if not db_messages:
        return []

    # Reverse to keep chronological order (oldest to newest)
    db_messages.reverse()

    # 3. Populate Redis Cache
    serialized_messages = [
        json.dumps({"role": msg.role, "content": msg.content}) for msg in db_messages
    ]

    pipeline = redis_client.pipeline()
    pipeline.rpush(key, *serialized_messages)  # Push all at once
    pipeline.ltrim(key, -20, -1)
    pipeline.expire(key, 3600)
    await pipeline.execute()

    return [{"role": msg.role, "content": msg.content} for msg in db_messages]


async def clear_short_term(session_id: str):
    """Clear Redis short-term memory for a session asynchronously."""
    await redis_client.delete(f"chat:short_term:{session_id}")
