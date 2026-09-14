import uuid
from pydantic import BaseModel
from datetime import datetime

class SessionResponse(BaseModel):
    id: uuid.UUID
    title: str
    updated_at: datetime

class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime

class ChatRequest(BaseModel):
    session_id: uuid.UUID
    message: str

class ChatResponse(BaseModel):
    response: str
