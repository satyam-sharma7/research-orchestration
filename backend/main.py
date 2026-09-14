from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.auth_routes import router as auth_router
from backend.api.chat_routes import router as chat_router

app = FastAPI(title="Research Orchestration API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth")
app.include_router(chat_router, prefix="/api/chat")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
