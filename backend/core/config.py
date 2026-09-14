import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────
POSTGRES_URL = os.environ.get("POSTGRES_URL")

# ── Redis ─────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

# ── Auth ──────────────────────────────────────────────────────────────
HASH_SECRET_KEY = os.environ.get("HASH_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 60

# ── External APIs ─────────────────────────────────────────────────────
FREE_LLM_API = os.getenv("FREE_LLM_API")
PINECONE_API_KEY = os.environ.get("pinecone")
