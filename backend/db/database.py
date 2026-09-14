from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from backend.core.config import POSTGRES_URL

async_engine = create_async_engine(url=POSTGRES_URL, pool_size=10, max_overflow=20)
AsyncSessionFactory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
