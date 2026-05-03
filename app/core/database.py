from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

# DB engine
engine = create_async_engine(settings.DATABASE_URL, echo=True)

# Session fabric
AsyncSessionLocal = sessionmaker (
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Basic class for models
Base = declarative_base()

# Function for getting session in API (Dependency Injection)
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session