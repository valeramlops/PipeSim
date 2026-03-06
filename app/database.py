import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator

# Get URL from docker-compose or using default

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@db:5432/pipesim_db"
)

# Create engine (echo=True show SQL-requests in console)
engine = create_async_engine(DATABASE_URL, echo=True)

# Session fabric
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Basic class for all models
Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Create new session for each request and garantly close her after preprocessing end
    """
    async with async_session() as session:
        yield session