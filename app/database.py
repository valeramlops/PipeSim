import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Get URL from docker-compose or using default

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@db:5432/pipesim_db"
)

# Create engine (echo=True show SQL-requests in console)
engine = create_async_engine(DATABASE_URL, echo=True)

# Session fabric
async_session = async_sessionmaker(engine, expire_on_commit=False)

# Basic class for all models
Base = declarative_base()