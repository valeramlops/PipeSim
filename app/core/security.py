from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import hashlib

from app.core.database import get_db
from app.schemas.auth import APIKey as APIKeyModel

# FastAPI will add this field to Swagger
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_key_hash(key: str) -> str:
    """
    Hashing key (SHA-256)
    """
    return hashlib.sha256(key.encode()).hexdigest()

async def verify_api_key(
    api_key: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db)
):
    """
    Dependency-function, which stay on endpoint guard
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key is missing in X-API-Key header"
        )
    # Hashing key from header
    hashed_key = get_key_hash(api_key)

    # Searching hash in DB
    result = await db.execute(select(APIKeyModel).where(APIKeyModel.key_hash == hashed_key))
    db_key = result.scalars().first()

    if not db_key or not db_key.is_active:
        raise HTTPException(
            status_code=403,
            detail="Invalid or inactive API Key"
        )
    
    # Return key object so that the endpoint knows who made the request
    return db_key