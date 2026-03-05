from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

import secrets

from app.core.database import get_db
from app.schemas.auth import APIKey
from app.core.security import get_key_hash

router = APIRouter()

@router.post("/keys")
async def generate_api_key(owner_name: str, access_level: str = "read-only", db: AsyncSession = Depends(get_db)):
    # Generating 32-digit security key
    raw_key = secrets.token_urlsafe(32)

    # Hash for database
    hashed_key = get_key_hash(raw_key)

    # Save only hash
    new_key = APIKey(
        key_hash=hashed_key,
        owner_name=owner_name,
        access_level=access_level
    )
    db.add(new_key)
    await db.commit()

    return {
        "message": "Save this key! It will not be show again",
        "api_key": raw_key,
        "owner": owner_name,
        "access_level": access_level
    }