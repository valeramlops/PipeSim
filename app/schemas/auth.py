from sqlalchemy import Column, Integer, String, Boolean, DateTime, func
from app.core.database import Base

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    # Store hash
    key_hash = Column(String, unique=True, index=True, nullable=False)
    # Owner name
    owner_name = Column(String, nullable=False)
    # Access level (predict - read-only   model training - full)
    access_level = Column(String, default="read-only")
    # Can use key
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())  
