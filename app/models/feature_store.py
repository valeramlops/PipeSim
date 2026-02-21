from sqlalchemy import Column, Integer, String, JSON, DateTime, func
from app.core.database import Base

class ProcessedFeature(Base):
    """
    Table for caching processed features (Feature Store).
    """
    __tablename__ = "processed_features"

    id = Column(Integer, primary_key=True, index=True)
    # Hash gray data
    raw_data_hash = Column(String, unique=True, index=True, nullable=False)
    # Features dict
    features = Column(JSON, nullable=True)
    # When features was created and added in hash
    created_at = Column(DateTime(timezone=True), server_default=func.now())
