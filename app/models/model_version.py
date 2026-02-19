from sqlalchemy import Column, Integer, String, DateTime, JSON, Float
from datetime import datetime
from app.core.database import Base

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    algorithm = Column(String) # Winner name
    metrics = Column(JSON) 
    filepath = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    