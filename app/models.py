import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, JSON
from app.database import Base

class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    result_json = Column(JSON, nullable=False) # Here is bbox, class and conf
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))