import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON
from app.database import Base

class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    result_json = Column(JSON, nullable=False) # Here is bbox, class and conf
    created_at = Column(DateTime, default=datetime.utcnow)

class VideoRecord(Base):
    __tablename__ = "video_records"

    id = Column(String, primary_key=True) # This is task.id form Celery
    filename = Column(String,nullable=False) # Source file name
    result_path = Column(String, nullable=True) # File path
    status = Column(String, default="processing") # Statuses
    created_at = Column(DateTime, default=datetime.utcnow)