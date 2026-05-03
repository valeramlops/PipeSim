from sqlalchemy import Column, Integer, Float, DateTime, JSON
from datetime import datetime
from app.core.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key = True, index = True)
    passenger_data = Column(JSON) # Input data
    prediction_result = Column(Integer) # 0 - Dead or 1 - survived
    probability = Column(Float) # Model confidence
    created_at = Column(DateTime, default=datetime.utcnow)