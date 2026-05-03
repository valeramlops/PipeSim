from sqlalchemy import Column, Integer, String, Float, Boolean
from app.core.database import Base

class Passenger(Base):
    __tablename__ = "passengers"

    id = Column(Integer, primary_key=True, index=True)
    PassengerId = Column(Integer, unique=True, index=True)
    Survived = Column(Integer)
    Pclass = Column(Integer)
    Name = Column(String)
    Sex = Column(String)
    Age = Column(Float, nullable=True)
    SibSp = Column(Integer)
    Parch = Column(Integer)
    Ticket = Column(String)
    Fare = Column(Float)
    Cabin = Column(String, nullable=True)
    Embarked = Column(String, nullable=True)
    is_processed = Column(Boolean, default=False)