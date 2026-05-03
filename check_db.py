import asyncio
from app.database import engine, Base
# Импортируем всё, что должно быть в базе
from app.models import DetectionRecord, VideoRecord
from app.schemas.auth import APIKey
from app.schemas.feature_store import ProcessedFeature
from app.schemas.model_version import ModelVersion
from app.schemas.passenger import Passenger
from app.schemas.prediction import Prediction

async def check():
    print("Проверка зарегистрированных таблиц в Base:")
    print(Base.metadata.tables.keys())
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Команда create_all выполнена.")

asyncio.run(check())