from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pathlib import Path
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import Optional, List, Dict, Tuple

import uuid

from app.core.database import AsyncSessionLocal, get_db
from app.models.prediction import Prediction as PredictionModel
from app.models.feature_store import ProcessedFeature
from app.api.data import preprocess_dataframe

router = APIRouter()

MODEL_PATH = Path("models/titanic_model.pkl")

jobs: Dict[str, dict] = {}

# For validation input data
class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Cabin: Optional[str] = None

def get_data_hash(data: dict) -> str:
    """
    Generate SHA-256 hash from dict, sorted key for stability
    """
    encoded = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()

@router.post("/status")
def predict_status():
    if not MODEL_PATH.exists():
        return {
            "status": "unavailable",
            "message": "Model not trained. Train model first"
        }
    
    return {
        "status": "available",
        "message": "Prediction service is ready"
    }


async def process_and_predict(passengers: List[PassengerData], db: AsyncSession) -> Tuple[List[Dict], list, list]:
    """
    [CHANGED] Now asynchronous. Checks the Feature Store before doing preprocessing.
    """
    if not MODEL_PATH.exists():
        raise HTTPException (
            status_code=404,
            detail="Model not found. Train the model first"
        )

    model = joblib.load(MODEL_PATH)
    raw_data_list = [p.dict() for p in passengers]

    final_features_list = []

    # 1. Checking cache
    for raw_data in raw_data_list:
        data_hash = get_data_hash(raw_data)

        # Search in db
        result = await db.execute(select(ProcessedFeature).where(ProcessedFeature.raw_data_hash == data_hash))
        cached_feature = result.scalars().first()

        if cached_feature:
            # Take ready features
            final_features_list.append(cached_feature.features)
        else:
            # Not in cache. Doing heavy preprocessing via Pandas
            df_raw = pd.DataFrame([raw_data])
            df_processed = preprocess_dataframe(df_raw)

            feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'isAlone']
            processed_dict = df_processed[feature_columns].iloc[0].to_dict()

            final_features_list.append(processed_dict)

            # Save result in db
            new_cache = ProcessedFeature(raw_data_hash=data_hash, features=processed_dict)
            db.add(new_cache)
            
            # Flush to write to the current transaction without closing her
            await db.flush()

    # 2. Collecting final DataFrame from features (cached + new) 
    X_input = pd.DataFrame(final_features_list)

    # 3. Prediction
    try:
        predictions = model.predict(X_input)
        # Get confidence
        probabilities = model.predict_proba(X_input)[:, 1]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    
    return raw_data_list, predictions.tolist(), probabilities.tolist()

@router.post("/")
async def make_prediction(data: PassengerData, db: AsyncSession = Depends(get_db)):
    """
    Making prediction for one passenger
    """

    # 1. Using process_and_predict function
    raw_data_list, predictions, probabilities = await process_and_predict([data], db)

    # Get results
    prediction = predictions[0]
    probability = probabilities[0]
    raw_input_dict = raw_data_list[0]

    # 2. Write in database
    new_prediction = PredictionModel(
        passenger_data = raw_input_dict,
        prediction_result = int(prediction),
        probability = float(probability)
    )

    db.add(new_prediction)
    await db.commit()
    await db.refresh(new_prediction)

    return {
        "status": "success",
        "prediction_id": new_prediction.id,
        "survived": bool(prediction),
        "probability": float(probability),
        "message": "Passenger survived" if prediction == 1 else "Passenger did not survive"
    }

async def run_batch_prediction(job_id: str, passengers: List[PassengerData]):
    """
    Background task for mass prediction
    """
    jobs[job_id]["status"] = "processing"

    try:
        # Save in database
        async with AsyncSessionLocal() as db:
            raw_data_list, predictions, probabilities = await process_and_predict(passengers, db)
        
            results = []
            db_predictions = []

            for idx, (raw_data, pred, prob) in enumerate(zip(raw_data_list, predictions, probabilities)):
                results.append({
                    "passenger_index": idx,
                    "survived": bool(pred),
                    "probability": float(prob)
                })

                db_predictions.append(PredictionModel(
                    passenger_data = raw_data,
                    prediction_result = int(pred),
                    probability = float(prob)
                ))

            db.add_all(db_predictions)
            await db.commit()

        jobs[job_id]['status'] = "completed"
        jobs[job_id]['result'] = results

    except Exception as e:
        jobs[job_id]['status'] = "failed"
        jobs[job_id]['error'] = str(e)
        print(f"[Background Task Error] Job {job_id} failed: {str(e)}")

@router.post("/batch")
async def make_batch_prediction(passengers: List[PassengerData], background_tasks: BackgroundTasks):
    """
    It accepts a list of passengers and immediately returns a number (job_id).
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet"
        )
    
    # Generating unique ticket number
    job_id = str(uuid.uuid4())

    # Registration task with pending status (in queue)
    jobs[job_id] = {"status": "pending", "result": None}

    # Send function working on background
    background_tasks.add_task(run_batch_prediction, job_id, passengers)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Received {len(passengers)} passengers. Processing started in background"
    }

@router.get("/batch/{job_id}/status")
async def get_batch_status(job_id: str):
    """
    # Get status and claim results by number
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )
    return jobs[job_id]

@router.get("/history")
async def get_prediction_history(db: AsyncSession = Depends(get_db)):
    """
    Get history of all predicts
    """
    result = await db.execute(select(PredictionModel).order_by(desc(PredictionModel.created_at)))
    return result.scalars().all()