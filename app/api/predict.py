from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
from pydantic import BaseModel
import pandas as pd
import joblib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Select, desc
from typing import Optional

from app.core.database import get_db
from app.models.prediction import Prediction as PredictionModel
from app.api.data import preprocess_dataframe

router = APIRouter()

MODEL_PATH = Path("models/titanic_model.pkl")

# For validation input data
class PassengerData(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int
    FamilySize: int
    isAlone: int
    Cabin: Optional[str] = None

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

@router.post("/")
async def make_prediction(data: PassengerData, db: AsyncSession = Depends(get_db)):
    """
    Making prediction and save in database
    """

    if not MODEL_PATH.exists():
        raise HTTPException (
            status_code=404,
            detail="Model not found. Train the model first"
        )
    
    # 1. Loading model
    model = joblib.load(MODEL_PATH)

    # 2. Convert the raw user data into a DataFrame
    raw_input_dict = data.dict()
    df_raw = pd.DataFrame([raw_input_dict])

    # 3. We run the data through a single preprocessing pipeline
    try:
        df_processed = preprocess_dataframe(df_raw)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing error: {str(e)}"
        )
    
    # 4. Make sure that we select exactly the features that the model needs
    feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'isAlone']

    missing_cols = [col for col in feature_columns if col not in df_processed.columns]
    if missing_cols:
        raise HTTPException(
            status_code=500, detail=f"Missing processed columns: {missing_cols}"
        )
    
    X_input = df_processed[feature_columns]

    # 5. Make predict
    try:
        prediction = model.predict(X_input)[0]
        # Get confidence
        probability = model.predict_proba(X_input)[0][1]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    
    # 6. Write in database
    new_prediction = PredictionModel(
        passenger_data = raw_input_dict,
        prediction_result = int(prediction),
        probability = float(probability)
    )
    db.add(new_prediction)
    await db.commit()
    await db.refresh(new_prediction)

    # 7. Return results to user
    return {
        "status": "success",
        "prediction_id": new_prediction.id,
        "survived": bool(prediction),
        "probability": float(probability),
        "message": "Passenger survived" if prediction == 1 else "Passenger did not survive"
    }

@router.get("/history")
async def get_prediction_history(db: AsyncSession = Depends(get_db)):
    """
    Get history of all prediction
    """
    # Sort by new to old
    result = await db.execute(Select(PredictionModel).order_by(desc(PredictionModel.created_at)))
    predictions = result.scalars().all()
    return predictions