from fastapi import APIRouter, HTTPException
from pathlib import Path
from pydantic import BaseModel
import pandas as pd
import joblib

router = APIRouter()

MODEL_PATH = Path("models/titanic_model.pkl")


class PredictionInput(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int
    FamilySize: int
    isAlone: int

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
def make_prediction(data: PredictionInput):
    """
    Making prediction
    """

    if not MODEL_PATH.exists():
        raise HTTPException (
            status_code=404,
            detail="Model not found. Train the model first"
        )
    
    # Loading model
    model = joblib.load(MODEL_PATH)

    # Preprocessing data
    input_data = pd.DataFrame([data.dict()])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    return {
        "survived": int(prediction),
        "probability": {
            "died": float(probability[0]),
            "survived": float(probability[1])
        }
    }