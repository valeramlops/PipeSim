from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from app.api.data import load_dataframe, preprocess_dataframe

router = APIRouter()

MODEL_PATH = Path("models/titanic_model.pkl")
MODEL_INFO_PATH = Path("models/model_info.pkl")

def get_model_info():
    """
    Loading information about model
    """
    if not MODEL_INFO_PATH.exists():
        return None
    return joblib.load(MODEL_INFO_PATH)

def save_model_info(info: dict):
    """
    Saving information about model
    """
    MODEL_INFO_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(info, MODEL_INFO_PATH)

@router.get("/status")
async def model_status():
    if not MODEL_PATH.exists():
        return {
            "status": "not_trained",
            "message": "Model has not been trained yet"
        }
    
    info = get_model_info()
    return {
        "status": "trained",
        "trained_at": info.get("trained_at") if info else None,
        "accuracy": info.get("accuracy") if info else None
    }

@router.post("/train")
def train_model():
    """
    Train model on Titanic dataset
    """

    # 1. Loading and processing data
    df = load_dataframe()
    df_processed = preprocess_dataframe(df)

    # 2. NaN checking
    if df_processed.isnull().sum().sum() > 0:
        raise HTTPException(
            status_code=500,
            detail="Processed data contain NaN values. Cannot train model"
        )
    
    # 3. Choosing features and targets
    feature_columns = ['Pclass', 'Sex', 'Age',
    'Parch', 'Fare', 'Embarked', 'FamilySize', 'isAlone']

    X = df_processed[feature_columns]
    y = df_processed['Survived']

    # 4. Splitting into train/test
    X_train, X_test, y_train, y_test = train_test_split (
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 5. Training model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 6. Quality assessment
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 7. Saving model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    # 8. Saving metadata
    model_info = {
        "trained_at": datetime.utcnow().isoformat(),
        "algorithm": "LogisticRegression",
        "features": feature_columns,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": conf_matrix.tolist()
    }

    save_model_info(model_info)

    return {
        "status": "success",
        "message": "Model trained successfully",
        "metrix": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

@router.get("/info")
def model_info():
    """
    Getting information about model
    """
    if not MODEL_PATH.exists():
        raise HTTPException (
            status_code=404,
            detail="Model not found. Train the model first"
        )
    
    info = get_model_info()
    if not info:
        raise HTTPException (
            status_code=404,
            detail="Model info not found"
        )
    
    return info

@router.get("/metrics")
def model_metrics():
    """
    Getting model metrics
    """
    info = get_model_info()

    if not info:
        raise HTTPException (
            status_code=404,
            detail="Model not trained yet"
        )
    
    conf_matrix = info.get("confusion_matrix", [[0, 0], [0, 0]])

    return {
        "accuracy": info.get("accuracy"),
        "precision": info.get("precision"),
        "recall": info.get("recall"),
        "confusion_matrix": {
            "true_negative": conf_matrix[0][0],
            "false_positive": conf_matrix[0][1],
            "false_negative": conf_matrix[1][0],
            "true_positive": conf_matrix[1][1]
        }
    }

