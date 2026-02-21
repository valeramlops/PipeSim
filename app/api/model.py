from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.api.data import load_dataframe, preprocess_dataframe
from app.core.database import get_db
from app.models.model_version import ModelVersion

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
        "algorithm": info.get("algorithm") if info else "Unknown",
        "accuracy": info.get("accuracy") if info else None
    }

@router.post("/train")
async def train_model(db: AsyncSession = Depends(get_db)):
    """
    Train model on Titanic dataset with version history
    """

    # 1. Loading and processing data
    try:
        df = load_dataframe()
        df_processed = preprocess_dataframe(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data error: {str(e)}"
        )

    # NaN checking
    if df_processed.isnull().sum().sum() > 0:
        raise HTTPException(
            status_code=500,
            detail="Processed data contain NaN values. Cannot train model"
        )
    
    # 2. Choosing features and targets
    feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp',
    'Parch', 'Fare', 'Embarked', 'FamilySize', 'isAlone']

    X = df_processed[feature_columns]
    y = df_processed['Survived']

    # Splitting into train/test
    X_train, X_test, y_train, y_test = train_test_split (
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 3. Candidate selection (List of models and their settings)
    model_config = [
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            "params": {
                "C": [0.1, 1.0, 10.0] # Regulation strenght
            }
        },
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(
                random_state=42,
            ),
            "params": {
                "n_estimators": [50, 100], # Tree value: 50 or 100
                "max_depth": [5, 10, None], # Tree's depth
                "min_samples_split": [2, 5] # Face-control
            }
        }
    ]

    best_model = None
    best_score = -1
    best_info = {}
    experiment_log = [] # Experiment log to return to the user

    # 4. Model training cycle
    for config in model_config:
        # GridSearch - parameter tuning with cross-validation (cv=3)
        grid = GridSearchCV(config["model"], config["params"], cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)

        # Picking the best model version
        current_best_model = grid.best_estimator_

        # Testing on deployed sample (Test Set)
        y_pred = current_best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Logging result
        experiment_log.append({
            "algorithm": config["name"],
            "best_params": grid.best_params_,
            "accuracy": float(accuracy)
        })

        # Comparison: if this model better than the previouse leader, it becomes the new leader
        if accuracy > best_score:
            best_score = accuracy
            best_model = current_best_model

            # Collecting winner's stats
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            best_info = {
                "trained_at": datetime.utcnow().isoformat(),
                "algorithm": config["name"],
                "best_params": grid.best_params_,
                "features": feature_columns,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "confusion_matrix": conf_matrix.tolist()
            }

    # 5. Saving winner and versioning
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    save_model_info(best_info)
    
    # Make unique name for version history
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version_filename = f"model_{best_info['algorithm']}_{timestamp}.pkl"
    version_path = Path("models") / version_filename
    
    # Save model copy for history
    joblib.dump(best_model, version_path)

    # 6. Write information about new version into DB
    new_version = ModelVersion(
        algorithm = best_info['algorithm'],
        metrics = best_info,
        filepath = str(version_path)
    )
    db.add(new_version)
    await db.commit()
    await db.refresh(new_version)

    return {
        "status": "success",
        "version_id": new_version.id,
        "message": f"Training complete. Winner: {best_info['algorithm']}",
        "winner_metrics": {
            "accuracy": best_info['accuracy'],
            "precision": best_info['precision']
        },
        "experiments": experiment_log
    }

@router.get("/history")
async def get_model_history(db: AsyncSession = Depends(get_db)):
    """
    Get history of all trained models
    """
    # Sort by ID (desc)
    result = await db.execute(select(ModelVersion).order_by(desc(ModelVersion.id)))
    versions = result.scalars().all()
    return versions

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
        "algorithm": info.get("algorithm"),
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