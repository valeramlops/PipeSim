from fastapi import APIRouter, HTTPException
import pandas as pd
from pathlib import Path

router = APIRouter()

DATA_PATH = Path("data/train.csv")

def load_dataframe() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Place train.csv in /data folder"
        )
    return pd.read_csv(DATA_PATH)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processing data for ML:
    - Filling gaps
    - encoding categories
    - creating features
    """
    df = df.copy() # Copy for not changing original DF

    #1. Filling gaps in digital columns
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    #2. Filling gaps in categorical columns
    df["Embarked"] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Cabin'] = df['Cabin'].fillna('Unknown')

    #3. Encoding categories features
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    #4. Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['isAlone'] = (df['FamilySize'] == 1).astype(int)

    return df


@router.get("/status")
def data_status():
    if not DATA_PATH.exists():
        return {
            "status": "not_loaded",
            "message": "Dataset not found"
        }
    df = pd.read_csv(DATA_PATH)
    return {
        "status": "loaded",
        "rows": len(df),
        "columns": list(df.columns)
    }

@router.get("/schema")
def data_schema():
    df = load_dataframe()
    return {
        "columns": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        }
    }

@router.get("/preview")
def data_preview(limit: int = 10):
    df = load_dataframe()
    return {
        "preview": df.head(limit).fillna("").to_dict(orient="records")
    }

@router.get("/stats")
def data_stats():
    df = load_dataframe()
    stats = df.describe(include="all").fillna("").to_dict()
    return {
        "statistics": stats
    }

@router.get("/processed")
def data_processed(limit: int = 10):
    """
    Returns processed data ready for ML
    """

    df = load_dataframe()
    df_processed = preprocess_dataframe(df)

    return {
        "original_shape": {"rows": len(df), "columns": len(df.columns)},
        "processed_shape": {"rows": len(df_processed), "columns": len(df_processed.columns)},
        "new_features": ["FamilySize", "IsAlone"],
        "preview": df_processed.head(limit).fillna("").to_dict(orient="records")
    }

@router.get("/missing")
def data_missing():
    """
    Comparison gaps before and after processing
    """

    df = load_dataframe()
    df_processed = preprocess_dataframe(df)

    missing_before = df.isnull().sum()
    missing_after = df_processed.isnull().sum()

    comparison = {
        col: {
            "before": int(missing_before[col]),
            "after": int(missing_after[col])
        }

        for col in df.columns
        if missing_before[col] > 0
    }

    return {
        "total_before": int(missing_before.sum()),
        "total_after": int(missing_after.sum()),
        "by_column": comparison
    }