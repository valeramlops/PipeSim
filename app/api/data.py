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
    return {
        "statistics": df.describe(include="all").fillna("").to_dict()
    }