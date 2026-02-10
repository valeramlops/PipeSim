from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def predict():
    return {
        "prediction": None,
        "message": "Prediction endpoint is not implemented yet"
    }
