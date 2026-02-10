from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def model_status():
    return {
        "status": "not_trained",
        "message": "Model is not trained yet"
    }
