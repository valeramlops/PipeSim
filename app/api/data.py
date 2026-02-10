from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def data_status():
    return {
        "status": "not_loaded",
        "message": "Dataset is not loaded yet"
    }
