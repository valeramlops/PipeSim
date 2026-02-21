from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.models.prediction import Prediction as PredictionModel

router = APIRouter()

@router.get("/stats")
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    """
    Get general statistics for the monitoring dashboard (Analytics)
    """
    # 1. Create a single query to collect metrics
    query = select(
        func.count(PredictionModel.id).label("total_predictions"),
        func.sum(PredictionModel.prediction_result).label("total_survived"),
        func.avg(PredictionModel.probability).label("average_probability")
    )
    
    # 2. Request
    result = await db.execute(query)
    stats = result.first()

    # 3. Get data
    total = stats.total_predictions or 0
    survived = stats.total_survived or 0
    avg_prob = stats.average_probability or 0

    # 4. Count production metrics
    survival_rate = (survived / total * 100) if total > 0 else 0.0
    perished = total - survived

    # 5. Return JSON for frontend
    return {
        "status": "success",
        "metrics": {
            "total_predictions": total, # Amount of predictions
            "total_survived": survived, # Amount of survived people
            "total_perished": perished, # Amount of dead people
            "survival_rate_percent": round(survival_rate, 2), # Percent of survived people
            "average_confidence": round(avg_prob, 4) # Cofidence avg percent
        }
    }