from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from typing import Dict
from pydantic import BaseModel, Field
from app.services.revenue_service import RevenueForecastService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class ForecastRequest(BaseModel):
    start_date: datetime = Field(..., format="date", description="Start date in YYYY-MM-DD format")
    end_date: datetime = Field(..., format="date", description="End date in YYYY-MM-DD format")
    include_analysis: bool = True
    granularity: str = "daily"

    class Config:
        json_encoders = {
            datetime: lambda v: v.strftime("%Y-%m-%d")
        }


@router.post("/api/revenue/forecast")
async def get_revenue_forecast(request: ForecastRequest) -> Dict:
    try:
        # Loại bỏ múi giờ để đồng bộ với dữ liệu trong DB
        start_date = request.start_date.replace(tzinfo=None)
        end_date = request.end_date.replace(tzinfo=None)

        service = RevenueForecastService()
        forecast, analysis = service.generate_forecast(start_date, end_date, request.granularity)

        response = {
            "status": "success",
            "forecast": forecast,
            "metadata": {
                "model_used": "Prophet + LSTM Ensemble",
                "last_trained": datetime.now().isoformat(),
                "max_forecast_date": (service.historical_data['TransactionDate'].max() + timedelta(days=30)).isoformat()
            }
        }
        if request.include_analysis:
            response["analysis"] = analysis
        return response
    except Exception as e:
        logger.error(f"Error in forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")