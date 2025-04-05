from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from typing import Dict
from pydantic import BaseModel
from app.services.revenue_service import RevenueForecastService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class ForecastRequest(BaseModel):
    start_date: datetime  # Trực tiếp dùng datetime thay vì str
    end_date: datetime  # Trực tiếp dùng datetime thay vì str
    include_analysis: bool = True
    granularity: str = "daily"

    class Config:
        json_encoders = {
            datetime: lambda v: v.strftime("%Y-%m-%d")  # Định dạng datetime trong JSON response
        }


@router.post("/api/revenue/forecast")
async def get_revenue_forecast(request: ForecastRequest) -> Dict:
    try:
        service = RevenueForecastService()
        forecast, analysis = service.generate_forecast(
            request.start_date,
            request.end_date,
            request.granularity
        )

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