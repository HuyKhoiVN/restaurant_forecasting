from fastapi import APIRouter, Depends, HTTPException
from datetime import date, datetime, timedelta
from typing import Optional
from pydantic import BaseModel, validator
from app.services.revenue_service import RevenueForecastService

router = APIRouter()


class RevenueForecastInput(BaseModel):
    start_date: date
    end_date: date
    include_analysis: bool = True
    granularity: str = "daily"

    @validator("end_date")
    def validate_end_date(cls, v, values):
        if v < values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class RevenueForecastOutput(BaseModel):
    status: str
    forecast: list
    analysis: Optional[dict]
    metadata: dict


@router.post("/forecast", response_model=RevenueForecastOutput)
async def get_revenue_forecast(input: RevenueForecastInput):
    try:
        service = RevenueForecastService()
        # Chuyển date thành datetime
        start_date = datetime.combine(input.start_date, datetime.min.time())
        end_date = datetime.combine(input.end_date, datetime.min.time())

        forecast = service.generate_forecast(
            start_date=start_date,
            end_date=end_date,
            granularity=input.granularity
        )

        analysis = service.generate_analysis(forecast) if input.include_analysis else None

        metadata = {
            "model_used": "Prophet + LSTM Ensemble",
            "last_trained": datetime.now().isoformat(),
            "max_forecast_date": (service.historical_data['TransactionDate'].max() + timedelta(days=30)).isoformat()
        }

        return {
            "status": "success",
            "forecast": forecast,
            "analysis": analysis,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))