from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, Field
from app.services.product_forecast_service import ProductForecastService
import logging

router = APIRouter(prefix="/api/product", tags=["product_forecast"])
logger = logging.getLogger(__name__)

class ProductForecastRequest(BaseModel):
    date: datetime = Field(..., format="date", description="Date in YYYY-MM-DD format")
    include_ingredients: bool = False
    time_ranges: List[str] = Field(default=None, description="Time ranges: lunch, dinner")

    class Config:
        json_encoders = {datetime: lambda v: v.strftime("%Y-%m-%d")}

@router.post("/forecast")
async def get_product_forecast(request: ProductForecastRequest) -> Dict:
    try:
        date = request.date.replace(tzinfo=None)
        service = ProductForecastService()
        response = service.generate_product_forecast(date, request.include_ingredients, request.time_ranges)
        return response
    except Exception as e:
        logger.error(f"Error in product forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating product forecast: {str(e)}")