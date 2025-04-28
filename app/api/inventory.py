from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import logging
from app.services.inventory_service import InventoryOptimizationService

router = APIRouter()
logger = logging.getLogger(__name__)


class InventoryOptimizationRequest(BaseModel):
    ingredient_id: str = Field(..., description="Mã nguyên liệu cần tối ưu hóa")
    forecast_days: int = Field(30, description="Số ngày dự báo")
    budget: Optional[float] = Field(None, description="Ngân sách tối đa (VND)")


class AllInventoryOptimizationRequest(BaseModel):
    forecast_days: int = Field(30, description="Số ngày dự báo")
    total_budget: Optional[float] = Field(None, description="Tổng ngân sách tối đa (VND)")


@router.get("/status")
async def get_inventory_status() -> Dict:
    """
    Lấy trạng thái hiện tại của kho hàng
    """
    try:
        service = InventoryOptimizationService()
        inventory_status = service.get_current_inventory_status()

        return {
            "status": "success",
            "data": inventory_status,
            "metadata": {
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting inventory status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting inventory status: {str(e)}")


@router.get("/usage/{ingredient_id}")
async def get_ingredient_usage(
        ingredient_id: str,
        days: int = Query(30, description="Số ngày lịch sử")
) -> Dict:
    """
    Lấy lịch sử sử dụng nguyên liệu
    """
    try:
        service = InventoryOptimizationService()
        usage_history = service.get_ingredient_usage_history(ingredient_id, days)

        return {
            "status": "success",
            "data": usage_history,
            "metadata": {
                "ingredient_id": ingredient_id,
                "days": days,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting ingredient usage: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting ingredient usage: {str(e)}")


@router.post("/optimize")
async def optimize_inventory(request: InventoryOptimizationRequest) -> Dict:
    """
    Tối ưu hóa kho hàng cho một nguyên liệu
    """
    try:
        service = InventoryOptimizationService()
        result = service.optimize_inventory(
            request.ingredient_id,
            request.forecast_days,
            request.budget
        )

        return {
            "status": "success",
            "data": result,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_used": "XGBoost + Linear Programming"
            }
        }
    except ValueError as e:
        logger.error(f"Value error in inventory optimization: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing inventory: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error optimizing inventory: {str(e)}")


@router.post("/optimize-all")
async def optimize_all_inventory(request: AllInventoryOptimizationRequest) -> Dict:
    """
    Tối ưu hóa kho hàng cho tất cả nguyên liệu
    """
    try:
        service = InventoryOptimizationService()
        result = service.optimize_all_inventory(
            request.forecast_days,
            request.total_budget
        )

        return {
            "status": "success",
            "data": result,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_used": "XGBoost + Linear Programming"
            }
        }
    except Exception as e:
        logger.error(f"Error optimizing all inventory: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error optimizing all inventory: {str(e)}")


@router.get("/insights")
async def get_inventory_insights() -> Dict:
    """
    Lấy các insights về kho hàng
    """
    try:
        service = InventoryOptimizationService()
        insights = service.get_inventory_insights()

        return {
            "status": "success",
            "data": insights,
            "metadata": {
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting inventory insights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting inventory insights: {str(e)}")
