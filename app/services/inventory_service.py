import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from app.database import get_db
# Sửa đổi import để sử dụng file ml.py ở thư mục gốc
from app.ml.ml import InventoryOptimizationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InventoryOptimizationService:
    """
    Service xử lý tối ưu hóa kho hàng
    """

    def __init__(self):
        logger.info("Initializing InventoryOptimizationService")
        self.model = InventoryOptimizationModel()
        self._load_data_and_train()

    def _load_data_and_train(self):
        """
        Tải dữ liệu từ database và huấn luyện mô hình
        """
        try:
            logger.info("Loading data from database")
            db = next(get_db())

            # Tải dữ liệu tồn kho
            inventory_query = """
                SELECT 
                    Date, IngredientId, IngredientName, StockQuantity, UnitCost, ExpirationDate
                FROM Inventory
                ORDER BY Date, IngredientId
            """
            inventory_data = pd.read_sql(inventory_query, db.connection())

            # Tải dữ liệu giao dịch
            transaction_query = """
                SELECT 
                    t.Id as TransactionId, t.TransactionDate, t.TransactionTime, 
                    td.ProductId, td.Quantity
                FROM [Transaction] t
                JOIN TransactionDetail td ON t.Id = td.TransactionId
                ORDER BY t.TransactionDate, t.TransactionTime
            """
            transaction_data = pd.read_sql(transaction_query, db.connection())

            # Tải dữ liệu mapping món ăn - nguyên liệu
            dish_ingredient_query = """
                SELECT 
                    Id, DishId, IngredientId, QuantityPerDish
                FROM DishIngredientMapping
            """
            dish_ingredient_data = pd.read_sql(dish_ingredient_query, db.connection())

            # Tải dữ liệu sản phẩm
            product_query = """
                SELECT 
                    Id, Category, Name, UnitPrice
                FROM Product
            """
            product_data = pd.read_sql(product_query, db.connection())

            # Đảm bảo các cột datetime được xử lý đúng
            inventory_data['Date'] = pd.to_datetime(inventory_data['Date'])
            inventory_data['ExpirationDate'] = pd.to_datetime(inventory_data['ExpirationDate'])
            transaction_data['TransactionDate'] = pd.to_datetime(transaction_data['TransactionDate'])

            # Huấn luyện mô hình
            self.model.train(
                inventory_data,
                transaction_data,
                dish_ingredient_data,
                product_data
            )

            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error during data loading and model training: {str(e)}", exc_info=True)
            # Không raise exception để service vẫn có thể khởi tạo

    def get_current_inventory_status(self) -> List[Dict]:
        """
        Lấy trạng thái hiện tại của kho hàng
        """
        try:
            logger.info("Getting current inventory status")
            db = next(get_db())

            query = """
                WITH LatestInventory AS (
                    SELECT 
                        IngredientId,
                        MAX(Date) as LatestDate
                    FROM Inventory
                    GROUP BY IngredientId
                )
                SELECT 
                    i.IngredientId,
                    i.IngredientName,
                    i.StockQuantity,
                    i.UnitCost,
                    i.ExpirationDate,
                    DATEDIFF(day, GETDATE(), i.ExpirationDate) as DaysToExpire
                FROM Inventory i
                JOIN LatestInventory li ON i.IngredientId = li.IngredientId AND i.Date = li.LatestDate
                ORDER BY i.IngredientId
            """

            inventory_status = pd.read_sql(query, db.connection())

            # Xử lý các giá trị NaT trong ExpirationDate
            inventory_status['DaysToExpire'] = inventory_status['DaysToExpire'].fillna(365)

            return inventory_status.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting inventory status: {str(e)}", exc_info=True)
            return []

    def get_ingredient_usage_history(self, ingredient_id: str, days: int = 30) -> List[Dict]:
        """
        Lấy lịch sử sử dụng nguyên liệu
        """
        try:
            logger.info(f"Getting usage history for ingredient {ingredient_id}")
            db = next(get_db())

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            query = f"""
                SELECT 
                    t.TransactionDate,
                    SUM(td.Quantity * dim.QuantityPerDish) as IngredientUsed
                FROM [Transaction] t
                JOIN TransactionDetail td ON t.Id = td.TransactionId
                JOIN DishIngredientMapping dim ON td.ProductId = dim.DishId
                WHERE dim.IngredientId = '{ingredient_id}'
                    AND t.TransactionDate BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY t.TransactionDate
                ORDER BY t.TransactionDate
            """

            usage_history = pd.read_sql(query, db.connection())

            # Đảm bảo có dữ liệu cho tất cả các ngày
            all_dates = pd.date_range(start=start_date, end=end_date)
            full_history = pd.DataFrame({'TransactionDate': all_dates})

            full_history = full_history.merge(
                usage_history,
                left_on='TransactionDate',
                right_on='TransactionDate',
                how='left'
            )

            full_history['IngredientUsed'] = full_history['IngredientUsed'].fillna(0)

            # Chuyển đổi datetime thành string để JSON serialization
            full_history['TransactionDate'] = full_history['TransactionDate'].dt.strftime('%Y-%m-%d')

            return full_history.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting ingredient usage history: {str(e)}", exc_info=True)
            return []

    def optimize_inventory(self, ingredient_id: str, forecast_days: int = 30,
                           budget: Optional[float] = None) -> Dict:
        """
        Tối ưu hóa kho hàng cho một nguyên liệu
        """
        try:
            logger.info(f"Optimizing inventory for ingredient {ingredient_id}")

            # Lấy trạng thái hiện tại của nguyên liệu
            inventory_status = self.get_current_inventory_status()
            ingredient_status = next((item for item in inventory_status if item['IngredientId'] == ingredient_id), None)

            if ingredient_status is None:
                raise ValueError(f"Ingredient {ingredient_id} not found in inventory")

            current_stock = ingredient_status['StockQuantity']
            current_cost = ingredient_status['UnitCost']
            days_to_expire = ingredient_status['DaysToExpire']

            # Nếu không có ngân sách, giả định là không giới hạn
            if budget is None:
                budget = float('inf')

            # Tạo dự báo nhu cầu
            future_dates = pd.date_range(start=datetime.now().date(), periods=forecast_days)
            demand_forecast = self.model.predict_demand(
                ingredient_id,
                future_dates,
                current_stock,
                current_cost,
                days_to_expire
            )

            # Tạo dự báo giá
            price_forecast = self.model.predict_price(ingredient_id, forecast_days)

            # Tối ưu hóa kho hàng
            optimization_result = self.model.optimize_inventory(
                ingredient_id,
                demand_forecast,
                price_forecast,
                current_stock,
                days_to_expire,
                budget
            )

            # Tạo khuyến nghị
            recommendations = self.model.generate_recommendations(ingredient_id, optimization_result)

            # Tạo kết quả
            result = {
                'ingredient_id': ingredient_id,
                'ingredient_name': ingredient_status['IngredientName'],
                'current_stock': float(current_stock),
                'current_cost': float(current_cost),
                'days_to_expire': int(days_to_expire) if days_to_expire is not None else None,
                'forecast_days': forecast_days,
                'demand_forecast': [float(x) for x in demand_forecast],
                'price_forecast': [float(x) for x in price_forecast],
                'optimization_result': {
                    'eoq': optimization_result['eoq'],
                    'rop': optimization_result['rop'],
                    'safety_stock': optimization_result['safety_stock'],
                    'inventory_plan': optimization_result['inventory_plan']
                },
                'recommendations': recommendations
            }

            return result
        except Exception as e:
            logger.error(f"Error optimizing inventory: {str(e)}", exc_info=True)
            raise

    def optimize_all_inventory(self, forecast_days: int = 30,
                               total_budget: Optional[float] = None) -> Dict:
        """
        Tối ưu hóa kho hàng cho tất cả nguyên liệu
        """
        try:
            logger.info("Optimizing inventory for all ingredients")

            # Lấy trạng thái hiện tại của kho hàng
            inventory_status = self.get_current_inventory_status()

            # Nếu không có ngân sách, giả định là không giới hạn
            if total_budget is None:
                total_budget = float('inf')

            # Phân bổ ngân sách theo tỷ lệ giá trị tồn kho
            total_inventory_value = sum(item['StockQuantity'] * item['UnitCost'] for item in inventory_status)

            results = []
            for item in inventory_status:
                ingredient_id = item['IngredientId']

                # Phân bổ ngân sách
                if total_inventory_value > 0:
                    ingredient_value = item['StockQuantity'] * item['UnitCost']
                    ingredient_budget = total_budget * (ingredient_value / total_inventory_value)
                else:
                    ingredient_budget = total_budget / len(inventory_status)

                try:
                    # Tối ưu hóa kho hàng cho nguyên liệu
                    result = self.optimize_inventory(ingredient_id, forecast_days, ingredient_budget)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error optimizing inventory for ingredient {ingredient_id}: {str(e)}")
                    # Tiếp tục với nguyên liệu tiếp theo

            # Tổng hợp kết quả
            summary = {
                'total_ingredients': len(results),
                'total_current_value': sum(result['current_stock'] * result['current_cost'] for result in results),
                'total_recommended_orders': sum(
                    len([plan for plan in result['optimization_result']['inventory_plan']
                         if plan['order_quantity'] > 0])
                    for result in results
                ),
                'total_stockout_risks': sum(
                    len([plan for plan in result['optimization_result']['inventory_plan']
                         if plan['remaining_stock'] < plan['demand']])
                    for result in results
                )
            }

            return {
                'summary': summary,
                'results': results
            }
        except Exception as e:
            logger.error(f"Error optimizing all inventory: {str(e)}", exc_info=True)
            raise

    def get_inventory_insights(self) -> Dict:
        """
        Tạo các insights về kho hàng
        """
        try:
            logger.info("Generating inventory insights")

            # Lấy trạng thái hiện tại của kho hàng
            inventory_status = self.get_current_inventory_status()

            # Tính toán các chỉ số
            total_value = sum(item['StockQuantity'] * item['UnitCost'] for item in inventory_status)
            expiring_soon = [item for item in inventory_status if item['DaysToExpire'] <= 7]
            low_stock = [item for item in inventory_status if item['StockQuantity'] <= 10]  # Ngưỡng tùy chỉnh

            # Sắp xếp nguyên liệu theo giá trị
            sorted_by_value = sorted(
                inventory_status,
                key=lambda x: x['StockQuantity'] * x['UnitCost'],
                reverse=True
            )

            # Top 5 nguyên liệu có giá trị cao nhất
            top_value_ingredients = sorted_by_value[:5]

            insights = {
                'total_ingredients': len(inventory_status),
                'total_value': float(total_value),
                'expiring_soon_count': len(expiring_soon),
                'expiring_soon_items': expiring_soon,
                'low_stock_count': len(low_stock),
                'low_stock_items': low_stock,
                'top_value_ingredients': top_value_ingredients
            }

            return insights
        except Exception as e:
            logger.error(f"Error getting inventory insights: {str(e)}", exc_info=True)
            return {
                'total_ingredients': 0,
                'total_value': 0,
                'expiring_soon_count': 0,
                'expiring_soon_items': [],
                'low_stock_count': 0,
                'low_stock_items': [],
                'top_value_ingredients': []
            }
