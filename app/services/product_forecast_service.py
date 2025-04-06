import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from app.database import get_db
from app.ml.product.xgboost_model import ProductForecaster
from app.ml.revenue.prophet_model import ProphetForecaster

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ProductForecastService:
    def __init__(self):
        logger.debug("Initializing ProductForecastService")
        prophet = ProphetForecaster()
        self.product_forecaster = ProductForecaster(holidays=prophet.holidays)
        self._load_historical_data()
        self._load_product_data()
        self._train_model()

    def _load_historical_data(self):
        logger.debug("Loading historical data for product forecast")
        db = next(get_db())
        query = """
            SELECT 
                t.TransactionDate,
                t.TransactionTime,
                td.ProductId AS ProductID,
                td.Quantity,
                r.IsHoliday,
                r.IsWeekend,
                r.Weather
            FROM [TransactionDetail] td
            INNER JOIN [Transaction] t ON td.TransactionId = t.Id
            LEFT JOIN [Revenue] r ON t.TransactionDate = r.Date
            ORDER BY t.TransactionDate
        """
        raw_data = pd.read_sql(query, db.connection())
        raw_data['TransactionDate'] = pd.to_datetime(raw_data['TransactionDate'])
        raw_data['TransactionTime'] = pd.to_timedelta(raw_data['TransactionTime'].astype(str))

        self.historical_data = raw_data
        logger.info(f"Loaded {len(self.historical_data)} rows of historical product data")
        self.product_forecaster.historical_data = raw_data

    def _load_product_data(self):
        db = next(get_db())
        self.products = pd.read_sql("SELECT Id, Category, Name FROM Product", db.connection())
        self.ingredients = pd.read_sql(
            "SELECT DishId AS ProductID, IngredientId, QuantityPerDish AS QuantityNeeded, 'kg' AS Unit FROM DishIngredientMapping",
            db.connection()
        )

    def _train_model(self):
        logger.debug("Training product forecast model")
        self.product_forecaster.train(self.historical_data)

    def generate_product_forecast(self, date: datetime, include_ingredients: bool = False,
                                  time_ranges: List[str] = None) -> Dict:
        logger.debug(f"Generating product forecast for {date}")
        product_predictions = self.product_forecaster.predict(date, time_ranges, self.products)

        if include_ingredients:
            for pred in product_predictions:
                product_id = pred['product_id']
                ingredients = self.ingredients[self.ingredients['ProductID'] == product_id]
                pred['ingredient_requirements'] = [
                    {
                        'ingredient_id': row['IngredientId'],
                        'quantity_needed': float(row['QuantityNeeded'] * pred['predicted_quantity']),
                        'unit': row['Unit']
                    }
                    for _, row in ingredients.iterrows()
                ]
        else:
            for pred in product_predictions:
                pred['ingredient_requirements'] = []

        menu_recommendations = self._analyze_trends(date)

        return {
            'status': 'success',
            'forecast': product_predictions,
            'menu_recommendations': menu_recommendations,
            'metadata': {
                'model_used': 'XGBoost',
                'last_trained': self.product_forecaster.last_trained.isoformat()
            }
        }

    def _analyze_trends(self, date: datetime) -> Dict:
        last_7_days = self.historical_data[
            (self.historical_data['TransactionDate'] > date - timedelta(days=7)) &
            (self.historical_data['TransactionDate'] <= date)
            ]
        trends = last_7_days.groupby('ProductID')['Quantity'].mean().to_dict()
        increase, decrease = [], []

        for product_id, avg_qty in trends.items():
            if product_id == 0:
                continue
            historical_avg = self.historical_data[
                self.historical_data['ProductID'] == product_id
                ]['Quantity'].mean()
            if pd.isna(historical_avg) or historical_avg == 0:
                continue
            if avg_qty > historical_avg * 1.25:
                increase.append({
                    'product_id': int(product_id),
                    'reason': f"Xu hướng tăng {(avg_qty / historical_avg - 1) * 100:.0f}%"
                })
            elif avg_qty < historical_avg * 0.75 and avg_qty > 0:
                decrease.append({
                    'product_id': int(product_id),
                    'reason': "Bán chậm trong 7 ngày qua"
                })

        return {'increase': increase, 'decrease': decrease}