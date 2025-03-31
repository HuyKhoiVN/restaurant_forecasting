import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from app.database import get_db
from app.ml.revenue.prophet_model import ProphetForecaster
from app.ml.revenue.lstm_model import LSTMForecaster
from app.ml.utils import aggregate_data, preprocess_data, apply_pca, apply_kmeans

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RevenueForecastService:
    def __init__(self):
        logger.debug("Initializing RevenueForecastService")
        self.prophet = ProphetForecaster()
        self.lstm = LSTMForecaster()
        self._load_historical_data()
        self._preprocess_and_train_models()

    def _load_historical_data(self):
        logger.debug("Loading historical data")
        db = next(get_db())
        query = """
            SELECT 
                t.TransactionDate,
                SUM(t.TotalAmount) as TotalAmount,
                COUNT(*) as TransactionCount,
                r.IsHoliday,
                r.IsWeekend,
                r.Weather
            FROM [Transaction] t
            LEFT JOIN Revenue r ON t.TransactionDate = r.Date
            GROUP BY t.TransactionDate, r.IsHoliday, r.IsWeekend, r.Weather
            ORDER BY t.TransactionDate
        """
        self.historical_data = pd.read_sql(query, db.connection())
        logger.info(f"Loaded {len(self.historical_data)} rows, TransactionDate type: {type(self.historical_data['TransactionDate'].iloc[0])}")

    def _preprocess_and_train_models(self):
        logger.debug("Preprocessing and training models")
        self.historical_data = preprocess_data(self.historical_data)
        features = ['TotalAmount', 'TransactionCount']
        self.historical_data, variance_ratio = apply_pca(self.historical_data, features)
        logger.info(f"PCA Explained Variance Ratio: {variance_ratio}")
        self.historical_data = apply_kmeans(self.historical_data)
        self.prophet.train(self.historical_data)
        self.lstm.train(self.historical_data)
        logger.info("Models trained successfully")

    def generate_forecast(self, start_date: datetime, end_date: datetime,
                          granularity: str = "daily") -> List[Dict]:
        logger.debug(f"Generating forecast from {start_date} to {end_date}")
        max_date = self.historical_data['TransactionDate'].max()
        # Chuyển max_forecast_date thành datetime.datetime
        max_forecast_date = datetime.combine(max_date, datetime.min.time()) + timedelta(days=30)
        if end_date > max_forecast_date:
            logger.info(f"Adjusting end_date from {end_date} to {max_forecast_date}")
            end_date = max_forecast_date

        prophet_results = self.prophet.predict(start_date, end_date)
        lstm_results = self.lstm.predict(self._get_last_values(), start_date, end_date)

        prophet_results['lstm'] = lstm_results
        prophet_results['combined'] = (prophet_results['yhat'] + prophet_results['lstm']) / 2
        forecast = aggregate_data(prophet_results, granularity)
        return self._add_analysis(forecast)

    def _get_last_values(self) -> np.ndarray:
        return self.historical_data['TotalAmount'].tail(self.lstm.look_back).values

    def _add_analysis(self, forecast: pd.DataFrame) -> List[Dict]:
        logger.debug(f"Forecast ds type: {type(forecast['ds'].iloc[0])}")
        results = []
        for i, row in forecast.iterrows():
            ds_value = row['ds']
            if isinstance(ds_value, pd.Timestamp):
                ds_value = ds_value.date()
            elif isinstance(ds_value, datetime):
                ds_value = ds_value.date()

            result = {
                "date": ds_value,
                "predicted_revenue": row['combined'],
                "confidence_interval": {
                    "lower": row['yhat_lower'],
                    "upper": row['yhat_upper']
                },
                "trend": self._calculate_trend(row, forecast, i),
                "trend_percentage": self._calculate_trend_percentage(row, forecast, i),
                "comparison": {
                    "previous_day": self._compare_with_previous(row, forecast, i, 1),
                    "previous_week": self._compare_with_previous(row, forecast, i, 7),
                    "previous_month": self._compare_with_previous(row, forecast, i, 30)
                }
            }
            results.append(result)
        return results

    def _calculate_trend(self, current_row, all_data, current_index):
        if current_index == 0:
            return "stable"
        prev_row = all_data.iloc[current_index - 1]
        if current_row['combined'] > prev_row['combined'] * 1.05:
            return "increase"
        elif current_row['combined'] < prev_row['combined'] * 0.95:
            return "decrease"
        return "stable"

    def _calculate_trend_percentage(self, current_row, all_data, current_index):
        avg = all_data['combined'].mean()
        return ((current_row['combined'] - avg) / avg) * 100

    def _compare_with_previous(self, current_row, all_data, current_index, days_back):
        if current_index < days_back:
            return 0.0
        prev_row = all_data.iloc[current_index - days_back]
        return ((current_row['combined'] - prev_row['combined']) / prev_row['combined']) * 100

    def generate_analysis(self, forecast_data):
        return {
            "seasonality_patterns": self._detect_seasonality(forecast_data),
            "action_recommendations": self._generate_recommendations(forecast_data)
        }

    def _detect_seasonality(self, forecast_data):
        patterns = []
        weekday_data = [x for x in forecast_data if x['date'].weekday() < 5]
        weekend_data = [x for x in forecast_data if x['date'].weekday() >= 5]

        if weekend_data and weekday_data:
            weekday_avg = np.mean([x['predicted_revenue'] for x in weekday_data])
            weekend_avg = np.mean([x['predicted_revenue'] for x in weekend_data])

            if weekend_avg > weekday_avg * 1.2:
                patterns.append({
                    "pattern": "weekend_high",
                    "impact": round(((weekend_avg / weekday_avg) - 1) * 100, 1)
                })

        return patterns

    def _generate_recommendations(self, forecast_data):
        recommendations = []
        max_day = max(forecast_data, key=lambda x: x['predicted_revenue'])
        avg_revenue = np.mean([x['predicted_revenue'] for x in forecast_data])

        if max_day['predicted_revenue'] > avg_revenue * 1.5:
            recommendations.append({
                "type": "staffing",
                "message": f"Tăng nhân viên vào {max_day['date'].strftime('%A %d/%m')}"
            })

        return recommendations