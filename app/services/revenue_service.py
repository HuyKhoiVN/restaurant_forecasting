import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from app.database import get_db
from app.ml.revenue.prophet_model import ProphetForecaster
from app.ml.revenue.lstm_model import LSTMForecaster
from app.ml.utils import aggregate_data, preprocess_data, apply_pca, apply_kmeans
from sklearn.metrics import mean_absolute_percentage_error

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RevenueForecastService:
    def __init__(self):
        logger.debug("Initializing RevenueForecastService")
        self.prophet = ProphetForecaster()
        self.lstm = LSTMForecaster(holidays=self.prophet.holidays)  # Truyền holidays từ Prophet
        self._load_historical_data()
        self._preprocess_and_train_models()
        self.weights = self._optimize_weights()

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
        raw_data = pd.read_sql(query, db.connection())
        raw_data['TransactionDate'] = pd.to_datetime(raw_data['TransactionDate'])
        all_dates = pd.date_range(start=raw_data['TransactionDate'].min(),
                                  end=raw_data['TransactionDate'].max(), freq='D')
        self.historical_data = pd.DataFrame({'TransactionDate': all_dates})
        self.historical_data = self.historical_data.merge(raw_data, on='TransactionDate', how='left')
        self.historical_data['TotalAmount'] = self.historical_data['TotalAmount'].fillna(0)
        self.historical_data.fillna({'TransactionCount': 0, 'IsHoliday': 0, 'IsWeekend': 0, 'Weather': 'Unknown'},
                                    inplace=True)
        logger.info(f"Loaded {len(self.historical_data)} rows after filling missing dates")

    def _preprocess_and_train_models(self):
        logger.debug("Preprocessing and training models")
        self.historical_data = preprocess_data(self.historical_data)
        features = ['TotalAmount', 'TransactionCount', 'weather_holiday_interaction']
        self.historical_data, variance_ratio = apply_pca(self.historical_data, features)
        logger.info(f"PCA Explained Variance Ratio: {variance_ratio}")
        self.historical_data = apply_kmeans(self.historical_data)

        prophet_data = self.historical_data[
            ['TransactionDate', 'TotalAmount', 'IsHoliday', 'IsWeekend', 'PCA1', 'PCA2']]
        self.prophet.train(prophet_data)
        self.lstm.train(prophet_data)

    def _optimize_weights(self):
        train_data = self.historical_data.iloc[:-30]
        val_data = self.historical_data.iloc[-30:]

        prophet_val = self.prophet.predict(val_data['TransactionDate'].min(), val_data['TransactionDate'].max())
        lstm_val = self.lstm.predict(train_data['TotalAmount'].tail(self.lstm.look_back).values,
                                     val_data['TransactionDate'].min(), val_data['TransactionDate'].max())

        prophet_mape = mean_absolute_percentage_error(val_data['TotalAmount'], prophet_val['yhat'])
        lstm_mape = mean_absolute_percentage_error(val_data['TotalAmount'], lstm_val)

        total_error = prophet_mape + lstm_mape
        w_prophet = lstm_mape / total_error
        w_lstm = prophet_mape / total_error
        logger.info(f"Optimized weights - Prophet: {w_prophet:.2f}, LSTM: {w_lstm:.2f}")
        return {'prophet': w_prophet, 'lstm': w_lstm}

    def generate_forecast(self, start_date: datetime, end_date: datetime, granularity: str = "daily") -> List[Dict]:
        logger.debug(f"Generating forecast from {start_date} to {end_date}")
        max_date = self.historical_data['TransactionDate'].max()
        max_forecast_date = datetime.combine(max_date, datetime.min.time()) + timedelta(days=30)
        if end_date > max_forecast_date:
            logger.info(f"Adjusting end_date from {end_date} to {max_forecast_date}")
            end_date = max_forecast_date

        prophet_results = self.prophet.predict(start_date, end_date)
        lstm_results = self.lstm.predict(self._get_last_values(), start_date, end_date)

        prophet_results['lstm'] = lstm_results
        prophet_results['combined'] = (self.weights['prophet'] * prophet_results['yhat'] +
                                       self.weights['lstm'] * prophet_results['lstm'])
        prophet_results['combined'] = prophet_results['combined'].clip(lower=0)
        forecast = aggregate_data(prophet_results, granularity)
        return self._add_analysis(forecast)

    def _get_last_values(self) -> np.ndarray:
        return self.historical_data['TotalAmount'].tail(self.lstm.look_back).values

    def _add_analysis(self, forecast: pd.DataFrame) -> List[Dict]:
        results = []
        for i, row in forecast.iterrows():
            ds_value = row['ds'].date() if isinstance(row['ds'], pd.Timestamp) else row['ds']
            result = {
                "date": ds_value,
                "predicted_revenue": row['combined'],
                "confidence_interval": {"lower": row['yhat_lower'], "upper": row['yhat_upper']},
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
        if current_index == 0: return "stable"
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
            current_date = current_row['ds']
            prev_date = current_date - timedelta(days=days_back)
            prev_data = self.historical_data[self.historical_data['TransactionDate'] == prev_date]
            if not prev_data.empty:
                prev_value = prev_data['TotalAmount'].iloc[0]
                return ((current_row['combined'] - prev_value) / prev_value) * 100
            return 0.0
        prev_row = all_data.iloc[current_index - days_back]
        return ((current_row['combined'] - prev_row['combined']) / prev_row['combined']) * 100