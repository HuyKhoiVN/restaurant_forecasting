import logging
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProphetForecaster:
    def __init__(self, model=None):
        logger.debug("Initializing ProphetForecaster")
        if model is None:
            self.model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                uncertainty_samples=500
            )
        else:
            self.model = model
        self.max_date = None

    def train(self, historical_data):
        logger.info("Starting Prophet training")
        prophet_df = historical_data.rename(columns={
            'TransactionDate': 'ds',
            'TotalAmount': 'y',
            'IsHoliday': 'holiday'
        })
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        self.max_date = prophet_df['ds'].max()

        holidays = prophet_df[prophet_df['holiday'] == 1][['ds']].drop_duplicates()
        holidays['holiday'] = 'holiday_event'
        holidays['lower_window'] = 0
        holidays['upper_window'] = 1

        self.model.add_country_holidays(country_name='VN')
        self.model.holidays = holidays

        self.model.fit(prophet_df)
        logger.info("Prophet training completed")

    def predict(self, start_date, end_date):
        logger.debug(f"Predicting from {start_date} to {end_date}")
        max_forecast_date = self.max_date + timedelta(days=30)
        if end_date > max_forecast_date.to_pydatetime():
            end_date = max_forecast_date.to_pydatetime()

        periods = (end_date - start_date).days + 1
        if periods <= 0:
            logger.error(f"Invalid date range: start_date {start_date} > end_date {end_date}")
            raise ValueError("start_date must be before or equal to end_date")

        # Tạo future DataFrame với ngày chính xác từ start_date
        future = pd.DataFrame({
            'ds': pd.date_range(start=start_date, periods=periods, freq='D')
        })
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]