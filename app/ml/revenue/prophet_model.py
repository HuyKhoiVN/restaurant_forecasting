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
                uncertainty_samples=1000
            )
        else:
            self.model = model
        self.max_date = None

    def train(self, historical_data):
        """Huấn luyện mô hình Prophet với yếu tố ngày lễ"""
        logger.info("Starting Prophet training")
        # Chuẩn hóa TransactionDate thành pd.Timestamp
        prophet_df = historical_data.rename(columns={
            'TransactionDate': 'ds',
            'TotalAmount': 'y',
            'IsHoliday': 'holiday'
        })
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])  # Chuyển thành Timestamp
        self.max_date = prophet_df['ds'].max()

        # Định nghĩa ngày lễ
        holidays = prophet_df[prophet_df['holiday'] == 1][['ds']].drop_duplicates()
        holidays['holiday'] = 'holiday_event'
        holidays['lower_window'] = 0
        holidays['upper_window'] = 1

        self.model.add_country_holidays(country_name='VN')
        self.model.holidays = holidays

        self.model.fit(prophet_df)
        logger.info("Prophet training completed")

    def predict(self, start_date, end_date):
        """Dự đoán doanh thu với giới hạn tối đa 1 tháng từ ngày cuối"""
        logger.debug(f"Predicting from {start_date} to {end_date}")
        max_forecast_date = self.max_date + timedelta(days=30)  # Giữ pd.Timestamp
        if end_date > max_forecast_date.to_pydatetime():  # Chuyển thành datetime.datetime
            end_date = max_forecast_date.to_pydatetime()

        periods = (end_date - start_date).days + 1
        if periods <= 0:
            logger.error(f"Invalid date range: start_date {start_date} > end_date {end_date}")
            raise ValueError("start_date must be before or equal to end_date")

        future = self.model.make_future_dataframe(periods=periods, include_history=False)
        future = future[future['ds'] >= start_date]
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]