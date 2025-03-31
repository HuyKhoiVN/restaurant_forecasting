from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta


class ProphetForecaster:
    def __init__(self):
        self.model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            uncertainty_samples=1000
        )
        self.max_date = None

    def train(self, historical_data):
        """Huấn luyện mô hình Prophet với yếu tố ngày lễ"""
        prophet_df = historical_data.rename(columns={
            'TransactionDate': 'ds',
            'TotalAmount': 'y',
            'IsHoliday': 'holiday'  # Thêm yếu tố ngày lễ
        })
        self.max_date = prophet_df['ds'].max()

        # Định nghĩa ngày lễ
        holidays = prophet_df[prophet_df['holiday'] == 1][['ds']].drop_duplicates()
        holidays['holiday'] = 'holiday_event'
        holidays['lower_window'] = 0
        holidays['upper_window'] = 1

        self.model.add_country_holidays(country_name='VN')  # Thêm ngày lễ Việt Nam
        self.model.holidays = holidays

        self.model.fit(prophet_df)

    def predict(self, start_date, end_date):
        """Dự đoán doanh thu với giới hạn tối đa 1 tháng từ ngày cuối"""
        max_forecast_date = self.max_date + timedelta(days=30)
        if end_date > max_forecast_date:
            end_date = max_forecast_date

        periods = (end_date - start_date).days + 1
        future = self.model.make_future_dataframe(periods=periods, include_history=False)
        future = future[future['ds'] >= start_date]
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]