import logging
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

HOLIDAYS = [
    {"date": "2024-04-30", "name": "Ngày Thống nhất"},
    {"date": "2024-05-01", "name": "Ngày Quốc tế Lao động"},
    {"date": "2024-09-02", "name": "Quốc khánh"},
    {"date": "2024-10-20", "name": "Phụ nữ Việt Nam"},
    {"date": "2025-01-29", "name": "Tết Nguyên Đán (Mùng 1)"},
    {"date": "2025-01-30", "name": "Tết Nguyên Đán (Mùng 2)"},
    {"date": "2025-01-31", "name": "Tết Nguyên Đán (Mùng 3)"},
    {"date": "2025-03-08", "name": "Ngày Quốc tế Phụ nữ"}
]


class ProphetForecaster:
    def __init__(self, model=None, max_date=None):
        logger.debug("Initializing ProphetForecaster")
        if model is None:
            self.model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                uncertainty_samples=1000
            )
            self.model.add_country_holidays(country_name='VN')
            # Thêm regressors cho IsHoliday và IsWeekend
            self.model.add_regressor('IsHoliday')
            self.model.add_regressor('IsWeekend')
        else:
            self.model = model
        self.max_date = max_date
        self.holidays = self._prepare_holidays()

    def _prepare_holidays(self):
        holidays_df = pd.DataFrame(HOLIDAYS)
        holidays_df['ds'] = pd.to_datetime(holidays_df['date'])
        holidays_df['holiday'] = holidays_df['name']
        holidays_df['lower_window'] = 0
        holidays_df['upper_window'] = 1

        tet_dates = {
            2024: "2024-02-10",
            2025: "2025-01-29",
            2026: "2026-02-17"
        }
        tet_holidays = []
        for year, tet_date in tet_dates.items():
            base_date = pd.to_datetime(tet_date)
            tet_holidays.extend([
                {'ds': base_date, 'holiday': 'Tết Nguyên Đán (Mùng 1)', 'lower_window': 0, 'upper_window': 1},
                {'ds': base_date + timedelta(days=1), 'holiday': 'Tết Nguyên Đán (Mùng 2)', 'lower_window': 0,
                 'upper_window': 1},
                {'ds': base_date + timedelta(days=2), 'holiday': 'Tết Nguyên Đán (Mùng 3)', 'lower_window': 0,
                 'upper_window': 1}
            ])

        all_holidays = pd.concat([holidays_df[['ds', 'holiday', 'lower_window', 'upper_window']],
                                  pd.DataFrame(tet_holidays)], ignore_index=True)
        return all_holidays.drop_duplicates(subset=['ds'])

    def train(self, historical_data):
        logger.info("Starting Prophet training")
        prophet_df = historical_data.rename(columns={'TransactionDate': 'ds', 'TotalAmount': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

        if prophet_df[['ds', 'y', 'IsHoliday', 'IsWeekend']].isna().any().any():
            logger.error("Input data contains NaN values")
            raise ValueError("Prophet cannot handle NaN in 'ds', 'y', 'IsHoliday', or 'IsWeekend'")

        self.max_date = prophet_df['ds'].max()
        self.model.holidays = self.holidays
        self.model.fit(prophet_df)
        logger.info("Prophet training completed")

    def predict(self, start_date, end_date):
        logger.debug(f"Predicting from {start_date} to {end_date}")
        if self.max_date is None:
            logger.error("max_date is not set. Please train the model first.")
            raise ValueError("max_date is not set")

        max_forecast_date = self.max_date + timedelta(days=30)
        if end_date > max_forecast_date.to_pydatetime():
            end_date = max_forecast_date.to_pydatetime()

        periods = (end_date - start_date).days + 1
        if periods <= 0:
            logger.error(f"Invalid date range: start_date {start_date} > end_date {end_date}")
            raise ValueError("start_date must be before or equal to end_date")

        future = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=periods, freq='D')})
        # Thêm regressors vào future
        future['IsHoliday'] = future['ds'].apply(
            lambda x: 1 if x in self.holidays['ds'].values else 0
        )
        future['IsWeekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)

        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]