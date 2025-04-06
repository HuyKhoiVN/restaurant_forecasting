import logging
import pandas as pd
from datetime import datetime
from prophet import Prophet
from lunarcalendar import Converter, Solar, Lunar

logger = logging.getLogger(__name__)

class ProphetForecaster:
    def __init__(self):
        logger.debug("Initializing ProphetForecaster")
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        self.model.add_country_holidays(country_name='VN')
        self.holidays = self._add_tet_holidays()
        self.model.holidays = self.holidays
        self.max_date = None
        self.training_data = None

    def _add_tet_holidays(self):
        tet_dates = []
        for year in range(2024, 2027):
            lunar_new_year = Lunar(year, 1, 1)
            solar_date = Converter.Lunar2Solar(lunar_new_year)
            tet_dates.append({
                'holiday': 'Tet Nguyen Dan',
                'ds': pd.to_datetime(f"{solar_date.year}-{solar_date.month}-{solar_date.day}"),
                'lower_window': -2,
                'upper_window': 3
            })
        holidays = pd.concat([self.model.holidays, pd.DataFrame(tet_dates)], ignore_index=True)
        return holidays

    def train(self, historical_data: pd.DataFrame):
        logger.info("Starting Prophet training")
        df = historical_data.rename(columns={'TransactionDate': 'ds', 'TotalAmount': 'y'})
        df['IsHoliday'] = df['IsHoliday'].astype(int)
        df['IsWeekend'] = df['IsWeekend'].astype(int)
        self.model.add_regressor('IsHoliday')
        self.model.add_regressor('IsWeekend')
        self.model.fit(df)
        self.max_date = df['ds'].max()
        self.training_data = df
        logger.info("Prophet training completed")

    def predict(self, start_date: datetime, end_date: datetime):
        logger.debug(f"Predicting from {start_date} to {end_date}")
        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        future = pd.DataFrame({'ds': future_dates})
        future['IsHoliday'] = future['ds'].apply(lambda x: 1 if x in self.holidays['ds'].values else 0)
        future['IsWeekend'] = future['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]