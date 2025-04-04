from datetime import timedelta, datetime
import  logging
import numpy as np
import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout, Input
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import optuna
import tensorflow as tf

logger = logging.getLogger(__name__)


class LSTMForecaster:
    def __init__(self, model=None, holidays=None):
        self.look_back = 7
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.max_date = None
        self.holidays = holidays  # Nhận holidays từ bên ngoài
        self.model = model if model else Sequential([
            Input(shape=(self.look_back, 3)),  # 3 features: TotalAmount, IsHoliday, IsWeekend
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        if not model:
            self.model.compile(optimizer='adam', loss='mse')

    def prepare_data(self, data):
        features = data[['TotalAmount', 'IsHoliday', 'IsWeekend']].values
        scaled_data = self.scaler.fit_transform(features)
        X, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i:(i + self.look_back), :])
            y.append(scaled_data[i + self.look_back, 0])  # Dự đoán TotalAmount
        return np.array(X), np.array(y)

    def train(self, historical_data: pd.DataFrame):
        logger.info("Starting LSTM training")
        self.max_date = pd.to_datetime(historical_data['TransactionDate'].max())
        X, y = self.prepare_data(historical_data)
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=1)
        logger.info("LSTM training completed")

    def predict(self, last_values: np.ndarray, start_date: datetime, end_date: datetime):
        logger.debug(f"Predicting from {start_date} to {end_date}")
        periods = (end_date - start_date).days + 1
        if periods <= 0:
            logger.error(f"Invalid date range: start_date {start_date} > end_date {end_date}")
            raise ValueError("start_date must be before or equal to end_date")

        if self.holidays is None:
            logger.error("Holidays not provided to LSTMForecaster")
            raise ValueError("Holidays must be provided for accurate forecasting")

        # Chuẩn bị dữ liệu đầu vào với IsHoliday và IsWeekend
        last_data = pd.DataFrame({
            'TotalAmount': last_values[-self.look_back:],
            'IsHoliday': [1 if d in self.holidays['ds'].values else 0 for d in
                          pd.date_range(end_date - timedelta(days=self.look_back - 1), periods=self.look_back)],
            'IsWeekend': [(d.weekday() >= 5) for d in
                          pd.date_range(end_date - timedelta(days=self.look_back - 1), periods=self.look_back)]
        })
        last_scaled = self.scaler.transform(last_data[['TotalAmount', 'IsHoliday', 'IsWeekend']])

        predictions = []
        current_input = last_scaled.reshape(1, self.look_back, 3)
        for _ in range(periods):
            next_pred = self.model.predict(current_input, verbose=0)
            predictions.append(next_pred[0, 0])
            next_date = start_date + timedelta(days=len(predictions))
            next_row = np.array([[next_pred[0, 0],
                                  1 if next_date in self.holidays['ds'].values else 0,
                                  1 if next_date.weekday() >= 5 else 0]])
            next_scaled = self.scaler.transform(next_row)[0]
            current_input = np.append(current_input[:, 1:, :], [[next_scaled]], axis=1)

        predictions = np.array(predictions).reshape(-1, 1)
        dummy = np.zeros((len(predictions), 2))  # Để inverse_transform
        predictions_full = np.hstack([predictions, dummy])
        predictions = self.scaler.inverse_transform(predictions_full)[:, 0]
        return predictions