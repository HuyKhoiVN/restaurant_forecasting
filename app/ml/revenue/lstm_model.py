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
        self.look_back = 30
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.max_date = None
        self.holidays = holidays
        self.model = model if model else Sequential([
            Input(shape=(self.look_back, 3)),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(100),
            Dropout(0.2),
            Dense(1)
        ])
        if not model:
            self.model.compile(optimizer='adam', loss='mse')

    def prepare_data(self, data):
        total_amount = data[['TotalAmount']].values.astype(float)
        scaled_total = self.scaler.fit_transform(total_amount)
        features = np.hstack([scaled_total, data[['IsHoliday', 'IsWeekend']].values.astype(float)])
        X, y = [], []
        for i in range(len(features) - self.look_back):
            X.append(features[i:(i + self.look_back), :])
            y.append(scaled_total[i + self.look_back, 0])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self, historical_data: pd.DataFrame):
        logger.info("Starting LSTM training")
        self.max_date = pd.to_datetime(historical_data['TransactionDate'].max())
        X, y = self.prepare_data(historical_data)
        self.model.fit(X, y, epochs=100, batch_size=32, verbose=1)  # Tăng epochs
        logger.info("LSTM training completed")

    def predict(self, last_values: np.ndarray, start_date: datetime, end_date: datetime):
        logger.debug(f"Predicting from {start_date} to {end_date}")
        periods = (end_date - start_date).days + 1
        if periods <= 0:
            raise ValueError("start_date must be before or equal to end_date")
        if self.holidays is None:
            raise ValueError("Holidays must be provided")

        dates = pd.date_range(end_date - timedelta(days=self.look_back - 1), periods=self.look_back)
        last_data = np.hstack([
            last_values[-self.look_back:].reshape(-1, 1),
            np.array([1 if d in self.holidays['ds'].values else 0 for d in dates]).reshape(-1, 1),
            np.array([(d.weekday() >= 5) for d in dates]).reshape(-1, 1)
        ]).astype(float)
        scaled_total = self.scaler.transform(last_data[:, [0]])
        last_scaled = np.hstack([scaled_total, last_data[:, 1:]]).astype(np.float32)

        predictions = []
        current_input = last_scaled.reshape(1, self.look_back, 3)
        for i in range(periods):
            next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            next_date = start_date + timedelta(days=i + 1)
            next_features = np.array([
                next_pred,
                1 if next_date in self.holidays['ds'].values else 0,
                1 if next_date.weekday() >= 5 else 0
            ]).reshape(1, -1).astype(float)
            next_scaled_total = self.scaler.transform(next_features[:, [0]])
            next_input = np.hstack([next_scaled_total, next_features[:, 1:]]).astype(np.float32)
            current_input = np.append(current_input[:, 1:, :], next_input.reshape(1, 1, 3), axis=1)

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        logger.debug(f"LSTM predictions: {predictions.tolist()}")
        return np.clip(predictions, a_min=0, a_max=None)