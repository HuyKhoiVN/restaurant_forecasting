from datetime import timedelta
import  logging
import numpy as np
import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

logger = logging.getLogger(__name__)


class LSTMForecaster:
    def __init__(self, look_back=30):
        self.look_back = look_back
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        self.max_date = None

    def _build_model(self) -> Sequential:
        logger.debug("Building LSTM model")
        model = Sequential([
            Input(shape=(self.look_back, 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        if len(data) < self.look_back:
            logger.error(f"Data length {len(data)} is less than look_back {self.look_back}")
            raise ValueError("Not enough data for LSTM look_back")
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i - self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def train(self, historical_data: pd.DataFrame):
        logger.info("Starting LSTM training")
        # Chuẩn hóa max_date thành pd.Timestamp
        self.max_date = pd.to_datetime(historical_data['TransactionDate'].max())
        data = historical_data.set_index('TransactionDate')['TotalAmount']
        X, y = self.prepare_data(data)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=1)
        logger.info("LSTM training completed")

    def predict(self, last_values: np.ndarray, start_date, end_date) -> np.ndarray:
        logger.debug(f"Predicting from {start_date} to {end_date}")
        max_forecast_date = self.max_date + timedelta(days=30)
        if end_date > max_forecast_date.to_pydatetime():  # Giữ to_pydatetime() vì max_date giờ là Timestamp
            end_date = max_forecast_date.to_pydatetime()

        periods = (end_date - start_date).days + 1
        predictions = []
        current_batch = last_values[-self.look_back:].reshape(1, self.look_back, 1)

        for _ in range(periods):
            current_pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred[0])
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()