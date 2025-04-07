import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

logger = logging.getLogger(__name__)


class ProductForecaster:
    def __init__(self, holidays=None):
        logger.debug("Initializing ProductForecaster")
        self.models = {}
        self.holidays = holidays
        self.label_encoders = {'Weather': LabelEncoder()}
        self.last_trained = None
        self.feature_columns = None
        self.historical_data = None

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Preprocessing data")
        df = data.copy()

        # Làm sạch: Loại bỏ Quantity = 0
        df = df[df['Quantity'] > 0]

        # Xử lý thiếu
        if 'Weather' not in df.columns:
            logger.warning("Column 'Weather' not found in input data. Adding with default 'Unknown'")
            df['Weather'] = 'Unknown'
        df['Weather'] = df['Weather'].fillna('Unknown')
        df['IsHoliday'] = df['IsHoliday'].astype(int)
        df['IsWeekend'] = df['IsWeekend'].astype(int)

        # Chuyển đổi định dạng
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        df['Hour'] = df['TransactionTime'].dt.components['hours']  # Trích xuất Hour

        # Tạo đặc trưng mới
        df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
        df['Month'] = df['TransactionDate'].dt.month
        df['DayIndex'] = (df['TransactionDate'] - df['TransactionDate'].min()).dt.days

        # Rolling average (7 ngày) và Lag features
        agg_dict = {
            'Quantity': 'sum',
            'IsHoliday': 'first',
            'IsWeekend': 'first',
            'Weather': 'first',
            'DayOfWeek': 'first',
            'Month': 'first',
            'DayIndex': 'first'
        }
        # Nhóm theo ngày, sản phẩm và giờ, không thêm Hour vào agg_dict vì nó đã là key
        df = df.groupby(['TransactionDate', 'ProductID', 'Hour']).agg(agg_dict).reset_index()
        df = df.sort_values(['TransactionDate', 'Hour'])

        for product_id in df['ProductID'].unique():
            mask = df['ProductID'] == product_id
            df.loc[mask, 'RollingAvg7'] = df.loc[mask, 'Quantity'].rolling(window=7, min_periods=1).mean()
            df.loc[mask, 'Lag1'] = df.loc[mask, 'Quantity'].shift(1)

        df['RollingAvg7'] = df['RollingAvg7'].fillna(df['Quantity'])
        df['Lag1'] = df['Lag1'].fillna(df['Quantity'])

        # Mã hóa Weather
        df['Weather'] = self.label_encoders['Weather'].fit_transform(df['Weather'])

        # Định nghĩa feature_columns với Hour
        self.feature_columns = ['DayOfWeek', 'Month', 'IsHoliday', 'IsWeekend', 'Weather',
                                'RollingAvg7', 'Lag1', 'DayIndex', 'Hour']
        return df

    def train(self, historical_data: pd.DataFrame):
        logger.info("Starting XGBoost training for product forecast")
        self.historical_data = historical_data  # Lưu dữ liệu để dùng trong predict
        df = self.preprocess_data(historical_data)
        products = df['ProductID'].unique()

        train_size = int(len(df['TransactionDate'].unique()) * 0.8)
        train_dates = df['TransactionDate'].unique()[:train_size]
        train_df = df[df['TransactionDate'].isin(train_dates)]
        test_df = df[~df['TransactionDate'].isin(train_dates)]

        for product_id in products:
            train_product = train_df[train_df['ProductID'] == product_id]
            test_product = test_df[test_df['ProductID'] == product_id]

            if len(train_product) < 10:
                logger.warning(f"Skipping ProductID {product_id} due to insufficient data")
                continue

            X_train = train_product[self.feature_columns].values
            y_train = train_product['Quantity'].values
            X_test = test_product[self.feature_columns].values
            y_test = test_product['Quantity'].values

            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            if len(test_product) > 0:
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                logger.info(f"Product {product_id}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2%}")

            self.models[product_id] = model

        self.last_trained = datetime.now()
        logger.info("XGBoost training completed")

    def predict(self, date: datetime, time_ranges: list = None, products_info: pd.DataFrame = None):
        logger.debug(f"Predicting product quantities for {date}")
        if not self.models:
            raise ValueError("Model not trained yet")
        if products_info is None or products_info.empty:
            raise ValueError("Product info required for prediction")

        predict_date = pd.DataFrame({
            'TransactionDate': [date],
            'TransactionTime': [timedelta(hours=12)],
            'IsHoliday': [1 if self.holidays is not None and date in self.holidays['ds'].values else 0],
            'IsWeekend': [1 if date.weekday() >= 5 else 0],
            'Weather': ['Nắng'],
            'RollingAvg7': [0],
            'Lag1': [0],
            'DayIndex': [(date - self.historical_data['TransactionDate'].min()).days],
            'Hour': [12]  # Thêm Hour mặc định
        })

        for product_id in self.models.keys():
            hist = self.preprocess_data(self.historical_data[self.historical_data['ProductID'] == product_id])
            if not hist.empty:
                last_day = hist[hist['TransactionDate'] < date].tail(1)
                if not last_day.empty:
                    predict_date['RollingAvg7'] = last_day['RollingAvg7'].values[0]
                    predict_date['Lag1'] = last_day['Quantity'].values[0]

        predict_date['DayOfWeek'] = predict_date['TransactionDate'].dt.dayofweek
        predict_date['Month'] = predict_date['TransactionDate'].dt.month
        predict_date['Weather'] = self.label_encoders['Weather'].transform(predict_date['Weather'])

        predictions = []
        time_ranges = time_ranges or ['lunch', 'dinner']
        hourly_map = {'lunch': [11, 12, 13, 14], 'dinner': [17, 18, 19, 20]}

        for product_id in self.models.keys():
            product = products_info[products_info['Id'] == product_id].iloc[0]
            model = self.models[product_id]
            total_pred = 0
            hourly_dist = []

            for time_range in time_ranges:
                hours = hourly_map.get(time_range, [12])
                for hour in hours:
                    pred_input = predict_date.copy()
                    pred_input['Hour'] = hour  # Cập nhật Hour cho mỗi dự đoán
                    pred = max(0, float(model.predict(pred_input[self.feature_columns].values)[0]))
                    total_pred += pred
                    hourly_dist.append({'hour': f"{hour:02d}:00-{(hour + 1) % 24:02d}:00", 'pred': pred})

            for dist in hourly_dist:
                dist['percentage'] = float(dist['pred'] / total_pred * 100) if total_pred > 0 else 0.0

            lower = max(0, float(total_pred * 0.9))
            upper = float(total_pred * 1.1)

            predictions.append({
                'product_id': int(product_id),
                'product_name': product['Name'],
                'category': product['Category'],
                'predicted_quantity': float(total_pred),
                'confidence_interval': {'lower': lower, 'upper': upper},
                'hourly_distribution': [{'hour': d['hour'], 'percentage': d['percentage']} for d in hourly_dist]
            })

        return predictions