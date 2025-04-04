import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Starting data preprocessing")
    df = df.copy()

    # Bảo vệ các ngày lễ
    holiday_mask = df['IsHoliday'] == 1
    non_holiday_df = df[~holiday_mask]

    # Xử lý outlier chỉ trên dữ liệu không phải ngày lễ
    clf = IsolationForest(contamination=0.05, random_state=42)
    outliers = clf.fit_predict(non_holiday_df[['TotalAmount']])
    cleaned_non_holiday_df = non_holiday_df[outliers == 1]

    # Gộp lại với dữ liệu ngày lễ
    df = pd.concat([cleaned_non_holiday_df, df[holiday_mask]], ignore_index=True)
    logger.info(f"Removed {sum(outliers == -1)} outliers from non-holiday data, kept {holiday_mask.sum()} holidays")

    # Tạo feature động
    df['weather_holiday_interaction'] = df['Weather'].map({'Rainy': 1, 'Sunny': 0, 'Unknown': 0}) * df['IsHoliday']
    logger.info("Added weather_holiday_interaction feature")

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_to_scale = ['TotalAmount', 'TransactionCount', 'weather_holiday_interaction']
    for feature in features_to_scale:
        if df[feature].std() == 0:
            logger.warning(f"Feature {feature} has zero variance, setting to 0 after scaling")
            df[feature] = 0
        else:
            df[[feature]] = scaler.fit_transform(df[[feature]])

    df[features_to_scale] = df[features_to_scale].fillna(0)
    logger.debug("Filled NaN values with 0 after scaling")

    return df


def apply_pca(df: pd.DataFrame, features: list) -> tuple:
    logger.debug("Applying PCA")
    if df[features].isna().any().any():
        logger.error(f"NaN values found in features {features} before PCA")
        raise ValueError("Input contains NaN before PCA")

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[features])
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    logger.info(f"PCA variance ratio: {pca.explained_variance_ratio_}")
    return df, pca.explained_variance_ratio_


def apply_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Applying KMeans")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['PCA1', 'PCA2']])
    return df


def aggregate_data(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    logger.debug(f"Aggregating data with {granularity} granularity")
    if granularity == "weekly":
        df['ds'] = pd.to_datetime(df['ds']).dt.to_period('W').dt.start_time
    elif granularity == "monthly":
        df['ds'] = pd.to_datetime(df['ds']).dt.to_period('M').dt.start_time
    return df.groupby('ds').agg({
        'yhat': 'mean',
        'yhat_lower': 'mean',
        'yhat_upper': 'mean',
        'lstm': 'mean',
        'combined': 'mean'
    }).reset_index()