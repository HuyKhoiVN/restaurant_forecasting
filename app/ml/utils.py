import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def preprocess_data(df):
    """Tiền xử lý dữ liệu: điền giá trị thiếu và xử lý outlier"""
    # Điền giá trị thiếu
    df = df.fillna({'IsHoliday': 0, 'IsWeekend': 0, 'Weather': 'Unknown'})
    df['Weather'] = df['Weather'].astype('category').cat.codes

    # Xử lý outlier trong TotalAmount
    Q1 = df['TotalAmount'].quantile(0.25)
    Q3 = df['TotalAmount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['TotalAmount'] = df['TotalAmount'].clip(lower=lower_bound, upper=upper_bound)

    return df


def aggregate_data(df: pd.DataFrame, granularity: str = "daily"):
    """Tổng hợp dữ liệu theo granularity"""
    if granularity == "daily":
        return df
    elif granularity == "weekly":
        df['ds'] = pd.to_datetime(df['ds']).dt.to_period('W').apply(lambda r: r.start_time)
        return df.groupby('ds').agg({
            'yhat': 'mean',
            'yhat_lower': 'min',
            'yhat_upper': 'max',
            'lstm': 'mean',
            'combined': 'mean'
        }).reset_index()
    elif granularity == "monthly":
        df['ds'] = pd.to_datetime(df['ds']).dt.to_period('M').apply(lambda r: r.start_time)
        return df.groupby('ds').agg({
            'yhat': 'mean',
            'yhat_lower': 'min',
            'yhat_upper': 'max',
            'lstm': 'mean',
            'combined': 'mean'
        }).reset_index()
    return df


def apply_pca(df, features):
    """Áp dụng PCA để giảm chiều dữ liệu"""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[features])
    df[['PCA1', 'PCA2']] = pca_result
    return df, pca.explained_variance_ratio_


def apply_kmeans(df):
    """Áp dụng K-means để phân cụm dữ liệu"""
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['PCA1', 'PCA2']])
    return df