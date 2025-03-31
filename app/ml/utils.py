import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def aggregate_data(df, granularity):
    """Tổng hợp dữ liệu theo daily/weekly/monthly"""
    df['ds'] = pd.to_datetime(df['ds'])

    if granularity == "daily":
        return df

    if granularity == "weekly":
        agg_func = lambda x: x - pd.Timedelta(days=x.dayofweek)
        grouped = df.groupby(df['ds'].apply(agg_func))
    elif granularity == "monthly":
        grouped = df.groupby(pd.Grouper(key='ds', freq='M'))

    aggregated = grouped.agg({
        'yhat': 'sum',
        'yhat_lower': 'sum',
        'yhat_upper': 'sum',
        'lstm': 'sum',
        'combined': 'sum'
    }).reset_index()

    return aggregated


def preprocess_data(df):
    """Tiền xử lý dữ liệu: loại bỏ giá trị thiếu và bất thường"""
    # Loại bỏ giá trị thiếu
    df = df.dropna(subset=['TotalAmount', 'TransactionDate'])

    # Phát hiện và loại bỏ bất thường bằng IQR
    Q1 = df['TotalAmount'].quantile(0.25)
    Q3 = df['TotalAmount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[(df['TotalAmount'] >= lower_bound) & (df['TotalAmount'] <= upper_bound)]

    return df_cleaned


def apply_pca(df, features, n_components=2):
    """Giảm chiều dữ liệu bằng PCA"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    df[f'pca_1'] = pca_result[:, 0]
    df[f'pca_2'] = pca_result[:, 1]
    return df, pca.explained_variance_ratio_


def apply_kmeans(df, n_clusters=3):
    """Phân cụm dữ liệu bằng K-means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df[['pca_1', 'pca_2']])
    df['cluster'] = clusters
    return df