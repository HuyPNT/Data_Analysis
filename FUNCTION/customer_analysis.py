import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def calculate_rfm(df):
    """
    Tính điểm RFM cho từng khách hàng.
    Recency: Số ngày kể từ lần mua cuối
    Frequency: Số lần giao dịch
    Monetary: Tổng chi tiêu
    """
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy['Ngày bán'] = pd.to_datetime(df_copy['Ngày bán'], errors='coerce')

    # Lọc bỏ khách hàng ẩn danh
    df_copy = df_copy.dropna(subset=['Mã khách hàng'])
    df_copy = df_copy[df_copy['Mã khách hàng'].astype(str).str.strip() != '']

    if df_copy.empty:
        logging.warning("Không có dữ liệu khách hàng hợp lệ để tính RFM")
        return pd.DataFrame()

    reference_date = df_copy['Ngày bán'].max() + pd.Timedelta(days=1)

    rfm = df_copy.groupby('Mã khách hàng').agg({
        'Ngày bán': lambda x: (reference_date - x.max()).days,
        'Mã giao dịch': 'nunique',
        'Thành tiền': 'sum'
    }).reset_index()

    rfm.columns = ['Mã khách hàng', 'Recency', 'Frequency', 'Monetary']

    # Tính điểm R, F, M (1-5) bằng qcut
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    except (ValueError, TypeError):
        rfm['R_Score'] = 3

    try:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    except (ValueError, TypeError):
        rfm['F_Score'] = 3

    try:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    except (ValueError, TypeError):
        rfm['M_Score'] = 3

    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    logging.info(f"Đã tính RFM cho {len(rfm)} khách hàng")
    return rfm


def segment_customers(df, n_clusters=4):
    """
    Phân khúc khách hàng sử dụng K-Means clustering trên dữ liệu RFM.
    Trả về (rfm_with_segment, segment_summary).
    """
    rfm = calculate_rfm(df)
    if rfm.empty:
        return pd.DataFrame(), pd.DataFrame()

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
    except ImportError:
        logging.warning("sklearn chưa được cài đặt. Sử dụng phân khúc đơn giản thay thế.")
        return _simple_segmentation(rfm)

    features = rfm[['Recency', 'Frequency', 'Monetary']].values

    # Chuẩn hóa
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Đảm bảo n_clusters không vượt quá số mẫu
    actual_clusters = min(n_clusters, len(rfm))
    if actual_clusters < 2:
        return _simple_segmentation(rfm)

    # K-Means
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(features_scaled)

    # Gán nhãn phân khúc dựa trên Monetary trung bình của cluster
    cluster_monetary = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
    segment_labels = ['VIP', 'Trung thành', 'Tiềm năng', 'Cần chú ý']
    label_map = {}
    for i, cluster_id in enumerate(cluster_monetary.index):
        label_map[cluster_id] = segment_labels[i] if i < len(segment_labels) else f'Nhóm {i + 1}'

    rfm['Phân khúc'] = rfm['Cluster'].map(label_map)

    # Tạo summary
    summary = rfm.groupby('Phân khúc').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Mã khách hàng': 'count'
    }).reset_index()
    summary.columns = ['Phân khúc', 'Recency TB', 'Frequency TB', 'Monetary TB', 'Số KH']
    summary = summary.sort_values('Monetary TB', ascending=False)

    logging.info(f"Đã phân khúc {len(rfm)} khách hàng thành {actual_clusters} nhóm")
    return rfm, summary


def _simple_segmentation(rfm):
    """Phân khúc đơn giản khi không có sklearn."""
    if rfm.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Phân khúc theo tổng monetary
    rfm_sorted = rfm.copy()
    monetary_median = rfm_sorted['Monetary'].median()
    frequency_median = rfm_sorted['Frequency'].median()

    conditions = [
        (rfm_sorted['Monetary'] >= rfm_sorted['Monetary'].quantile(0.75)),
        (rfm_sorted['Monetary'] >= monetary_median) & (rfm_sorted['Frequency'] >= frequency_median),
        (rfm_sorted['Frequency'] >= frequency_median),
    ]
    choices = ['VIP', 'Trung thành', 'Tiềm năng']
    rfm_sorted['Phân khúc'] = np.select(conditions, choices, default='Cần chú ý')

    summary = rfm_sorted.groupby('Phân khúc').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Mã khách hàng': 'count'
    }).reset_index()
    summary.columns = ['Phân khúc', 'Recency TB', 'Frequency TB', 'Monetary TB', 'Số KH']
    summary = summary.sort_values('Monetary TB', ascending=False)

    return rfm_sorted, summary


def customer_purchase_patterns(df):
    """
    Phân tích thói quen mua hàng của từng khách hàng:
    SP yêu thích, PT thanh toán, thời điểm mua, số đơn, giá trị TB.
    """
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy['Ngày bán'] = pd.to_datetime(df_copy['Ngày bán'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Mã khách hàng'])
    df_copy = df_copy[df_copy['Mã khách hàng'].astype(str).str.strip() != '']

    if df_copy.empty:
        return pd.DataFrame()

    # Phân loại thời điểm mua
    def classify_time(hour):
        if 5 <= hour < 12:
            return 'Sáng'
        elif 12 <= hour < 17:
            return 'Chiều'
        elif 17 <= hour < 21:
            return 'Tối'
        else:
            return 'Đêm'

    df_copy['Thời điểm'] = df_copy['Ngày bán'].dt.hour.apply(classify_time)

    patterns = df_copy.groupby('Mã khách hàng').agg(
        sp_yeu_thich=('Tên mặt hàng', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'),
        pt_thanh_toan=('Phương thức thanh toán', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'),
        thoi_diem=('Thời điểm', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'),
        so_don=('Mã giao dịch', 'nunique'),
        gia_tri_tb=('Thành tiền', 'mean')
    ).reset_index()

    patterns.columns = ['Mã khách hàng', 'SP yêu thích', 'PT thanh toán',
                         'Thời điểm mua', 'Số đơn', 'Giá trị TB']

    logging.info(f"Đã phân tích thói quen cho {len(patterns)} khách hàng")
    return patterns


def customer_lifetime_value(df):
    """
    Tính giá trị vòng đời khách hàng (CLV).
    CLV dự đoán = DT trung bình/đơn × Tần suất ước tính/năm.
    """
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy['Ngày bán'] = pd.to_datetime(df_copy['Ngày bán'], errors='coerce')
    df_copy = df_copy.dropna(subset=['Mã khách hàng'])
    df_copy = df_copy[df_copy['Mã khách hàng'].astype(str).str.strip() != '']

    if df_copy.empty:
        return pd.DataFrame()

    # Số ngày trong khoảng dữ liệu
    date_range_days = (df_copy['Ngày bán'].max() - df_copy['Ngày bán'].min()).days
    if date_range_days <= 0:
        date_range_days = 365

    clv = df_copy.groupby('Mã khách hàng').agg(
        tong_dt=('Thành tiền', 'sum'),
        dt_tb=('Thành tiền', 'mean'),
        so_don=('Mã giao dịch', 'nunique'),
    ).reset_index()

    clv['tan_suat_nam'] = clv['so_don'] / date_range_days * 365
    clv['gia_tri_nam'] = clv['dt_tb'] * clv['tan_suat_nam']

    # Chọn đúng 5 cột cần hiển thị
    clv = clv[['Mã khách hàng', 'tong_dt', 'dt_tb', 'tan_suat_nam', 'gia_tri_nam']]
    clv.columns = ['Mã khách hàng', 'Tổng DT', 'DT TB/đơn', 'Tần suất mua', 'Giá trị dự đoán/năm']
    clv = clv.sort_values('Tổng DT', ascending=False)

    logging.info(f"Đã tính CLV cho {len(clv)} khách hàng")
    return clv


def customer_summary(df):
    """
    Tổng hợp thông tin khách hàng tổng quan.
    """
    if df.empty:
        return {}

    df_copy = df.copy()
    df_copy = df_copy.dropna(subset=['Mã khách hàng'])
    df_copy = df_copy[df_copy['Mã khách hàng'].astype(str).str.strip() != '']

    if df_copy.empty:
        return {'total_customers': 0}

    customer_stats = df_copy.groupby('Mã khách hàng').agg(
        num_transactions=('Mã giao dịch', 'nunique'),
        total_revenue=('Thành tiền', 'sum')
    )

    top_customer = customer_stats['total_revenue'].idxmax()
    repeat_customers = (customer_stats['num_transactions'] > 1).sum()

    summary = {
        'total_customers': df_copy['Mã khách hàng'].nunique(),
        'avg_transactions_per_customer': customer_stats['num_transactions'].mean(),
        'avg_revenue_per_customer': customer_stats['total_revenue'].mean(),
        'top_customer_id': top_customer,
        'top_customer_revenue': customer_stats.loc[top_customer, 'total_revenue'],
        'retention_rate_estimate': repeat_customers / len(customer_stats) * 100 if len(customer_stats) > 0 else 0,
    }

    logging.info(f"Tổng hợp: {summary['total_customers']} khách hàng, "
                 f"tỷ lệ quay lại {summary['retention_rate_estimate']:.1f}%")
    return summary
