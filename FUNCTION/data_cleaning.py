"""
Module làm sạch và đánh giá chất lượng dữ liệu bán hàng cửa hàng tiện lợi.
Cung cấp các hàm phát hiện ngoại lệ (IQR, Z-score), xử lý khuyết thiếu,
chuẩn hóa định dạng và báo cáo chất lượng dữ liệu.
"""

import logging
import numpy as np
import pandas as pd

# Thiết lập logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def detect_outliers_iqr(df, column='Thành tiền'):
    """
    Phát hiện các giá trị ngoại lệ (outliers) bằng phương pháp khoảng tứ phân vị (IQR).

    Công thức:
        - Q1: Phân vị 25%
        - Q3: Phân vị 75%
        - IQR = Q3 - Q1
        - Ngưỡng dưới (lower_bound) = Q1 - 1.5 * IQR
        - Ngưỡng trên (upper_bound) = Q3 + 1.5 * IQR

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.
        column (str): Tên cột cần kiểm tra giá trị ngoại lệ (mặc định 'Thành tiền').

    Trả về:
        tuple: (outliers_df, lower_bound, upper_bound)
            - outliers_df (pd.DataFrame): Các dòng chứa giá trị ngoại lệ.
            - lower_bound (float): Ngưỡng dưới xác định ngoại lệ.
            - upper_bound (float): Ngưỡng trên xác định ngoại lệ.
    """
    if df is None or df.empty or column not in df.columns:
        logger.warning(f"DataFrame rỗng hoặc không tìm thấy cột '{column}'.")
        empty_df = pd.DataFrame(columns=df.columns if df is not None else [])
        return empty_df, 0.0, 0.0

    series = pd.to_numeric(df[column], errors='coerce').dropna()
    if series.empty:
        logger.warning(f"Cột '{column}' không có dữ liệu số hợp lệ để tính IQR.")
        return df.iloc[0:0].copy(), 0.0, 0.0

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)

    # Lọc các dòng ngoại lệ dựa trên giá trị của cột
    col_numeric = pd.to_numeric(df[column], errors='coerce')
    outliers_mask = (col_numeric < lower_bound) | (col_numeric > upper_bound)
    outliers_df = df[outliers_mask.fillna(False)].copy()

    logger.info(
        f"IQR [{column}]: Q1={q1:,.2f}, Q3={q3:,.2f}, IQR={iqr:,.2f}, "
        f"Ngưỡng=[{lower_bound:,.2f}, {upper_bound:,.2f}], Phát hiện {len(outliers_df)} outliers."
    )
    return outliers_df, lower_bound, upper_bound


def detect_outliers_zscore(df, column='Thành tiền', threshold=3):
    """
    Phát hiện các giá trị ngoại lệ bằng phương pháp Z-score.

    Công thức:
        z = (x - mean) / std
        Ngoại lệ khi: |z| > threshold

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.
        column (str): Cột cần kiểm tra giá trị ngoại lệ (mặc định 'Thành tiền').
        threshold (float): Ngưỡng độ lệch chuẩn để phân loại ngoại lệ (mặc định 3).

    Trả về:
        tuple: (outliers_df, z_scores_series)
            - outliers_df (pd.DataFrame): Các dòng chứa giá trị ngoại lệ.
            - z_scores_series (pd.Series): Chuỗi giá trị Z-score tương ứng với từng dòng trong df.
    """
    if df is None or df.empty or column not in df.columns:
        logger.warning(f"DataFrame rỗng hoặc không tìm thấy cột '{column}'.")
        empty_df = pd.DataFrame(columns=df.columns if df is not None else [])
        return empty_df, pd.Series(dtype=float)

    series = pd.to_numeric(df[column], errors='coerce')
    valid_series = series.dropna()

    if valid_series.empty or len(valid_series) < 2:
        logger.warning(f"Cột '{column}' không đủ dữ liệu để tính Z-score.")
        z_scores = pd.Series(0.0, index=df.index)
        return df.iloc[0:0].copy(), z_scores

    mean = float(valid_series.mean())
    std = float(valid_series.std(ddof=0))

    if std == 0.0 or np.isnan(std):
        logger.warning(f"Độ lệch chuẩn của cột '{column}' bằng 0, không có biến thiên.")
        z_scores = pd.Series(0.0, index=df.index)
        return df.iloc[0:0].copy(), z_scores

    z_scores = (series - mean) / std
    outliers_mask = z_scores.abs() > threshold
    outliers_df = df[outliers_mask.fillna(False)].copy()

    logger.info(
        f"Z-score [{column}]: Mean={mean:,.2f}, Std={std:,.2f}, "
        f"Ngưỡng={threshold}, Phát hiện {len(outliers_df)} outliers."
    )
    return outliers_df, z_scores


def handle_missing_values(df):
    """
    Xử lý các giá trị bị khuyết thiếu trong DataFrame theo quy tắc:
        - Cột số (numeric): Điền bằng giá trị trung vị (median).
        - Cột 'Mã khách hàng': Điền bằng 'KH_UNKNOWN'.
        - Cột 'Thành phố': Điền bằng giá trị xuất hiện nhiều nhất (mode).
        - Các cột chuỗi (string/object) khác: Điền bằng 'N/A'.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.

    Trả về:
        tuple: (df_cleaned, missing_report_dict)
            - df_cleaned (pd.DataFrame): DataFrame sau khi điền khuyết thiếu.
            - missing_report_dict (dict): Báo cáo số lượng khuyết thiếu trước khi xử lý {tên_cột: số_lượng}.
    """
    if df is None or df.empty:
        logger.warning("DataFrame rỗng khi gọi handle_missing_values.")
        return pd.DataFrame() if df is None else df.copy(), {}

    df_cleaned = df.copy()
    missing_report_dict = {str(col): int(df_cleaned[col].isna().sum()) for col in df_cleaned.columns}

    # 1. Cột 'Mã khách hàng'
    if 'Mã khách hàng' in df_cleaned.columns:
        df_cleaned['Mã khách hàng'] = df_cleaned['Mã khách hàng'].fillna('KH_UNKNOWN')

    # 2. Cột 'Thành phố'
    if 'Thành phố' in df_cleaned.columns:
        non_null_cities = df_cleaned['Thành phố'].dropna()
        modes = non_null_cities.mode()
        mode_val = modes.iloc[0] if not modes.empty else 'Không xác định'
        df_cleaned['Thành phố'] = df_cleaned['Thành phố'].fillna(mode_val)

    # 3. Các cột kiểu số
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df_cleaned[col].median()
        if pd.isna(median_val):
            median_val = 0
        df_cleaned[col] = df_cleaned[col].fillna(median_val)

    # 4. Các cột chuỗi/object khác
    object_cols = df_cleaned.select_dtypes(include=['object', 'string', 'category']).columns
    for col in object_cols:
        if col not in ['Mã khách hàng', 'Thành phố']:
            df_cleaned[col] = df_cleaned[col].fillna('N/A')

    # 5. Các cột thời gian nếu có khuyết thiếu
    datetime_cols = df_cleaned.select_dtypes(include=['datetime64', 'datetimetz']).columns
    for col in datetime_cols:
        if df_cleaned[col].isna().any():
            df_cleaned[col] = df_cleaned[col].bfill().ffill()

    total_missing_fixed = sum(missing_report_dict.values())
    logger.info(f"Đã xử lý xong giá trị khuyết thiếu. Tổng số giá trị khuyết đã khắc phục: {total_missing_fixed}.")
    return df_cleaned, missing_report_dict


def detect_duplicates(df):
    """
    Phát hiện các dòng dữ liệu trùng lặp trong DataFrame.
    Ưu tiên kiểm tra theo cột 'Mã giao dịch'. Nếu không có cột này, kiểm tra toàn bộ các cột.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.

    Trả về:
        pd.DataFrame: DataFrame chứa các dòng bị trùng lặp.
    """
    if df is None or df.empty:
        logger.warning("DataFrame rỗng khi gọi detect_duplicates.")
        return pd.DataFrame(columns=df.columns if df is not None else [])

    if 'Mã giao dịch' in df.columns:
        duplicates_mask = df.duplicated(subset=['Mã giao dịch'], keep=False)
    else:
        duplicates_mask = df.duplicated(keep=False)

    duplicates_df = df[duplicates_mask].copy()
    logger.info(f"Phát hiện {len(duplicates_df)} dòng dữ liệu trùng lặp.")
    return duplicates_df


def normalize_data(df):
    """
    Chuẩn hóa dữ liệu:
        - Cắt bỏ khoảng trắng thừa ở đầu/cuối của các cột kiểu chuỗi.
        - Chuẩn hóa các giá trị trong cột 'Phương thức thanh toán' về dạng chuẩn:
          'Tiền mặt', 'Thẻ', 'Chuyển khoản', 'MoMo', 'ZaloPay'.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.

    Trả về:
        pd.DataFrame: DataFrame sau khi chuẩn hóa.
    """
    if df is None or df.empty:
        logger.warning("DataFrame rỗng khi gọi normalize_data.")
        return pd.DataFrame() if df is None else df.copy()

    df_normalized = df.copy()

    # Cắt khoảng trắng các cột dạng chuỗi/object
    str_cols = df_normalized.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df_normalized[col] = df_normalized[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    # Chuẩn hóa cột Phương thức thanh toán
    if 'Phương thức thanh toán' in df_normalized.columns:
        payment_map = {
            'tiền mặt': 'Tiền mặt',
            'tien mat': 'Tiền mặt',
            'cash': 'Tiền mặt',
            'thẻ': 'Thẻ',
            'the': 'Thẻ',
            'card': 'Thẻ',
            'thẻ tín dụng': 'Thẻ',
            'thẻ ghi nợ': 'Thẻ',
            'credit card': 'Thẻ',
            'debit card': 'Thẻ',
            'chuyển khoản': 'Chuyển khoản',
            'chuyen khoan': 'Chuyển khoản',
            'ck': 'Chuyển khoản',
            'banking': 'Chuyển khoản',
            'internet banking': 'Chuyển khoản',
            'bank transfer': 'Chuyển khoản',
            'momo': 'MoMo',
            'ví momo': 'MoMo',
            'vi momo': 'MoMo',
            'zalopay': 'ZaloPay',
            'zalo pay': 'ZaloPay',
            'zalo': 'ZaloPay',
            'ví zalopay': 'ZaloPay',
        }

        def _standardize_payment(val):
            if not isinstance(val, str):
                return val
            val_clean = val.strip()
            val_lower = val_clean.lower()
            return payment_map.get(val_lower, val_clean)

        df_normalized['Phương thức thanh toán'] = df_normalized['Phương thức thanh toán'].apply(_standardize_payment)

    logger.info("Đã hoàn thành chuẩn hóa dữ liệu chuỗi và phương thức thanh toán.")
    return df_normalized


def data_quality_report(df):
    """
    Tạo báo cáo tổng quan về chất lượng của tập dữ liệu.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.

    Trả về:
        dict: Báo cáo chất lượng với các trường:
            - 'total_rows' (int): Tổng số dòng.
            - 'total_columns' (int): Tổng số cột.
            - 'missing_values' (dict): Số lượng giá trị thiếu theo từng cột {cột: số_lượng}.
            - 'missing_percentage' (dict): Tỷ lệ % giá trị thiếu theo từng cột {cột: phần_trăm}.
            - 'duplicates' (int): Số lượng giao dịch bị trùng lặp.
            - 'outliers_count' (int): Số lượng giá trị ngoại lệ của cột 'Thành tiền'.
            - 'data_types' (dict): Kiểu dữ liệu tương ứng của từng cột {cột: kiểu_dữ_liệu}.
            - 'unique_values' (dict): Số lượng giá trị duy nhất của từng cột {cột: số_giá_trị_duy_nhất}.
    """
    if df is None or df.empty:
        logger.warning("DataFrame rỗng khi tạo báo cáo chất lượng dữ liệu.")
        return {
            'total_rows': 0,
            'total_columns': 0,
            'missing_values': {},
            'missing_percentage': {},
            'duplicates': 0,
            'outliers_count': 0,
            'data_types': {},
            'unique_values': {},
        }

    total_rows = int(len(df))
    total_columns = int(len(df.columns))

    missing_values = {str(col): int(val) for col, val in df.isna().sum().items()}
    missing_percentage = {
        col: round(float((count / total_rows) * 100), 2) if total_rows > 0 else 0.0
        for col, count in missing_values.items()
    }

    if 'Mã giao dịch' in df.columns:
        duplicates = int(df.duplicated(subset=['Mã giao dịch']).sum())
    else:
        duplicates = int(df.duplicated().sum())

    if 'Thành tiền' in df.columns:
        outliers_df, _, _ = detect_outliers_iqr(df, column='Thành tiền')
        outliers_count = int(len(outliers_df))
    else:
        outliers_count = 0

    data_types = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    unique_values = {str(col): int(df[col].nunique(dropna=False)) for col in df.columns}

    report = {
        'total_rows': total_rows,
        'total_columns': total_columns,
        'missing_values': missing_values,
        'missing_percentage': missing_percentage,
        'duplicates': duplicates,
        'outliers_count': outliers_count,
        'data_types': data_types,
        'unique_values': unique_values,
    }

    logger.info(
        f"Báo cáo chất lượng dữ liệu: {total_rows:,} dòng, {total_columns} cột, "
        f"{duplicates} trùng lặp, {outliers_count} ngoại lệ."
    )
    return report


def full_clean_pipeline(df):
    """
    Quy trình làm sạch dữ liệu toàn diện:
        1. Chuẩn hóa dữ liệu chuỗi và phương thức thanh toán (normalize_data).
        2. Xử lý giá trị bị khuyết thiếu (handle_missing_values).
        3. Loại bỏ các dòng giao dịch bị trùng lặp (drop_duplicates).
        4. Tạo báo cáo đánh giá chất lượng dữ liệu sau khi làm sạch (data_quality_report).

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu thô ban đầu.

    Trả về:
        tuple: (df_cleaned, report_dict)
            - df_cleaned (pd.DataFrame): DataFrame sạch sau toàn bộ quy trình.
            - report_dict (dict): Báo cáo chất lượng dữ liệu của DataFrame sau khi làm sạch.
    """
    if df is None or df.empty:
        logger.warning("DataFrame rỗng được truyền vào full_clean_pipeline.")
        empty_df = pd.DataFrame() if df is None else df.copy()
        return empty_df, data_quality_report(empty_df)

    logger.info("=== Bắt đầu quy trình làm sạch dữ liệu toàn diện (full_clean_pipeline) ===")

    # Bước 1: Chuẩn hóa dữ liệu
    logger.info("Bước 1: Chuẩn hóa chuỗi và định dạng phương thức thanh toán...")
    df_step1 = normalize_data(df)

    # Bước 2: Xử lý giá trị thiếu
    logger.info("Bước 2: Điền các giá trị bị khuyết thiếu...")
    df_step2, missing_report = handle_missing_values(df_step1)

    # Bước 3: Loại bỏ bản ghi trùng lặp
    logger.info("Bước 3: Loại bỏ các giao dịch trùng lặp...")
    if 'Mã giao dịch' in df_step2.columns:
        df_cleaned = df_step2.drop_duplicates(subset=['Mã giao dịch'], keep='first').copy()
    else:
        df_cleaned = df_step2.drop_duplicates(keep='first').copy()
    num_dropped = len(df_step2) - len(df_cleaned)
    logger.info(f"Đã loại bỏ {num_dropped} dòng trùng lặp.")

    # Bước 4: Tạo báo cáo chất lượng dữ liệu sau làm sạch
    logger.info("Bước 4: Sinh báo cáo chất lượng dữ liệu sau làm sạch...")
    report = data_quality_report(df_cleaned)

    logger.info(f"=== Hoàn thành quy trình làm sạch. Dữ liệu sạch có {len(df_cleaned):,} dòng. ===")
    return df_cleaned, report
