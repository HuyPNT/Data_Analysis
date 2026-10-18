import pandas as pd
import os
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def validate_csv(file_path):
    """
    Kiểm tra file CSV có đủ cột bắt buộc không.
    Trả về True nếu hợp lệ, False nếu không.
    """
    required_columns = [
        'Mã giao dịch', 'Ngày bán', 'Tên mặt hàng', 'Mã sản phẩm',
        'Số lượng', 'Đơn giá', 'Thành tiền', 'Mã nhân viên',
        'Phương thức thanh toán', 'Mã khách hàng'
    ]
    try:
        df = pd.read_csv(file_path)
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logging.error(f"File {file_path} thiếu cột: {missing_columns}")
            return False
        return True
    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra file {file_path}: {e}")
        return False

def load_single_file(file_path):
    """
    Tải và làm sạch một file CSV bán hàng.
    Trả về DataFrame hoặc DataFrame rỗng nếu lỗi.
    """
    if not os.path.exists(file_path):
        logging.warning(f"File không tồn tại: {file_path}")
        return pd.DataFrame()

    if not validate_csv(file_path):
        logging.warning(f"File không hợp lệ: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        df.columns = [col.strip() for col in df.columns]

        # Xóa hàng trống
        initial_rows = len(df)
        df.dropna(inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logging.info(f"Đã xóa {dropped_rows} hàng trống trong {file_path}")

        # Chuyển đổi kiểu dữ liệu
        df['Số lượng'] = pd.to_numeric(df['Số lượng'], errors='coerce')
        df['Đơn giá'] = pd.to_numeric(df['Đơn giá'], errors='coerce')
        df['Thành tiền'] = pd.to_numeric(df['Thành tiền'], errors='coerce')
        df['Ngày bán'] = pd.to_datetime(df['Ngày bán'], errors='coerce')

        # Xóa hàng lỗi
        initial_rows = len(df)
        df.dropna(subset=['Số lượng', 'Đơn giá', 'Thành tiền', 'Ngày bán', 'Tên mặt hàng'], inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logging.info(f"Đã xóa {dropped_rows} hàng lỗi trong {file_path}")

        if df.empty:
            logging.warning(f"File {file_path} không có dữ liệu hợp lệ sau khi làm sạch")
            return pd.DataFrame()

        # Tạo thêm cột thời gian phân tích
        df['Ngày'] = df['Ngày bán'].dt.date
        df['Thời gian'] = df['Ngày bán']

        logging.info(f"Tải thành công: {file_path} ({len(df)} dòng)")
        return df

    except Exception as e:
        logging.error(f"Lỗi khi nạp file {file_path}: {e}")
        return pd.DataFrame()

def load_multiple_files(file_paths):
    """
    Tải nhiều file CSV, trả về DataFrame gộp lại toàn bộ.
    """
    all_dataframes = []

    for path in file_paths:
        df = load_single_file(path)
        if not df.empty:
            all_dataframes.append(df)

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logging.info(f"Gộp {len(all_dataframes)} file thành {len(combined_df)} dòng")
        return combined_df
    else:
        logging.warning("Không có dữ liệu nào được tải.")
        return pd.DataFrame()

def load_sales_data(file_or_files):
    """
    Hàm tổng quát để tải dữ liệu: có thể là một file hoặc nhiều file.
    """
    if isinstance(file_or_files, str):
        return load_single_file(file_or_files)
    elif isinstance(file_or_files, (list, tuple)):
        return load_multiple_files(file_or_files)
    else:
        logging.error("Đầu vào không hợp lệ cho load_sales_data.")
        return pd.DataFrame()


def load_all_sales_data():
    """
    Tự động tải toàn bộ file CSV trong thư mục data/sales_data/.
    Không cần chọn file thủ công.
    """
    # Tìm thư mục sales_data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sales_dir = os.path.join(base_dir, 'sales_data')

    if not os.path.exists(sales_dir):
        logging.error(f"Thư mục không tồn tại: {sales_dir}")
        return pd.DataFrame()

    # Lấy tất cả file CSV
    csv_files = sorted([
        os.path.join(sales_dir, f)
        for f in os.listdir(sales_dir)
        if f.endswith('.csv')
    ])

    if not csv_files:
        logging.warning(f"Không tìm thấy file CSV trong {sales_dir}")
        return pd.DataFrame()

    logging.info(f"Tìm thấy {len(csv_files)} file CSV trong {sales_dir}")
    return load_multiple_files(csv_files)
