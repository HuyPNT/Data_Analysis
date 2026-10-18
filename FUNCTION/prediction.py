"""
Module dự báo doanh thu và phân tích xu hướng bán hàng bằng Machine Learning và Thống kê.
Cung cấp các hàm dự báo doanh thu, phân tích xu hướng mặt hàng, tính mùa vụ và phát hiện bất thường.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# Thiết lập logging
logger = logging.getLogger(__name__)

# Thử nghiệm import scikit-learn
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Thư viện scikit-learn chưa được cài đặt. Hệ thống sẽ sử dụng phương pháp trung bình thay thế.")


def predict_revenue_linear(df: pd.DataFrame, months_ahead: int = 3) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Dự báo doanh thu theo tháng sử dụng mô hình Hồi quy tuyến tính (Linear Regression).

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng chứa các cột 'Ngày bán' và 'Thành tiền'.
        months_ahead (int): Số tháng trong tương lai cần dự báo (mặc định là 3).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - predictions_df: DataFrame gồm các cột ['Tháng', 'Doanh thu thực tế', 'Doanh thu dự đoán'].
            - metrics_dict: Từ điển chứa các chỉ số đánh giá {'r2_score': float, 'mae': float, 'trend': str}.
    """
    empty_predictions = pd.DataFrame(columns=['Tháng', 'Doanh thu thực tế', 'Doanh thu dự đoán'])
    default_metrics = {'r2_score': 0.0, 'mae': 0.0, 'trend': 'Ổn định'}

    if df is None or df.empty:
        logger.warning("predict_revenue_linear: DataFrame rỗng.")
        return empty_predictions, default_metrics

    required_cols = {'Ngày bán', 'Thành tiền'}
    if not required_cols.issubset(df.columns):
        logger.error(f"predict_revenue_linear: Thiếu cột bắt buộc: {required_cols - set(df.columns)}")
        return empty_predictions, default_metrics

    try:
        work_df = df[['Ngày bán', 'Thành tiền']].copy()
        work_df['Ngày bán'] = pd.to_datetime(work_df['Ngày bán'], errors='coerce')
        work_df['Thành tiền'] = pd.to_numeric(work_df['Thành tiền'], errors='coerce')
        work_df = work_df.dropna(subset=['Ngày bán', 'Thành tiền'])

        if work_df.empty:
            logger.warning("predict_revenue_linear: Không có dữ liệu hợp lệ sau khi làm sạch.")
            return empty_predictions, default_metrics

        # Gom nhóm doanh thu theo tháng
        work_df['Period'] = work_df['Ngày bán'].dt.to_period('M')
        monthly_rev = work_df.groupby('Period')['Thành tiền'].sum().reset_index()
        monthly_rev = monthly_rev.sort_values('Period').reset_index(drop=True)
        n_months = len(monthly_rev)

        if n_months == 0:
            return empty_predictions, default_metrics

        y_hist = monthly_rev['Thành tiền'].values.astype(float)
        last_period = monthly_rev['Period'].iloc[-1]
        future_periods = [last_period + i for i in range(1, months_ahead + 1)]

        # Trường hợp chỉ có 1 tháng dữ liệu
        if n_months < 2:
            avg_val = float(y_hist[0])
            hist_periods_str = [p.strftime('%m-%Y') for p in monthly_rev['Period']]
            future_periods_str = [p.strftime('%m-%Y') for p in future_periods]

            hist_df = pd.DataFrame({
                'Tháng': hist_periods_str,
                'Doanh thu thực tế': y_hist,
                'Doanh thu dự đoán': [avg_val]
            })
            future_df = pd.DataFrame({
                'Tháng': future_periods_str,
                'Doanh thu thực tế': [np.nan] * months_ahead,
                'Doanh thu dự đoán': [avg_val] * months_ahead
            })
            predictions_df = pd.concat([hist_df, future_df], ignore_index=True)
            return predictions_df, default_metrics

        X_hist = np.arange(1, n_months + 1).reshape(-1, 1)

        # Huấn luyện mô hình Linear Regression hoặc sử dụng fallback
        if SKLEARN_AVAILABLE:
            try:
                model = LinearRegression()
                model.fit(X_hist, y_hist)
                y_pred_hist = model.predict(X_hist)

                X_future = np.arange(n_months + 1, n_months + 1 + months_ahead).reshape(-1, 1)
                y_pred_future = model.predict(X_future)
                y_pred_future = np.maximum(y_pred_future, 0.0)

                r2 = float(r2_score(y_hist, y_pred_hist))
                mae = float(mean_absolute_error(y_hist, y_pred_hist))
                slope = float(model.coef_[0])
            except Exception as e:
                logger.warning(f"Lỗi khi huấn luyện LinearRegression ({e}), chuyển sang trung bình đơn giản.")
                avg_rev = float(np.mean(y_hist))
                y_pred_hist = np.full(n_months, avg_rev)
                y_pred_future = np.full(months_ahead, avg_rev)
                r2 = 0.0
                mae = float(np.mean(np.abs(y_hist - avg_rev)))
                slope = 0.0
        else:
            avg_rev = float(np.mean(y_hist))
            y_pred_hist = np.full(n_months, avg_rev)
            y_pred_future = np.full(months_ahead, avg_rev)
            r2 = 0.0
            mae = float(np.mean(np.abs(y_hist - avg_rev)))
            slope = 0.0

        # Xác định xu hướng dựa trên độ dốc hồi quy tương đối
        mean_rev = float(np.mean(y_hist)) if len(y_hist) > 0 and np.mean(y_hist) != 0 else 1.0
        rel_slope = (slope / mean_rev) * 100.0

        if rel_slope >= 1.0:
            trend = 'Tăng'
        elif rel_slope <= -1.0:
            trend = 'Giảm'
        else:
            trend = 'Ổn định'

        # Tạo DataFrame kết quả
        hist_periods_str = [p.strftime('%m-%Y') for p in monthly_rev['Period']]
        future_periods_str = [p.strftime('%m-%Y') for p in future_periods]

        hist_df = pd.DataFrame({
            'Tháng': hist_periods_str,
            'Doanh thu thực tế': np.round(y_hist, 2),
            'Doanh thu dự đoán': np.round(y_pred_hist, 2)
        })

        future_df = pd.DataFrame({
            'Tháng': future_periods_str,
            'Doanh thu thực tế': [np.nan] * months_ahead,
            'Doanh thu dự đoán': np.round(y_pred_future, 2)
        })

        predictions_df = pd.concat([hist_df, future_df], ignore_index=True)
        metrics_dict = {
            'r2_score': round(r2, 4),
            'mae': round(mae, 2),
            'trend': trend
        }
        return predictions_df, metrics_dict

    except Exception as e:
        logger.error(f"Lỗi không xác định trong predict_revenue_linear: {e}")
        return empty_predictions, default_metrics


def product_trend_analysis(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Phân tích xu hướng tiêu thụ của từng mặt hàng theo độ dốc hồi quy tuyến tính.

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng chứa các cột 'Tên mặt hàng', 'Số lượng', 'Ngày bán'.
        top_n (int): Số lượng mặt hàng có biến động xu hướng mạnh nhất cần trả về.

    Returns:
        pd.DataFrame: DataFrame gồm các cột:
            ['Tên mặt hàng', 'Tổng SL', 'SL TB/tháng', 'Xu hướng', 'Độ thay đổi (%)']
    """
    empty_df = pd.DataFrame(columns=['Tên mặt hàng', 'Tổng SL', 'SL TB/tháng', 'Xu hướng', 'Độ thay đổi (%)'])

    if df is None or df.empty:
        logger.warning("product_trend_analysis: DataFrame rỗng.")
        return empty_df

    required_cols = {'Tên mặt hàng', 'Số lượng', 'Ngày bán'}
    if not required_cols.issubset(df.columns):
        logger.error(f"product_trend_analysis: Thiếu cột bắt buộc: {required_cols - set(df.columns)}")
        return empty_df

    try:
        work_df = df[['Tên mặt hàng', 'Số lượng', 'Ngày bán']].copy()
        work_df['Ngày bán'] = pd.to_datetime(work_df['Ngày bán'], errors='coerce')
        work_df['Số lượng'] = pd.to_numeric(work_df['Số lượng'], errors='coerce')
        work_df = work_df.dropna(subset=['Tên mặt hàng', 'Số lượng', 'Ngày bán'])

        if work_df.empty:
            return empty_df

        work_df['Period'] = work_df['Ngày bán'].dt.to_period('M')
        all_periods = sorted(work_df['Period'].unique())
        n_periods = len(all_periods)

        # Tạo bảng pivot: dòng là sản phẩm, cột là các tháng
        pivot_qty = work_df.pivot_table(
            index='Tên mặt hàng',
            columns='Period',
            values='Số lượng',
            aggfunc='sum',
            fill_value=0
        )

        if pivot_qty.empty:
            return empty_df

        total_qty = pivot_qty.sum(axis=1)
        mean_qty = pivot_qty.mean(axis=1)

        slopes = []
        change_pcts = []
        trends = []

        if n_periods >= 2:
            x = np.arange(1, n_periods + 1)
            x_mean = np.mean(x)
            x_diff = x - x_mean
            x_denom = np.sum(x_diff ** 2)

            for prod, row in pivot_qty.iterrows():
                y = row.values.astype(float)
                y_mean = np.mean(y)
                y_diff = y - y_mean

                slope = float(np.sum(x_diff * y_diff) / x_denom) if x_denom != 0 else 0.0
                slopes.append(slope)

                # Tính phần trăm thay đổi trên toàn chu kỳ
                pct_change = (slope * (n_periods - 1) / y_mean) * 100.0 if y_mean > 0 else 0.0
                change_pcts.append(pct_change)

                # Phân loại xu hướng
                if pct_change >= 20.0:
                    trend_label = 'Tăng mạnh'
                elif pct_change >= 5.0:
                    trend_label = 'Tăng nhẹ'
                elif pct_change > -5.0:
                    trend_label = 'Ổn định'
                elif pct_change > -20.0:
                    trend_label = 'Giảm nhẹ'
                else:
                    trend_label = 'Giảm mạnh'
                trends.append(trend_label)
        else:
            slopes = [0.0] * len(pivot_qty)
            change_pcts = [0.0] * len(pivot_qty)
            trends = ['Ổn định'] * len(pivot_qty)

        result_df = pd.DataFrame({
            'Tên mặt hàng': pivot_qty.index,
            'Tổng SL': total_qty.values,
            'SL TB/tháng': np.round(mean_qty.values, 2),
            'Xu hướng': trends,
            'Độ thay đổi (%)': np.round(change_pcts, 2),
            '_abs_slope': np.abs(slopes)
        })

        # Sắp xếp theo độ lớn của độ dốc (abs slope)
        result_df = result_df.sort_values(by=['_abs_slope', 'Tổng SL'], ascending=[False, False])
        result_df = result_df.drop(columns=['_abs_slope']).head(top_n).reset_index(drop=True)
        return result_df

    except Exception as e:
        logger.error(f"Lỗi trong product_trend_analysis: {e}")
        return empty_df


def seasonal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phân tích tính mùa vụ theo tháng trong năm (từ tháng 1 đến tháng 12).
    Xác định các tháng Cao điểm, Bình thường hoặc Thấp điểm dựa trên độ lệch chuẩn.

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng chứa các cột 'Ngày bán' và 'Thành tiền'.

    Returns:
        pd.DataFrame: DataFrame gồm các cột ['Tháng', 'Doanh thu TB', 'Loại'].
    """
    empty_df = pd.DataFrame(columns=['Tháng', 'Doanh thu TB', 'Loại'])

    if df is None or df.empty:
        logger.warning("seasonal_analysis: DataFrame rỗng.")
        return empty_df

    required_cols = {'Ngày bán', 'Thành tiền'}
    if not required_cols.issubset(df.columns):
        logger.error(f"seasonal_analysis: Thiếu cột bắt buộc: {required_cols - set(df.columns)}")
        return empty_df

    try:
        work_df = df[['Ngày bán', 'Thành tiền']].copy()
        work_df['Ngày bán'] = pd.to_datetime(work_df['Ngày bán'], errors='coerce')
        work_df['Thành tiền'] = pd.to_numeric(work_df['Thành tiền'], errors='coerce')
        work_df = work_df.dropna(subset=['Ngày bán', 'Thành tiền'])

        if work_df.empty:
            return empty_df

        work_df['Tháng'] = work_df['Ngày bán'].dt.month
        work_df['Năm'] = work_df['Ngày bán'].dt.year

        # Tính tổng doanh thu theo từng tháng của từng năm
        monthly_year_df = work_df.groupby(['Năm', 'Tháng'])['Thành tiền'].sum().reset_index()

        # Tính doanh thu trung bình cho từng tháng (1-12)
        seasonal = monthly_year_df.groupby('Tháng')['Thành tiền'].mean().reset_index()
        seasonal.columns = ['Tháng', 'Doanh thu TB']
        seasonal = seasonal.sort_values('Tháng').reset_index(drop=True)

        overall_mean = float(seasonal['Doanh thu TB'].mean())
        overall_std = float(seasonal['Doanh thu TB'].std(ddof=0))
        if np.isnan(overall_std):
            overall_std = 0.0

        upper_bound = overall_mean + 0.5 * overall_std
        lower_bound = overall_mean - 0.5 * overall_std

        def classify_month(val):
            if val > upper_bound:
                return 'Cao điểm'
            elif val < lower_bound:
                return 'Thấp điểm'
            else:
                return 'Bình thường'

        seasonal['Loại'] = seasonal['Doanh thu TB'].apply(classify_month)
        seasonal['Doanh thu TB'] = np.round(seasonal['Doanh thu TB'], 2)
        return seasonal

    except Exception as e:
        logger.error(f"Lỗi trong seasonal_analysis: {e}")
        return empty_df


def detect_anomalies(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Phát hiện các ngày có doanh thu bất thường dựa trên khoảng cách độ lệch chuẩn
    so với giá trị trung bình trượt (rolling mean, cửa sổ 7 ngày).

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng chứa các cột 'Ngày bán' và 'Thành tiền'.
        threshold (float): Ngưỡng số độ lệch chuẩn để xác định bất thường (mặc định là 2.0).

    Returns:
        pd.DataFrame: DataFrame gồm các cột:
            ['Ngày', 'Doanh thu', 'Trung bình', 'Độ lệch', 'Loại']
    """
    empty_df = pd.DataFrame(columns=['Ngày', 'Doanh thu', 'Trung bình', 'Độ lệch', 'Loại'])

    if df is None or df.empty:
        logger.warning("detect_anomalies: DataFrame rỗng.")
        return empty_df

    required_cols = {'Ngày bán', 'Thành tiền'}
    if not required_cols.issubset(df.columns):
        logger.error(f"detect_anomalies: Thiếu cột bắt buộc: {required_cols - set(df.columns)}")
        return empty_df

    try:
        work_df = df[['Ngày bán', 'Thành tiền']].copy()
        work_df['Ngày bán'] = pd.to_datetime(work_df['Ngày bán'], errors='coerce')
        work_df['Thành tiền'] = pd.to_numeric(work_df['Thành tiền'], errors='coerce')
        work_df = work_df.dropna(subset=['Ngày bán', 'Thành tiền'])

        if work_df.empty:
            return empty_df

        work_df['Ngày'] = work_df['Ngày bán'].dt.date
        daily_rev = work_df.groupby('Ngày')['Thành tiền'].sum().reset_index()
        daily_rev = daily_rev.sort_values('Ngày').reset_index(drop=True)
        daily_rev.columns = ['Ngày', 'Doanh thu']

        if len(daily_rev) == 0:
            return empty_df

        # Tính trung bình trượt và độ lệch chuẩn trượt cửa sổ 7 ngày
        daily_rev['Trung bình'] = daily_rev['Doanh thu'].rolling(window=7, min_periods=1).mean()
        daily_rev['Độ lệch chuẩn'] = daily_rev['Doanh thu'].rolling(window=7, min_periods=1).std().fillna(0.0)

        daily_rev['Độ lệch'] = daily_rev['Doanh thu'] - daily_rev['Trung bình']

        # Lọc các ngày vượt ngưỡng bất thường
        is_anomaly = (daily_rev['Độ lệch chuẩn'] > 0) & (daily_rev['Độ lệch'].abs() > threshold * daily_rev['Độ lệch chuẩn'])
        anomalies = daily_rev[is_anomaly].copy()

        if anomalies.empty:
            return empty_df

        anomalies['Loại'] = np.where(anomalies['Độ lệch'] > 0, 'Cao bất thường', 'Thấp bất thường')

        anomalies['Ngày'] = anomalies['Ngày'].astype(str)
        anomalies['Doanh thu'] = np.round(anomalies['Doanh thu'], 2)
        anomalies['Trung bình'] = np.round(anomalies['Trung bình'], 2)
        anomalies['Độ lệch'] = np.round(anomalies['Độ lệch'], 2)

        result_df = anomalies[['Ngày', 'Doanh thu', 'Trung bình', 'Độ lệch', 'Loại']].reset_index(drop=True)
        return result_df

    except Exception as e:
        logger.error(f"Lỗi trong detect_anomalies: {e}")
        return empty_df


def revenue_forecast_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Tạo báo cáo tóm tắt dự báo doanh thu tháng tới, tốc độ tăng trưởng,
    mức độ tin cậy và đề xuất hành động kinh doanh phù hợp.

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng.

    Returns:
        Dict[str, Any]: Từ điển chứa:
            - 'du_doan_thang_toi': Doanh thu dự đoán tháng tiếp theo (float).
            - 'tang_truong': Tỷ lệ tăng trưởng dự kiến (%) so với tháng gần nhất (float).
            - 'do_tin_cay': Mức độ tin cậy ('Cao', 'Trung bình', 'Thấp').
            - 'khuyen_nghi': Đề xuất hành động kinh doanh thực tế (str).
    """
    default_summary = {
        'du_doan_thang_toi': 0.0,
        'tang_truong': 0.0,
        'do_tin_cay': 'Không có dữ liệu',
        'khuyen_nghi': 'Cần nạp dữ liệu bán hàng để thực hiện dự báo doanh thu.'
    }

    if df is None or df.empty:
        logger.warning("revenue_forecast_summary: DataFrame rỗng.")
        return default_summary

    try:
        pred_df, metrics = predict_revenue_linear(df, months_ahead=1)
        if pred_df.empty:
            return default_summary

        # Tìm dòng tháng dự báo trong tương lai (Doanh thu thực tế là NaN)
        future_rows = pred_df[pred_df['Doanh thu thực tế'].isna()]
        if future_rows.empty:
            return default_summary

        next_month_pred = float(future_rows['Doanh thu dự đoán'].iloc[0])

        # Tìm doanh thu thực tế của tháng gần nhất
        hist_rows = pred_df[pred_df['Doanh thu thực tế'].notna()]
        if not hist_rows.empty:
            last_actual = float(hist_rows['Doanh thu thực tế'].iloc[-1])
            if last_actual > 0:
                growth_rate = ((next_month_pred - last_actual) / last_actual) * 100.0
            else:
                growth_rate = 0.0
        else:
            growth_rate = 0.0

        r2 = metrics.get('r2_score', 0.0)
        if r2 >= 0.7:
            confidence = 'Cao'
        elif r2 >= 0.4:
            confidence = 'Trung bình'
        else:
            confidence = 'Thấp'

        # Đưa ra khuyến nghị kinh doanh cụ thể
        if growth_rate > 5.0:
            recommendation = (
                f"Doanh thu tháng tới dự báo tăng trưởng tốt (+{growth_rate:.1f}%). "
                "Cửa hàng nên chủ động gia tăng lượng hàng tồn kho cho các mặt hàng chủ lực, "
                "bố trí thêm nhân sự trực ca và đẩy mạnh các chiến dịch quảng cáo kích cầu."
            )
        elif growth_rate < -5.0:
            recommendation = (
                f"Doanh thu tháng tới dự báo sụt giảm ({growth_rate:.1f}%). "
                "Cần kiểm tra kỹ các nhóm hàng bán chậm, áp dụng ngay các chương trình ưu đãi, combo "
                "tiết kiệm và xem xét tối ưu lại chi phí vận hành cửa hàng."
            )
        else:
            recommendation = (
                f"Doanh thu tháng tới dự kiến duy trì ổn định (biến động {growth_rate:+.1f}%). "
                "Doanh nghiệp nên duy trì mức tồn kho an toàn, nâng cao chất lượng dịch vụ "
                "và đẩy mạnh chăm sóc tệp khách hàng thân thiết để giữ vững phong độ."
            )

        return {
            'du_doan_thang_toi': round(next_month_pred, 2),
            'tang_truong': round(growth_rate, 2),
            'do_tin_cay': confidence,
            'khuyen_nghi': recommendation
        }

    except Exception as e:
        logger.error(f"Lỗi trong revenue_forecast_summary: {e}")
        return default_summary
